"""Mutual Fund Quant Ranker -- single-file Streamlit app.

Deploy: put this file + requirements.txt in a GitHub repo, point Streamlit Cloud at app.py.
Data: pick "Live fetch" in the sidebar (MFAPI + yfinance), or "Synthetic demo" to explore offline.
Edit CATEGORIES / BENCHMARKS / defaults in the CONFIG section below.
"""
from __future__ import annotations

import logging
import re
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import statsmodels.api as sm
import streamlit as st
from requests.adapters import HTTPAdapter
from scipy.signal import lfilter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from urllib3.util.retry import Retry

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


# ===========================================================================
# CONFIG
# ===========================================================================
# --------------------------------------------------------------------------- #
# Category definitions
# key   -> label used everywhere in the app
# match -> substring(s) searched (case-insensitive) in MFAPI's `scheme_category`
#          e.g. "Equity Scheme - Large Cap Fund"
# bench -> key into BENCHMARKS below
# --------------------------------------------------------------------------- #
CATEGORIES: dict[str, dict] = {
    "Large Cap": {"match": ["equity scheme - large cap"], "bench": "NIFTY100"},
    "Mid Cap": {"match": ["equity scheme - mid cap"], "bench": "NIFTYMID150"},
    "Small Cap": {"match": ["equity scheme - small cap"], "bench": "NIFTYSMALL250"},
    "Flexi Cap": {"match": ["equity scheme - flexi cap"], "bench": "NIFTY500"},
    "Large & Mid Cap": {"match": ["equity scheme - large & mid cap"], "bench": "NIFTYLMC250"},
    "Multi Cap": {"match": ["equity scheme - multi cap"], "bench": "NIFTY500"},
    "Value / Contra": {"match": ["equity scheme - value", "equity scheme - contra"], "bench": "NIFTY500"},
    "Balanced Advantage (BAF)": {
        "match": ["hybrid scheme - dynamic asset allocation", "balanced advantage"],
        "bench": "NIFTY50",
    },
}

# --------------------------------------------------------------------------- #
# Benchmarks. Two ways to supply data (CSV wins if both exist):
#   1. CSV in ./benchmarks/<KEY>.csv with columns  Date,Close   (recommended: use
#      NSE *Total Return* Index history -- the fund NAVs are dividend-inclusive)
#   2. yfinance ticker below. Yahoo only carries PRICE indices and its coverage of
#      NSE sub-indices is patchy -- run `python build_dataset.py --check-benchmarks`
#      and fix any ticker that comes back empty.
# --------------------------------------------------------------------------- #
BENCHMARKS: dict[str, dict] = {
    "NIFTY50": {"name": "Nifty 50", "ticker": "^NSEI"},
    "NIFTY500": {"name": "Nifty 500", "ticker": "^CRSLDX"},
    "NIFTY100": {"name": "Nifty 100", "ticker": "^CNX100"},
    "NIFTYMID150": {"name": "Nifty Midcap 150", "ticker": "NIFTYMIDCAP150.NS"},
    "NIFTYSMALL250": {"name": "Nifty Smallcap 250", "ticker": "NIFTYSMLCAP250.NS"},
    "NIFTYLMC250": {"name": "Nifty LargeMidcap 250", "ticker": "NIFTY_LARGEMID250.NS"},
}

# --------------------------------------------------------------------------- #
# Modelling defaults
# --------------------------------------------------------------------------- #
RISK_FREE_ANNUAL = 0.065      # flat proxy for the 91-day T-bill; swap for a series if you like
METRIC_WINDOW_MONTHS = 36     # trailing window used to compute each metric
MIN_HISTORY_MONTHS = 36       # a fund needs at least this much NAV history to enter the panel
FORWARD_HORIZON_MONTHS = 12   # prediction horizon
MIN_FUNDS_PER_CROSS_SECTION = 8  # drop (category, date) groups with fewer funds than this

# Features fed to the models (all computed in metrics.py)
FEATURES = [
    "sharpe", "sortino", "treynor", "info_ratio", "alpha", "beta",
    "up_capture", "down_capture", "capture_ratio",
    "volatility", "max_drawdown", "ret_3m", "ret_6m", "ret_12m",
]

# Features that go into the naive "normalise and rank" composite score (baseline).
# Signs: +1 higher is better, -1 lower is better.
COMPOSITE_SIGNS = {
    "sharpe": 1, "sortino": 1, "treynor": 1, "info_ratio": 1,
    "up_capture": 1, "down_capture": -1,
}


# ===========================================================================
# DATA: MFAPI + benchmarks
# ===========================================================================
log = logging.getLogger(__name__)

BASE_URL = "https://api.mfapi.in"
BENCH_DIR = Path(__file__).parent / "benchmarks"   # optional Date,Close CSV overrides (e.g. TRI)

# Names we never want (saves thousands of pointless API calls)
_EXCLUDE = re.compile(
    r"idcw|dividend|bonus|payout|reinvest|regular|"
    r"liquid|overnight|gilt|debt|bond|money market|fixed maturity|fmp|fixed term|"
    r"\betf\b|index|nifty|sensex|fund of fund|fof|arbitrage|"
    r"credit|duration|floater|banking & psu|ultra short|low duration|corporate|"
    r"segregated|series|interval|capital protection",
    re.I,
)


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET",))
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=32))
    s.headers.update({"User-Agent": "mf-quant-ranker/1.0"})
    return s


# --------------------------------------------------------------------------- #
# MFAPI
# --------------------------------------------------------------------------- #
def list_schemes(session: Optional[requests.Session] = None, page: int = 1000) -> pd.DataFrame:
    """All scheme codes + names on MFAPI (paged)."""
    s = session or _session()
    seen: dict[int, str] = {}
    offset = 0
    while True:
        r = s.get(f"{BASE_URL}/mf", params={"limit": page, "offset": offset}, timeout=60)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        new = 0
        for row in rows:
            code = int(row["schemeCode"])
            if code not in seen:
                seen[code] = row["schemeName"]
                new += 1
        if new == 0:                      # API ignored offset / we've seen everything
            break
        offset += page
    return pd.DataFrame({"scheme_code": list(seen), "scheme_name": list(seen.values())})


def direct_growth_candidates(schemes: pd.DataFrame) -> pd.DataFrame:
    """Keep only Direct-Growth plans and drop obviously irrelevant names."""
    n = schemes["scheme_name"]
    keep = n.str.contains("direct", case=False) & n.str.contains("growth", case=False)
    keep &= ~n.str.contains(_EXCLUDE)
    return schemes[keep].reset_index(drop=True)


def assign_category(scheme_category: str | None) -> Optional[str]:
    if not scheme_category:
        return None
    c = scheme_category.lower()
    for label, spec in CATEGORIES.items():
        if any(m in c for m in spec["match"]):
            return label
    return None


def _parse_nav(payload: dict) -> pd.Series:
    df = pd.DataFrame(payload.get("data", []))
    if df.empty:
        return pd.Series(dtype=float)
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna().query("nav > 0").drop_duplicates("date").sort_values("date")
    return df.set_index("date")["nav"]


def fetch_scheme(code: int, session: Optional[requests.Session] = None) -> tuple[dict, pd.Series]:
    """(meta, NAV series) for one scheme."""
    s = session or _session()
    r = s.get(f"{BASE_URL}/mf/{code}", timeout=60)
    r.raise_for_status()
    payload = r.json()
    return payload.get("meta", {}), _parse_nav(payload)


def build_universe(
    categories: Optional[list[str]] = None,
    min_history_months: int = MIN_HISTORY_MONTHS,
    max_workers: int = 8,
    progress: Optional[Callable[[int, int], None]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Discover, download and clean the fund universe.

    Returns
    -------
    meta : DataFrame  index=scheme_code; cols = scheme_name, fund_house, scheme_category, category, inception
    nav  : DataFrame  daily NAVs, index=date, columns=scheme_code
    """
    cats = set(categories or CATEGORIES)
    sess = _session()
    cand = direct_growth_candidates(list_schemes(sess))
    log.info("%d Direct-Growth candidates after name filter", len(cand))

    metas, navs = {}, {}
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fetch_scheme, int(c), sess): int(c) for c in cand["scheme_code"]}
        for fut in as_completed(futs):
            code = futs[fut]
            done += 1
            if progress:
                progress(done, len(futs))
            try:
                meta, s = fut.result()
            except Exception as e:  # network hiccup, malformed payload...
                log.warning("scheme %s failed: %s", code, e)
                continue
            cat = assign_category(meta.get("scheme_category"))
            if cat is None or cat not in cats or s.empty:
                continue
            if (s.index[-1] - s.index[0]).days < min_history_months * 30.4:
                continue
            if s.pct_change().abs().max() > 0.30:      # data error / unadjusted corporate action
                log.warning("dropping %s: >30%% one-day NAV jump", code)
                continue
            metas[code] = {
                "scheme_name": meta.get("scheme_name"),
                "fund_house": meta.get("fund_house"),
                "scheme_category": meta.get("scheme_category"),
                "category": cat,
                "inception": s.index[0],
                "last_nav_date": s.index[-1],
            }
            navs[code] = s

    meta_df = pd.DataFrame.from_dict(metas, orient="index")
    meta_df.index.name = "scheme_code"
    nav_df = pd.DataFrame(navs).sort_index()
    nav_df.columns.name = "scheme_code"
    return meta_df, nav_df


# --------------------------------------------------------------------------- #
# Benchmarks
# --------------------------------------------------------------------------- #
def load_benchmark(key: str, start: str = "2005-01-01") -> pd.Series:
    """Daily benchmark level series. CSV in ./benchmarks/ overrides yfinance."""
    spec = BENCHMARKS[key]
    csv = BENCH_DIR / f"{key}.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        df.columns = [c.strip().lower() for c in df.columns]
        dcol = next(c for c in df.columns if c.startswith("date"))
        vcol = "close" if "close" in df.columns else df.columns[1]
        df[dcol] = pd.to_datetime(df[dcol], dayfirst=True, errors="coerce")
        df[vcol] = pd.to_numeric(df[vcol].astype(str).str.replace(",", ""), errors="coerce")
        s = df.dropna(subset=[dcol, vcol]).drop_duplicates(dcol).set_index(dcol)[vcol].sort_index()
        s.name = key
        return s

    import yfinance as yf  # imported lazily so the Streamlit app doesn't need network for cached data

    raw = yf.download(spec["ticker"], start=start, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise ValueError(
            f"No data for {key} ({spec['ticker']}). Fix the ticker in BENCHMARKS or drop a "
            f"Date,Close CSV at benchmarks/{key}.csv"
        )
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.name = key
    return close


def load_benchmarks(keys: list[str], start: str = "2005-01-01") -> pd.DataFrame:
    return pd.concat({k: load_benchmark(k, start) for k in keys}, axis=1).sort_index()



# ===========================================================================
# METRICS
# ===========================================================================
FWD_HORIZONS = (3, 6, 12)
TRAIL_HORIZONS = (3, 6, 12)


# --------------------------------------------------------------------------- #
# single-window maths
# --------------------------------------------------------------------------- #
def _gmean(x: np.ndarray) -> float:
    """Geometric mean return per period."""
    return float(np.expm1(np.mean(np.log1p(x))))


def window_metrics(r: np.ndarray, b: np.ndarray, rf_p: float, ppy: int,
                   min_side_obs: int) -> dict:
    """Metrics for one window. r, b: fund / benchmark simple returns (no NaN)."""
    ex = r - rf_p
    bex = b - rf_p
    sq = np.sqrt(ppy)

    mean_ex = ex.mean() * ppy
    vol = r.std(ddof=1) * sq
    dd = np.sqrt(np.mean(np.minimum(ex, 0.0) ** 2)) * sq
    var_b = b.var(ddof=1)
    beta = np.cov(r, b, ddof=1)[0, 1] / var_b if var_b > 0 else np.nan
    alpha = mean_ex - beta * bex.mean() * ppy if np.isfinite(beta) else np.nan
    act = r - b
    te = act.std(ddof=1) * sq

    up, dn = b > 0, b < 0
    up_cap = dn_cap = np.nan
    if up.sum() >= min_side_obs:
        gb = _gmean(b[up])
        up_cap = _gmean(r[up]) / gb if gb != 0 else np.nan
    if dn.sum() >= min_side_obs:
        gb = _gmean(b[dn])
        dn_cap = _gmean(r[dn]) / gb if gb != 0 else np.nan

    cum = np.cumprod(1.0 + r)
    mdd = float(np.min(cum / np.maximum.accumulate(cum) - 1.0))

    return {
        "sharpe": mean_ex / vol if vol > 0 else np.nan,
        "sortino": mean_ex / dd if dd > 0 else np.nan,
        # Treynor explodes as beta -> 0 (e.g. defensive BAFs); refuse to compute below |beta| 0.2
        "treynor": mean_ex / beta if np.isfinite(beta) and abs(beta) >= 0.2 else np.nan,
        "info_ratio": act.mean() * ppy / te if te > 0 else np.nan,
        "alpha": alpha,
        "beta": beta,
        "up_capture": up_cap,
        "down_capture": dn_cap,
        "capture_ratio": up_cap / dn_cap if np.isfinite(up_cap) and np.isfinite(dn_cap) and dn_cap > 0 else np.nan,
        "volatility": vol,
        "max_drawdown": mdd,
    }


# --------------------------------------------------------------------------- #
# calendar helpers
# --------------------------------------------------------------------------- #
def month_end_dates(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last available date in each calendar month of `idx`."""
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.groupby(idx.to_period("M")).max().values)


# --------------------------------------------------------------------------- #
# panel builder
# --------------------------------------------------------------------------- #
def compute_panel_metrics(
    nav: pd.DataFrame,
    meta: pd.DataFrame,
    bench: pd.DataFrame,
    freq: str = "M",
    window_months: int = METRIC_WINDOW_MONTHS,
    rf_annual: float = RISK_FREE_ANNUAL,
) -> pd.DataFrame:
    """Build the (date, scheme_code) panel of features + forward outcomes.

    nav   : daily NAV, columns = scheme_code
    meta  : index = scheme_code, needs a `category` column
    bench : daily benchmark levels, columns = benchmark keys from BENCHMARKS
    freq  : "M" (monthly returns, 12 obs/yr) or "D" (daily returns, 252 obs/yr)
    """
    if freq not in ("M", "D"):
        raise ValueError("freq must be 'M' or 'D'")
    ppy = 12 if freq == "M" else 252
    rf_p = (1 + rf_annual) ** (1 / ppy) - 1
    min_obs = int(0.9 * window_months * ppy / 12)
    min_side = max(6, int(0.2 * window_months * ppy / 12))

    frames = []
    for cat, spec in CATEGORIES.items():
        codes = [c for c in meta.index[meta["category"] == cat] if c in nav.columns]
        bkey = spec["bench"]
        if not codes or bkey not in bench.columns:
            continue
        bser = bench[bkey].dropna()
        cal = bser.index
        # align fund NAVs to the benchmark trading calendar (carry NAV over short gaps only)
        nv = nav[codes].reindex(cal.union(nav.index)).ffill(limit=5).reindex(cal)

        me = month_end_dates(cal)
        lvl_m = nv.loc[me]
        b_m = bser.loc[me]

        if freq == "M":
            R, B, cal_r = lvl_m.pct_change(fill_method=None), b_m.pct_change(), me
        else:
            R, B, cal_r = nv.pct_change(fill_method=None), bser.pct_change(), cal
        Rv, Bv = R.to_numpy(), B.to_numpy()

        # trailing / forward point-to-point returns on the monthly grid
        fund_extra, bench_extra = {}, {}      # numpy arrays on the month-end grid
        for k in TRAIL_HORIZONS:
            fund_extra[f"ret_{k}m"] = (lvl_m / lvl_m.shift(k) - 1).to_numpy()
        for k in FWD_HORIZONS:
            fund_extra[f"fwd_ret_{k}m"] = (lvl_m.shift(-k) / lvl_m - 1).to_numpy()
            bench_extra[f"fwd_bench_{k}m"] = (b_m.shift(-k) / b_m - 1).to_numpy()
        me_pos = {d: i for i, d in enumerate(me)}
        pos_r = {d: i for i, d in enumerate(cal_r)}
        rows = []
        for d in me:
            p = pos_r[d]
            if freq == "M":
                s = p - window_months + 1
            else:
                s = cal_r.searchsorted(d - pd.DateOffset(months=window_months), side="right")
            if s < 1:
                continue
            bw = Bv[s:p + 1]
            for j, code in enumerate(codes):
                rw = Rv[s:p + 1, j]
                ok = np.isfinite(rw) & np.isfinite(bw)
                if ok.sum() < min_obs:
                    continue
                m = window_metrics(rw[ok], bw[ok], rf_p, ppy, min_side)
                m["date"], m["scheme_code"] = d, code
                q = me_pos[d]
                for name, arr in fund_extra.items():
                    m[name] = arr[q, j]
                for name, arr in bench_extra.items():
                    m[name] = arr[q]
                rows.append(m)
        if rows:
            f = pd.DataFrame(rows)
            f["category"] = cat
            frames.append(f)

    if not frames:
        raise ValueError("No fund had enough history to compute metrics -- check data / window length")
    out = pd.concat(frames, ignore_index=True)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.sort_values(["category", "date", "scheme_code"]).reset_index(drop=True)


# ===========================================================================
# PANEL PREP
# ===========================================================================
TARGETS = {
    "cat_rank": "Forward percentile rank within category (0-1)",
    "cat_excess": "Forward return minus category median",
    "bench_excess": "Forward return minus benchmark",
}


def add_targets(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = df.copy()
    fr, fb = f"fwd_ret_{horizon}m", f"fwd_bench_{horizon}m"
    g = df.groupby(["category", "date"])[fr]
    df["y_cat_rank"] = g.rank(pct=True)
    df["y_cat_excess"] = df[fr] - g.transform("median")
    df["y_bench_excess"] = df[fr] - df[fb]
    df["fwd_ret"] = df[fr]
    return df


def cross_sectional_standardise(
    df: pd.DataFrame,
    features: list[str],
    method: str = "rank",
    min_funds: int = MIN_FUNDS_PER_CROSS_SECTION,
) -> pd.DataFrame:
    """Standardise each feature within (category, date). Missing values -> 0 (cross-section neutral).

    method="rank"   : percentile rank centred on 0  (range -0.5..0.5) -- robust to outliers
    method="zscore" : z-score winsorised at +/-3
    Adds `<feature>_raw` copies so tables in the app can still show the real numbers.
    """
    df = df.copy()
    keys = ["category", "date"]
    n = df.groupby(keys)["sharpe"].transform("count")
    df = df[n >= min_funds].copy()

    for f in features:
        df[f + "_raw"] = df[f]
    g = df.groupby(keys)
    for f in features:
        if method == "rank":
            df[f] = g[f].rank(pct=True) - 0.5
        elif method == "zscore":
            mu, sd = g[f].transform("mean"), g[f].transform("std").replace(0, np.nan)
            df[f] = ((df[f] - mu) / sd).clip(-3, 3)
        else:
            raise ValueError(method)
    df[features] = df[features].fillna(0.0)
    return df.reset_index(drop=True)


def composite_score(df: pd.DataFrame, signs: dict[str, int] | None = None) -> pd.Series:
    """The 'normalise everything and average' ranking -- our benchmark to beat."""
    signs = signs or COMPOSITE_SIGNS
    return sum(df[f] * s for f, s in signs.items()) / len(signs)


def prepare(df: pd.DataFrame, horizon: int, features: list[str] | None = None,
            method: str = "rank") -> pd.DataFrame:
    features = features or FEATURES
    out = add_targets(df, horizon)
    out = cross_sectional_standardise(out, features, method)
    out["composite"] = composite_score(out)
    return out


# ===========================================================================
# MODELS + DIAGNOSTICS
# ===========================================================================
MODEL_NAMES = ["Ridge", "PCA + Ridge", "Gradient Boosting"]


def make_model(name: str, ridge_alpha: float = 50.0, pca_var: float = 0.90, seed: int = 0):
    if name == "Ridge":
        return Ridge(alpha=ridge_alpha)
    if name == "PCA + Ridge":
        return Pipeline([("pca", PCA(n_components=pca_var, svd_solver="full")),
                         ("ridge", Ridge(alpha=ridge_alpha))])
    if name == "Gradient Boosting":
        return HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.03, max_iter=250, min_samples_leaf=100,
            l2_regularization=5.0, random_state=seed,
        )
    raise ValueError(f"unknown model {name}")


# --------------------------------------------------------------------------- #
# Information-coefficient helpers
# --------------------------------------------------------------------------- #
def nw_tstat(x: pd.Series, lags: int) -> tuple[float, float]:
    """Mean and Newey-West t-stat of a time series (HAC handles overlapping forward windows)."""
    x = x.dropna()
    if len(x) < max(8, lags + 3):
        return float(x.mean()) if len(x) else np.nan, np.nan
    res = sm.OLS(x.to_numpy(), np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.params[0]), float(res.tvalues[0])


def univariate_ic(df: pd.DataFrame, features: list[str], target: str = "fwd_ret",
                  horizon: int = 12) -> pd.DataFrame:
    """Per-date Spearman rank correlation of each feature with the forward outcome.

    Correlations are computed within (category, date), averaged across categories per date, then
    summarised over time. Features in `df` must be the standardised ones.
    """
    tgt = df[target]
    rows = []
    d = df[["category", "date"] + features].copy()
    d["_t"] = tgt
    d = d.dropna(subset=["_t"])

    def _ic(g: pd.DataFrame) -> pd.Series:
        return g[features].rank().corrwith(g["_t"].rank())

    ic = d.groupby(["category", "date"]).apply(_ic, include_groups=False)
    ic_t = ic.groupby(level="date").mean().sort_index()
    for f in features:
        m, t = nw_tstat(ic_t[f], lags=max(horizon - 1, 1))
        s = ic_t[f].std()
        rows.append({"feature": f, "mean_IC": m, "IC_std": s, "IC_IR": m / s if s else np.nan,
                     "t_stat_NW": t, "pct_positive": float((ic_t[f] > 0).mean()), "n_dates": int(ic_t[f].notna().sum())})
    return pd.DataFrame(rows).set_index("feature").sort_values("mean_IC", ascending=False)


def fama_macbeth(df: pd.DataFrame, features: list[str], target: str = "y_cat_rank",
                 horizon: int = 12, min_obs: int = 15) -> pd.DataFrame:
    """Fama-MacBeth: OLS across funds for every (category, date), then average over time."""
    d = df.dropna(subset=[target])
    recs = []
    for (cat, date), g in d.groupby(["category", "date"]):
        if len(g) < max(min_obs, len(features) + 5):
            continue
        X = np.column_stack([np.ones(len(g)), g[features].to_numpy()])
        b, *_ = np.linalg.lstsq(X, g[target].to_numpy(), rcond=None)
        recs.append((date, *b))
    if not recs:
        return pd.DataFrame()
    coefs = pd.DataFrame(recs, columns=["date", "const"] + features).groupby("date").mean().sort_index()
    out = []
    for f in features:
        m, t = nw_tstat(coefs[f], lags=max(horizon - 1, 1))
        out.append({"feature": f, "coef": m, "t_stat_NW": t, "n_dates": len(coefs)})
    return pd.DataFrame(out).set_index("feature").sort_values("t_stat_NW", ascending=False)


# --------------------------------------------------------------------------- #
# Unsupervised: descriptive fund "archetypes" (NOT used for ranking)
# --------------------------------------------------------------------------- #
CLUSTER_FEATURES = ["beta", "volatility", "up_capture", "down_capture", "alpha", "max_drawdown"]


def cluster_snapshot(snapshot: pd.DataFrame, k: int = 4, seed: int = 0) -> pd.DataFrame:
    """K-means on raw risk-profile metrics for one date. Adds cluster, pc1, pc2, silhouette."""
    cols = [c + "_raw" if c + "_raw" in snapshot.columns else c for c in CLUSTER_FEATURES]
    X = snapshot[cols].replace([np.inf, -np.inf], np.nan)
    ok = X.notna().all(axis=1)
    out = snapshot.loc[ok].copy()
    if len(out) <= k:
        out["cluster"], out["pc1"], out["pc2"] = 0, 0.0, 0.0
        return out
    Z = StandardScaler().fit_transform(X.loc[ok])
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(Z)
    P = PCA(n_components=2).fit_transform(Z)
    out["cluster"] = km.labels_
    out["pc1"], out["pc2"] = P[:, 0], P[:, 1]
    out.attrs["silhouette"] = float(silhouette_score(Z, km.labels_))
    return out


# ===========================================================================
# WALK-FORWARD EVALUATION
# ===========================================================================
TARGET_COL = {"cat_rank": "y_cat_rank", "cat_excess": "y_cat_excess", "bench_excess": "y_bench_excess"}

BASELINES = {
    "Composite (normalise & rank)": "composite",
    "Trailing 12M return": "ret_12m",
    "Sharpe only": "sharpe",
}


def walk_forward(
    df: pd.DataFrame,
    features: list[str],
    horizon: int,
    model_names: list[str],
    target: str = "cat_rank",
    min_train_months: int = 60,
    step: int = 6,
    ridge_alpha: float = 50.0,
    progress: Optional[Callable[[float], None]] = None,
) -> pd.DataFrame:
    """Out-of-sample predictions for every model + the baselines.

    Returns long DataFrame: date, scheme_code, category, fwd_ret, target, and one column per
    model/baseline holding the *score* (higher = better predicted).
    """
    ycol = TARGET_COL[target]
    dates = np.sort(df["date"].unique())
    first_test = min_train_months + horizon
    if len(dates) <= first_test:
        raise ValueError(f"Only {len(dates)} monthly dates in the panel; need > {first_test}. "
                         "Reduce min_train_months / horizon or use more history.")
    starts = list(range(first_test, len(dates), step))
    preds = []
    for n, i0 in enumerate(starts):
        test_dates = dates[i0:i0 + step]
        train_dates = dates[: i0 - horizon + 1]                 # purge: forward window done by dates[i0]
        tr = df[df["date"].isin(train_dates)].dropna(subset=[ycol])
        te = df[df["date"].isin(test_dates)].copy()
        if len(tr) < 200 or te.empty:
            continue
        for m in model_names:
            mdl = make_model(m, ridge_alpha=ridge_alpha)
            mdl.fit(tr[features].to_numpy(), tr[ycol].to_numpy())
            te[m] = mdl.predict(te[features].to_numpy())
        for name, col in BASELINES.items():
            te[name] = te[col]
        preds.append(te)
        if progress:
            progress((n + 1) / len(starts))
    if not preds:
        raise ValueError("Walk-forward produced no folds -- not enough data")
    out = pd.concat(preds, ignore_index=True)
    keep = ["date", "scheme_code", "category", "fwd_ret", ycol] + model_names + list(BASELINES)
    out = out[keep].rename(columns={ycol: "target"})
    return out.dropna(subset=["fwd_ret"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
_EMPTY = {"ic": np.nan, "top_vs_avg": np.nan, "bottom_vs_avg": np.nan, "spread": np.nan, "top_beats_median": np.nan}


def _cs(preds: pd.DataFrame, col: str) -> pd.DataFrame:
    """Per-(category, date) stats for one score column, empty cross-sections removed."""
    st = preds.groupby(["category", "date"]).apply(_cross_section_stats, col=col, include_groups=False)
    return st.dropna(subset=["ic"])


def _cross_section_stats(g: pd.DataFrame, col: str, top_frac: float = 0.2) -> pd.Series:
    s = g[[col, "fwd_ret"]].dropna()
    n = len(s)
    if n < MIN_FUNDS_PER_CROSS_SECTION or s[col].nunique() < 3:
        return pd.Series(_EMPTY)
    ic = s[col].rank().corr(s["fwd_ret"].rank())
    k = max(3, int(round(n * top_frac)))
    order = s.sort_values(col, ascending=False)
    top, bot = order["fwd_ret"].iloc[:k].mean(), order["fwd_ret"].iloc[-k:].mean()
    avg = s["fwd_ret"].mean()
    return pd.Series({"ic": ic, "top_vs_avg": top - avg, "bottom_vs_avg": bot - avg,
                      "spread": top - bot, "top_beats_median": float(order["fwd_ret"].iloc[:k].mean() > s["fwd_ret"].median())})


def score_predictions(preds: pd.DataFrame, score_cols: list[str], horizon: int,
                      by_category: bool = False) -> pd.DataFrame:
    """Per-date cross-sectional stats -> summarised over time.

    mean_IC          average rank correlation between score and realised forward return
    IC_IR            mean_IC / std_IC  (consistency)
    t_stat_NW        Newey-West t-stat of mean_IC (lag = horizon-1 because forward windows overlap)
    top_vs_avg       avg fwd return of the top-20% ranked funds minus the category average
    spread           top-20% minus bottom-20%
    """
    rows = []
    for cat in (preds["category"].unique() if by_category else [None]):
        sub = preds if cat is None else preds[preds["category"] == cat]
        for col in score_cols:
            st = _cs(sub, col)
            if st.empty:
                continue
            per_date = st.groupby(level="date").mean().sort_index()
            m, t = nw_tstat(per_date["ic"], lags=max(horizon - 1, 1))
            sd = per_date["ic"].std()
            row = {"model": col, "mean_IC": m, "IC_IR": m / sd if sd else np.nan, "t_stat_NW": t,
                   "pct_dates_IC>0": float((per_date["ic"] > 0).mean()),
                   "top_vs_avg": per_date["top_vs_avg"].mean(), "bottom_vs_avg": per_date["bottom_vs_avg"].mean(),
                   "spread": per_date["spread"].mean(), "top_beats_median": per_date["top_beats_median"].mean(),
                   "n_dates": len(per_date)}
            if by_category:
                row["category"] = cat
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if by_category:
        return out.set_index(["category", "model"]).sort_index()
    return out.set_index("model").sort_values("mean_IC", ascending=False)


def ic_time_series(preds: pd.DataFrame, score_cols: list[str]) -> pd.DataFrame:
    """Per-date IC for each score (for the rolling-IC chart)."""
    out = {}
    for col in score_cols:
        st = _cs(preds, col)
        if not st.empty:
            out[col] = st["ic"].groupby(level="date").mean()
    return pd.DataFrame(out).sort_index()


# --------------------------------------------------------------------------- #
# Live ranking
# --------------------------------------------------------------------------- #
def rank_latest(
    df: pd.DataFrame,
    features: list[str],
    horizon: int,
    model_name: str,
    target: str = "cat_rank",
    ridge_alpha: float = 50.0,
    as_of: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Fit on every row whose outcome is already known, score the most recent cross-section."""
    ycol = TARGET_COL[target]
    dates = np.sort(df["date"].unique())
    as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(dates[-1])
    i = int(np.searchsorted(dates, np.datetime64(as_of)))
    train_dates = dates[: max(i - horizon + 1, 0)]
    tr = df[df["date"].isin(train_dates)].dropna(subset=[ycol])
    now = df[df["date"] == dates[i]].copy()
    mdl = make_model(model_name, ridge_alpha=ridge_alpha)
    mdl.fit(tr[features].to_numpy(), tr[ycol].to_numpy())
    now["score"] = mdl.predict(now[features].to_numpy())
    now["rank_model"] = now.groupby("category")["score"].rank(ascending=False, method="first").astype(int)
    now["rank_composite"] = now.groupby("category")["composite"].rank(ascending=False, method="first").astype(int)
    now["n_in_category"] = now.groupby("category")["score"].transform("count")
    now.attrs["train_rows"] = len(tr)
    now.attrs["train_end"] = pd.Timestamp(train_dates[-1]) if len(train_dates) else None
    return now.sort_values(["category", "rank_model"]).reset_index(drop=True)


# ===========================================================================
# SYNTHETIC DEMO DATA
# ===========================================================================
def demo_universe(funds_per_category: int = 25, start: str = "2008-01-01", end: str = "2026-08-31",
                  seed: int = 7, categories: list[str] | None = None):
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start, end)
    T = len(cal)
    cats = categories or list(CATEGORIES)

    # market factor with volatility regimes
    regime = np.repeat(rng.choice([0.7, 1.0, 1.8], size=T // 60 + 1, p=[0.4, 0.45, 0.15]), 60)[:T]
    mkt = rng.normal(0.13 / 252, 0.15 / np.sqrt(252), T) * regime

    bench = {}
    for key in BENCHMARKS:
        load = {"NIFTY50": 1.0, "NIFTY100": 1.0, "NIFTY500": 1.03, "NIFTYMID150": 1.2,
                "NIFTYSMALL250": 1.3, "NIFTYLMC250": 1.1}.get(key, 1.0)
        r = load * mkt + rng.normal(0.01 / 252, 0.05 / np.sqrt(252), T)
        bench[key] = 1000 * np.cumprod(1 + r)
    bench_df = pd.DataFrame(bench, index=cal)
    bench_ret = bench_df.pct_change().fillna(0.0)

    navs, metas = {}, {}
    code = 100000
    for cat in cats:
        bkey = CATEGORIES[cat]["bench"]
        base_beta = 0.55 if "BAF" in cat else 0.95
        for i in range(funds_per_category):
            code += 1
            beta = np.clip(rng.normal(base_beta, 0.08), 0.3, 1.3)
            alpha0 = rng.normal(0.0, 0.03)                       # annual, persistent
            # slowly drifting annual alpha (AR(1), stationary sd ~3%)
            drift = lfilter([1.0], [1.0, -0.998], rng.normal(0, 0.03 * np.sqrt(1 - 0.998 ** 2), T))  # AR(1), annual units
            te = rng.uniform(0.02, 0.06)
            r = beta * bench_ret[bkey].to_numpy() + (alpha0 + drift) / 252 + rng.normal(0, te / np.sqrt(252), T)
            incep = rng.integers(0, int(T * 0.55))
            death = T if rng.random() > 0.08 else rng.integers(int(T * 0.7), T)
            nav = 10 * np.cumprod(1 + r)
            s = pd.Series(nav, index=cal)
            s.iloc[:incep] = np.nan
            s.iloc[death:] = np.nan
            s = s.dropna()
            if len(s) < 800:
                continue
            navs[code] = s
            metas[code] = {"scheme_name": f"DEMO {cat} Fund {i + 1} - Direct Plan - Growth",
                           "fund_house": f"Demo AMC {i % 6 + 1}", "scheme_category": cat, "category": cat,
                           "inception": s.index[0], "last_nav_date": s.index[-1]}
    meta = pd.DataFrame.from_dict(metas, orient="index")
    meta.index.name = "scheme_code"
    nav = pd.DataFrame(navs).sort_index()
    nav.columns.name = "scheme_code"
    return meta, nav, bench_df


# ===========================================================================
# STREAMLIT APP
# ===========================================================================
METHODOLOGY = r"""
## How this works

**1. Data.** Fund NAVs and category labels come from [MFAPI](https://www.mfapi.in) (Direct-Growth plans only).
Benchmarks come from CSVs / yfinance. MFAPI has *no* AUM, expense-ratio or manager data.

**2. Features (at each month-end *t*, using only data up to *t*).** Over a trailing window (default 36 months):
Sharpe, Sortino, Treynor, information ratio, Jensen's alpha, beta, upside / downside capture, capture ratio,
volatility, max drawdown, and trailing 3/6/12-month returns.

**3. Target (what we predict).** A fund's *relative* outcome over the next *H* months: its percentile rank
inside its category (or excess over the category median / benchmark). We do **not** predict absolute returns --
those are dominated by the market, which fund-level metrics can't forecast.

**4. Cross-sectional scaling.** Every feature is ranked (or z-scored) *within each (category, date)*. This is the
ML equivalent of time fixed effects in a panel regression: it removes market-wide moves and keeps only
"fund A vs its peers on that date".

**5. Models.**
- *Ridge* -- linear panel model with shrinkage (metrics are highly collinear, plain OLS would be unstable).
- *PCA + Ridge* -- compress the collinear metrics into a few factors first.
- *Gradient boosting* -- shallow trees to test for non-linearities / interactions.
- *Composite (normalise & rank)*, *Trailing 12M return*, *Sharpe only* -- **baselines**. A model is only
  useful if it beats these out-of-sample.

**6. Validation.** Expanding-window walk-forward. Rows are purged so no training target overlaps the test period
(the *H*-month forward windows overlap, so ordinary train/test splits leak). Scores:
mean **rank IC** (correlation of score with realised forward return), IC IR, Newey-West t-stat, and the return of
the top-20% ranked funds vs the category average.

**7. K-means** is used only to describe fund archetypes -- it has no target, so it can't rank.

## Caveats you should not skip

- **Persistence is weak in real fund data.** Published work (e.g. Carhart 1997; S&P SPIVA India scorecards) finds
  little year-to-year persistence in *returns*; *risk* characteristics (volatility, beta, downside capture) persist
  far better. An out-of-sample IC of 0.05-0.10 would be a good result; anything much higher deserves suspicion.
- **Survivorship bias.** Funds that were merged or closed drop out of the sample, flattering history. Check how many
  dead schemes MFAPI actually returns for your universe.
- **Overlapping windows** mean the effective sample size is far smaller than the row count.
- **Benchmarks.** Yahoo has price indices only; fund NAVs include dividends, which biases alpha / IR upward by
  roughly the index dividend yield. Use Total Return Index CSVs where you can.
- **Category drift and reclassification**, style changes and fund-manager changes are not modelled.
- **Not investment advice.** This ranks funds statistically; it doesn't consider taxes, exit loads, suitability or
  liquidity.
"""

st.set_page_config(page_title="MF Quant Ranker", page_icon="📈", layout="wide")
DATA_DIR = Path(__file__).parent / "data"
HAVE_FILE = (DATA_DIR / "panel.parquet").exists() and (DATA_DIR / "meta.parquet").exists()


# ---------------------------- cached loaders ------------------------------ #
@st.cache_data(show_spinner="Loading saved panel…")
def load_saved(mtime: float):
    raw = pd.read_parquet(DATA_DIR / "panel.parquet")
    meta = pd.read_parquet(DATA_DIR / "meta.parquet").set_index("scheme_code")
    return raw, meta


@st.cache_data(show_spinner="Generating SYNTHETIC demo universe…")
def load_demo():
    meta, nav, bench = demo_universe(funds_per_category=25)
    return compute_panel_metrics(nav, meta, bench, freq="M"), meta


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def live_build(cats: tuple, freq: str, window: int, _progress=None):
    """Download NAVs + benchmarks and compute the metric panel (slow; cached for 24h)."""
    say = (lambda frac, txt: _progress.progress(min(frac, 1.0), text=txt)) if _progress else (lambda *a: None)
    meta, nav = build_universe(list(cats), progress=lambda i, n: say(0.7 * i / n, f"MFAPI: {i}/{n} schemes"))
    if meta.empty:
        raise ValueError("No funds matched the selected categories -- MFAPI may be unreachable or rate-limiting.")
    say(0.75, "Downloading benchmarks…")
    bench = load_benchmarks(sorted({CATEGORIES[c]["bench"] for c in cats}))
    say(0.85, "Computing rolling metrics…")
    raw = compute_panel_metrics(nav, meta, bench, freq=freq, window_months=window)
    return raw, meta


@st.cache_data(show_spinner="Standardising cross-sections…")
def get_prepared(_raw, src, cats, horizon, method):
    return prepare(_raw[_raw["category"].isin(cats)], horizon, FEATURES, method)


@st.cache_data(show_spinner="Running walk-forward backtest (can take a minute)…")
def get_backtest(_df, src, cats, horizon, method, features, model_names, target, min_train, step, alpha):
    return walk_forward(_df, list(features), horizon, list(model_names), target,
                        min_train_months=min_train, step=step, ridge_alpha=alpha)


@st.cache_data(show_spinner="Scoring the latest month…")
def get_latest(_df, src, cats, horizon, method, features, model_name, target, alpha):
    out = rank_latest(_df, list(features), horizon, model_name, target, alpha)
    return out, dict(out.attrs)


@st.cache_data(show_spinner=False)
def get_ic(_df, src, cats, horizon, method, features):
    return univariate_ic(_df, list(features), "fwd_ret", horizon)


@st.cache_data(show_spinner=False)
def get_fm(_df, src, cats, horizon, method, features, target):
    return fama_macbeth(_df, list(features), TARGET_COL[target], horizon)


# ------------------------------- sidebar ---------------------------------- #
st.title("📈 Mutual Fund Quant Ranker")

with st.sidebar:
    st.header("Data")
    choices = (["Saved file (data/panel.parquet)"] if HAVE_FILE else []) + \
              ["Live fetch (MFAPI + yfinance)", "Synthetic demo"]
    choice = st.radio("Source", choices, label_visibility="collapsed")
    use_demo = choice == "Synthetic demo"

    if choice.startswith("Saved"):
        raw, meta = load_saved((DATA_DIR / "panel.parquet").stat().st_mtime)
        src = f"saved-{(DATA_DIR / 'panel.parquet').stat().st_mtime:.0f}"
    elif use_demo:
        raw, meta = load_demo()
        src = "demo"
    else:
        fetch_cats = st.multiselect("Categories to fetch", list(CATEGORIES), default=list(CATEGORIES))
        freq = st.radio("Return frequency", ["M", "D"], horizontal=True,
                        format_func=lambda x: "Monthly" if x == "M" else "Daily")
        window = st.slider("Metric window (months)", 12, 60, METRIC_WINDOW_MONTHS, step=6)
        if st.button("Fetch / refresh data", type="primary"):
            st.session_state["live_args"] = (tuple(fetch_cats), freq, window)
        if "live_args" not in st.session_state:
            st.info("Choose categories and press **Fetch / refresh data**. First fetch takes several minutes "
                    "(~2,000 MFAPI calls); it is then cached for 24h.")
            st.stop()
        bar = st.progress(0.0, text="Starting…")
        try:
            raw, meta = live_build(*st.session_state["live_args"], _progress=bar)
        except Exception as e:
            bar.empty()
            st.error(f"Live fetch failed: {e}")
            st.stop()
        bar.empty()
        src = "live-" + "-".join(map(str, st.session_state["live_args"]))
        st.download_button("⬇ Save panel.parquet", raw.to_parquet(index=False), "panel.parquet",
                           help="Commit both files into a data/ folder in your repo to skip live fetching.")
        st.download_button("⬇ Save meta.parquet", meta.reset_index().to_parquet(index=False), "meta.parquet")

    st.header("Model")
    with st.form("controls"):
        all_cats = [c for c in CATEGORIES if c in raw["category"].unique()]
        cats = st.multiselect("Categories", all_cats, default=all_cats)
        horizon = st.selectbox("Forecast horizon (months)", [3, 6, 12], index=2)
        target = st.selectbox("Prediction target", list(TARGETS), index=0, format_func=lambda k: TARGETS[k])
        method = st.radio("Cross-sectional scaling", ["rank", "zscore"], horizontal=True,
                          help="Applied within each (category, date) group.")
        features = st.multiselect("Features", FEATURES, default=FEATURES)
        model_names = st.multiselect("Models", MODEL_NAMES, default=MODEL_NAMES)
        min_train = st.slider("Min training history (months)", 36, 96, 60, step=6)
        step = st.slider("Re-fit every N months", 3, 12, 6, step=3)
        alpha = st.number_input("Ridge alpha", 0.1, 1000.0, 50.0)
        st.form_submit_button("Apply / re-run", type="primary")

if use_demo:
    st.warning("**SYNTHETIC DEMO DATA** – randomly generated, not real funds. Nothing here is a finding.", icon="⚠️")

if not cats or not features or not model_names:
    st.info("Pick at least one category, feature and model, then press **Apply / re-run**.")
    st.stop()

cats_t, feats_t, mods_t = tuple(cats), tuple(features), tuple(model_names)
df = get_prepared(raw, src, cats_t, horizon, method)
if df.empty:
    st.error("No cross-sections with enough funds. Lower MIN_FUNDS_PER_CROSS_SECTION or add categories.")
    st.stop()

tab_rank, tab_bt, tab_sig, tab_clu, tab_about = st.tabs(
    ["🏆 Rankings", "🧪 Backtest", "🔬 What predicts?", "🧩 Archetypes", "ℹ️ Method & caveats"])

# ------------------------------- Rankings --------------------------------- #
with tab_rank:
    c1, c2 = st.columns(2)
    cat = c1.selectbox("Category", cats)
    rank_model = c2.selectbox("Ranking model", model_names)
    latest, attrs = get_latest(df, src, cats_t, horizon, method, feats_t, rank_model, target, alpha)
    asof = latest["date"].max()
    st.caption(f"Scores as of **{pd.Timestamp(asof).date()}**. Trained on rows whose {horizon}-month outcome was "
               f"already realised (last training date {pd.Timestamp(attrs['train_end']).date()}, "
               f"{attrs['train_rows']:,} fund-months). Score = predicted {TARGETS[target].lower()}.")
    t = latest[latest["category"] == cat].merge(
        meta[["scheme_name", "fund_house"]], left_on="scheme_code", right_index=True, how="left")
    show = t[["rank_model", "rank_composite", "scheme_name", "fund_house", "score",
              "sharpe_raw", "sortino_raw", "info_ratio_raw", "alpha_raw",
              "up_capture_raw", "down_capture_raw", "ret_12m_raw"]].rename(columns=lambda c: c.replace("_raw", ""))
    st.dataframe(
        show, hide_index=True, width="stretch", height=560,
        column_config={
            "rank_model": st.column_config.NumberColumn("Model rank", format="%d"),
            "rank_composite": st.column_config.NumberColumn("Composite rank", format="%d",
                                                            help="Naive normalise-and-average ranking"),
            "score": st.column_config.NumberColumn("Score", format="%.3f"),
            "alpha": st.column_config.NumberColumn("alpha", format="%.3f"),
            "ret_12m": st.column_config.NumberColumn("ret_12m", format="percent"),
            **{k: st.column_config.NumberColumn(k, format="%.2f")
               for k in ["sharpe", "sortino", "info_ratio", "up_capture", "down_capture"]},
        })
    st.download_button("Download ranking (CSV)", show.to_csv(index=False).encode(),
                       f"ranking_{cat.replace(' ', '_')}_{pd.Timestamp(asof).date()}.csv", "text/csv")
    st.caption("A high model rank is a statistical tilt, not a guarantee -- check the Backtest tab for how much "
               "signal there actually is.")

# ------------------------------- Backtest --------------------------------- #
with tab_bt:
    try:
        preds = get_backtest(df, src, cats_t, horizon, method, feats_t, mods_t, target, min_train, step, alpha)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    cols = list(model_names) + list(BASELINES)
    score = score_predictions(preds, cols, horizon)
    st.subheader("Out-of-sample scorecard")
    st.caption(f"Expanding-window walk-forward, purged by the {horizon}-month horizon, refit every {step} months. "
               f"{preds['date'].nunique()} test dates from {pd.Timestamp(preds['date'].min()).date()} to "
               f"{pd.Timestamp(preds['date'].max()).date()}.")
    st.dataframe(
        score, width="stretch",
        column_config={
            "mean_IC": st.column_config.NumberColumn("Mean rank IC", format="%.3f",
                                                     help="Avg Spearman corr between score and realised forward return"),
            "IC_IR": st.column_config.NumberColumn("IC IR", format="%.2f"),
            "t_stat_NW": st.column_config.NumberColumn("t (Newey-West)", format="%.2f"),
            "pct_dates_IC>0": st.column_config.NumberColumn("% dates IC>0", format="percent"),
            "top_vs_avg": st.column_config.NumberColumn("Top 20% vs avg", format="percent"),
            "bottom_vs_avg": st.column_config.NumberColumn("Bottom 20% vs avg", format="percent"),
            "spread": st.column_config.NumberColumn("Top-Bottom spread", format="percent"),
            "top_beats_median": st.column_config.NumberColumn("Top20% beats median", format="percent"),
        })

    ml = score.loc[[m for m in model_names if m in score.index]]
    if "Composite (normalise & rank)" in score.index and not ml.empty:
        best = ml["mean_IC"].idxmax()
        d = ml.loc[best, "mean_IC"] - score.loc["Composite (normalise & rank)", "mean_IC"]
        if d > 0.02 and ml.loc[best, "t_stat_NW"] > 2:
            st.success(f"**{best}** beats the plain normalise-and-rank composite by {d:+.3f} IC. "
                       "Worth keeping if it holds across categories and horizons (see below).")
        else:
            st.info(f"Best model (**{best}**) is only {d:+.3f} IC ahead of the plain composite -- no material "
                    "uplift. Prefer the simpler, more explainable composite unless this changes with more data.")

    st.subheader("Rolling 12-month average IC")
    roll = ic_time_series(preds, cols).rolling(12, min_periods=6).mean()
    fig = px.line(roll, labels={"value": "IC (12M avg)", "date": "", "variable": ""})
    fig.add_hline(y=0, line_dash="dot", line_color="grey")
    fig.update_layout(height=380, legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Top-20% minus category average (forward return)")
    figb = px.bar(score.reset_index(), x="model", y="top_vs_avg", labels={"top_vs_avg": "avg excess of top-20%", "model": ""})
    figb.update_yaxes(tickformat=".1%")
    figb.update_layout(height=320)
    st.plotly_chart(figb, width="stretch")

    with st.expander("Breakdown by category"):
        by_cat = score_predictions(preds, cols, horizon, by_category=True)
        st.dataframe(by_cat[["mean_IC", "IC_IR", "t_stat_NW", "top_vs_avg", "spread", "n_dates"]].round(3),
                     width="stretch")

# ---------------------------- What predicts? ------------------------------ #
with tab_sig:
    st.subheader("Which single metric carries information about the future?")
    ic = get_ic(df, src, cats_t, horizon, method, feats_t)
    figi = px.bar(ic.reset_index().sort_values("mean_IC"), x="mean_IC", y="feature", orientation="h",
                  color="t_stat_NW", color_continuous_scale="RdBu", color_continuous_midpoint=0,
                  labels={"mean_IC": f"mean rank IC vs forward {horizon}M return"})
    figi.update_layout(height=430)
    st.plotly_chart(figi, width="stretch")
    st.dataframe(ic.round(3), width="stretch")
    st.caption("Newey-West t-stats use lag = horizon-1 because consecutive months share most of their forward "
               "window. |t| > 2 is the usual bar; be sceptical of anything near it.")

    st.subheader("Fama-MacBeth: joint effect (controls for the other metrics)")
    st.dataframe(get_fm(df, src, cats_t, horizon, method, feats_t, target).round(3), width="stretch")
    st.caption("Sharpe, Sortino, Treynor and IR are highly correlated, so individual coefficients can be unstable "
               "or flip sign -- that's multicollinearity, and the reason PCA / Ridge exist.")

    st.subheader("Redundancy between metrics")
    corr = df[list(features)].corr(method="spearman")
    st.plotly_chart(px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", text_auto=".2f",
                              aspect="auto").update_layout(height=520), width="stretch")
    if len(features) >= 3:
        ev = PCA().fit(df[list(features)].to_numpy()).explained_variance_ratio_
        lab = [f"PC{i + 1}" for i in range(len(ev))]
        figp = go.Figure()
        figp.add_bar(x=lab, y=ev, name="explained")
        figp.add_scatter(x=lab, y=np.cumsum(ev), name="cumulative")
        figp.update_layout(height=320, yaxis_tickformat=".0%", title="PCA on the standardised metrics")
        st.plotly_chart(figp, width="stretch")

# ------------------------------ Archetypes -------------------------------- #
with tab_clu:
    st.subheader("Fund archetypes (K-means on risk profile) -- descriptive, not predictive")
    c1, c2 = st.columns(2)
    cc = c1.selectbox("Category ", cats, key="clu_cat")
    k = c2.slider("Clusters", 2, 6, 4)
    snap = df[(df["date"] == df["date"].max()) & (df["category"] == cc)].merge(
        meta[["scheme_name"]], left_on="scheme_code", right_index=True, how="left")
    cl = cluster_snapshot(snap, k)
    if cl.empty or "pc1" not in cl:
        st.info("Not enough funds for clustering in this category.")
    else:
        cl["cluster"] = cl["cluster"].astype(str)
        st.plotly_chart(px.scatter(cl, x="pc1", y="pc2", color="cluster", hover_name="scheme_name",
                                   hover_data={"pc1": False, "pc2": False}).update_layout(height=460),
                        width="stretch")
        prof = cl.groupby("cluster")[[c + "_raw" for c in CLUSTER_FEATURES]].mean()
        prof.columns = CLUSTER_FEATURES
        prof["n_funds"] = cl.groupby("cluster").size()
        st.dataframe(prof.round(3), width="stretch")
        st.caption(f"Silhouette = {cl.attrs.get('silhouette', float('nan')):.2f} (below ~0.25 means the "
                   "'archetypes' are mostly arbitrary slices of a continuum).")

with tab_about:
    st.markdown(METHODOLOGY)
