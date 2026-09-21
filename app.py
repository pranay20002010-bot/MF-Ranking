"""Mutual Fund Quant Ranker -- single-file Streamlit app.

Deploy: put this file + requirements.txt in a GitHub repo, point Streamlit Cloud at app.py.
Data: pick "Live fetch" in the sidebar (MFAPI + yfinance), or "Synthetic demo" to explore offline.
Optional: assets/vika_logo.png (used in the PDF header; falls back to a text wordmark if missing).
Edit CATEGORIES / BENCHMARKS / defaults in the CONFIG section below.
"""
from __future__ import annotations

import io
import logging
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import statsmodels.api as sm
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.platypus import (BaseDocTemplate, Flowable, Frame, HRFlowable, KeepTogether, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)
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
FWD_HORIZONS = (3, 6, 12, 24, 36)
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

MIN_TRAIN_ROWS, MIN_TRAIN_DATES = 200, 12

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
        raise ValueError(
            f"Not enough history for this setup: the data has {len(dates)} monthly dates, but the backtest needs "
            f"more than {first_test} ({min_train_months} months of training + {horizon} months of forecast). "
            "Try a shorter look-back window, a shorter forecast horizon, or less training history.")
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
        raise ValueError("Not enough history to test the ranking: no test period had enough completed past data. "
                         "Use a shorter forecast horizon or a shorter look-back window.")
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
    if not isinstance(st, pd.DataFrame) or "ic" not in st.columns:      # every cross-section was too small
        return pd.DataFrame(columns=list(_EMPTY))
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
    if len(tr) < MIN_TRAIN_ROWS or tr["date"].nunique() < MIN_TRAIN_DATES:
        raise ValueError(
            f"Not enough history to train the ranking method: only {tr['date'].nunique()} past months "
            f"({len(tr)} fund-months) have a completed {horizon}-month outcome, and at least {MIN_TRAIN_DATES} months / "
            f"{MIN_TRAIN_ROWS} fund-months are needed. Use a shorter forecast horizon or a shorter look-back window.")
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
# VIKA-BRANDED PDF REPORT
# ===========================================================================
VIKA_NAVY, VIKA_GOLD_LINE, VIKA_GOLD_LOGO = "#2A0284", "#FFBD59", "#F3D74F"
VIKA_GREEN, VIKA_MAROON = "#1E9A51", "#A81538"
VIKA_PURPLE, VIKA_CHART_GOLD = "#534191", "#F6BA47"
VIKA_CREAM, VIKA_LAVENDER = "#FFF9EF", "#EAE6F3"
VIKA_PASTELS = ["#99ACFF", "#FFC099", "#BFECAC", "#FFEB99", "#CEA8F0"]   # blue, orange, green, yellow, purple
VIKA_TEXT = "#262524"
LOGO_PATH = Path(__file__).parent / "assets" / "vika_logo.png"

# >>> EDIT THESE: client-facing wording should be approved by compliance before use <<<
DISCLAIMER_TEXT = (
    "Mutual fund investments are subject to market risks; please read all scheme related documents carefully. "
    "Past performance is not indicative of future results. The rankings in this document are produced by a "
    "statistical model using historical NAV data and are provided for information only. They are not a "
    "recommendation to buy, sell or hold any scheme, and do not consider your individual objectives, risk "
    "profile, taxes or exit loads."
)
CONTACT_LINE_1 = "For further details, contact us at: +91 - 8978880016  |  contact@vikawealth.in  |  www.vikawealth.in"
CONTACT_LINE_2 = ("AMFI-registered Mutual Fund Distributor \u2013 ARN 257866  |  AMFI-registered SIF Distributor \u2013 "
                  "ARN 257866  |  APMI PMS Distributor \u2013 APRN00458")


def _c(hex_: str):
    return colors.HexColor(hex_)


def _latin(txt) -> str:
    """Standard PDF fonts are Latin-1 only -- replace anything else so nothing renders as a black box."""
    return str(txt).encode("cp1252", "replace").decode("cp1252")


def _safe(txt) -> str:
    """_latin + escaping for reportlab Paragraph markup."""
    return _latin(txt).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class _BottomAnchored(Flowable):
    """Draws `inner` flush with the bottom of the frame (like the brand's floating contact box)."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def wrap(self, aw, ah):
        self._w, self._h = self.inner.wrap(aw, ah)
        self.width = aw
        # if it can't fit on this page, ask for a new one; otherwise swallow the leftover space
        self.height = ah if ah >= self._h else self._h + 1
        return self.width, self.height

    def draw(self):
        self.inner.drawOn(self.canv, 0, 0)

    def split(self, aw, ah):
        return []


def _bar_chart(items: list[tuple[str, float]], width: float, color: str = VIKA_PURPLE) -> Drawing:
    """Diverging horizontal bars (labels left, bars off a zero line, bold white value at inner end)."""
    row_h, gap, label_w = 20, 6, 190
    h = len(items) * (row_h + gap) + 6
    d = Drawing(width, h)
    vals = [v for _, v in items]
    lo, hi = min(0.0, min(vals)), max(0.0, max(vals))
    span = (hi - lo) or 1.0
    plot_w = width - label_w - 10
    x0 = label_w + (0 - lo) / span * plot_w
    for i, (lab, v) in enumerate(items):
        y = h - (i + 1) * (row_h + gap) + gap / 2
        x1 = label_w + (v - lo) / span * plot_w
        left, w = min(x0, x1), max(abs(x1 - x0), 1.5)
        d.add(String(label_w - 8, y + 6, _latin(lab), fontName="Helvetica", fontSize=8.5,
                     fillColor=_c(VIKA_TEXT), textAnchor="end"))
        d.add(Rect(left, y, w, row_h, fillColor=_c(color if v >= 0 else VIKA_MAROON), strokeColor=None))
        txt = f"{v * 100:+.1f}%"
        if w > 38:
            d.add(String(left + w - 4 if v >= 0 else left + 4, y + 6, txt, fontName="Helvetica-Bold", fontSize=8.5,
                         fillColor=colors.white, textAnchor="end" if v >= 0 else "start"))
        else:
            d.add(String(left + w + 4 if v >= 0 else left - 4, y + 6, txt, fontName="Helvetica-Bold", fontSize=8.5,
                         fillColor=_c(VIKA_TEXT), textAnchor="start" if v >= 0 else "end"))
    d.add(Line(x0, 0, x0, h, strokeColor=_c("#888888"), strokeWidth=0.6))
    return d


def build_pdf_report(s: dict) -> bytes:
    """Build the VIKA-branded PDF. `s` is a plain dict (see make_report_summary) so the content is easy to change."""
    W, H = letter
    LM = RM = 36
    HEADER_H, GOLD_H, FOOT_H, FOOT_GOLD = 58, 4, 18, 3.5
    CW = W - LM - RM

    base = ParagraphStyle("base", fontName="Helvetica", fontSize=9, leading=12.5, textColor=_c(VIKA_TEXT))
    small = ParagraphStyle("small", parent=base, fontSize=8, leading=10.5)
    h2 = ParagraphStyle("h2", fontName="Times-Bold", fontSize=14, leading=16, textColor=colors.black,
                        spaceBefore=10, spaceAfter=0)
    cell = ParagraphStyle("cell", parent=base, fontSize=8, leading=10)
    cell_b = ParagraphStyle("cell_b", parent=cell, fontName="Helvetica-Bold")
    th = ParagraphStyle("th", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
    box_lab = ParagraphStyle("box_lab", parent=base, fontSize=8, leading=10, alignment=TA_CENTER)
    box_val = ParagraphStyle("box_val", parent=base, fontName="Helvetica-Bold", fontSize=13, leading=16,
                             alignment=TA_CENTER)

    def heading(txt):
        return [Paragraph(_safe(txt), h2), HRFlowable(width="100%", thickness=2.5, color=_c(VIKA_GOLD_LINE),
                                                       spaceBefore=1, spaceAfter=6)]

    def cream(paras, style=base, pad=9):
        t = Table([[paras]], colWidths=[CW])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), _c(VIKA_CREAM)),
                               ("LEFTPADDING", (0, 0), (-1, -1), pad + 3), ("RIGHTPADDING", (0, 0), (-1, -1), pad + 3),
                               ("TOPPADDING", (0, 0), (-1, -1), pad), ("BOTTOMPADDING", (0, 0), (-1, -1), pad)]))
        return t

    story = []

    # ---- At a glance: pastel metric boxes (cycle blue, orange, green, yellow, purple)
    boxes = [("Funds ranked", str(s["n_funds"])), ("Looking ahead", f"{s['horizon']} months"),
             ("Data as of", pd.Timestamp(s["as_of"]).strftime("%d %b %Y")), ("Evidence from past tests", s["evidence"])]
    n = len(boxes)
    bt = Table([[[Paragraph(_safe(a), box_lab), Paragraph(_safe(b), box_val)] for a, b in boxes]],
               colWidths=[CW / n] * n)
    bt.setStyle(TableStyle(
        [("BACKGROUND", (i, 0), (i, 0), _c(VIKA_PASTELS[i % 5])) for i in range(n)] +
        [("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
         ("LINEAFTER", (0, 0), (-2, -1), 5, colors.white), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story += heading("At a glance") + [bt, Spacer(1, 6)]
    story += [cream([Paragraph(_safe(s["intro"]), base)])]

    # ---- Top funds table
    story += heading(f"Top-ranked funds: {s['category']}")
    hdr = ["Rank", "Fund", "Outlook", "What stands out", "Last 12M", "Return per risk*"]
    rows = [[Paragraph(h, th) for h in hdr]]
    for r in s["top_rows"]:
        rows.append([Paragraph(str(r["rank"]), cell_b), Paragraph(_safe(r["fund"]), cell),
                     Paragraph(_safe(r["outlook"]), cell), Paragraph(_safe(r["strengths"] or "-"), cell),
                     Paragraph(_safe(r["ret_12m"]), cell), Paragraph(_safe(r["sharpe"]), cell)])
    cw = [30, 168, 70, 160, 48, 60]
    tb = Table(rows, colWidths=cw, repeatRows=1)
    style = [("BACKGROUND", (0, 0), (-1, 0), _c(VIKA_NAVY)), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
             ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]
    for i in range(1, len(rows)):
        if i <= 3:
            style.append(("BACKGROUND", (0, i), (-1, i), _c(VIKA_PASTELS[2])))       # highlight the top 3
        elif i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), _c("#F7F6FB")))
    tb.setStyle(TableStyle(style))
    story += [tb, Spacer(1, 3),
              Paragraph("*Extra return over a bank-deposit-like rate for every unit of ups-and-downs (Sharpe ratio). "
                        "Higher is better. Ranks are within the category only.", small)]

    # ---- Did it work?
    sec = heading("Has this approach worked before?") + [Paragraph(_safe(s["backtest_intro"]), base), Spacer(1, 6)]
    if s.get("method_edges"):
        sec += [_bar_chart(s["method_edges"], CW), Spacer(1, 2), Paragraph(_safe(s["chart_caption"]), small), Spacer(1, 6)]
    story += [KeepTogether(sec)]          # heading, chart and caption stay on one page
    spaced = ParagraphStyle("spaced", parent=base, spaceAfter=3)
    story += [cream([Paragraph(_safe(t), spaced) for t in s["backtest_points"]])]

    # ---- How to read
    story += heading("How to read this")
    for k, v in s["glossary"]:
        story.append(Paragraph(f"<b>{_safe(k)}:</b> {_safe(v)}", small))
        story.append(Spacer(1, 2))

    # ---- disclaimer + contact (anchored to page bottom)
    story += [Spacer(1, 8), KeepTogether([cream([Paragraph(_safe(DISCLAIMER_TEXT),
                                                           ParagraphStyle("disc", parent=base, fontSize=7.5, leading=10))],
                                                pad=7)])]
    contact = Table([[[Paragraph(f"<b>{CONTACT_LINE_1}</b>", ParagraphStyle("c1", parent=base, fontSize=10, leading=13, alignment=TA_CENTER)),
                       Paragraph(CONTACT_LINE_2, ParagraphStyle("c2", parent=base, fontSize=7.5, leading=10, alignment=TA_CENTER))]]],
                    colWidths=[CW])
    contact.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), _c(VIKA_LAVENDER)),
                                 ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    story += [Spacer(1, 8), _BottomAnchored(contact)]

    title = f"Fund Ranking Insights | {s['category']} | {pd.Timestamp(s['as_of']).strftime('%B %Y')}"

    def draw_page(canv, doc):
        canv.saveState()
        # header: full-bleed navy band + gold trim
        canv.setFillColor(_c(VIKA_NAVY))
        canv.rect(0, H - HEADER_H, W, HEADER_H, stroke=0, fill=1)
        canv.setFillColor(_c(VIKA_GOLD_LINE))
        canv.rect(0, H - HEADER_H - GOLD_H, W, GOLD_H, stroke=0, fill=1)
        if LOGO_PATH.exists():
            iw, ih = ImageReader(str(LOGO_PATH)).getSize()
            lh = 40
            canv.drawImage(str(LOGO_PATH), LM, H - HEADER_H + (HEADER_H - lh) / 2, width=lh * iw / ih,
                           height=lh, mask="auto")
        else:                                                   # fallback wordmark if assets/vika_logo.png is missing
            canv.setFillColor(_c(VIKA_GOLD_LOGO))
            canv.setFont("Times-Bold", 20)
            canv.drawString(LM, H - HEADER_H + 26, "VIKA WEALTH")
            canv.setFillColor(colors.white)
            canv.setFont("Helvetica", 8)
            canv.drawString(LM, H - HEADER_H + 14, "Managing Risk & Return")
        canv.setFillColor(colors.white)
        canv.setFont("Helvetica-Bold", 11)
        canv.drawRightString(W - RM, H - HEADER_H / 2 - 4, _latin(title))
        # footer: gold trim + navy band
        canv.setFillColor(_c(VIKA_GOLD_LINE))
        canv.rect(0, FOOT_H, W, FOOT_GOLD, stroke=0, fill=1)
        canv.setFillColor(_c(VIKA_NAVY))
        canv.rect(0, 0, W, FOOT_H, stroke=0, fill=1)
        canv.restoreState()

    buf = io.BytesIO()
    doc = BaseDocTemplate(buf, pagesize=letter, leftMargin=LM, rightMargin=RM, title=title,
                          author="VIKA Wealth")
    top = H - HEADER_H - GOLD_H - 6
    bottom = FOOT_H + FOOT_GOLD + 8
    frame = Frame(LM, bottom, CW, top - bottom, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([PageTemplate(id="vika", frames=[frame], onPage=draw_page)])
    doc.build(story)
    return buf.getvalue()


# ===========================================================================
# STREAMLIT APP
# ===========================================================================
st.set_page_config(page_title="VIKA Fund Ranker", page_icon="📈", layout="wide")
DATA_DIR = Path(__file__).parent / "data"
HAVE_FILE = (DATA_DIR / "panel.parquet").exists() and (DATA_DIR / "meta.parquet").exists()

# =========================================================================== #
# Plain-English layer: friendly names, wording, verdict logic
# =========================================================================== #
GREEN_UI, MAROON_UI, BLUE_UI, AMBER_UI = "#1E9A51", "#A81538", "#2f6fdb", "#d98a00"

# key: (friendly name, what it means, "high"/"low"/None = which direction is 'better')
FEATURE_INFO = {
    "sharpe": ("Return for the risk taken (Sharpe)",
               "Extra return over a bank-deposit-like rate, per unit of ups-and-downs. Higher = more reward for the risk.", "high"),
    "sortino": ("Return for the downside risk (Sortino)",
                "Like Sharpe, but only counts falls as 'risk'. Higher is better.", "high"),
    "treynor": ("Return for the market risk (Treynor)",
                "Extra return per unit of sensitivity to the market. Higher is better.", "high"),
    "info_ratio": ("Steadiness in beating the benchmark (Information ratio)",
                   "How consistently the fund stays ahead of its benchmark. Higher = steadier outperformance.", "high"),
    "alpha": ("Value added beyond the market (Alpha)",
              "Annual return the fund earned over and above what its market exposure alone would explain.", "high"),
    "beta": ("Market sensitivity (Beta)",
             "1.0 = moves like the market; 0.8 = moves ~20% less. Neither good nor bad by itself.", None),
    "up_capture": ("Share of market gains captured (Upside capture)",
                   "In months the benchmark rose, how much of that rise the fund kept. 100% = kept pace.", "high"),
    "down_capture": ("Share of market falls suffered (Downside capture)",
                     "In months the benchmark fell, how much of that fall the fund took. Lower is better.", "low"),
    "capture_ratio": ("Gain-versus-pain balance (Capture ratio)",
                      "Upside capture divided by downside capture. Above 1 = gains more than it loses.", "high"),
    "volatility": ("Ups and downs (Volatility)", "How bumpy the ride was. Lower = smoother.", "low"),
    "max_drawdown": ("Worst fall (Max drawdown)",
                     "Biggest peak-to-bottom fall in the look-back period. A smaller fall is better.", "high"),
    "ret_3m": ("Last 3-month return", "Recent momentum.", "high"),
    "ret_6m": ("Last 6-month return", "Recent momentum.", "high"),
    "ret_12m": ("Last 12-month return", "Recent momentum.", "high"),
}
STRONG = {"sharpe": "Good return for the risk", "sortino": "Good return for its downside risk",
          "info_ratio": "Steadily beats its benchmark", "alpha": "Adds value beyond the market",
          "up_capture": "Captures market gains well", "down_capture": "Cushions market falls",
          "volatility": "Smoother ride than peers", "max_drawdown": "Shallower worst fall", "ret_12m": "Strong last 12 months"}
WEAK = {"sharpe": "Low return for the risk", "sortino": "Weak return for its downside risk",
        "info_ratio": "Inconsistent versus benchmark", "alpha": "Adds little beyond the market",
        "up_capture": "Misses market gains", "down_capture": "Falls hard in down markets",
        "volatility": "Bumpier ride than peers", "max_drawdown": "Deeper worst fall", "ret_12m": "Weak last 12 months"}
TRAITS = list(STRONG)

MODEL_LABELS = {"Ridge": "Blended score (linear)", "PCA + Ridge": "Blended score, overlap removed",
                "Gradient Boosting": "Machine-learning model"}
BASELINE_LABELS = {"Composite (normalise & rank)": "Simple average of the metrics",
                   "Trailing 12M return": "Just the last 12-month return", "Sharpe only": "Just the Sharpe ratio"}
TARGET_LABELS = {"cat_rank": "Beat category peers (recommended)", "cat_excess": "Return above the category median",
                 "bench_excess": "Return above the benchmark"}
SIMPLE = "Composite (normalise & rank)"


def pretty(name: str) -> str:
    return MODEL_LABELS.get(name) or BASELINE_LABELS.get(name) or name


def evidence_level(ic: float, t: float) -> tuple[str, str]:
    """(label, colour): how much to trust a ranking method, from its out-of-sample record."""
    if pd.isna(ic) or pd.isna(t):
        return "Not enough data", "#888888"
    if t >= 3 and ic >= 0.08:
        return "Strong", GREEN_UI
    if t >= 2 and ic >= 0.04:
        return "Moderate", BLUE_UI
    if t >= 1 and ic >= 0.02:
        return "Weak", AMBER_UI
    return "None found", MAROON_UI


def top_bucket_series(preds: pd.DataFrame, col: str) -> pd.Series:
    """Per-date edge of the top-ranked 20% over the category average."""
    return _cs(preds, col)["top_vs_avg"].groupby(level="date").mean().sort_index()


def add_plain_columns(latest: pd.DataFrame) -> pd.DataFrame:
    """Outlook band + 'what stands out' / 'watch-outs' text for each fund (vs its category peers)."""
    d = latest.copy()
    n = d.groupby("category")["score"].transform("count")
    d["pct_rank"] = 1 - (d["rank_model"] - 1) / n
    d["outlook"] = pd.cut(d["pct_rank"], [-1, 0.25, 0.5, 0.75, 1.01],
                          labels=["Bottom quarter", "Below average", "Above average", "Top quarter"]).astype(str)
    good = pd.DataFrame(index=d.index)
    for f in TRAITS:
        p = d.groupby("category")[f + "_raw"].rank(pct=True)
        good[f] = 1 - p + (1 / n) if FEATURE_INFO[f][2] == "low" else p
    strengths, watch = [], []
    for i in d.index:
        g = good.loc[i].dropna()
        s = g[g >= 0.8].sort_values(ascending=False).index[:2]
        w = g[g <= 0.2].sort_values().index[:2]
        strengths.append("; ".join(STRONG[k] for k in s))
        watch.append("; ".join(WEAK[k] for k in w))
    d["strengths"], d["watchouts"] = strengths, watch
    return d


def composite_fallback(df: pd.DataFrame) -> pd.DataFrame:
    """Ranking that needs no training: the simple average of the metrics, latest month only."""
    d = df[df["date"] == df["date"].max()].copy()
    d["score"] = d["composite"]
    d["rank_model"] = d.groupby("category")["score"].rank(ascending=False, method="first").astype(int)
    d["rank_composite"] = d["rank_model"]
    d["n_in_category"] = d.groupby("category")["score"].transform("count")
    return d.sort_values(["category", "rank_model"]).reset_index(drop=True)


def method_summary(score: pd.DataFrame, method: str, horizon: int) -> dict:
    r = score.loc[method]
    label, colour = evidence_level(r["mean_IC"], r["t_stat_NW"])
    edge, hit, bot = r["top_vs_avg"], r["top_beats_median"], r["bottom_vs_avg"]
    simple_edge = score.loc[SIMPLE, "top_vs_avg"] if SIMPLE in score.index else np.nan
    pts = [
        f"In past test periods, the funds ranked in the top 20% of their category went on to "
        f"{'beat' if edge >= 0 else 'trail'} the category average by {abs(edge):.1%} over the next {horizon} months, on average.",
        f"They did better than the typical (median) fund in {hit:.0%} of test periods.",
        f"The bottom-ranked 20% {'lagged' if bot <= 0 else 'beat'} the average by {abs(bot):.1%}.",
    ]
    vs_simple = None
    if method != SIMPLE and pd.notna(simple_edge):
        diff = edge - simple_edge
        vs_simple = ("Compared with simply averaging the metrics, this method was "
                     + ("clearly better." if diff > 0.005 else
                        "not meaningfully better, so the simpler ranking is just as good." if diff > -0.005 else "worse."))
        pts.append(vs_simple)
    pts.append("These are averages of past results. Individual funds and periods varied a lot, and the past may not repeat.")
    return {"label": label, "colour": colour, "edge": edge, "hit": hit, "bottom": bot, "points": pts,
            "vs_simple": vs_simple, "ic": r["mean_IC"], "t": r["t_stat_NW"]}


GLOSSARY = [
    ("Outlook", "Where the fund sits among its category peers on our ranking: Top quarter, Above average, Below average or "
                "Bottom quarter."),
    ("What stands out", "Areas where the fund is in the best fifth of its category (e.g. cushions market falls)."),
    ("Return per risk", "The Sharpe ratio: extra return over a bank-deposit-like rate per unit of ups-and-downs. Higher is better."),
    ("Evidence from past tests", "Whether, in past periods, funds ranked highly by this method actually went on to do better. "
                                 "Strong / Moderate / Weak / None found."),
]


def make_report_summary(cat, asof, horizon, table, n_funds, ms, score, cols, win_months, n_dates) -> dict:
    """Everything the PDF shows, as a plain dict. Edit THIS (and pdf layout above) to change PDF content."""
    top = table.sort_values("rank_model").head(10)
    when = f"the last {win_months} months" if win_months else "recent years"
    return {
        "category": cat, "as_of": asof, "horizon": horizon, "n_funds": n_funds, "evidence": ms["label"],
        "intro": (f"This note compares {n_funds} Direct-Growth {cat} mutual funds against each other using how they behaved "
                  f"over {when}: the return delivered for the risk taken, how well they held up in falling markets, and how "
                  f"steadily they beat their benchmark. The ranking aims to tilt the odds towards funds that have tended to keep "
                  f"doing better than their peers over the next {horizon} months. It is not a forecast for any single fund."),
        "top_rows": [{"rank": int(r.rank_model), "fund": r.scheme_name, "outlook": r.outlook, "strengths": r.strengths,
                      "ret_12m": f"{r.ret_12m_raw:.1%}" if pd.notna(r.ret_12m_raw) else "-",
                      "sharpe": f"{r.sharpe_raw:.2f}" if pd.notna(r.sharpe_raw) else "-"} for r in top.itertuples()],
        "backtest_intro": (f"We tested the method the way it would have been used in real life: at each past month-end, using only "
                           f"information available at that time, then checking what happened over the following {horizon} months."),
        "method_edges": [(pretty(m), float(score.loc[m, "top_vs_avg"])) for m in cols if m in score.index],
        "chart_caption": (f"Average gain of the top-ranked 20% of funds over their category average ({horizon}-month outcome, "
                          f"{n_dates} past test dates)."),
        "backtest_points": ms["points"], "glossary": GLOSSARY,
    }


GUIDE = """
### What this app does, in plain words
It looks at how each mutual fund behaved over the last few years (return, risk, how it held up in market falls) and asks
one question: **do funds that look good on these numbers tend to keep doing better than their peers?**
We test that honestly on past data, and only then use it to rank funds today.

### How to read the results
1. **Rankings tab**: funds ranked inside their own category (never across categories). *Outlook* groups them into four bands.
   *What stands out* / *Watch-outs* explain the ranking in words. The button at the bottom creates the VIKA-branded PDF.
2. **Has it worked before?**: the honesty check. If the top-ranked funds in the past did not beat the rest, the app says so.
3. **Which metrics matter?**: which of the familiar numbers (Sharpe, downside capture, ...) actually pointed to better
   future funds, and which did nothing.
4. **Fund personalities**: groups funds by how they behave (aggressive, defensive, ...). Descriptive only; it does not rank.

### The important caveats
- **Past patterns weaken.** Research on mutual funds generally finds that past *returns* are a weak guide to future returns.
  Risk traits (how much a fund falls in bad markets, how bumpy it is) tend to persist better. Treat rankings as one input, not a verdict.
- **A modest edge is realistic.** If the second tab shows "Weak" or "None found", believe it.
- **Survivorship.** Funds that were merged or shut down may be missing from the data, which flatters history.
- **Benchmarks.** Yahoo Finance gives price indices without dividends, which slightly flatters fund-vs-benchmark numbers.
  Where possible, put Total Return Index CSVs (columns Date, Close) in a `benchmarks/` folder named after the benchmark key.
- Not investment advice. Taxes, exit loads, suitability and liquidity are not considered.

### Glossary
| Term | Meaning |
|---|---|
| Category peers | Funds in the same SEBI category (e.g. Large Cap). Rankings are only ever within a category. |
| Look-back window | How many past months are used to compute each fund's metrics. |
| Forecast horizon | How far ahead we check whether the ranking worked (e.g. next 12 months). |
| Top 20% edge | Average extra return of the funds our method ranked in the top fifth, versus the category average. |
| Hit rate | Share of test periods in which those top-ranked funds beat the typical fund. |
| Backtest | Replaying history month by month using only what was known at the time. No peeking. |
"""


# =========================================================================== #
# Cached loaders / computations
# =========================================================================== #
@st.cache_data(show_spinner="Loading saved panel…")
def load_saved(mtime: float):
    raw = pd.read_parquet(DATA_DIR / "panel.parquet")
    meta = pd.read_parquet(DATA_DIR / "meta.parquet").set_index("scheme_code")
    return raw, meta


@st.cache_data(show_spinner="Generating SYNTHETIC demo universe…")
def load_demo():
    meta, nav, bench = demo_universe(funds_per_category=25)
    return compute_panel_metrics(nav, meta, bench, freq="M"), meta


@st.cache_resource
def _live_store() -> dict:
    """Shared in-memory store for fetched data. (Deliberately NOT st.cache_data: the fetch reports progress through a
    Streamlit element, which cache_data refuses to replay.)"""
    return {}


def live_build(cats: tuple, freq: str, window: int, progress=None):
    """Download NAVs + benchmarks and compute the metric panel (slow; result kept for 24h)."""
    key, store = (cats, freq, window), _live_store()
    hit = store.get(key)
    if hit and time.time() - hit[0] < 24 * 3600:
        return hit[1], hit[2]
    say = (lambda frac, txt: progress.progress(min(frac, 1.0), text=txt)) if progress else (lambda *a: None)
    meta, nav = build_universe(list(cats), min_history_months=window,
                               progress=lambda i, n: say(0.7 * i / n, f"Fund data: {i}/{n} schemes"))
    if meta.empty:
        raise ValueError("No funds matched. MFAPI may be unreachable/rate-limiting, or no fund has "
                         f"{window} months of history -- try a shorter look-back window.")
    say(0.75, "Downloading benchmark indices…")
    bench = load_benchmarks(sorted({CATEGORIES[c]["bench"] for c in cats}))
    say(0.85, "Computing rolling metrics…")
    raw = compute_panel_metrics(nav, meta, bench, freq=freq, window_months=window)
    store[key] = (time.time(), raw, meta)
    while len(store) > 2:
        store.pop(next(iter(store)))
    return raw, meta


@st.cache_data(show_spinner="Preparing the data…")
def get_prepared(_raw, src, cats, horizon, method):
    return prepare(_raw[_raw["category"].isin(cats)], horizon, FEATURES, method)


@st.cache_data(show_spinner="Replaying history to test the rankings (this can take a minute)…")
def get_backtest(_df, src, cats, horizon, method, features, model_names, target, min_train, step, alpha):
    return walk_forward(_df, list(features), horizon, list(model_names), target,
                        min_train_months=min_train, step=step, ridge_alpha=alpha)


@st.cache_data(show_spinner="Scoring funds…")
def get_latest(_df, src, cats, horizon, method, features, model_name, target, alpha):
    out = rank_latest(_df, list(features), horizon, model_name, target, alpha)
    return out, dict(out.attrs)


@st.cache_data(show_spinner=False)
def get_ic(_df, src, cats, horizon, method, features):
    return univariate_ic(_df, list(features), "fwd_ret", horizon)


@st.cache_data(show_spinner=False)
def get_fm(_df, src, cats, horizon, method, features, target):
    return fama_macbeth(_df, list(features), TARGET_COL[target], horizon)


# =========================================================================== #
# Sidebar
# =========================================================================== #
st.title("📈 VIKA Fund Ranker")
st.caption("Ranks mutual funds within their category, and shows honestly whether that ranking has worked in the past.")

win_months = None
with st.sidebar:
    st.header("1. Data")
    choices = (["Saved file (data/panel.parquet)"] if HAVE_FILE else []) + \
              ["Live fetch (MFAPI + yfinance)", "Synthetic demo"]
    choice = st.radio("Source", choices, label_visibility="collapsed")
    use_demo = choice == "Synthetic demo"

    if choice.startswith("Saved"):
        raw, meta = load_saved((DATA_DIR / "panel.parquet").stat().st_mtime)
        src = f"saved-{(DATA_DIR / 'panel.parquet').stat().st_mtime:.0f}"
    elif use_demo:
        raw, meta = load_demo()
        src, win_months = "demo", METRIC_WINDOW_MONTHS
    else:
        fetch_cats = st.multiselect("Categories to fetch", list(CATEGORIES), default=list(CATEGORIES))
        freq = st.radio("Return frequency", ["M", "D"], horizontal=True,
                        format_func=lambda x: "Monthly" if x == "M" else "Daily")
        window = st.slider("Look-back window for each fund's metrics (months)", 12, 120, METRIC_WINDOW_MONTHS, step=6,
                           help="Up to 10 years. Longer = steadier numbers, but only funds with that much history qualify.")
        if st.button("Fetch / refresh data", type="primary"):
            st.session_state["live_args"] = (tuple(fetch_cats), freq, window)
        if "live_args" not in st.session_state:
            st.info("Choose categories and press **Fetch / refresh data**. The first fetch takes several minutes "
                    "(~2,000 MFAPI calls) and is then kept for 24 hours.")
            st.stop()
        bar = st.progress(0.0, text="Starting…")
        try:
            raw, meta = live_build(*st.session_state["live_args"], progress=bar)
        except Exception as e:
            bar.empty()
            st.error(f"Live fetch failed: {e}")
            st.stop()
        bar.empty()
        src = "live-" + "-".join(map(str, st.session_state["live_args"]))
        win_months = st.session_state["live_args"][2]
        st.download_button("⬇ Save panel.parquet", raw.to_parquet(index=False), "panel.parquet",
                           help="Commit both saved files into a data/ folder in your repo to skip live fetching next time.")
        st.download_button("⬇ Save meta.parquet", meta.reset_index().to_parquet(index=False), "meta.parquet")

    st.header("2. Question")
    with st.form("controls"):
        all_cats = [c for c in CATEGORIES if c in raw["category"].unique()]
        cats = st.multiselect("Categories", all_cats, default=all_cats)
        horizon = st.selectbox("How far ahead should the ranking look?", [3, 6, 12, 24, 36], index=2,
                               format_func=lambda m: f"{m} months")
        target = st.selectbox("What should a good ranking predict?", list(TARGET_LABELS), index=0,
                              format_func=lambda k: TARGET_LABELS[k])
        with st.expander("Advanced settings"):
            method = st.radio("Scaling within category", ["rank", "zscore"], horizontal=True,
                              help="How each metric is compared against peers on the same date.")
            features = st.multiselect("Metrics to use", FEATURES, default=FEATURES,
                                      format_func=lambda k: FEATURE_INFO[k][0])
            model_names = st.multiselect("Ranking methods to compare", MODEL_NAMES, default=MODEL_NAMES, format_func=pretty)
            min_train = st.slider("History used before the first test (months)", 24, 180, 60, step=6)
            step = st.slider("Re-learn every N months", 3, 12, 6, step=3)
            alpha = st.number_input("Smoothing strength (ridge alpha)", 0.1, 1000.0, 50.0)
        st.form_submit_button("Apply / re-run", type="primary")

if use_demo:
    st.warning("**SYNTHETIC DEMO DATA**: randomly generated, not real funds. Nothing here is a finding.", icon="⚠️")

if not cats or not features or not model_names:
    st.info("Pick at least one category, metric and method, then press **Apply / re-run**.")
    st.stop()
if f"fwd_ret_{horizon}m" not in raw.columns:
    st.error(f"This saved data file predates the {horizon}-month horizon. Rebuild it with Live fetch and save it again.")
    st.stop()

cats_t, feats_t, mods_t = tuple(cats), tuple(features), tuple(model_names)
df = get_prepared(raw, src, cats_t, horizon, method)
if df.empty:
    st.error("No category has enough funds on the same dates. Add categories, or shorten the look-back window.")
    st.stop()

# ---- backtest (shared by several tabs) ----
n_dates = df["date"].nunique()
min_train_eff = min_train
if n_dates <= min_train + horizon + 12:
    min_train_eff = max(24, n_dates - horizon - 12)
cols = list(model_names) + list(BASELINES)
preds, score, bt_error = None, None, None
try:
    preds = get_backtest(df, src, cats_t, horizon, method, feats_t, mods_t, target, min_train_eff, step, alpha)
    score = score_predictions(preds, cols, horizon)
    if score is None or score.empty:
        raise ValueError("Too few funds have data on the same dates to test the ranking (each category needs at least "
                         f"{MIN_FUNDS_PER_CROSS_SECTION} funds at once). Try a shorter look-back window, more categories, "
                         "or a shorter horizon.")
except ValueError as e:
    bt_error, score = str(e), None

st.caption(f"{df['scheme_code'].nunique():,} funds · {df['category'].nunique()} categories · "
           f"{pd.Timestamp(df['date'].min()).strftime('%b %Y')} to {pd.Timestamp(df['date'].max()).strftime('%b %Y')}"
           + (f" · history before first test reduced to {min_train_eff} months (data is short)"
              if min_train_eff != min_train else ""))

tab_rank, tab_bt, tab_sig, tab_clu, tab_about = st.tabs(
    ["🏆 Rankings", "🧪 Has it worked before?", "🔬 Which metrics matter?", "🧩 Fund personalities", "ℹ️ How to read this"])

# =========================================================================== #
# Rankings
# =========================================================================== #
with tab_rank:
    c1, c2 = st.columns(2)
    cat = c1.selectbox("Category", cats)
    rank_model = c2.selectbox("Ranking method", model_names, format_func=pretty)
    fit_error = None
    try:
        latest, attrs = get_latest(df, src, cats_t, horizon, method, feats_t, rank_model, target, alpha)
    except ValueError as e:                     # e.g. not enough history to train: fall back to the untrained ranking
        fit_error = str(e)
        latest = composite_fallback(df)
    asof = pd.Timestamp(latest["date"].max())
    ms = method_summary(score, rank_model, horizon) if score is not None and rank_model in score.index and not fit_error else None
    if fit_error:
        st.warning(fit_error + " **Showing the simple average of the metrics instead** (needs no training, "
                   "but has no track-record check).")

    if ms:
        st.markdown(
            f"<div style='border-left:6px solid {ms['colour']};background:#F7F6FB;padding:10px 14px;border-radius:4px'>"
            f"<b>Evidence this ranking has worked before: <span style='color:{ms['colour']}'>{ms['label']}</span></b><br>"
            f"{ms['points'][0]} {ms['points'][1]}</div>", unsafe_allow_html=True)
        st.write("")
    elif bt_error and not fit_error:
        st.warning(bt_error)

    plain = add_plain_columns(latest)
    t = plain[plain["category"] == cat].merge(meta[["scheme_name", "fund_house"]], left_on="scheme_code",
                                              right_index=True, how="left")
    show = pd.DataFrame({
        "Rank": t["rank_model"], "Fund": t["scheme_name"], "Fund house": t["fund_house"], "Outlook": t["outlook"],
        "What stands out": t["strengths"], "Watch-outs": t["watchouts"],
        "Return for risk (Sharpe)": t["sharpe_raw"], "Share of falls taken": t["down_capture_raw"],
        "Share of gains kept": t["up_capture_raw"], "Last 12M return": t["ret_12m_raw"]})
    st.caption(f"As of **{asof.strftime('%d %b %Y')}** · {len(show)} funds in {cat} · "
               "Ranks are within the category only · Hover over column names for meanings.")
    st.dataframe(
        show, hide_index=True, width="stretch", height=560,
        column_config={
            "Rank": st.column_config.NumberColumn(format="%d", width="small"),
            "Outlook": st.column_config.TextColumn(help="Top quarter / Above average / Below average / Bottom quarter of the category"),
            "What stands out": st.column_config.TextColumn(width="medium", help="Where the fund is in the best fifth of its category"),
            "Watch-outs": st.column_config.TextColumn(width="medium", help="Where the fund is in the weakest fifth of its category"),
            "Return for risk (Sharpe)": st.column_config.NumberColumn(format="%.2f", help=FEATURE_INFO["sharpe"][1]),
            "Share of falls taken": st.column_config.NumberColumn(format="percent", help=FEATURE_INFO["down_capture"][1]),
            "Share of gains kept": st.column_config.NumberColumn(format="percent", help=FEATURE_INFO["up_capture"][1]),
            "Last 12M return": st.column_config.NumberColumn(format="percent"),
        })

    d1, d2, _ = st.columns([1, 1, 2])
    full = t[["rank_model", "scheme_name", "fund_house", "outlook", "strengths", "watchouts", "score", "rank_composite"]
             + [f + "_raw" for f in FEATURES]].rename(columns={"rank_model": "rank", "rank_composite": "rank_simple_average"})
    d1.download_button("Download table (CSV)", full.to_csv(index=False).encode(),
                       f"ranking_{cat.replace(' ', '_')}_{asof.date()}.csv", "text/csv")
    if ms:
        try:
            pdf = build_pdf_report(make_report_summary(
                cat, asof, horizon, t, len(t), ms, score, cols, win_months, preds["date"].nunique()))
            d2.download_button("📄 Download VIKA PDF", pdf, f"VIKA_Fund_Ranking_{cat.replace(' ', '_')}_{asof.date()}.pdf",
                               "application/pdf", type="primary")
        except Exception as e:                       # never let a PDF problem break the page
            d2.error(f"PDF failed: {e}")
    st.caption("A high rank is a statistical tilt, not a guarantee. The next tab shows how much the ranking has been worth.")

# =========================================================================== #
# Has it worked before?
# =========================================================================== #
with tab_bt:
    if bt_error:
        st.error(bt_error)
    else:
        bt_model = st.selectbox("Ranking method to check", model_names, format_func=pretty, key="bt_model")
        ms2 = method_summary(score, bt_model, horizon)
        st.caption(f"We replayed history month by month ({preds['date'].nunique()} test dates, "
                   f"{pd.Timestamp(preds['date'].min()).strftime('%b %Y')} to {pd.Timestamp(preds['date'].max()).strftime('%b %Y')}), "
                   f"ranking funds using only what was known at the time, then looking {horizon} months ahead.")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Top-ranked 20% vs category average", f"{ms2['edge']:+.1%}", help=f"Average extra return over the next {horizon} months")
        m2.metric("Bottom-ranked 20% vs average", f"{ms2['bottom']:+.1%}")
        m3.metric("Top picks beat the typical fund", f"{ms2['hit']:.0%}", help="Share of test periods")
        m4.metric("Evidence", ms2["label"])
        if ms2["vs_simple"]:
            st.markdown("**" + ms2["vs_simple"] + "**")
        st.markdown(f"<div style='border-left:6px solid {ms2['colour']};background:#FFF9EF;padding:10px 14px;border-radius:4px'>"
                    + "<br>".join(ms2["points"][:3] + ms2["points"][-1:]) + "</div>", unsafe_allow_html=True)

        st.subheader("Our methods versus simple rules")
        bar_df = score.reset_index()[["model", "top_vs_avg"]]
        bar_df["Type"] = np.where(bar_df["model"].isin(BASELINES), "Simple rule", "Our method")
        bar_df["Method"] = bar_df["model"].map(pretty)
        figb = px.bar(bar_df, x="Method", y="top_vs_avg", color="Type",
                      color_discrete_map={"Our method": "#534191", "Simple rule": "#F6BA47"},
                      labels={"top_vs_avg": f"Top-20% edge over category average ({horizon}M)"})
        figb.update_yaxes(tickformat=".1%")
        figb.update_layout(height=360, xaxis_title="")
        st.plotly_chart(figb, width="stretch")
        st.caption("Taller is better. If our methods are not clearly taller than the simple rules, prefer the simple rule: "
                   "it is easier to explain and just as good.")

        st.subheader("Did it work consistently, or only in some periods?")
        ts = pd.DataFrame({pretty(bt_model): top_bucket_series(preds, bt_model),
                           pretty(SIMPLE): top_bucket_series(preds, SIMPLE)}).rolling(12, min_periods=6).mean()
        figt = px.line(ts, labels={"value": "Top-20% edge (12-month average)", "date": "", "variable": ""},
                       color_discrete_sequence=["#534191", "#F6BA47"])
        figt.add_hline(y=0, line_dash="dot", line_color="grey")
        figt.update_yaxes(tickformat=".0%")
        figt.update_layout(height=340, legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(figt, width="stretch")
        st.caption("Above the dotted line = the top-ranked funds beat the average in that stretch. "
                   "A line that dips below zero shows the ranking does not work in every market.")

        with st.expander("By category"):
            bc = score_predictions(preds, [bt_model], horizon, by_category=True)
            bc = bc.reset_index().rename(columns={"top_vs_avg": "Top-20% edge", "top_beats_median": "Top picks beat typical fund",
                                                  "n_dates": "Test dates"})
            st.dataframe(bc[["category", "Top-20% edge", "Top picks beat typical fund", "Test dates"]], hide_index=True,
                         width="stretch", column_config={"Top-20% edge": st.column_config.NumberColumn(format="percent"),
                                                         "Top picks beat typical fund": st.column_config.NumberColumn(format="percent")})
        with st.expander("Technical scorecard (for the quant-minded)"):
            tech = score.rename(index=pretty).rename(columns={
                "mean_IC": "Rank correlation with later returns (IC)", "IC_IR": "Consistency of that correlation",
                "t_stat_NW": "Statistical confidence (t-stat)", "pct_dates_IC>0": "Share of dates with positive correlation",
                "top_vs_avg": "Top 20% edge", "bottom_vs_avg": "Bottom 20% edge", "spread": "Top minus bottom",
                "top_beats_median": "Top picks beat typical fund", "n_dates": "Test dates"})
            st.dataframe(tech.round(3), width="stretch")
            st.caption("IC = how closely the ranking order matched the order of actual later returns (0 = no link, 1 = perfect). "
                       "Realistic values for fund data are 0.03 to 0.10. Confidence above ~2 means the link is unlikely to be luck; "
                       "it is adjusted for the fact that consecutive months overlap.")

# =========================================================================== #
# Which metrics matter?
# =========================================================================== #
def _signals_tab():
    st.subheader("Which of the familiar numbers actually pointed to better funds later?")
    try:
        ic = get_ic(df, src, cats_t, horizon, method, feats_t).reset_index()
    except Exception:
        st.info("Not enough completed history to judge the metrics for this horizon. "
                "Try a shorter forecast horizon or a shorter look-back window.")
        return

    def verdict(r):
        if r["t_stat_NW"] >= 2 and r["mean_IC"] >= 0.03:
            return "Useful: higher has gone with better later results"
        if r["t_stat_NW"] <= -2 and r["mean_IC"] <= -0.03:
            return "Reverse: higher has gone with WORSE later results"
        return "No reliable signal"

    ic["What history says"] = ic.apply(verdict, axis=1)
    ic["Metric"] = ic["feature"].map(lambda k: FEATURE_INFO[k][0])
    ic["What it means"] = ic["feature"].map(lambda k: FEATURE_INFO[k][1])
    figi = px.bar(ic.sort_values("mean_IC"), x="mean_IC", y="Metric", orientation="h", color="What history says",
                  color_discrete_map={"Useful: higher has gone with better later results": "#1E9A51",
                                      "Reverse: higher has gone with WORSE later results": "#A81538",
                                      "No reliable signal": "#B8B8C4"},
                  labels={"mean_IC": "Link with later results (-1 to +1)"})
    figi.update_layout(height=460, legend=dict(orientation="h", y=-0.2, title=""), yaxis_title="")
    st.plotly_chart(figi, width="stretch")
    st.caption(f"Bars to the right: funds with a higher value of that metric tended to do better over the next {horizon} months "
               "than category peers. Bars to the left: the opposite. Short or grey bars = do not rely on it. "
               "For 'lower is better' metrics (downside capture, volatility), a bar to the LEFT is the good news.")
    st.dataframe(ic[["Metric", "What history says", "mean_IC", "What it means"]].rename(columns={"mean_IC": "Link (-1 to +1)"}),
                 hide_index=True, width="stretch", column_config={"Link (-1 to +1)": st.column_config.NumberColumn(format="%.2f")})

    with st.expander("Advanced: do metrics still matter after accounting for each other? (Fama-MacBeth)"):
        try:
            fm = get_fm(df, src, cats_t, horizon, method, feats_t, target)
        except Exception:
            fm = pd.DataFrame()
        fm = fm.rename(index=lambda k: FEATURE_INFO[k][0]).rename(
            columns={"coef": "Independent effect", "t_stat_NW": "Confidence (t-stat)", "n_dates": "Dates"})
        st.dataframe(fm.round(3), width="stretch")
        st.caption("Many metrics say nearly the same thing, so individual effects here can look odd or flip sign. That is expected.")
    with st.expander("Advanced: which metrics say the same thing?"):
        corr = df[list(features)].corr(method="spearman")
        short = [FEATURE_INFO[k][0].split(" (")[-1].rstrip(")") if "(" in FEATURE_INFO[k][0] else FEATURE_INFO[k][0]
                 for k in corr.columns]
        corr.index, corr.columns = short, short
        st.plotly_chart(px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", text_auto=".2f",
                                  aspect="auto").update_layout(height=520), width="stretch")
        st.caption("Dark red = the two metrics almost always agree. This is why blending them beats picking one.")


with tab_sig:
    _signals_tab()


# =========================================================================== #
# Fund personalities
# =========================================================================== #
def label_groups(prof: pd.DataFrame) -> dict:
    """Give each K-means group a plain description from how it differs from the other groups."""
    z = ((prof - prof.mean()) / prof.std(ddof=0).replace(0, np.nan)).fillna(0)
    out = {}
    for g in prof.index:
        risk = (z.loc[g, "beta"] + z.loc[g, "volatility"]) / 2
        alpha_z, prot = z.loc[g, "alpha"], -z.loc[g, "down_capture"]
        dims = {"risk": risk, "alpha": alpha_z, "prot": prot}
        k = max(dims, key=lambda x: abs(dims[x]))
        if abs(dims[k]) < 0.5:
            desc = "Middle of the pack"
        elif k == "risk":
            desc = "Aggressive: bigger swings than peers" if risk > 0 else "Defensive: steadier than peers"
        elif k == "alpha":
            desc = "Value-adders: more return than their risk explains" if alpha_z > 0 else "Laggards: less return than their risk explains"
        else:
            desc = "Cushioned: fall less in down markets" if prot > 0 else "Exposed: fall more in down markets"
        out[g] = f"Group {int(g) + 1}: {desc}"
    return out


with tab_clu:
    st.subheader("Fund personalities: how funds behave (this does not rank them)")
    c1, c2 = st.columns(2)
    cc = c1.selectbox("Category ", cats, key="clu_cat")
    k = c2.slider("Number of groups", 2, 6, 4)
    snap = df[(df["date"] == df["date"].max()) & (df["category"] == cc)].merge(
        meta[["scheme_name"]], left_on="scheme_code", right_index=True, how="left")
    cl = cluster_snapshot(snap, k)
    if cl.empty or "pc1" not in cl:
        st.info("Not enough funds for grouping in this category.")
    else:
        prof = cl.groupby("cluster")[[c + "_raw" for c in CLUSTER_FEATURES]].mean()
        prof.columns = CLUSTER_FEATURES
        names = label_groups(prof)
        cl["Group"] = cl["cluster"].map(names)
        figc = px.scatter(cl, x="pc1", y="pc2", color="Group", hover_name="scheme_name",
                          hover_data={"pc1": False, "pc2": False},
                          labels={"pc1": "Funds further apart = behave more differently", "pc2": ""})
        figc.update_layout(height=460, legend=dict(orientation="h", y=-0.2, title=""))
        figc.update_xaxes(showticklabels=False)
        figc.update_yaxes(showticklabels=False)
        st.plotly_chart(figc, width="stretch")
        tbl = prof.rename(index=names)
        tbl["Funds"] = cl.groupby("Group").size()
        tbl = tbl.rename(columns={"beta": "Market sensitivity", "volatility": "Ups and downs", "up_capture": "Share of gains kept",
                                  "down_capture": "Share of falls taken", "alpha": "Value added (alpha)", "max_drawdown": "Worst fall"})
        pct = st.column_config.NumberColumn(format="percent")
        st.dataframe(tbl, width="stretch", column_config={
            "Ups and downs": pct, "Share of gains kept": pct, "Share of falls taken": pct, "Value added (alpha)": pct,
            "Worst fall": pct, "Market sensitivity": st.column_config.NumberColumn(format="%.2f")})
        with st.expander("Funds in each group"):
            for g in sorted(cl["Group"].unique()):
                st.markdown(f"**{g}**: " + ", ".join(sorted(cl.loc[cl["Group"] == g, "scheme_name"].dropna().astype(str))))
        st.caption(f"Grouping clarity score: {cl.attrs.get('silhouette', float('nan')):.2f} (0 to 1). Below about 0.25 the groups are "
                   "soft: funds sit on a continuum and the boundaries are somewhat arbitrary.")

with tab_about:
    st.markdown(GUIDE)
