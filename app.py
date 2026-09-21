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
    # `tickers` are tried IN ORDER; the first that returns >= 250 daily rows wins. Index tickers come first, then ETF
    # proxies (Indian index ETFs are growth plans, so their price ~ total return). The app records which one was used.
    "NIFTY50": {"name": "Nifty 50", "tickers": ["^NSEI", "NIFTYBEES.NS"]},
    "NIFTY500": {"name": "Nifty 500", "tickers": ["^CRSLDX"]},
    "NIFTY100": {"name": "Nifty 100", "tickers": ["^CNX100", "NIF100IETF.NS", "NIF100BEES.NS"]},
    "NIFTYMID150": {"name": "Nifty Midcap 150", "tickers": ["NIFTYMIDCAP150.NS", "MOM100.NS", "MID150BEES.NS"]},
    "NIFTYSMALL250": {"name": "Nifty Smallcap 250", "tickers": ["NIFTYSMLCAP250.NS"]},
    "NIFTYLMC250": {"name": "Nifty LargeMidcap 250", "tickers": ["NIFTY_LARGEMID250.NS"]},
}
# Appended to EVERY chain as a last resort so one dead ticker can't stop the whole fetch (flagged in the app + PDF).
LAST_RESORT_PROXY = "NIFTYBEES.NS"   # Nippon India Nifty 50 ETF

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
    "volatility", "max_drawdown", "ret_3m", "ret_6m", "ret_12m", "cagr_2y", "cagr_3y",
]

# Features that go into the naive "normalise and rank" composite score (baseline).
# Signs: +1 higher is better, -1 lower is better.
COMPOSITE_SIGNS = {
    "sharpe": 1, "sortino": 1, "treynor": 1, "info_ratio": 1,
    "up_capture": 1, "down_capture": -1,
    "ret_12m": 1, "cagr_2y": 1, "cagr_3y": 1,          # 1-, 2- and 3-year returns (CAGR)
}

# --------------------------------------------------------------------------- #
# Newer funds (less history than the metric window)
# --------------------------------------------------------------------------- #
NEW_FUND_MIN_SHOW_MONTHS = 3    # funds younger than this are ignored entirely
NEW_FUND_MIN_RANK_MONTHS = 12   # below this we show return since launch but do not rank
# like-for-like score used to rank newer funds against peers over the SAME period
LFL_SIGNS = {"sharpe": 1, "sortino": 1, "info_ratio": 1, "up_capture": 1, "down_capture": -1, "ann_return": 1}


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
def _yahoo_close(ticker: str, start: str) -> pd.Series:
    import yfinance as yf  # imported lazily so the Streamlit app doesn't need network for cached data

    raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        return pd.Series(dtype=float)
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close


def load_benchmark(key: str, start: str = "2005-01-01") -> tuple[pd.Series, str]:
    """(daily level series, source label). CSV in ./benchmarks/ wins; else try each ticker in order, then the
    last-resort ETF proxy. Raises only if every option fails."""
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
        return s, f"CSV benchmarks/{key}.csv"

    tried = []
    for tk in list(spec["tickers"]) + [LAST_RESORT_PROXY]:
        if tk in tried:
            continue
        tried.append(tk)
        try:
            s = _yahoo_close(tk, start)
        except Exception as e:                     # rate limit, delisted symbol, network...
            log.warning("benchmark %s: %s failed (%s)", key, tk, e)
            continue
        if len(s) >= 250:
            s.name = key
            return s, tk
        log.warning("benchmark %s: %s returned only %d rows", key, tk, len(s))
    raise ValueError(f"No data for benchmark {key}; tried {tried}. Add a Date,Close CSV at benchmarks/{key}.csv")


def load_benchmarks(keys: list[str], start: str = "2005-01-01") -> pd.DataFrame:
    """Benchmark levels for `keys`; df.attrs['sources'] records the ticker actually used for each."""
    series, sources = {}, {}
    for k in keys:
        series[k], sources[k] = load_benchmark(k, start)
    df = pd.concat(series, axis=1).sort_index()
    df.attrs["sources"] = sources
    return df



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
        fund_extra["cagr_2y"] = ((lvl_m / lvl_m.shift(24)) ** (1 / 2) - 1).to_numpy()
        fund_extra["cagr_3y"] = ((lvl_m / lvl_m.shift(36)) ** (1 / 3) - 1).to_numpy()
        fund_extra["history_months"] = lvl_m.notna().cumsum().to_numpy().astype(float)
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
            f["benchmark"] = bkey
            f["bench_source"] = bench.attrs.get("sources", {}).get(bkey, "")
            f["kind"] = "full"
            frames.append(f)

    if not frames:
        raise ValueError("No fund had enough history to compute metrics -- check data / window length")
    out = pd.concat(frames, ignore_index=True)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.sort_values(["category", "date", "scheme_code"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Newer funds: fewer months of history than the metric window
# --------------------------------------------------------------------------- #
def _growth_ann(r: np.ndarray, ppy: int) -> float:
    return float(np.prod(1.0 + r) ** (ppy / len(r)) - 1.0)


def compute_new_funds(
    nav: pd.DataFrame,
    meta: pd.DataFrame,
    bench: pd.DataFrame,
    full_panel: pd.DataFrame,
    freq: str = "M",
    rf_annual: float = RISK_FREE_ANNUAL,
) -> pd.DataFrame:
    """Rows (kind='new') for live funds that are too young for the main panel.

    Each newer fund is compared LIKE-FOR-LIKE: its metrics over its own life are ranked against those of every
    category peer computed over exactly the same dates. Funds under NEW_FUND_MIN_RANK_MONTHS are listed with
    their return since launch but not ranked.
    """
    ppy = 12 if freq == "M" else 252
    rf_p = (1 + rf_annual) ** (1 / ppy) - 1
    have = set(full_panel.loc[full_panel["date"] == full_panel["date"].max(), "scheme_code"]) if len(full_panel) else set()
    out = []
    for cat, spec in CATEGORIES.items():
        codes = [c for c in meta.index[meta["category"] == cat] if c in nav.columns]
        bkey = spec["bench"]
        if not codes or bkey not in bench.columns:
            continue
        bser = bench[bkey].dropna()
        cal = bser.index
        nv = nav[codes].reindex(cal.union(nav.index)).ffill(limit=5).reindex(cal)
        me = month_end_dates(cal)
        d = me[-1]
        if freq == "M":
            R, B = nv.loc[me].pct_change(fill_method=None), bser.loc[me].pct_change()
        else:
            R, B = nv.pct_change(fill_method=None), bser.pct_change()
        Rv, Bv = R.to_numpy(), B.to_numpy()
        p = Rv.shape[0] - 1
        last = nv.loc[d]
        src = bench.attrs.get("sources", {}).get(bkey, "")

        peer_cache: dict[int, pd.DataFrame] = {}

        def peers_over(s0: int) -> pd.DataFrame:
            """Metrics of every category fund over the SAME slice [s0:p], plus a like-for-like composite."""
            if s0 in peer_cache:
                return peer_cache[s0]
            bw = Bv[s0:p + 1]
            n_obs = p + 1 - s0
            rows = {}
            for k, code in enumerate(codes):
                rw = Rv[s0:p + 1, k]
                ok = np.isfinite(rw) & np.isfinite(bw)
                if ok.sum() < 0.9 * n_obs:
                    continue
                m = window_metrics(rw[ok], bw[ok], rf_p, ppy, max(3, int(0.2 * n_obs)))
                m["ann_return"] = _growth_ann(rw[ok], ppy)
                rows[code] = m
            pm = pd.DataFrame.from_dict(rows, orient="index")
            if len(pm) >= 5:
                z = pd.DataFrame({f: pm[f].rank(pct=True) - 0.5 for f in LFL_SIGNS})
                pm["lfl_score"] = sum(z[f].fillna(0) * sg for f, sg in LFL_SIGNS.items()) / len(LFL_SIGNS)
            peer_cache[s0] = pm
            return pm

        for j, code in enumerate(codes):
            if code in have or not np.isfinite(last.iloc[j]):
                continue
            first = nv[code].first_valid_index()
            months = (d - first).days / 30.44
            if months < NEW_FUND_MIN_SHOW_MONTHS:
                continue
            row = {"date": d, "scheme_code": code, "category": cat, "kind": "new", "history_months": round(months, 1),
                   "since_launch": float(last.iloc[j] / nv[code].loc[first] - 1),
                   "benchmark": bkey, "bench_source": src}
            if months >= NEW_FUND_MIN_RANK_MONTHS:
                s0 = int(np.flatnonzero(np.isfinite(Rv[:, j]))[0])
                pm = peers_over(s0)
                if code in pm.index:
                    row.update(pm.loc[code].drop(labels=["lfl_score"], errors="ignore").to_dict())
                    if "lfl_score" in pm.columns:
                        row["lfl_rank"] = int((pm["lfl_score"] > pm.loc[code, "lfl_score"]).sum() + 1)
                        row["lfl_peers"] = int(len(pm))
            out.append(row)
    return pd.DataFrame(out)


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
MODEL_NAMES = ["Ridge", "PCA + Ridge"]


def make_model(name: str = "Ridge", ridge_alpha: float = 50.0, pca_var: float = 0.90, **_):
    if name == "Ridge":
        return Ridge(alpha=ridge_alpha)
    if name == "PCA + Ridge":
        return Pipeline([("pca", PCA(n_components=pca_var, svd_solver="full")), ("ridge", Ridge(alpha=ridge_alpha))])
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
                 horizon: int = 12, min_obs: int = 0) -> pd.DataFrame:
    """Fama-MacBeth: OLS across funds on each date, then average the coefficients over time.

    Features are already standardised within (category, date), so categories can be pooled on each date. That
    gives every date enough funds to run the cross-sectional regression even when single categories are small.
    """
    d = df.dropna(subset=[target])
    need = max(min_obs, len(features) + 10)
    recs = []
    for date, g in d.groupby("date"):
        if len(g) < need:
            continue
        X = np.column_stack([np.ones(len(g)), g[features].to_numpy(float)])
        b, *_ = np.linalg.lstsq(X, g[target].to_numpy(float), rcond=None)
        recs.append((date, *b))
    cols = ["coef", "t_stat_NW", "n_dates"]
    if not recs:
        return pd.DataFrame(np.nan, index=features, columns=cols)
    coefs = pd.DataFrame(recs, columns=["date", "const"] + features).set_index("date").sort_index()
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


# --------------------------------------------------------------------------- #
# Regression results (what the ranking is built from)
# --------------------------------------------------------------------------- #
def regression_report(df: pd.DataFrame, features: list[str], target_col: str, horizon: int,
                      alpha: float = 50.0) -> dict:
    """Fit the ranking regression on every fund-month whose outcome is known and report everything about it."""
    d = df.dropna(subset=[target_col])
    X, y = d[features].to_numpy(float), d[target_col].to_numpy(float)
    ridge = Ridge(alpha=alpha).fit(X, y)
    fm = fama_macbeth(d, features, target_col, horizon)
    tab = pd.DataFrame({"ridge_coef": ridge.coef_}, index=features)
    tab = tab.join(fm[["coef", "t_stat_NW"]].rename(columns={"coef": "fm_coef", "t_stat_NW": "fm_t"}))
    try:                                              # each metric ON ITS OWN (robust to metrics overlapping)
        uni = univariate_ic(d, features, "fwd_ret", horizon)
        tab = tab.join(uni[["mean_IC", "t_stat_NW"]].rename(columns={"mean_IC": "uni_ic", "t_stat_NW": "uni_t"}))
    except Exception:
        tab["uni_ic"], tab["uni_t"] = np.nan, np.nan
    ols_text = ""
    try:                                              # pooled OLS with standard errors clustered by date
        Xc = sm.add_constant(d[features].astype(float))
        groups = pd.factorize(d["date"])[0]
        ols_text = str(sm.OLS(y, Xc).fit(cov_type="cluster", cov_kwds={"groups": groups}).summary())
    except Exception as e:                            # collinear design etc.
        ols_text = f"(full OLS output unavailable: {e})"
    return {"table": tab, "r2_in": float(ridge.score(X, y)), "n_obs": int(len(d)),
            "n_funds": int(d["scheme_code"].nunique()), "n_dates": int(d["date"].nunique()),
            "first": pd.Timestamp(d["date"].min()), "last": pd.Timestamp(d["date"].max()),
            "intercept": float(ridge.intercept_), "ols_summary": ols_text}


def pca_report(df: pd.DataFrame, features: list[str], target_col: str, alpha: float = 50.0, var: float = 0.90) -> dict:
    """PCA view of Ranking 3: the factors, what they are made of, and how the regression uses them."""
    d = df.dropna(subset=[target_col])
    X, y = d[features].to_numpy(float), d[target_col].to_numpy(float)
    pca = PCA(n_components=var, svd_solver="full").fit(X)
    Z = pca.transform(X)
    k = int(pca.n_components_)
    ridge = Ridge(alpha=alpha).fit(Z, y)
    names = [f"Factor {i + 1}" for i in range(k)]
    load = pd.DataFrame(pca.components_, columns=features, index=names)
    tvals = pd.Series(np.nan, index=names)
    ols_text = ""
    try:
        res = sm.OLS(y, sm.add_constant(pd.DataFrame(Z, columns=names))).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(d["date"])[0]})
        tvals = res.tvalues.drop("const")
        ols_text = str(res.summary())
    except Exception as e:
        ols_text = f"(full OLS output unavailable: {e})"
    return {"n_components": k, "n_features": len(features), "explained": pca.explained_variance_ratio_,
            "load": load, "factor_coef": pd.Series(ridge.coef_, index=names), "factor_t": tvals,
            "implied": pd.Series(pca.components_.T @ ridge.coef_, index=features),   # weight per original metric
            "r2_in": float(ridge.score(Z, y)), "ols_summary": ols_text}


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
    bench_df.attrs["sources"] = {k: "SYNTHETIC" for k in BENCHMARKS}
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
            young = i % 6 == 5                                     # every 6th fund is a recent launch (3 to ~30 months old)
            incep = T - int(rng.integers(65, 650)) if young else rng.integers(0, int(T * 0.55))
            death = T if (young or rng.random() > 0.08) else rng.integers(int(T * 0.7), T)
            nav = 10 * np.cumprod(1 + r)
            s = pd.Series(nav, index=cal)
            s.iloc[:incep] = np.nan
            s.iloc[death:] = np.nan
            s = s.dropna()
            if not young and len(s) < 800:
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
    row_h, gap, label_w = 16, 5, 190
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
        d.add(String(label_w - 8, y + 5, _latin(lab), fontName="Helvetica", fontSize=8.5,
                     fillColor=_c(VIKA_TEXT), textAnchor="end"))
        d.add(Rect(left, y, w, row_h, fillColor=_c(color if v >= 0 else VIKA_MAROON), strokeColor=None))
        txt = f"{v * 100:+.1f}%"
        if w > 38:
            d.add(String(left + w - 4 if v >= 0 else left + 4, y + 5, txt, fontName="Helvetica-Bold", fontSize=8.5,
                         fillColor=colors.white, textAnchor="end" if v >= 0 else "start"))
        else:
            d.add(String(left + w + 4 if v >= 0 else left - 4, y + 5, txt, fontName="Helvetica-Bold", fontSize=8.5,
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
    box_val = ParagraphStyle("box_val", parent=base, fontName="Helvetica-Bold", fontSize=12, leading=15,
                             alignment=TA_CENTER)
    spaced = ParagraphStyle("spaced", parent=base, spaceAfter=3)

    def heading(txt):
        return [Paragraph(_safe(txt), h2), HRFlowable(width="100%", thickness=2.5, color=_c(VIKA_GOLD_LINE),
                                                       spaceBefore=1, spaceAfter=6)]

    def cream(paras, pad=9):
        t = Table([[paras]], colWidths=[CW])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), _c(VIKA_CREAM)),
                               ("LEFTPADDING", (0, 0), (-1, -1), pad + 3), ("RIGHTPADDING", (0, 0), (-1, -1), pad + 3),
                               ("TOPPADDING", (0, 0), (-1, -1), pad), ("BOTTOMPADDING", (0, 0), (-1, -1), pad)]))
        return t

    def data_table(header, rows, widths, highlight_top=0, bold_first=True):
        data = [[Paragraph(_safe(h), th) for h in header]]
        for i, r in enumerate(rows):
            data.append([Paragraph(_safe(v), cell_b if (k == 0 and bold_first) else cell) for k, v in enumerate(r)])
        t = Table(data, colWidths=widths, repeatRows=1)
        st = [("BACKGROUND", (0, 0), (-1, 0), _c(VIKA_NAVY)), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
              ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
              ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]
        for i in range(1, len(data)):
            if i <= highlight_top:
                st.append(("BACKGROUND", (0, i), (-1, i), _c(VIKA_PASTELS[2])))      # highlight the top 3
            elif i % 2 == 0:
                st.append(("BACKGROUND", (0, i), (-1, i), _c("#F7F6FB")))
        t.setStyle(TableStyle(st))
        return t

    story = []

    # ---- At a glance: five pastel boxes (blue, orange, green, yellow, purple)
    boxes = [("Funds ranked", str(s["n_funds"])), ("Looking ahead", f"{s['horizon']} months"),
             ("Data as of", pd.Timestamp(s["as_of"]).strftime("%d %b %Y")),
             ("Ranking 1 evidence", s["ev1"]), ("Ranking 2 evidence", s["ev2"]), ("Ranking 3 evidence", s["ev3"])]
    n = len(boxes)
    box_val.fontSize = 11
    bt = Table([[[Paragraph(_safe(a), box_lab), Paragraph(_safe(b), box_val)] for a, b in boxes]],
               colWidths=[CW / n] * n)
    bt.setStyle(TableStyle(
        [("BACKGROUND", (i, 0), (i, 0), _c(VIKA_PASTELS[i % 5])) for i in range(n)] +
        [("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
         ("LINEAFTER", (0, 0), (-2, -1), 5, colors.white), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story += heading("At a glance") + [bt, Spacer(1, 6)]
    story += [cream([Paragraph(_safe(t), spaced) for t in s["intro"]])]

    # ---- Ranking 1 and Ranking 2 tables
    hdr = ["Rank", "Fund", "What stands out", "Last 12M", "2Y CAGR", "3Y CAGR", "Return per risk*"]
    cw = [34, 170, 150, 46, 46, 46, 48]
    for key, title, sub in (("r1_rows", f"Ranking 1: {s['category']}, by observed performance", s["r1_sub"]),
                            ("r2_rows", f"Ranking 2: {s['category']}, by panel regression (forward-looking)", s["r2_sub"]),
                            ("r3_rows", f"Ranking 3: {s['category']}, by PCA regression (forward-looking)", s["r3_sub"])):
        rows = [[r["rank"], r["fund"], r["strengths"] or "-", r["ret_12m"], r["cagr_2y"], r["cagr_3y"], r["sharpe"]]
                for r in s[key]]
        block = heading(title) + [Paragraph(_safe(sub), small), Spacer(1, 4)]
        if rows:
            block.append(data_table(hdr, rows, cw, highlight_top=3))
        else:
            block.append(Paragraph("Not available: " + _safe(s.get("r2_unavailable", "not enough history to train.")), base))
        story += [KeepTogether(block)]
    story += [Spacer(1, 3),
              Paragraph("*Extra return over a bank-deposit-like rate for every unit of ups-and-downs (Sharpe ratio). "
                        "CAGR = compound annual growth rate. Ranks are within the category only.", small)]

    # ---- newer funds
    if s.get("new_rows"):
        nrows = [[r["fund"], r["months"], r["since_launch"], r["sharpe"], r["lfl"]] for r in s["new_rows"]]
        blk = heading("Newer funds (not ranked against the funds above)") + [Paragraph(_safe(s["new_note"]), small), Spacer(1, 4),
              data_table(["Fund", "Months of data", "Since launch", "Return per risk*", "Like-for-like rank"], nrows,
                         [230, 65, 70, 70, 105], bold_first=False)]
        story += [KeepTogether(blk)]

    # ---- Did it work?
    sec = heading("Have these rankings worked before?") + [Paragraph(_safe(s["backtest_intro"]), base), Spacer(1, 6)]
    if s.get("method_edges"):
        sec += [_bar_chart(s["method_edges"], CW), Spacer(1, 2), Paragraph(_safe(s["chart_caption"]), small), Spacer(1, 6)]
    story += [KeepTogether(sec)]
    paras = [Paragraph(f"<b>Ranking 1 (observed performance).</b> {_safe(' '.join(s['points1']))}", spaced),
             Paragraph(f"<b>Ranking 2 (panel regression).</b> {_safe(' '.join(s['points2']))}", spaced)]
    if s.get("points3"):
        paras.append(Paragraph(f"<b>Ranking 3 (PCA regression).</b> {_safe(' '.join(s['points3']))}", spaced))
    story += [cream(paras)]

    # ---- regression results
    if s.get("reg_rows"):
        rr = [[m, w, t_, v] for m, w, t_, v in s["reg_rows"]]
        blk = heading("What the regression found") + [Paragraph(_safe(s["reg_text"]), base), Spacer(1, 5),
              data_table(["Metric", "Weight", "Confidence (t)", "What history says"], rr, [230, 50, 70, 190], bold_first=False), Spacer(1, 3),
              Paragraph(_safe(s["reg_foot"]), small)]
        story += [KeepTogether(blk)]

    # ---- How to read
    story += heading("How to read this")
    for k, v in s["glossary"]:
        story.append(Paragraph(f"<b>{_safe(k)}:</b> {_safe(v)}", small))
        story.append(Spacer(1, 1))
    if s.get("benchmark_note"):
        story.append(Paragraph(f"<b>Benchmarks:</b> {_safe(s['benchmark_note'])}", small))

    # ---- disclaimer + contact (anchored to page bottom)
    story += [Spacer(1, 5), KeepTogether([cream([Paragraph(_safe(DISCLAIMER_TEXT),
                                                           ParagraphStyle("disc", parent=base, fontSize=7.5, leading=10))],
                                                pad=7)])]
    contact = Table([[[Paragraph(f"<b>{CONTACT_LINE_1}</b>", ParagraphStyle("c1", parent=base, fontSize=10, leading=13, alignment=TA_CENTER)),
                       Paragraph(CONTACT_LINE_2, ParagraphStyle("c2", parent=base, fontSize=7.5, leading=10, alignment=TA_CENTER))]]],
                    colWidths=[CW])
    contact.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), _c(VIKA_LAVENDER)),
                                 ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    story += [Spacer(1, 4), _BottomAnchored(contact)]

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
    "ret_12m": ("Last 12-month return", "Return over the last year.", "high"),
    "cagr_2y": ("Last 2-year return (CAGR)", "Compound annual growth rate over the last 2 years.", "high"),
    "cagr_3y": ("Last 3-year return (CAGR)", "Compound annual growth rate over the last 3 years.", "high"),
}
STRONG = {"sharpe": "Good return for the risk", "sortino": "Good return for its downside risk",
          "info_ratio": "Steadily beats its benchmark", "alpha": "Adds value beyond the market",
          "up_capture": "Captures market gains well", "down_capture": "Cushions market falls",
          "volatility": "Smoother ride than peers", "max_drawdown": "Shallower worst fall", "ret_12m": "Strong last 12 months",
          "cagr_2y": "Strong 2-year return", "cagr_3y": "Strong 3-year return"}
WEAK = {"sharpe": "Low return for the risk", "sortino": "Weak return for its downside risk",
        "info_ratio": "Inconsistent versus benchmark", "alpha": "Adds little beyond the market",
        "up_capture": "Misses market gains", "down_capture": "Falls hard in down markets",
        "volatility": "Bumpier ride than peers", "max_drawdown": "Deeper worst fall", "ret_12m": "Weak last 12 months",
        "cagr_2y": "Weak 2-year return", "cagr_3y": "Weak 3-year return"}
TRAITS = list(STRONG)

R1, R2, R3 = "Composite (normalise & rank)", "Ridge", "PCA + Ridge"   # Ranking 1 (observed), 2 (regression), 3 (PCA regression)
LABELS = {R1: "Ranking 1: observed performance", R2: "Ranking 2: panel regression", R3: "Ranking 3: PCA regression",
          "Trailing 12M return": "Just the last 12-month return", "Sharpe only": "Just the Sharpe ratio"}
TARGET_LABELS = {"cat_rank": "Beat category peers (recommended)", "cat_excess": "Return above the category median",
                 "bench_excess": "Return above the benchmark"}


def pretty(name: str) -> str:
    return LABELS.get(name, name)


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
    """'What stands out' / 'watch-outs' text for each fund (vs its category peers, on observed metrics)."""
    d = latest.copy()
    n = d.groupby("category")["scheme_code"].transform("count")
    good = pd.DataFrame(index=d.index)
    for f in TRAITS:
        p = d.groupby("category")[f + "_raw"].rank(pct=True)
        good[f] = 1 - p + (1 / n) if FEATURE_INFO[f][2] == "low" else p
    strengths, watch = [], []
    for i in d.index:
        g = good.loc[i].dropna()
        strengths.append("; ".join(STRONG[k] for k in g[g >= 0.8].sort_values(ascending=False).index[:2]))
        watch.append("; ".join(WEAK[k] for k in g[g <= 0.2].sort_values().index[:2]))
    d["strengths"], d["watchouts"] = strengths, watch
    return d


def composite_fallback(df: pd.DataFrame) -> pd.DataFrame:
    """Ranking that needs no training: the simple average of the metrics, latest month only."""
    d = df[df["date"] == df["date"].max()].copy()
    d["score"] = d["composite"]
    d["rank_composite"] = d.groupby("category")["score"].rank(ascending=False, method="first").astype(int)
    d["rank_model"] = np.nan                       # Ranking 2 could not be trained
    d["n_in_category"] = d.groupby("category")["score"].transform("count")
    return d.sort_values(["category", "rank_model"]).reset_index(drop=True)


def method_summary(score: pd.DataFrame, method: str, horizon: int) -> dict:
    r = score.loc[method]
    label, colour = evidence_level(r["mean_IC"], r["t_stat_NW"])
    edge, hit, bot = r["top_vs_avg"], r["top_beats_median"], r["bottom_vs_avg"]
    simple_edge = score.loc[R1, "top_vs_avg"] if R1 in score.index else np.nan
    pts = [
        f"In past test periods, the funds ranked in the top 20% of their category went on to "
        f"{'beat' if edge >= 0 else 'trail'} the category average by {abs(edge):.1%} over the next {horizon} months, on average.",
        f"They did better than the typical (median) fund in {hit:.0%} of test periods.",
        f"The bottom-ranked 20% {'lagged' if bot <= 0 else 'beat'} the average by {abs(bot):.1%}.",
    ]
    vs_simple = None
    if method in (R2, R3) and pd.notna(simple_edge):
        diff = edge - simple_edge
        vs_simple = ("Compared with Ranking 1 (observed performance), this ranking was "
                     + ("clearly better." if diff > 0.005 else
                        "not meaningfully better." if diff > -0.005 else "worse."))
        pts.append(vs_simple)
    pts.append("These are averages of past results. Individual funds and periods varied a lot, and the past may not repeat.")
    return {"label": label, "colour": colour, "edge": edge, "hit": hit, "bottom": bot, "points": pts,
            "vs_simple": vs_simple, "ic": r["mean_IC"], "t": r["t_stat_NW"]}


GLOSSARY = [
    ("Ranking 1: observed performance", "Who did best over the look-back window: return for risk, protection in falls, steadiness vs "
                                        "benchmark and 1/2/3-year returns. No forecasting."),
    ("Ranking 2: panel regression", "Forward-looking: a regression learnt from history, applied to today's numbers."),
    ("Ranking 3: PCA regression", "Same idea, but first condenses the overlapping metrics into a few independent factors (PCA)."),
    ("What stands out", "Areas where the fund is in the best fifth of its category (e.g. cushions market falls)."),
    ("Return per risk", "Sharpe ratio: extra return over a bank-deposit-like rate per unit of ups-and-downs. Higher is better."),
    ("Evidence", "Whether funds ranked highly in the past actually beat peers afterwards: Strong / Moderate / Weak / None found."),
]


SHORT_NAMES = {"sharpe": "Sharpe", "sortino": "Sortino", "treynor": "Treynor", "info_ratio": "Info ratio", "alpha": "Alpha",
               "beta": "Beta", "up_capture": "Upside capture", "down_capture": "Downside capture", "capture_ratio": "Capture ratio",
               "volatility": "Volatility", "max_drawdown": "Max drawdown", "ret_3m": "3M return", "ret_6m": "6M return",
               "ret_12m": "12M return", "cagr_2y": "2Y CAGR", "cagr_3y": "3Y CAGR"}


def fmt_pct(x) -> str:
    return f"{x:.1%}" if pd.notna(x) else "-"


def fmt_num(x, d=2) -> str:
    return f"{x:.{d}f}" if pd.notna(x) else "-"


UP_TXT, DOWN_TXT, NONE_TXT = ("Higher values go with better later results", "Higher values go with worse later results",
                               "No reliable link")


def reg_table(rep: dict) -> pd.DataFrame:
    """Regression results in reader-friendly form, sorted by how reliably each metric links to later results."""
    tb = rep["table"].copy()
    tb["Metric"] = [FEATURE_INFO[k][0] for k in tb.index]
    tb["What it means"] = [FEATURE_INFO[k][1] for k in tb.index]
    tb["Weight"] = tb["ridge_coef"] * 100

    def verdict(r):
        if pd.notna(r["uni_t"]) and r["uni_t"] >= 2 and r["uni_ic"] >= 0.03:
            return UP_TXT
        if pd.notna(r["uni_t"]) and r["uni_t"] <= -2 and r["uni_ic"] <= -0.03:
            return DOWN_TXT
        return NONE_TXT

    tb["What history says"] = tb.apply(verdict, axis=1)
    return tb.reindex(tb["uni_t"].abs().sort_values(ascending=False, na_position="last").index)


def benchmark_note(df: pd.DataFrame, cat: str) -> str:
    r = df[df["category"] == cat][["benchmark", "bench_source"]].drop_duplicates() if "benchmark" in df.columns else pd.DataFrame()
    if r.empty:
        return ""
    key, srcname = r.iloc[0]["benchmark"], str(r.iloc[0]["bench_source"])
    name = BENCHMARKS[key]["name"]
    if srcname in ("", "nan"):
        return f"{cat} funds are compared with the {name}."
    if srcname == BENCHMARKS[key]["tickers"][0] or srcname.startswith("CSV"):
        return f"{cat} funds are compared with the {name} ({srcname})."
    return f"{cat} funds are compared with a proxy for the {name}: the ETF/index {srcname}."


def make_report_summary(cat, asof, horizon, t, yg, ms1, ms2, score, cols, win_months, n_dates, reg, reg_tb, fit_error, bnote,
                        ms3=None, pca_rep=None):
    """Everything the PDF shows, as a plain dict. Edit THIS (and the layout in build_pdf_report) to change PDF content."""
    when = f"the last {win_months} months" if win_months else "recent years"

    def rows(col):
        top = t.sort_values(col).head(10)
        return [{"rank": int(getattr(r, col)), "fund": r.scheme_name, "strengths": r.strengths,
                 "ret_12m": fmt_pct(r.ret_12m_raw), "cagr_2y": fmt_pct(r.cagr_2y_raw), "cagr_3y": fmt_pct(r.cagr_3y_raw),
                 "sharpe": fmt_num(r.sharpe_raw)} for r in top.itertuples()]

    new_rows = []
    if yg is not None and len(yg):
        for r in yg.sort_values(["lfl_rank", "history_months"], ascending=[True, False], na_position="last").itertuples():
            new_rows.append({"fund": r.scheme_name, "months": f"{r.history_months:.0f}", "since_launch": fmt_pct(r.since_launch),
                             "sharpe": fmt_num(r.sharpe),
                             "lfl": f"{int(r.lfl_rank)} of {int(r.lfl_peers)}" if pd.notna(r.lfl_rank) else "Too new to rank"})
    p2 = ms2["points"][:3] + ([ms2["vs_simple"]] if ms2["vs_simple"] else []) + [ms2["points"][-1]] if ms2 else \
        ["Ranking 2 could not be tested or trained on the available history."]
    summ = {
        "category": cat, "as_of": asof, "horizon": horizon, "n_funds": len(t),
        "ev1": ms1["label"] if ms1 else "n/a", "ev2": ms2["label"] if ms2 else "n/a", "ev3": ms3["label"] if ms3 else "n/a",
        "intro": [f"This note compares {len(t)} Direct-Growth {cat} mutual funds. Ranking 1 ranks them on what they actually delivered "
                  f"over {when}: return for the risk taken, protection in falling markets, steadiness versus the benchmark, and "
                  f"1-, 2- and 3-year returns (CAGR).",
                  f"Ranking 2 is forward-looking. A panel regression learns from history which combinations of those numbers were "
                  f"followed by better performance than category peers over the next {horizon} months, and applies that to today's "
                  f"numbers. Ranking 3 uses the same idea but first condenses the overlapping metrics into a few independent factors "
                  f"(PCA). None of the rankings is a forecast for any single fund."],
        "r1_rows": rows("rank_composite"), "r1_sub": f"How each fund did over {when}, averaged into one score. Rank 1 = strongest.",
        "r2_rows": [] if fit_error else rows("rank_model"),
        "r2_sub": f"Learnt from history: which metric patterns preceded better peer-relative results over the next {horizon} months.",
        "r2_unavailable": fit_error or "",
        "r3_rows": [] if (fit_error or "rank_pca" not in t or t["rank_pca"].isna().all()) else rows("rank_pca"),
        "r3_sub": "Same regression idea, but run on a few independent factors (PCA) instead of every overlapping metric.",
        "new_rows": new_rows,
        "new_note": ("These funds are too young for the full look-back window, so each is compared like-for-like: its record since "
                     "launch is ranked against category peers over exactly the same period. Under 12 months of data is too short to rank."),
        "backtest_intro": (f"We replayed history month by month, ranking funds using only what was known at the time, then checking "
                           f"what happened over the following {horizon} months."),
        "method_edges": [(pretty(m), float(score.loc[m, "top_vs_avg"])) for m in [R1, R2, R3, "Trailing 12M return", "Sharpe only"]
                         if score is not None and m in score.index],
        "chart_caption": (f"Average gain of the top-ranked 20% of funds over their category average ({horizon}-month outcome, "
                          f"{n_dates} past test dates)."),
        "points1": ms1["points"][:3] if ms1 else [], "points2": p2,
        "points3": (ms3["points"][:3] + ([ms3["vs_simple"]] if ms3["vs_simple"] else [])) if ms3 else [],
        "glossary": GLOSSARY, "benchmark_note": bnote,
    }
    if reg is not None and reg_tb is not None:
        top = reg_tb.head(5)
        summ["reg_rows"] = [(row["Metric"], f"{row['Weight']:+.1f}", "-" if pd.isna(row["uni_t"]) else f"{row['uni_t']:+.1f}",
                             row["What history says"]) for _, row in top.iterrows()]
        summ["reg_text"] = (f"Ranking 2 is built from this regression, fitted on {reg['n_obs']:,} fund-months ({reg['n_funds']} funds, "
                            f"{reg['n_dates']} months of history). It weighs each metric by how much it has helped predict later "
                            f"peer-relative performance. The five metrics with the most reliable link to later results are shown.")
        if pca_rep is not None:
            summ["reg_text"] += (f" Ranking 3 first condenses the {pca_rep['n_features']} overlapping metrics into {pca_rep['n_components']} "
                                 f"independent factors that keep {pca_rep['explained'].sum():.0%} of their information, then runs the same regression.")
        summ["reg_foot"] = (f"Weight = shift in predicted peer percentile between the best and worst fund on that metric, all else equal. "
                            f"R-squared {reg['r2_in']:.2f} in-sample: small values are normal when predicting fund performance.")
    return summ


GUIDE = """
### What this app does, in plain words
It looks at how each mutual fund behaved over the last few years (return, risk, how it held up in market falls, 1/2/3-year
returns) and produces **three rankings inside each category**:

- **Ranking 1: observed performance.** Simply: who did best on these measures over the look-back window. No forecasting.
- **Ranking 2: panel regression (forward-looking).** A regression learns from history which of these measures were followed by
  better results versus peers, then applies those lessons to today's numbers.
- **Ranking 3: PCA regression (forward-looking).** Many of the measures say nearly the same thing. PCA first condenses them into
  a handful of independent factors, and the same kind of regression is run on those. It often ranks similarly to Ranking 2; the
  backtest shows whether it is any steadier.

We then test all three honestly on past data (the **Have these worked before?** tab). If none has a track record, the app says so.

### The tabs
1. **Rankings**: the three rankings side by side, plus **newer funds** that are too young for the full window (see below).
   The button creates the VIKA-branded PDF.
2. **Have these worked before?**: replays history month by month with no peeking.
3. **Regression results**: what the regressions found: which metrics they lean on, how confident we can be, how well they fit, plus the PCA factors.
4. **Fund personalities**: groups funds by behaviour (aggressive, defensive, ...). Descriptive only; it does not rank.

### Newer funds
Funds younger than the look-back window cannot be scored on the same footing. Instead each is compared **like-for-like**: its
record since launch is ranked against every category peer over *exactly the same dates*. Funds with under 12 months of data are
listed with their return since launch but not ranked. Short records are noisy, so treat these with extra caution.

### Caveats
- **Past patterns weaken.** Research generally finds past *returns* are a weak guide to future returns; risk traits persist better.
- **A modest edge is realistic.** If the tab says "Weak" or "None found", believe it.
- **Survivorship.** Merged or closed funds may be missing from the data, which flatters history.
- **Benchmarks.** Where the Yahoo index ticker fails, an ETF proxy is used (noted in the app and PDF). ETF prices are market
  prices with a small cost drag. For best results put Total Return Index CSVs (columns Date, Close) in a `benchmarks/` folder.
- Not investment advice. Taxes, exit loads, suitability and liquidity are not considered.

### Glossary
| Term | Meaning |
|---|---|
| Category peers | Funds in the same SEBI category (e.g. Large Cap). Rankings are only ever within a category. |
| Look-back window | How many past months are used to compute each fund's metrics. |
| Forecast horizon | How far ahead we check whether the ranking worked (e.g. next 12 months). |
| CAGR | Compound annual growth rate: the steady yearly return that would produce the same total gain. |
| Top 20% edge | Average extra return of the funds ranked in the top fifth, versus the category average. |
| Backtest | Replaying history month by month using only what was known at the time. |
"""


# =========================================================================== #
# Cached loaders / computations
# =========================================================================== #
def add_new_funds(raw: pd.DataFrame, nav, meta, bench, freq: str) -> pd.DataFrame:
    new = compute_new_funds(nav, meta, bench, raw, freq=freq)
    return pd.concat([raw, new], ignore_index=True) if len(new) else raw


@st.cache_data(show_spinner="Loading saved panel…")
def load_saved(mtime: float):
    raw = pd.read_parquet(DATA_DIR / "panel.parquet")
    meta = pd.read_parquet(DATA_DIR / "meta.parquet").set_index("scheme_code")
    return raw, meta


@st.cache_data(show_spinner="Generating SYNTHETIC demo universe…")
def load_demo():
    meta, nav, bench = demo_universe(funds_per_category=25)
    raw = compute_panel_metrics(nav, meta, bench, freq="M")
    return add_new_funds(raw, nav, meta, bench, "M"), meta


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
    meta, nav = build_universe(list(cats), min_history_months=NEW_FUND_MIN_SHOW_MONTHS,
                               progress=lambda i, n: say(0.7 * i / n, f"Fund data: {i}/{n} schemes"))
    if meta.empty:
        raise ValueError("No funds matched. MFAPI may be unreachable or rate-limiting.")
    say(0.75, "Downloading benchmark indices / ETF proxies…")
    bench = load_benchmarks(sorted({CATEGORIES[c]["bench"] for c in cats}))
    say(0.85, "Computing rolling metrics…")
    raw = compute_panel_metrics(nav, meta, bench, freq=freq, window_months=window)
    say(0.95, "Scoring newer funds…")
    raw = add_new_funds(raw, nav, meta, bench, freq)
    store[key] = (time.time(), raw, meta)
    while len(store) > 2:
        store.pop(next(iter(store)))
    return raw, meta


@st.cache_data(show_spinner="Preparing the data…")
def get_prepared(_raw, src, cats, horizon, method):
    return prepare(_raw[_raw["category"].isin(cats)], horizon, FEATURES, method)


@st.cache_data(show_spinner="Replaying history to test the rankings (this can take a minute)…")
def get_backtest(_df, src, cats, horizon, method, features, target, min_train, step, alpha):
    return walk_forward(_df, list(features), horizon, [R2, R3], target,
                        min_train_months=min_train, step=step, ridge_alpha=alpha)


@st.cache_data(show_spinner="Fitting the regression…")
def get_latest(_df, src, cats, horizon, method, features, target, alpha, model_name=R2):
    out = rank_latest(_df, list(features), horizon, model_name, target, alpha)
    return out, dict(out.attrs)


@st.cache_data(show_spinner=False)
def get_pca(_df, src, cats, horizon, method, features, target, alpha):
    return pca_report(_df, list(features), TARGET_COL[target], alpha)


@st.cache_data(show_spinner=False)
def get_regression(_df, src, cats, horizon, method, features, target, alpha):
    return regression_report(_df, list(features), TARGET_COL[target], horizon, alpha)


@st.cache_data(show_spinner=False)
def get_ic(_df, src, cats, horizon, method, features):
    return univariate_ic(_df, list(features), "fwd_ret", horizon)


# =========================================================================== #
# Sidebar
# =========================================================================== #
st.title("📈 VIKA Fund Ranker")
st.caption("Ranks mutual funds within their category two ways, and shows honestly whether those rankings have worked before.")

win_months = None
has_new_info = True
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
                           help="Up to 10 years. Longer = steadier numbers. Funds with less history are shown separately as newer funds.")
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

    # newer funds travel in the same file (kind == 'new'); split them off from the main panel
    if "kind" in raw.columns:
        young_all, raw = raw[raw["kind"] == "new"].copy(), raw[raw["kind"] != "new"].copy()
    else:
        young_all, has_new_info = raw.iloc[0:0].copy(), False
    missing = [c for c in FEATURES if c not in raw.columns]
    if missing:
        st.error("This data was built before the 2- and 3-year CAGR metrics were added. Re-fetch (Live fetch) and save again.")
        st.stop()

    st.header("2. Question")
    with st.form("controls"):
        all_cats = [c for c in CATEGORIES if c in raw["category"].unique()]
        cats = st.multiselect("Categories", all_cats, default=all_cats)
        horizon = st.selectbox("How far ahead should the regression look?", [3, 6, 12, 24, 36], index=2,
                               format_func=lambda m: f"{m} months")
        target = st.selectbox("What should a good ranking predict?", list(TARGET_LABELS), index=0,
                              format_func=lambda k: TARGET_LABELS[k])
        with st.expander("Advanced settings"):
            method = st.radio("Scaling within category", ["rank", "zscore"], horizontal=True,
                              help="How each metric is compared against peers on the same date.")
            features = st.multiselect("Metrics to use", FEATURES, default=FEATURES,
                                      format_func=lambda k: FEATURE_INFO[k][0])
            min_train = st.slider("History used before the first test (months)", 24, 180, 60, step=6)
            step = st.slider("Re-learn every N months", 3, 12, 6, step=3)
            alpha = st.number_input("Smoothing strength (ridge alpha)", 0.1, 1000.0, 50.0)
        st.form_submit_button("Apply / re-run", type="primary")

if use_demo:
    st.warning("**SYNTHETIC DEMO DATA**: randomly generated, not real funds. Nothing here is a finding.", icon="⚠️")

if not cats or not features:
    st.info("Pick at least one category and metric, then press **Apply / re-run**.")
    st.stop()
if f"fwd_ret_{horizon}m" not in raw.columns:
    st.error(f"This saved data file predates the {horizon}-month horizon. Rebuild it with Live fetch and save it again.")
    st.stop()

cats_t, feats_t = tuple(cats), tuple(features)
df = get_prepared(raw, src, cats_t, horizon, method)
if df.empty:
    st.error("No category has enough funds on the same dates. Add categories, or shorten the look-back window.")
    st.stop()

# ---- backtest (shared by several tabs) ----
n_dates = df["date"].nunique()
min_train_eff = min_train
if n_dates <= min_train + horizon + 12:
    min_train_eff = max(24, n_dates - horizon - 12)
cols = [R2, R3] + list(BASELINES)                   # R1 is one of the BASELINES
preds, score, bt_error = None, None, None
try:
    preds = get_backtest(df, src, cats_t, horizon, method, feats_t, target, min_train_eff, step, alpha)
    score = score_predictions(preds, cols, horizon)
    if score is None or score.empty:
        raise ValueError("Too few funds have data on the same dates to test the rankings (each category needs at least "
                         f"{MIN_FUNDS_PER_CROSS_SECTION} funds at once). Try a shorter look-back window, more categories, "
                         "or a shorter horizon.")
except ValueError as e:
    bt_error, score = str(e), None

st.caption(f"{df['scheme_code'].nunique():,} funds in the main ranking · {df['category'].nunique()} categories · "
           f"{pd.Timestamp(df['date'].min()).strftime('%b %Y')} to {pd.Timestamp(df['date'].max()).strftime('%b %Y')}"
           + (f" · {len(young_all)} newer funds listed separately" if len(young_all) else "")
           + (f" · history before first test reduced to {min_train_eff} months (data is short)"
              if min_train_eff != min_train else ""))
with st.expander("Benchmarks used"):
    for c in cats:
        st.write("• " + (benchmark_note(df, c) or f"{c}: n/a"))

tab_rank, tab_bt, tab_reg, tab_clu, tab_about = st.tabs(
    ["🏆 Rankings", "🧪 Have these worked before?", "📐 Regression results", "🧩 Fund personalities", "ℹ️ How to read this"])

# =========================================================================== #
# Rankings
# =========================================================================== #
with tab_rank:
    cat = st.selectbox("Category", cats)
    fit_error, pca_error = None, None
    try:
        latest, attrs = get_latest(df, src, cats_t, horizon, method, feats_t, target, alpha, R2)
    except ValueError as e:                     # e.g. not enough history to train: Rankings 2 and 3 unavailable
        fit_error = str(e)
        latest = composite_fallback(df)
    if fit_error:
        latest["rank_pca"], latest["score_pca"] = np.nan, np.nan
    else:
        try:
            lp, _ = get_latest(df, src, cats_t, horizon, method, feats_t, target, alpha, R3)
            latest = latest.merge(lp[["scheme_code", "score", "rank_model"]].rename(
                columns={"score": "score_pca", "rank_model": "rank_pca"}), on="scheme_code", how="left")
        except Exception as e:
            pca_error = str(e)
            latest["rank_pca"], latest["score_pca"] = np.nan, np.nan
    asof = pd.Timestamp(latest["date"].max())
    ms1 = method_summary(score, R1, horizon) if score is not None and R1 in score.index else None
    ms2 = method_summary(score, R2, horizon) if score is not None and R2 in score.index and not fit_error else None
    ms3 = (method_summary(score, R3, horizon) if score is not None and R3 in score.index and not fit_error and not pca_error
           else None)
    win_txt = f"the last {win_months} months" if win_months else "the look-back window"

    def card(title, blurb, ms, unavailable=""):
        colour = ms["colour"] if ms else "#888888"
        body = (f"<b>Track record: <span style='color:{colour}'>{ms['label']}</span></b><br>{ms['points'][0]}" if ms
                else unavailable or "Track record not available for this setup.")
        return (f"<div style='border-left:6px solid {colour};background:#F7F6FB;padding:10px 14px;border-radius:4px;"
                f"min-height:170px'><b>{title}</b><br><span style='color:#555;font-size:0.9em'>{blurb}</span><br><br>{body}</div>")

    k1, k2, k3 = st.columns(3)
    k1.markdown(card("Ranking 1: observed performance",
                     f"Who did best over {win_txt}: return for risk, protection in falls, steadiness vs benchmark, and 1/2/3-year "
                     "returns (CAGR), averaged into one score. No forecasting.", ms1), unsafe_allow_html=True)
    k2.markdown(card("Ranking 2: panel regression (forward-looking)",
                     f"A regression learnt from history: which of these numbers were followed by better results than peers over "
                     f"the next {horizon} months. Details in the Regression results tab.", ms2,
                     "Could not be trained on this history. " + (fit_error or "")), unsafe_allow_html=True)
    k3.markdown(card("Ranking 3: PCA regression (forward-looking)",
                     "The same regression idea, but the overlapping metrics are first condensed into a few independent factors "
                     "(PCA), which can be steadier.", ms3,
                     "Could not be trained on this history. " + (fit_error or pca_error or "")), unsafe_allow_html=True)
    if fit_error:
        st.warning(fit_error)
    st.write("")

    plain = add_plain_columns(latest)
    t = plain[plain["category"] == cat].merge(meta[["scheme_name", "fund_house"]], left_on="scheme_code",
                                              right_index=True, how="left")
    order_opts = ["Ranking 1: observed performance"] + ([] if fit_error else ["Ranking 2: panel regression"]) + \
                 ([] if (fit_error or pca_error) else ["Ranking 3: PCA regression"])
    order = st.radio("Order the table by", order_opts, horizontal=True)
    t = t.sort_values({"Ranking 2": "rank_model", "Ranking 3": "rank_pca"}.get(order[:9], "rank_composite"))
    show = pd.DataFrame({
        "Rank 1 (observed)": t["rank_composite"].astype("Int64"), "Rank 2 (regression)": t["rank_model"].astype("Int64"),
        "Rank 3 (PCA)": t["rank_pca"].astype("Int64"),
        "Fund": t["scheme_name"], "Fund house": t["fund_house"], "What stands out": t["strengths"], "Watch-outs": t["watchouts"],
        "Return for risk (Sharpe)": t["sharpe_raw"], "Share of falls taken": t["down_capture_raw"],
        "Share of gains kept": t["up_capture_raw"], "Last 12M": t["ret_12m_raw"], "2Y CAGR": t["cagr_2y_raw"],
        "3Y CAGR": t["cagr_3y_raw"]})
    if fit_error:
        show = show.drop(columns=["Rank 2 (regression)", "Rank 3 (PCA)"])
    elif pca_error:
        show = show.drop(columns=["Rank 3 (PCA)"])
    st.caption(f"As of **{asof.strftime('%d %b %Y')}** · {len(show)} funds in {cat} · Ranks are within the category only · "
               "Hover over column names for meanings.")
    pct = st.column_config.NumberColumn(format="percent")
    st.dataframe(
        show, hide_index=True, width="stretch", height=520,
        column_config={
            "Rank 1 (observed)": st.column_config.NumberColumn(format="%d", width="small", help="Ranking 1: who did best over the window"),
            "Rank 2 (regression)": st.column_config.NumberColumn(format="%d", width="small", help="Ranking 2: forward-looking panel regression"),
            "Rank 3 (PCA)": st.column_config.NumberColumn(format="%d", width="small", help="Ranking 3: regression on PCA factors"),
            "What stands out": st.column_config.TextColumn(width="medium", help="Where the fund is in the best fifth of its category"),
            "Watch-outs": st.column_config.TextColumn(width="medium", help="Where the fund is in the weakest fifth of its category"),
            "Return for risk (Sharpe)": st.column_config.NumberColumn(format="%.2f", help=FEATURE_INFO["sharpe"][1]),
            "Share of falls taken": st.column_config.NumberColumn(format="percent", help=FEATURE_INFO["down_capture"][1]),
            "Share of gains kept": st.column_config.NumberColumn(format="percent", help=FEATURE_INFO["up_capture"][1]),
            "Last 12M": pct, "2Y CAGR": pct, "3Y CAGR": pct})

    # ---- newer funds ----
    yg = young_all[young_all["category"] == cat].merge(meta[["scheme_name", "fund_house"]], left_on="scheme_code",
                                                       right_index=True, how="left")
    if len(yg):
        st.subheader(f"Newer {cat} funds ({len(yg)})")
        st.caption("Too young for the full look-back window, so **not ranked against the funds above**. Each is compared "
                   "like-for-like: its record since launch versus category peers over exactly the same dates. "
                   f"Under {NEW_FUND_MIN_RANK_MONTHS} months of data is too short to rank. Short records are noisy: treat with caution.")
        yg = yg.sort_values(["lfl_rank", "history_months"], ascending=[True, False], na_position="last")
        newshow = pd.DataFrame({
            "Fund": yg["scheme_name"], "Fund house": yg["fund_house"], "Months of data": yg["history_months"],
            "Return since launch": yg["since_launch"], "Annualised return": yg["ann_return"],
            "Return for risk (Sharpe)": yg["sharpe"], "Share of falls taken": yg["down_capture"],
            "Share of gains kept": yg["up_capture"],
            "Rank vs peers over the same period": [f"{int(r)} of {int(n)}" if pd.notna(r) else "Too new to rank"
                                                   for r, n in zip(yg["lfl_rank"], yg["lfl_peers"])]})
        st.dataframe(newshow, hide_index=True, width="stretch", column_config={
            "Months of data": st.column_config.NumberColumn(format="%.0f"), "Return since launch": pct, "Annualised return": pct,
            "Return for risk (Sharpe)": st.column_config.NumberColumn(format="%.2f"),
            "Share of falls taken": pct, "Share of gains kept": pct})
    elif not has_new_info:
        st.caption("Newer funds are not in this saved file. Re-fetch with Live fetch to include them.")

    d1, d2, _ = st.columns([1, 1, 2])
    full = t[["rank_composite", "rank_model", "rank_pca", "scheme_name", "fund_house", "strengths", "watchouts", "score", "score_pca"]
             + [f + "_raw" for f in FEATURES]].rename(columns={"rank_composite": "rank1_observed", "rank_model": "rank2_regression",
                                                               "rank_pca": "rank3_pca", "score": "score_regression"})
    d1.download_button("Download table (CSV)", full.to_csv(index=False).encode(),
                       f"ranking_{cat.replace(' ', '_')}_{asof.date()}.csv", "text/csv")
    if ms1:
        try:
            rep = None if fit_error else get_regression(df, src, cats_t, horizon, method, feats_t, target, alpha)
            prep = None if (fit_error or pca_error) else get_pca(df, src, cats_t, horizon, method, feats_t, target, alpha)
            pdf = build_pdf_report(make_report_summary(
                cat, asof, horizon, plain[plain["category"] == cat].merge(
                    meta[["scheme_name"]], left_on="scheme_code", right_index=True, how="left"),
                yg, ms1, ms2, score, cols, win_months, preds["date"].nunique(), rep,
                reg_table(rep) if rep else None, fit_error, benchmark_note(df, cat), ms3, prep))
            d2.download_button("📄 Download VIKA PDF", pdf, f"VIKA_Fund_Ranking_{cat.replace(' ', '_')}_{asof.date()}.pdf",
                               "application/pdf", type="primary")
        except Exception as e:                       # never let a PDF problem break the page
            d2.error(f"PDF failed: {e}")
    st.caption("A high rank is a statistical tilt, not a guarantee. The next tab shows how much each ranking has been worth.")

# =========================================================================== #
# Have these worked before?
# =========================================================================== #
with tab_bt:
    if bt_error:
        st.error(bt_error)
    else:
        st.caption(f"We replayed history month by month ({preds['date'].nunique()} test dates, "
                   f"{pd.Timestamp(preds['date'].min()).strftime('%b %Y')} to {pd.Timestamp(preds['date'].max()).strftime('%b %Y')}), "
                   f"ranking funds using only what was known at the time, then looking {horizon} months ahead.")
        for m in (R1, R2, R3):
            ms_ = method_summary(score, m, horizon)
            st.subheader(pretty(m))
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Top-ranked 20% vs category average", f"{ms_['edge']:+.1%}", help=f"Average extra return over the next {horizon} months")
            m2.metric("Bottom-ranked 20% vs average", f"{ms_['bottom']:+.1%}")
            m3.metric("Top picks beat the typical fund", f"{ms_['hit']:.0%}", help="Share of test periods")
            m4.metric("Evidence", ms_["label"])
            if ms_["vs_simple"]:
                st.markdown("**" + ms_["vs_simple"] + "**")
            st.markdown(f"<div style='border-left:6px solid {ms_['colour']};background:#FFF9EF;padding:10px 14px;border-radius:4px'>"
                        + "<br>".join(ms_["points"][:3] + ms_["points"][-1:]) + "</div>", unsafe_allow_html=True)

        st.subheader("Side by side")
        bar_df = score.reset_index()[["model", "top_vs_avg"]]
        bar_df["Type"] = np.where(bar_df["model"].isin([R1, R2, R3]), "Our rankings", "Simple rule")
        bar_df["Method"] = bar_df["model"].map(pretty)
        figb = px.bar(bar_df, x="Method", y="top_vs_avg", color="Type",
                      color_discrete_map={"Our rankings": "#534191", "Simple rule": "#F6BA47"},
                      labels={"top_vs_avg": f"Top-20% edge over category average ({horizon}M)"})
        figb.update_yaxes(tickformat=".1%")
        figb.update_layout(height=360, xaxis_title="")
        st.plotly_chart(figb, width="stretch")
        st.caption("Taller is better. The two simple rules (gold) are the yardstick: a ranking that cannot beat them is not adding value.")

        st.subheader("Did it work consistently, or only in some periods?")
        ts = pd.DataFrame({pretty(m): top_bucket_series(preds, m) for m in (R1, R2, R3)}).rolling(12, min_periods=6).mean()
        figt = px.line(ts, labels={"value": "Top-20% edge (12-month average)", "date": "", "variable": ""},
                       color_discrete_sequence=["#F6BA47", "#534191", "#1E9A51"])
        figt.add_hline(y=0, line_dash="dot", line_color="grey")
        figt.update_yaxes(tickformat=".0%")
        figt.update_layout(height=340, legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(figt, width="stretch")
        st.caption("Above the dotted line = the top-ranked funds beat the average in that stretch. "
                   "A line that dips below zero shows the ranking does not work in every market.")

        with st.expander("By category"):
            bc = score_predictions(preds, [R1, R2, R3], horizon, by_category=True).reset_index()
            bc["model"] = bc["model"].map(pretty)
            bc = bc.rename(columns={"model": "Ranking", "top_vs_avg": "Top-20% edge", "top_beats_median": "Top picks beat typical fund",
                                    "n_dates": "Test dates"})
            st.dataframe(bc[["category", "Ranking", "Top-20% edge", "Top picks beat typical fund", "Test dates"]], hide_index=True,
                         width="stretch", column_config={"Top-20% edge": pct, "Top picks beat typical fund": pct})
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
# Regression results
# =========================================================================== #
def _regression_tab():
    st.subheader("The regression behind Ranking 2")
    st.markdown(
        f"For every fund and every past month we recorded its metrics, then what happened next: **how it ranked against its category "
        f"peers over the following {horizon} months** ({TARGET_LABELS[target].lower()}). The regression finds the weighting of metrics that "
        "best lined up with what happened next. That weighting is then applied to today's numbers to produce Ranking 2.")
    try:
        rep = get_regression(df, src, cats_t, horizon, method, feats_t, target, alpha)
    except Exception as e:
        st.info("Not enough completed history to fit the regression for this horizon. Try a shorter forecast horizon or "
                f"look-back window. ({e})")
        return
    r2_oos = ic_oos = None
    if preds is not None and score is not None and R2 in score.index:
        p = preds[[R2, "target"]].dropna()
        if len(p) > 30:
            r2_oos = 1 - ((p["target"] - p[R2]) ** 2).sum() / ((p["target"] - p["target"].mean()) ** 2).sum()
        ic_oos = score.loc[R2, "mean_IC"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations", f"{rep['n_obs']:,}", help=f"Fund-months with a completed {horizon}-month outcome")
    c2.metric("Funds / months", f"{rep['n_funds']} / {rep['n_dates']}")
    c3.metric("R-squared (in-sample)", f"{rep['r2_in']:.3f}", help="Share of the variation in later peer-rank explained on the data it was fitted to")
    c4.metric("R-squared (out-of-sample)", "n/a" if r2_oos is None else f"{r2_oos:.3f}",
              help="Same, but on months the regression had never seen (the honest number). Can be negative.")
    st.caption("Predicting fund performance is hard: R-squared of 0.01 to 0.05 out-of-sample is typical, and a negative value means "
               "the regression did worse than guessing 'average' for every fund." + (f" Out-of-sample rank correlation (IC): {ic_oos:.3f}." if ic_oos is not None else ""))

    tb = reg_table(rep)
    unit = "percentile points" if method == "rank" else "points per standard deviation"
    st.markdown("#### Which metrics does it lean on?")
    cmap = {UP_TXT: "#1E9A51", DOWN_TXT: "#d98a00", NONE_TXT: "#B8B8C4"}
    figr = px.bar(tb.reset_index().sort_values("Weight"), x="Weight", y="Metric", orientation="h", color="What history says",
                  color_discrete_map=cmap, labels={"Weight": f"Weight in Ranking 2 ({unit})"})
    figr.update_layout(height=520, legend=dict(orientation="h", y=-0.2, title=""), yaxis_title="")
    st.plotly_chart(figr, width="stretch")
    st.caption("**Weight** = how far a fund's predicted peer-percentile moves between the worst and best fund on that metric, all else equal "
               f"({unit}). Positive = higher values push a fund UP the forward-looking ranking. **Colour** = whether that metric, on its own, "
               "has reliably lined up with later results (green: higher went with better, amber: higher went with worse, grey: no reliable link). "
               "For 'lower is better' metrics such as downside capture or volatility, amber is the expected, healthy result. "
               "Many metrics overlap, so weights are shrunk (ridge) to stay stable.")
    out = tb[["Metric", "Weight", "uni_ic", "uni_t", "What history says", "What it means"]].rename(
        columns={"uni_ic": "Link on its own (-1 to +1)", "uni_t": "Confidence (t-stat)"})
    st.dataframe(out, hide_index=True, width="stretch", column_config={
        "Weight": st.column_config.NumberColumn(format="%+.1f"),
        "Link on its own (-1 to +1)": st.column_config.NumberColumn(format="%+.2f"),
        "Confidence (t-stat)": st.column_config.NumberColumn(format="%.1f", help="Newey-West adjusted. Beyond +/-2 is unlikely to be luck.")})

    with st.expander("After accounting for the other metrics (Fama-MacBeth)"):
        fmt = tb[["Metric", "fm_coef", "fm_t"]].rename(columns={"fm_coef": "Independent effect", "fm_t": "Confidence (t-stat)"})
        st.dataframe(fmt, hide_index=True, width="stretch", column_config={
            "Independent effect": st.column_config.NumberColumn(format="%+.3f"), "Confidence (t-stat)": st.column_config.NumberColumn(format="%.1f")})
        st.caption("Many metrics say nearly the same thing, so once the others are held constant each one's independent effect is small and "
                   "noisy. Low numbers here do not mean the regression is useless: it works with the metrics as a group.")
    st.markdown("#### Ranking 3: the PCA view")
    try:
        pr = get_pca(df, src, cats_t, horizon, method, feats_t, target, alpha)
    except Exception as e:
        st.info(f"PCA view unavailable for this setup. ({e})")
        pr = None
    if pr is not None:
        st.markdown(
            f"Many metrics say nearly the same thing (Sharpe, Sortino, alpha, ...). **PCA** condenses the {pr['n_features']} metrics into "
            f"**{pr['n_components']} independent factors** that keep {pr['explained'].sum():.0%} of their information. The regression is then "
            "run on those factors instead of on every overlapping metric, which can make it steadier.")
        exp_df = pd.DataFrame({"Factor": pr["load"].index, "Share of information": pr["explained"]})
        figp = px.bar(exp_df, x="Factor", y="Share of information", text=exp_df["Share of information"].map(lambda v: f"{v:.0%}"),
                      color_discrete_sequence=["#534191"])
        figp.update_yaxes(tickformat=".0%")
        figp.update_layout(height=300, xaxis_title="")
        st.plotly_chart(figp, width="stretch")
        rows = []
        for i, fn in enumerate(pr["load"].index):
            lo = pr["load"].loc[fn]
            top3 = lo.abs().sort_values(ascending=False).index[:3]
            rows.append({"Factor": fn, "Mostly made of": ", ".join(("+" if lo[k] > 0 else "-") + SHORT_NAMES[k] for k in top3),
                         "Share of information": pr["explained"][i], "Weight in Ranking 3": pr["factor_coef"][fn] * 100,
                         "Confidence (t-stat)": pr["factor_t"].get(fn, np.nan)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
            "Share of information": pct, "Weight in Ranking 3": st.column_config.NumberColumn(format="%+.1f"),
            "Confidence (t-stat)": st.column_config.NumberColumn(format="%.1f")})
        st.caption("'+Sortino' means a fund scores higher on that factor when its Sortino is higher; '-Volatility' means when volatility is lower. "
                   f"Fit: R-squared {pr['r2_in']:.3f} in-sample (Ranking 2: {rep['r2_in']:.3f}). PCA discards small factors, so it usually fits "
                   "slightly less on past data but can hold up better on new data. Compare the two in the Have these worked before? tab."
                   + (f" Out-of-sample rank correlation (IC): Ranking 2 {score.loc[R2, 'mean_IC']:.3f}, Ranking 3 {score.loc[R3, 'mean_IC']:.3f}."
                      if score is not None and R3 in score.index and R2 in score.index else ""))
        keys = list(tb.index)
        cmp = pd.DataFrame({"Metric": [FEATURE_INFO[k][0] for k in keys], "Ranking 2 weight": tb["Weight"].to_numpy(),
                            "Ranking 3 weight": (pr["implied"].reindex(keys) * 100).to_numpy()})
        with st.expander("Implied weight per metric: Ranking 2 vs Ranking 3"):
            st.dataframe(cmp, hide_index=True, width="stretch", column_config={
                "Ranking 2 weight": st.column_config.NumberColumn(format="%+.1f"), "Ranking 3 weight": st.column_config.NumberColumn(format="%+.1f")})
            st.caption("Ranking 3's factor weights translated back into a weight on each original metric, so the two regressions can be compared directly.")
        with st.expander("Full PCA regression output (statsmodels OLS on the factors, errors clustered by month)"):
            st.code(pr["ols_summary"], language=None)

    with st.expander("Full regression output (statsmodels OLS, errors clustered by month)"):
        st.code(rep["ols_summary"], language=None)
        st.caption("This pooled OLS is shown for transparency. Ranking 2 itself uses the ridge-shrunk version (the Weight column), "
                   "which is steadier when metrics overlap.")
    with st.expander("Which metrics say the same thing?"):
        corr = df[list(features)].corr(method="spearman")
        corr.index = corr.columns = [SHORT_NAMES[k] for k in corr.columns]
        st.plotly_chart(px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", text_auto=".2f",
                                  aspect="auto").update_layout(height=560), width="stretch")
        st.caption("Dark red = the two metrics almost always agree. This is why the regression shrinks weights instead of trusting each one blindly.")


with tab_reg:
    _regression_tab()


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
