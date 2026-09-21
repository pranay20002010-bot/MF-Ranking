"""Streamlit front-end. Run:  streamlit run app.py"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

from mfquant import config, evaluate, metrics, models, panel, synthetic

warnings.filterwarnings("ignore")
st.set_page_config(page_title="MF Quant Ranker", page_icon="📈", layout="wide")

DATA = Path(__file__).parent / "data"
HAVE_REAL = (DATA / "panel.parquet").exists() and (DATA / "meta.parquet").exists()


# --------------------------------------------------------------------------- #
# Cached loaders / computations
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading panel…")
def load_real(mtime: float):
    raw = pd.read_parquet(DATA / "panel.parquet")
    meta = pd.read_parquet(DATA / "meta.parquet").set_index("scheme_code")
    info = json.loads((DATA / "build_info.json").read_text()) if (DATA / "build_info.json").exists() else {}
    return raw, meta, info


@st.cache_data(show_spinner="Generating SYNTHETIC demo universe…")
def load_demo():
    meta, nav, bench = synthetic.demo_universe(funds_per_category=25)
    raw = metrics.compute_panel_metrics(nav, meta, bench, freq="M")
    return raw, meta, {"demo": True, "freq": "M", "window_months": config.METRIC_WINDOW_MONTHS}


@st.cache_data(show_spinner="Standardising cross-sections…")
def get_prepared(_raw: pd.DataFrame, src: str, cats: tuple, horizon: int, method: str) -> pd.DataFrame:
    return panel.prepare(_raw[_raw["category"].isin(cats)], horizon, config.FEATURES, method)


@st.cache_data(show_spinner="Running walk-forward backtest (this can take a minute)…")
def get_backtest(_df, src, cats, horizon, method, features, model_names, target, min_train, step, alpha):
    return evaluate.walk_forward(_df, list(features), horizon, list(model_names), target,
                                 min_train_months=min_train, step=step, ridge_alpha=alpha)


@st.cache_data(show_spinner="Fitting on all realised history and scoring latest month…")
def get_latest(_df, src, cats, horizon, method, features, model_name, target, alpha):
    out = evaluate.rank_latest(_df, list(features), horizon, model_name, target, alpha)
    return out, dict(out.attrs)


@st.cache_data(show_spinner=False)
def get_ic(_df, src, cats, horizon, method, features):
    return models.univariate_ic(_df, list(features), "fwd_ret", horizon)


@st.cache_data(show_spinner=False)
def get_fm(_df, src, cats, horizon, method, features, target):
    return models.fama_macbeth(_df, list(features), evaluate.TARGET_COL[target], horizon)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.title("📈 Mutual Fund Quant Ranker")

with st.sidebar:
    st.header("Setup")
    use_demo = True
    if HAVE_REAL:
        use_demo = st.toggle("Use SYNTHETIC demo data", value=False,
                             help="Random data to explore the app. Not real funds.")
    if use_demo:
        raw, meta, info = load_demo()
        src = "demo"
    else:
        raw, meta, info = load_real((DATA / "panel.parquet").stat().st_mtime)
        src = f"real-{(DATA / 'panel.parquet').stat().st_mtime:.0f}"

    with st.form("controls"):
        all_cats = [c for c in config.CATEGORIES if c in raw["category"].unique()]
        cats = st.multiselect("Categories", all_cats, default=all_cats)
        horizon = st.selectbox("Forecast horizon (months)", [3, 6, 12], index=2)
        target = st.selectbox("Prediction target", list(panel.TARGETS), index=0,
                              format_func=lambda k: panel.TARGETS[k])
        method = st.radio("Cross-sectional scaling", ["rank", "zscore"], horizontal=True,
                          help="Applied within each (category, date) group.")
        features = st.multiselect("Features", config.FEATURES, default=config.FEATURES)
        model_names = st.multiselect("Models", models.MODEL_NAMES, default=models.MODEL_NAMES)
        min_train = st.slider("Min training history (months)", 36, 96, 60, step=6)
        step = st.slider("Re-fit every N months", 3, 12, 6, step=3)
        alpha = st.number_input("Ridge alpha", 0.1, 1000.0, 50.0)
        st.form_submit_button("Apply / re-run", type="primary")

if use_demo:
    st.warning("**SYNTHETIC DEMO DATA** – randomly generated, not real funds. Build the real dataset with "
               "`python build_dataset.py` and commit `data/panel.parquet`.", icon="⚠️")
elif info:
    st.caption(f"Data built {info.get('built_utc', '?')} · {info.get('freq', '?')} returns · "
               f"{info.get('window_months', '?')}M metric window · {info.get('n_funds', '?')} funds · "
               f"{info.get('first_date', '?')} → {info.get('last_date', '?')}")

if not cats or not features or not model_names:
    st.info("Pick at least one category, feature and model, then press **Apply / re-run**.")
    st.stop()

cats_t, feats_t, mods_t = tuple(cats), tuple(features), tuple(model_names)
df = get_prepared(raw, src, cats_t, horizon, method)
if df.empty:
    st.error("No cross-sections with enough funds. Lower MIN_FUNDS_PER_CROSS_SECTION in config.py or add categories.")
    st.stop()

tab_rank, tab_bt, tab_sig, tab_clu, tab_about = st.tabs(
    ["🏆 Rankings", "🧪 Backtest", "🔬 What predicts?", "🧩 Archetypes", "ℹ️ Method & caveats"])

# --------------------------------------------------------------------------- #
# Rankings
# --------------------------------------------------------------------------- #
with tab_rank:
    c1, c2 = st.columns([1, 1])
    cat = c1.selectbox("Category", cats)
    rank_model = c2.selectbox("Ranking model", model_names)
    latest, attrs = get_latest(df, src, cats_t, horizon, method, feats_t, rank_model, target, alpha)
    asof = latest["date"].max()
    st.caption(f"Scores as of **{pd.Timestamp(asof).date()}**. Model trained on rows whose {horizon}-month "
               f"outcome was already realised (last training date {pd.Timestamp(attrs['train_end']).date()}, "
               f"{attrs['train_rows']:,} fund-months). Score = predicted "
               f"{panel.TARGETS[target].lower()}.")
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
    st.caption("Metrics shown are the raw values that fed the model. A high model rank is a *statistical tilt*, "
               "not a guarantee -- check the Backtest tab for how much signal there actually is.")

# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
with tab_bt:
    try:
        preds = get_backtest(df, src, cats_t, horizon, method, feats_t, mods_t, target, min_train, step, alpha)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    cols = list(model_names) + list(evaluate.BASELINES)
    score = evaluate.score_predictions(preds, cols, horizon)
    st.subheader("Out-of-sample scorecard")
    st.caption(f"Expanding-window walk-forward, purged by the {horizon}-month horizon, refit every {step} months. "
               f"{preds['date'].nunique()} test dates from {pd.Timestamp(preds['date'].min()).date()} to "
               f"{pd.Timestamp(preds['date'].max()).date()}.")
    st.dataframe(
        score, width="stretch",
        column_config={
            "mean_IC": st.column_config.NumberColumn("Mean rank IC", format="%.3f", help="Avg Spearman corr between score and realised forward return"),
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
    ic_ts = evaluate.ic_time_series(preds, cols)
    roll = ic_ts.rolling(12, min_periods=6).mean()
    fig = px.line(roll, labels={"value": "IC (12M avg)", "date": "", "variable": ""})
    fig.add_hline(y=0, line_dash="dot", line_color="grey")
    fig.update_layout(height=380, legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Top-20% minus category average (forward return)")
    bar = score.reset_index()
    figb = px.bar(bar, x="model", y="top_vs_avg", labels={"top_vs_avg": "avg excess of top-20%", "model": ""})
    figb.update_yaxes(tickformat=".1%")
    figb.update_layout(height=320)
    st.plotly_chart(figb, width="stretch")

    with st.expander("Breakdown by category"):
        by_cat = evaluate.score_predictions(preds, cols, horizon, by_category=True)
        st.dataframe(by_cat[["mean_IC", "IC_IR", "t_stat_NW", "top_vs_avg", "spread", "n_dates"]].round(3),
                     width="stretch")

# --------------------------------------------------------------------------- #
# What predicts?
# --------------------------------------------------------------------------- #
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
               "window. |t| > 2 is the usual bar; with overlapping data be sceptical of anything near it.")

    st.subheader("Fama-MacBeth: joint effect (controls for the other metrics)")
    fm = get_fm(df, src, cats_t, horizon, method, feats_t, target)
    st.dataframe(fm.round(3), width="stretch")
    st.caption("Sharpe, Sortino, Treynor and IR are highly correlated, so individual coefficients here can be "
               "unstable and even flip sign -- that's multicollinearity, and the reason PCA / Ridge exist.")

    st.subheader("Redundancy between metrics")
    corr = df[list(features)].corr(method="spearman")
    st.plotly_chart(px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", text_auto=".2f",
                              aspect="auto").update_layout(height=520), width="stretch")
    if len(features) >= 3:
        p = PCA().fit(df[list(features)].to_numpy())
        ev = p.explained_variance_ratio_
        figp = go.Figure()
        figp.add_bar(x=[f"PC{i + 1}" for i in range(len(ev))], y=ev, name="explained")
        figp.add_scatter(x=[f"PC{i + 1}" for i in range(len(ev))], y=np.cumsum(ev), name="cumulative")
        figp.update_layout(height=320, yaxis_tickformat=".0%", title="PCA on the standardised metrics")
        st.plotly_chart(figp, width="stretch")

# --------------------------------------------------------------------------- #
# Archetypes
# --------------------------------------------------------------------------- #
with tab_clu:
    st.subheader("Fund archetypes (K-means on risk profile) -- descriptive, not predictive")
    c1, c2 = st.columns(2)
    cc = c1.selectbox("Category ", cats, key="clu_cat")
    k = c2.slider("Clusters", 2, 6, 4)
    snap = df[(df["date"] == df["date"].max()) & (df["category"] == cc)].merge(
        meta[["scheme_name"]], left_on="scheme_code", right_index=True, how="left")
    cl = models.cluster_snapshot(snap, k)
    if cl.empty or "pc1" not in cl:
        st.info("Not enough funds for clustering in this category.")
    else:
        cl["cluster"] = cl["cluster"].astype(str)
        st.plotly_chart(px.scatter(cl, x="pc1", y="pc2", color="cluster", hover_name="scheme_name",
                                   hover_data={"pc1": False, "pc2": False}).update_layout(height=460),
                        width="stretch")
        prof = cl.groupby("cluster")[[c + "_raw" for c in models.CLUSTER_FEATURES]].mean()
        prof.columns = models.CLUSTER_FEATURES
        prof["n_funds"] = cl.groupby("cluster").size()
        st.dataframe(prof.round(3), width="stretch")
        st.caption(f"Silhouette = {cl.attrs.get('silhouette', float('nan')):.2f} (closer to 1 = cleaner clusters; "
                   "below ~0.25 means the 'archetypes' are mostly arbitrary slices of a continuum).")

# --------------------------------------------------------------------------- #
# About
# --------------------------------------------------------------------------- #
with tab_about:
    st.markdown(Path(__file__).with_name("METHODOLOGY.md").read_text())
