# Mutual Fund Quant Ranker

Pulls Direct-Growth NAV history from [MFAPI](https://www.mfapi.in), benchmarks from yfinance / CSV, computes rolling
risk-adjusted metrics per fund, and tests whether a **panel model** can rank funds better than the usual
"normalise everything and average" approach - using a proper walk-forward backtest.

```
mfquant/
  config.py      categories, benchmark map, defaults           <- edit this first
  data.py        MFAPI + benchmark download, on-disk cache
  metrics.py     Sharpe, Sortino, Treynor, IR, alpha, beta, up/down capture, vol, MDD, momentum
  panel.py       forward targets + within-(category,date) standardisation
  models.py      Ridge / PCA+Ridge / Gradient Boosting, IC + Fama-MacBeth diagnostics, K-means
  evaluate.py    purged walk-forward backtest, scoring, latest ranking
  synthetic.py   fake data generator for smoke tests / demo
build_dataset.py offline pipeline -> data/panel.parquet
app.py           Streamlit UI (reads data/, never hits the APIs)
tests/           pytest sanity checks (formulas, no look-ahead, no leakage)
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-build.txt

python build_dataset.py --check-benchmarks     # 1. make sure every benchmark resolves
python build_dataset.py --categories "Large Cap" "Flexi Cap"   # 2. small first run
python build_dataset.py                        # 3. full universe (a few minutes)
streamlit run app.py
```

No network yet? `python build_dataset.py --demo` (or just run the app with no `data/` folder) uses synthetic data.
The app shows a banner whenever it's on synthetic data. **Don't read anything into results from it.**

## Benchmarks - do this before trusting any number

Yahoo Finance only has *price* indices and patchy NSE sub-index coverage. Fund NAVs are dividend-inclusive, so a
price-index benchmark overstates alpha / information ratio slightly. Best practice: download **Total Return Index**
history from niftyindices.com and save it as `benchmarks/<KEY>.csv` (columns `Date,Close`; keys are in
`config.BENCHMARKS`). A CSV always wins over the yfinance ticker. Tickers in `config.py` are best guesses -
`--check-benchmarks` tells you which ones fail.

## Deploy on Streamlit Community Cloud

1. Run `build_dataset.py` locally, then commit `data/panel.parquet`, `data/meta.parquet`, `data/build_info.json`.
2. Push to GitHub, then on share.streamlit.io choose the repo and `app.py` as the entry point.
3. Optional: `.github/workflows/refresh.yml` rebuilds and commits the data on the 3rd of each month.

The app is deliberately data-light at runtime: it needs `requirements.txt` only (no yfinance, no live API calls).

## Design decisions worth knowing

| Decision | Why |
|---|---|
| Target = rank within category, not absolute return | Absolute returns are market-regime driven; fund metrics can't predict the market |
| Features standardised within (category, date) | Same effect as time fixed effects in a panel regression; no look-ahead |
| Purged expanding-window walk-forward | Overlapping H-month targets leak badly under random K-fold |
| Newey-West t-stats, lag = H-1 | Consecutive months share most of their forward window |
| Baselines in every backtest | Model must beat the naive composite to earn its complexity |
| K-means only for archetypes | No target -> can't rank; it's a descriptive tool |

## Extending

- Add a feature: compute it in `metrics.window_metrics`, add its name to `config.FEATURES`.
- Add a category: add an entry to `config.CATEGORIES` (`match` is a substring of MFAPI's `scheme_category`).
- Add a model: add a branch in `models.make_model` and its name to `MODEL_NAMES`.
- Real risk-free series: replace the flat `RISK_FREE_ANNUAL` in `metrics.compute_panel_metrics`.

## Known limitations

MFAPI carries NAV only (no AUM, expense ratio, manager tenure - all known predictors). Survivorship bias depends on
how many defunct schemes MFAPI returns. Not investment advice.
