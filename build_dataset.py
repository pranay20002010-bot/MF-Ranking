"""Build the metric panel offline and save it to ./data/ (the Streamlit app only reads these files).

Why offline?  Downloading ~1,500 NAV histories takes minutes; a Streamlit page should never do that
on every visit, and Streamlit Community Cloud has an ephemeral disk. So: run this locally (or in the
GitHub Action), commit data/panel.parquet + data/meta.parquet, and let the app read them.

Usage
-----
    python build_dataset.py --check-benchmarks          # verify Yahoo tickers / CSVs first
    python build_dataset.py                             # all categories, monthly returns, 36M window
    python build_dataset.py --freq D --window 36        # daily returns instead of monthly
    python build_dataset.py --categories "Large Cap" "Mid Cap" "Flexi Cap"
    python build_dataset.py --demo                      # synthetic data, no network
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from mfquant import config, data, metrics, synthetic

OUT = Path(__file__).parent / "data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="*", default=None, help="subset of config.CATEGORIES keys")
    ap.add_argument("--freq", choices=["M", "D"], default="M", help="return frequency for the metrics")
    ap.add_argument("--window", type=int, default=config.METRIC_WINDOW_MONTHS, help="lookback in months")
    ap.add_argument("--rf", type=float, default=config.RISK_FREE_ANNUAL, help="annual risk-free rate")
    ap.add_argument("--max-age-days", type=float, default=7.0, help="reuse cached NAV downloads younger than this")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--check-benchmarks", action="store_true")
    ap.add_argument("--demo", action="store_true", help="synthetic data (offline smoke test)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if a.check_benchmarks:
        for k, v in data.check_benchmarks().items():
            print(f"{k:15s} {v}")
        return

    cats = a.categories or list(config.CATEGORIES)
    if a.demo:
        meta, nav, bench = synthetic.demo_universe(categories=cats)
    else:
        def prog(i, n):
            if i % 100 == 0 or i == n:
                print(f"  downloaded {i}/{n} schemes", flush=True)

        print("1/3  fund universe + NAVs from MFAPI ...")
        meta, nav = data.build_universe(cats, max_workers=a.workers, max_age_days=a.max_age_days, progress=prog)
        print(f"     kept {len(meta)} funds:\n{meta['category'].value_counts().to_string()}")
        print("2/3  benchmarks ...")
        keys = sorted({config.CATEGORIES[c]["bench"] for c in cats})
        bench = data.load_benchmarks(keys)

    print("3/3  rolling metrics ...")
    panel = metrics.compute_panel_metrics(nav, meta, bench, freq=a.freq, window_months=a.window, rf_annual=a.rf)

    OUT.mkdir(exist_ok=True)
    panel.to_parquet(OUT / "panel.parquet", index=False)
    meta.reset_index().to_parquet(OUT / "meta.parquet", index=False)
    (OUT / "build_info.json").write_text(json.dumps({
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "demo": bool(a.demo), "freq": a.freq, "window_months": a.window, "rf_annual": a.rf,
        "n_funds": int(meta.shape[0]), "n_rows": int(panel.shape[0]),
        "first_date": str(panel["date"].min().date()), "last_date": str(panel["date"].max().date()),
        "categories": cats,
    }, indent=2))
    print(f"saved {len(panel):,} rows -> {OUT / 'panel.parquet'}")


if __name__ == "__main__":
    main()
