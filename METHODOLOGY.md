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
