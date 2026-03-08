# Trader Sentiment Analysis — Summary Write-Up

## Methodology

We investigated the relationship between the **Crypto Fear & Greed Index (FGI)** and trader performance on **HyperLiquid** using two datasets: daily FGI readings and trade-level execution data.

**Data Preparation.** Timestamps were converted from IST (`DD-MM-YYYY HH:MM`) to datetime objects and each trade was mapped to its calendar date for daily aggregation. The FGI was binned into four sentiment regimes — *Extreme Fear*, *Fear*, *Greed*, *Extreme Greed* — and merged with trade data on the date key. Key metrics were computed per trader per day: total PnL, win rate, average trade size (USD), number of trades, and long/short ratio.

**Segmentation.** Traders were segmented along three independent axes:
- **Frequency** — *Frequent* (above-median daily trade count) vs. *Infrequent*.
- **Consistency** — *Consistent Winners* (win rate ≥ 55%) vs. *Inconsistent*.
- **Position Size** — *Large* (above-median average trade size) vs. *Small*.

Each segment's performance was then cross-tabulated against Fear vs. Greed days.

**Modelling (Bonus).** A supervised classifier (Random Forest / Gradient Boosting with TimeSeriesSplit cross-validation) was trained to predict whether next-day aggregate PnL would be positive or negative, using lagged features (FGI value, rolling PnL, trade count, win rate). KMeans clustering on trader profiles identified natural behavioural archetypes.

---

## Key Insights

1. **Performance diverges sharply by sentiment.** Average daily PnL and win rate differ materially between Fear and Greed regimes, confirming that market mood is a meaningful conditioning variable for trader outcomes.

2. **Traders shift behaviour with sentiment.** The long/short ratio and average trade size both change between Fear and Greed days — traders increase long exposure and position sizes during periods of greed, signalling higher confidence (and potentially higher risk).

3. **Consistent Winners trade differently.** High-win-rate traders exhibit distinct position-sizing and frequency patterns compared to inconsistent traders, and their edge is more resilient across sentiment regimes.

4. **Risk exposure varies by regime.** The percentage of negative-PnL days shifts between Fear and Greed periods, underscoring the need for regime-aware risk management.

---

## Strategy Recommendations

### Strategy 1 — Sentiment-Adaptive Position Sizing

| Condition | Action |
|-----------|--------|
| Win rate < 50% **and** FGI < 45 (Fear) | Cut position size 30–50%; tighten stops to 2% |
| Win rate ≥ 50% **and** FGI < 45 (Fear) | Maintain sizing — contrarian opportunity; high-conviction setups only |
| FGI > 75 (Extreme Greed) | Reduce exposure 25–40%; take partial profits |

### Strategy 2 — Frequency Tuning by Sentiment Regime

| Condition | Action |
|-----------|--------|
| FGI < 30 (Extreme Fear) | Cut trade count 50%; longer time-frames (4H/Daily); avoid scalping |
| FGI 45–55 (Neutral) | Normal frequency & sizing — best risk-reward window |
| FGI > 75 (Extreme Greed) | Reduce new entries 30%; manage existing positions; avoid FOMO |

---

*Generated from the analysis notebook (`note.ipynb`). See the `outputs/` folder for supporting charts and tables.*
