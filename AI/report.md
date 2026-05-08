# Bitcoin Trader Performance vs Market Sentiment
## Data Science Assignment — Primetrade.ai
---

## 1. Dataset Overview

| Metric | Value |
|---|---|
| Total Trades (after merge) | 211,218 |
| Closing Trades (PnL ≠ 0) | 104,402 |
| Unique Traders | see insights.json |
| Unique Coins | see insights.json |
| Date Range | 2023-05-01 → 2025-05-01 |
| Fear & Greed Records | 2,644 |

---

## 2. Sentiment Distribution in Trade Data

| Sentiment | Closing Trades | Win Rate | Avg PnL ($) | Total PnL ($) | Volume ($) |
|---|---|---|---|---|---|
| Extreme Fear  | 10,406  | **76.2%** | 71.03   | 739,110     | 56.9M  |
| Fear          | 29,808  | 87.3%     | 112.63  | **3,357,155** | **239.7M** |
| Neutral       | 18,159  | 82.4%     | 71.20   | 1,292,921   | 100.9M |
| Greed         | 25,176  | 76.9%     | 85.40   | 2,150,129   | 136.9M |
| Extreme Greed | 20,853  | **89.2%** | **130.21** | 2,715,171  | 57.96M |

---

## 3. Key Findings

### 🔑 Finding 1: Extreme Greed = Highest Win Rate
Traders achieve their **highest win rate (89.17%)** during Extreme Greed periods and the highest average PnL per trade ($130.21). This suggests that strong bullish momentum creates more profitable opportunities, possibly because trend-following strategies dominate.

### 🔑 Finding 2: Fear Regime = Highest Total PnL & Volume
Despite not having the best per-trade metrics, **Fear sentiment generates the most total PnL ($3.36M)** and volume ($239.7M). This is because Fear periods are the most active — traders place more trades (29,808 closing trades), likely capitalising on volatility.

### 🔑 Finding 3: Extreme Fear = Lowest Win Rate but Recoverable
Extreme Fear has the lowest win rate (76.2%), yet traders still generate positive aggregate PnL. This indicates that even in the worst sentiment conditions, traders on Hyperliquid are net-profitable — consistent with the platform's derivatives nature where smart money often fades crowd fear.

### 🔑 Finding 4: Weak Negative Correlation Between FG Score & Daily PnL
Pearson r = **-0.098** (p = 0.044, statistically significant). Higher Fear-Greed values are very slightly associated with **lower** daily PnL. This counter-intuitive result suggests traders may be contrarian (shorting greed, buying fear) or that high-greed periods attract more risk-taking with mixed results.

### 🔑 Finding 5: Buy/Sell Behaviour Shifts with Sentiment
- During **Fear** periods: SELL trades dominate → traders prefer shorting bearish momentum
- During **Greed** periods: BUY trades increase → traders ride bullish trends
- This shows sentiment-aware directional bias in the trader population

### 🔑 Finding 6: HYPE Dominates Activity
**HYPE coin** accounts for ~68,000 trades — nearly 1/3 of all activity. Its PnL performance varies significantly by sentiment regime, making it the single most important asset to model for sentiment-adjusted strategies.

---

## 4. Trading Strategy Recommendations

### Strategy A — Sentiment-Adaptive Position Sizing
Increase position size during **Extreme Greed** (89% win rate) and reduce during **Extreme Fear** (76% win rate). A simple 1.5x multiplier on Extreme Greed days would historically compound returns.

### Strategy B — Volatility Harvesting in Fear Regimes
Fear periods have 2-3x more trades and volume than Extreme Fear. Deploy mean-reversion or volatility strategies (straddles, high-frequency scalping) during Fear — the volume supports execution and the win rates remain above 87%.

### Strategy C — Contrarian Entry on Extreme Fear
The data shows net-positive PnL even at the lowest win-rate sentiment. Long entries on Extreme Fear (index < 25) with tight stops can capture the mean-reversion bounce that historically follows extreme pessimism.

### Strategy D — HYPE-Focused Sentiment Model
Given HYPE's dominance in trade count, build a dedicated sentiment-signal model for HYPE using the Fear-Greed index as a feature. Even a 1-2% win-rate improvement at this scale translates to significant PnL.

---

## 5. Charts Generated

| # | File | Description |
|---|---|---|
| 1 | `1_win_rate_by_sentiment.png` | Win rate per sentiment category |
| 2 | `2_avg_pnl_by_sentiment.png` | Average PnL per trade by sentiment |
| 3 | `3_pnl_and_volume.png` | Total PnL and volume side-by-side |
| 4 | `4_pnl_violin.png` | PnL distribution violin plot |
| 5 | `5_buy_sell_ratio.png` | Buy/sell ratio per sentiment |
| 6 | `6_coin_sentiment_heatmap.png` | Top 10 coins PnL heatmap |
| 7 | `7_fg_vs_daily_pnl.png` | FG score vs daily PnL scatter |
| 8 | `8_daily_trade_activity.png` | Trade activity timeline by sentiment |
| 9 | `9_top10_traders.png` | Top 10 traders by total PnL |
| 10 | `10_summary_dashboard.png` | Full summary dashboard |

---

## 6. Files in This Submission

```
trading-analysis/
├── analysis.py                      ← full analysis code
├── report.md                        ← this report
├── sentiment_stats.csv              ← aggregated stats table
├── trader_sentiment_breakdown.csv   ← per-trader per-sentiment metrics
├── insights.json                    ← machine-readable key findings
├── data.csv                         ← (fear greed index)
└── charts/
    ├── 1_win_rate_by_sentiment.png
    ├── 2_avg_pnl_by_sentiment.png
    ├── 3_pnl_and_volume.png
    ├── 4_pnl_violin.png
    ├── 5_buy_sell_ratio.png
    ├── 6_coin_sentiment_heatmap.png
    ├── 7_fg_vs_daily_pnl.png
    ├── 8_daily_trade_activity.png
    ├── 9_top10_traders.png
    └── 10_summary_dashboard.png
```

---

*Analysis by candidate — submitted for Primetrade.ai Data Science internship*
