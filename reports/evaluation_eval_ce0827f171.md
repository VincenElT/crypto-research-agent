# Historical Strategy Evaluation Report: BTC/USDT (eval_ce0827f171)

> **STALE / SUPERSEDED:** Generated before Milestone 1 accounting, candle validation, and cutoff clarifications. Figures below are historical artifacts and have not been re-run or verified against the corrected implementation.

- **Generated UTC:** 2026-10-25T10:25:48.880831+00:00
- **Asset Symbol:** BTC/USDT | **Timeframe:** 1d
- **Market Data Source:** Verified Public Binance Vision API (https://data-api.binance.vision/api/v3/klines)
- **Total Candles Analyzed:** 1826 (2021-10-25 to 2026-10-24)
- **Indicator Warm-up:** 219 closed daily bars (2021-10-25 to 2022-05-31)

---

## 1. Multi-Period Walk-Forward Performance Comparison

| Metric | Development (In-Sample) | Validation | Out-of-Sample (Holdout) | Full Compound |
| :--- | :---: | :---: | :---: | :---: |
| **Date Range** | 2022-06-01 to 2023-12-31 | 2024-01-01 to 2025-06-30 | 2025-07-01 to 2026-10-24 | 2022-06-01 to 2026-10-24 |
| **Decision Bars** | 579 | 547 | 481 | 1607 |
| **Strategy Net Return** | +3.83% | +2.50% | +0.26% | +7.66% |
| **BTC Buy & Hold Return** | +41.86% | +142.53% | -23.46% | +171.38% |
| **Exposure-Matched Return** | +1.83% | +5.75% | -0.41% | +5.74% |
| **Cash-Only Return** | +0.00% | +0.00% | +0.00% | +0.00% |
| **Strategy Max Drawdown** | 1.29% | 3.63% | 2.12% | 3.65% |
| **BTC B&H Max Drawdown** | 49.70% | 28.10% | 52.97% | 52.97% |
| **Exposure-Matched Max DD** | 1.90% | 1.57% | 1.76% | 2.49% |
| **Annualized Sharpe (Daily)** | 0.90 | 0.42 | 0.12 | 0.56 |
| **BTC B&H Sharpe** | 0.69 | 1.41 | -0.28 | 0.71 |
| **Exposure-Matched Sharpe** | 0.69 | 1.41 | -0.28 | 0.71 |
| **Completed Round Trips** | 3 | 10 | 4 | 17 |
| **Win Rate** | 66.7% | 30.0% | 50.0% | 47.1% |
| **Expectancy / Trade** | $+53.71 | $+26.36 | $+7.50 | $+46.13 |
| **Profit Factor** | 56.64 | 1.54 | 1.39 | 2.64 |
| **Average Daily Exposure** | 3.28% | 5.23% | 2.63% | 3.73% |
| **Time Invested** | 29.7% | 47.3% | 24.9% | 34.2% |
| **Total Friction (Fees+Slip)** | $10.92 | $32.74 | $12.23 | $54.67 |

---

## 2. Sensitivity Analysis (Development Period Friction Matrix)

Tests strategy fragility against adverse execution costs without parameter tuning:

| Scenario | Fee Rate | Slippage | Net Return | Net Profit | Max Drawdown | Sharpe | Profit Factor | Total Friction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (10bps fee / 5bps slip)** | 0.10% | 0.05% | +3.83% | $+383.07 | 1.29% | 0.90 | 56.64 | $10.92 |
| **Moderate Friction (15bps fee / 10bps slip)** | 0.15% | 0.10% | +3.76% | $+375.75 | 1.29% | 0.88 | 35.86 | $18.20 |
| **High Friction (20bps fee / 20bps slip)** | 0.20% | 0.20% | +3.65% | $+364.78 | 1.29% | 0.86 | 21.81 | $29.12 |
| **Stress Friction (30bps fee / 30bps slip)** | 0.30% | 0.30% | +3.50% | $+350.16 | 1.29% | 0.83 | 14.47 | $43.67 |

---

## 3. Methodology, Assumptions & Safeguards

- **Chronological Integrity:** The timeline is strictly partitioned into distinct Development, Validation, and Out-of-Sample intervals without overlap.
- **Zero Look-Ahead Bias:** Indicators (SMA50, SMA200, Momentum) are calculated strictly on past closed daily candles. 218 days of preceding historical data were reserved exclusively for indicator warm-up without generating trade signals.
- **Next-Bar Execution:** All candidate orders are executed at the `open` price of the bar *following* the signal bar close, preventing same-bar fill artifacts.
- **Exposure-Matched Benchmark Definition:** A dynamically rebalanced portfolio holding a fixed weight $\bar{w} = \text{avg\_exposure}$ in BTC and $(1 - \bar{w})$ in cash, matching the strategy's average capital commitment.
- **Sharpe Ratio Convention:** Daily portfolio close-to-close returns annualized using $\sqrt{365} \approx 19.105$ for 24/7 cryptocurrency markets, assuming a 0.0% cash risk-free benchmark rate.
- **Safeguards Enforced:** `is_live=False` hardcoded invariant; strict TLS verification; independent SQLite `:memory:` database state for each run.

---

## 4. Objective Findings & Identified Strategy Limitations

1. **Trade Scarcity & Low Turnover:**
   Across 1,607 decision bars (~4.4 years), the strategy completed only 17 round-trip trades (49 total fills). The strategy spends 65.8% of all trading days flat in 100% cash.
2. **Cash Drag in Strong Bull Markets:**
   With an average portfolio exposure of only 3.73% (governed by the 10% maximum allocation cap per symbol), the strategy earned +7.66% over the full timeline, substantially trailing the 100% BTC Buy-and-Hold return (+171.38%). However, it slightly outperformed the exposure-matched benchmark (+5.74%).
3. **Capital Preservation Advantage:**
   During the severe 2022 bear market (where BTC suffered a 49.70% maximum drawdown), the strategy suffered a maximum drawdown of only 1.29%, demonstrating effective downside protection.
4. **Out-of-Sample Statistical Insufficiency:**
   In the final holdout period (2025-07 to 2026-10), the strategy achieved +0.26% net return (vs -23.46% for BTC B&H) across 4 completed trades. With so few completed trades in the out-of-sample window, there is insufficient statistical sample size to reject randomness or claim an unambiguous edge.
5. **Execution Liquidity Assumption:**
   Simulated fills assume next-bar open liquidity without market impact beyond the fixed 0.05% slippage and 0.10% fee assumptions.

---
**Disclaimer:** This historical evaluation report is strictly educational and for quantitative research validation. Past performance does not predict future returns. Live exchange trading is disabled.
