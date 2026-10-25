# Historical Strategy Evaluation Report: BTC/USDT (eval_83c3d96a00)

- **Generated UTC:** 2026-10-25T10:47:04.833476+00:00
- **Asset Symbol:** BTC/USDT | **Timeframe:** 1d
- **Market Data Source:** Public binance OHLCV API (https://data-api.binance.vision/api/v3)
- **Total Candles Analyzed:** 1826 (2021-10-25 to 2026-10-24)
- **Indicator Warm-up:** 219 closed daily bars (2021-10-25 to 2022-05-31)
- **Dataset SHA-256:** `1814957552c8fd1fd1b796d281970a8fbe3dab510b08e75889eeaefddf370aef`
- **Code revision:** `b3550f1b6b89670489f8386199f41e4baf043373` | **Source fingerprint:** `55acaf5a26ce49ded8c1b3ba3031197b3066cc32fcc52b0ccde5a958827c0d2d`
- **Dependency versions:** `{"ccxt": "4.5.90", "openai": "3.37.0", "pandas": "3.1.0", "pydantic": "2.14.1", "python": "3.13.5", "python-dotenv": "1.2.4"}`
- **Experiment configuration:**
```json
{
  "benchmark_assumptions": {
    "buy_and_hold": "100% BTC, gross of transaction costs",
    "cash": "0% nominal yield",
    "exposure_matched": "constant average strategy weight, rebalanced daily, gross of transaction costs"
  },
  "candle_limit": 1826,
  "candle_policy": "1d candles become decision-eligible only after close; end_date is an inclusive candle-open label whose close is available at the next UTC midnight",
  "decision_provider": "DeterministicRuleProvider",
  "fee_rate": 0.001,
  "indicator_windows": {
    "momentum_days": 30,
    "sma_fast": 50,
    "sma_slow": 200
  },
  "initial_cash_usd": 10000.0,
  "pending_order_policy": "A signal on the last eligible candle remains unexecuted if no following candle open is present; it does not affect equity or trade metrics",
  "periods": [
    {
      "description": "Development / In-Sample: 2022 bear market (LUNA/FTX) & 2023 recovery",
      "end_date": "2023-12-31",
      "name": "development",
      "start_date": "2022-06-01"
    },
    {
      "description": "Validation: 2024 Spot ETF launch, halving bull run & consolidations",
      "end_date": "2025-06-30",
      "name": "validation",
      "start_date": "2024-01-01"
    },
    {
      "description": "Final Out-of-Sample / Holdout: Recent cycle (Strictly never tuned against)",
      "end_date": "2026-10-24",
      "name": "out_of_sample",
      "start_date": "2025-07-01"
    },
    {
      "description": "Full multi-year compound evaluation timeline",
      "end_date": "2026-10-24",
      "name": "full",
      "start_date": "2022-06-01"
    }
  ],
  "risk_limits": {
    "cooldown_bars": 5,
    "max_drawdown_pct": 0.15,
    "max_portfolio_allocation_pct": 0.1,
    "max_position_size_usd": 1000.0,
    "max_volatility_pct": 0.08,
    "min_scout_confidence": 0.5,
    "min_trade_size_usd": 10.0,
    "require_skeptic_approval": true
  },
  "sensitivity_enabled": true,
  "sensitivity_scenarios": [
    {
      "fee_rate": 0.001,
      "slippage_pct": 0.0005
    },
    {
      "fee_rate": 0.0015,
      "slippage_pct": 0.001
    },
    {
      "fee_rate": 0.002,
      "slippage_pct": 0.002
    },
    {
      "fee_rate": 0.003,
      "slippage_pct": 0.003
    }
  ],
  "slippage_pct": 0.0005
}
```

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
| **Realized Exit Tranches** | 3 | 10 | 4 | 17 |
| **Unexecuted End Signals** | 0 | 0 | 0 | 0 |
| **Win Rate** | 66.7% | 30.0% | 50.0% | 47.1% |
| **Expectancy / Trade** | $+52.69 | $+25.33 | $+6.49 | $+45.08 |
| **Profit Factor** | 40.98 | 1.51 | 1.32 | 2.58 |
| **Average Daily Exposure** | 3.28% | 5.23% | 2.63% | 3.73% |
| **Time Invested** | 29.7% | 47.3% | 24.9% | 34.2% |
| **Total Friction (Fees+Slip)** | $10.92 | $32.74 | $12.23 | $54.67 |

---

## 2. Sensitivity Analysis (Development Period Friction Matrix)

Tests strategy fragility against adverse execution costs without parameter tuning:

| Scenario | Fee Rate | Slippage | Net Return | Net Profit | Max Drawdown | Sharpe | Profit Factor | Total Friction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (10bps fee / 5bps slip)** | 0.10% | 0.05% | +3.83% | $+383.07 | 1.29% | 0.90 | 40.98 | $10.92 |
| **Moderate Friction (15bps fee / 10bps slip)** | 0.15% | 0.10% | +3.76% | $+375.75 | 1.29% | 0.88 | 25.99 | $18.20 |
| **High Friction (20bps fee / 20bps slip)** | 0.20% | 0.20% | +3.65% | $+364.78 | 1.29% | 0.86 | 16.38 | $29.12 |
| **Stress Friction (30bps fee / 30bps slip)** | 0.30% | 0.30% | +3.50% | $+350.16 | 1.29% | 0.83 | 10.62 | $43.67 |

---

## 3. Methodology, Assumptions & Safeguards

- **Chronological Integrity:** Development, validation, and holdout use disjoint UTC candle-open labels and independent cash-started portfolios.
- **Candle availability:** Daily bars become eligible after close at the next UTC midnight. The inclusive end-date candle closes at the following midnight. A last-bar signal without a following candle open remains unexecuted and excluded from equity and trade metrics.
- **Next-Bar Execution:** Signals use only closed candles; fills use the following candle's open with adverse slippage and fees.
- **Exposure-Matched Benchmark:** Holds the strategy's average exposure in BTC and rebalances daily; benchmark transaction costs are excluded.
- **Buy-and-Hold Benchmark:** Holds 100% BTC from the first evaluated close; benchmark transaction costs are excluded.
- **Sharpe Ratio Convention:** Daily close-to-close returns use sample standard deviation, zero risk-free rate, and $\sqrt{365}$ annualization. Only 1d evaluation is supported.
- **Safeguards Enforced:** `is_live=False` hardcoded invariant; strict TLS verification; independent SQLite `:memory:` database state for each run.

---

## 4. Objective Findings & Identified Strategy Limitations

1. **Trade Scarcity & Low Turnover:**
   Across 1,607 decision bars (~4.4 years), the strategy completed only 17 realized sell tranches (49 total fills). The strategy spends 65.8% of all trading days flat in 100% cash.
2. **Cash Drag in Strong Bull Markets:**
   With an average portfolio exposure of only 3.73% (governed by the 10% maximum allocation cap per symbol), the strategy earned +7.66% over the full timeline, substantially trailing the 100% BTC Buy-and-Hold return (+171.38%). However, it slightly outperformed the exposure-matched benchmark (+5.74%).
3. **Capital Preservation Advantage:**
   During the severe 2022 bear market (where BTC suffered a 49.70% maximum drawdown), the strategy suffered a maximum drawdown of only 1.29%, demonstrating effective downside protection.
4. **Out-of-Sample Statistical Insufficiency:**
   In the final holdout period (2025-07 to 2026-10), the strategy achieved +0.26% net return (vs -23.46% for BTC B&H) across 4 realized sell tranches. With so few realized tranches in the out-of-sample window, there is insufficient statistical sample size to reject randomness or claim an unambiguous edge.
5. **Execution Liquidity Assumption:**
   Simulated fills assume next-bar open liquidity without market impact beyond the fixed 0.05% slippage and 0.10% fee assumptions.

---
**Disclaimer:** This historical evaluation report is strictly educational and for quantitative research validation. Past performance does not predict future returns. Live exchange trading is disabled.