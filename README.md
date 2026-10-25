# AI-Assisted Crypto Trading Research Agent

Educational research system for historical crypto analysis, persistent paper trading, and historical backtesting. **It does not place live orders.** AI agents will never send exchange orders or override Python risk limits.

This is not financial advice. Indicator output and backtest metrics are historical research tools, not a guarantee of profitability. Do not treat Scout confidence as a calibrated probability of profit.

## Project structure

```
crypto-research-agent/
  crypto_research/
    config.py           # environment settings (no secrets printed)
    market_data.py      # public CCXT OHLCV fetch
    indicators.py       # SMA 50, SMA 200, 30-day momentum
    summary.py          # CLI formatting
    schemas.py          # strict Scout/Skeptic JSON schemas
    llm.py              # OpenAI Responses API + local re-validation
    market_context.py   # closed-candle payload for the agents
    scout.py            # BUY/SELL/HOLD proposal
    skeptic.py          # independent APPROVE/REJECT critique
    research.py         # Scout then Skeptic orchestration
    risk.py             # deterministic Python risk filters
    paper_trading.py    # persistent virtual portfolio and execution
    storage.py          # SQLite decision-log and audit trail
    backtest.py         # historical backtesting engine & CLI
    ml_pipeline.py      # reproducible hourly ML training and walk-forward evaluation
    dashboard.py        # local research dashboard and safe API
    dashboard_static/   # responsive browser UI (no frontend build required)
  tests/
  .env.example
  requirements.txt
```

## Implementation plan

| Milestone | Status | Scope |
| --- | --- | --- |
| 1. Market snapshot | **Implemented** | Fetch public candles, compute indicators, print a summary |
| 2. Dual agents | **Implemented** | Scout + Skeptic via OpenAI, strict JSON schema validation |
| 3. Risk + paper trade | **Implemented** | Deterministic risk filters, persistent virtual portfolio, SQLite logs |
| 4. Backtest | **Implemented** | Chronological replay, next-candle execution, fees/slippage, performance metrics vs buy-and-hold |
| 5. Model research + dashboard | **Implemented** | Hourly BTC/ETH model comparisons, persistent review labels, and local interactive research console |

## Implemented in this version

- **Public historical OHLCV via CCXT (`fetch_ohlcv`)**: Drops in-progress candles, validates strict price integrity, and rejects gapped series.
- **Indicators**: SMA 50, SMA 200, and 30-day momentum computed strictly on closed bars.
- **Dual AI agents**: Scout and Skeptic via OpenAI Responses API with local schema re-validation.
- **Deterministic Python risk filters**: Position sizing, allocation capacity, cooldown bars, max drawdown, and candle volatility guards.
- **Persistent virtual paper trading**: SQLite-backed virtual accounts, position tracking, average entry price, realized and unrealized P&L, fees, slippage, and audit logging surviving application restarts.
- **Historical backtesting engine (`crypto_research/backtest.py`)**:
  - Chronological replay over historical OHLCV candles without look-ahead bias.
  - Generates signals on closed bars, simulating executions at the next candle's open price.
  - Reuses persistent portfolio accounting, slippage, trading fees, and deterministic risk limits.
  - Multi-symbol support with isolated backtest state (`account_id=f"backtest_{run_id}"`) preventing contamination of the normal paper-trading account.
  - Offline deterministic replay mode using synthetic candles or recorded decisions without API credentials or network calls.
  - Comprehensive performance reporting: total return, buy-and-hold benchmark return, maximum drawdown, completed trades, win rate, gross profit/loss, net realized P&L, profit factor, equity curve series, and symbol breakdowns.
- **Fail-closed integrity**: Rejects non-finite, negative, or corrupt market data; never fabricates fills or market prices.
- **Live trading strictly disabled**: No live orders can be placed, and no exchange private keys exist in the codebase.

## Setup

Requires **Python 3.11+** (developed against 3.13), Git, and public internet access for exchange market data. No exchange API keys are required or supported. **Never put exchange credentials in `.env`.**

Clone the repository on each computer:

```sh
git clone https://github.com/VincenElT/crypto-research-agent.git
cd crypto-research-agent
```

Create a virtual environment and install dependencies.

**Windows PowerShell:**

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

**macOS / Linux:**

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

If Python 3.11 is not available as `py -3.11` or `python3`, install a supported Python version first. If PowerShell blocks environment activation, run the project commands with `.venv\Scripts\python.exe -m ...` without activating it.

Edit `.env` to choose a public symbol or exchange. `OPENAI_API_KEY` is optional and only needed for Scout/Skeptic calls; the backtest and theory comparison use deterministic local rules. Each computer stores its own ignored SQLite files and research artifacts by default. Do not share `.env`, database files, model files, or API keys through Git. These setup steps are intended to be repeated independently on every Windows, macOS, or Linux computer; do not copy the `.venv` folder between machines.

## Start the interactive research dashboard

Start it after installing the project dependencies:

```powershell
python -m crypto_research.dashboard
```

Then open <http://127.0.0.1:8765>. The dashboard reads the same local SQLite paper account and logs used by the CLI. It includes Mission Control, Agent Activity, Decision Review with append-only labels and CSV export, Market Explorer, Model Lab, Paper Journal, and Agent Scorecards. Missing records are shown as not available rather than being filled with demo numbers. All generated trades are simulated; no exchange-order endpoint exists.

To use the project on a friend's computer, clone the repository there and repeat the setup above. The local dashboard intentionally listens only on that computer by default; it does not expose an unauthenticated dashboard over a shared network. SQLite history and locally saved models are not synchronized between devices. If you want to show your existing data, use the dashboard's reviewed-label CSV export or prepare a separate sanitized export; never send `.env` or credentials.

Decision labels are human annotations, separate from market outcomes and model targets. Changing a label adds a revision. The CSV export documents this distinction so annotations are not silently treated as supervised truth.

## Train and evaluate hourly predictors

The Model Lab can start a fixed training job, or use the CLI:

```powershell
python -m crypto_research.ml_pipeline --symbols BTC/USDT ETH/USDT --candle-limit 10000
```

This fetches public closed 1-hour OHLCV candles and compares scaled logistic regression with histogram gradient boosting. Features use information in a fully closed candle. The target is whether the close four hourly bars later is above the current close (flat is labeled down); that is a direction classification task, not a claim that returns cover costs. Training uses ordered expanding walk-forward folds and 60/20/20 train, validation, and reserved out-of-sample blocks with a four-bar purge at split boundaries. Do not tune using the reserved out-of-sample period.

The Model Lab reports Brier score, log loss, precision/recall and calibration, alongside a separate next-open paper strategy simulation with modeled fees/slippage, cash-only, buy-and-hold, and existing fixed-rule controls. Model binaries, exact fetched datasets, metadata, predictions, and logs are written under the ignored `artifacts/` directory and local SQLite database; they are not committed to Git. Training needs public internet but no exchange credentials.

## Compare fixed strategy theories

The existing experiment harness compares the current rule control with fixed time-series momentum, prior-nine-candle Donchian breakout, and SMA50 mean-reversion hypotheses on development and validation periods only:

```powershell
python -m crypto_research.experiments --symbol BTC/USDT --timeframe 1d --candle-limit 1826
```

The thresholds are predeclared and not optimized. Compare them using identical evaluation periods, fees, slippage, and risk limits; the final holdout is not used by this command. Experiment history is local and ignored by Git. This is research tooling, not evidence that a strategy has an edge.

## Running current-market research and paper trading

```powershell
python -m crypto_research
```

This fetches current public market data, executes any pending **simulated paper** trades from prior closed candles, optionally runs Scout and Skeptic, applies deterministic risk filters, and records the virtual portfolio locally. The project does not place real exchange orders; this is suitable for a live-data paper test on another device. Without `OPENAI_API_KEY`, it prints the market summary and does not generate AI proposals.

For repeated paper updates, set `TIMEFRAME=1h` in your local `.env` and run:

```powershell
python -m crypto_research --watch
```

Watch mode runs one research cycle immediately, then waits until just after each configured candle boundary before checking the next closed candle. It stops on a failed data/research cycle and can be stopped with Ctrl+C. It only creates virtual paper trades; the account is isolated from exchange balances and no exchange order API is called. With `OPENAI_API_KEY` configured, each cycle also calls Scout and Skeptic, which may incur API usage costs.

## Running historical backtests

The backtester can be run directly from the command line without modifying any source code.

### 1. Offline deterministic test (no network or API keys required)
```powershell
python -m crypto_research.backtest --synthetic --symbols BTC/USDT --initial-cash 10000
```
or via the main entrypoint:
```powershell
python -m crypto_research --backtest --synthetic --symbols BTC/USDT
```

### 2. Backtest using public historical exchange data
```powershell
python -m crypto_research.backtest --symbols BTC/USDT --timeframe 1d --candle-limit 400 --initial-cash 10000
```

### 3. Backtest using a custom CSV file
```powershell
python -m crypto_research.backtest --csv-file path/to/candles.csv --symbols BTC/USDT
```

### Available CLI options

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--symbols` | `BTC/USDT` | Comma-separated asset symbols to test. |
| `--timeframe` | `1d` | Bar timeframe (e.g. `1d`, `1h`). |
| `--candle-limit` | `400` | Historical closed candles to fetch if using CCXT. |
| `--initial-cash` | `10000.0` | Initial virtual cash in USD. |
| `--fee-rate` | `0.001` | Trading fee rate (0.001 = 0.10%). |
| `--slippage` | `0.0005` | Slippage percentage applied to fills (0.0005 = 0.05%). |
| `--start-date` | `None` | Optional ISO start date filter (e.g. `2024-01-01`). |
| `--end-date` | `None` | Optional ISO end date filter (e.g. `2024-12-31`). |
| `--synthetic` | `False` | Run network-free with deterministic synthetic candles. |
| `--csv-file` | `None` | Path to CSV containing historical OHLCV data. |
| `--db-path` | `:memory:` | SQLite path for backtest audit log. |

## Interpreting the backtest report

- **Total Return**: Cumulative percent gain or loss based on final equity vs initial cash.
- **Buy & Hold Benchmark**: Return of an equal-weighted buy-and-hold strategy across the tested symbols from period start to end.
- **Max Drawdown**: Maximum peak-to-trough drop in total equity across the timeline.
- **Realized Exit Tranches**: Number of sell executions realizing P&L; a partial exit counts as its own tranche.
- **Win Rate**: Percentage of exit tranches with positive net realized P&L after allocated entry fees, exit fees, and modeled slippage.
- **Profit Factor**: Gross profits divided by gross losses. Displays `inf (no losses)` if all trades are profitable.
- **Execution History**: Chronological log of simulated fills with exact fees, slippage, and P&L.

## Controlled strategy experiments

`crypto_research.experiments.run_controlled_experiments` accepts explicitly named strategy hypotheses with a provider factory and JSON-serializable configuration. It evaluates each hypothesis on the development and validation splits only; it has no option to evaluate the final holdout. Runs are persisted to a separate SQLite research store with dataset/code/dependency fingerprints, configuration, parent experiment ID, period summaries, and a separate random run ID. Failed runs are saved before the original error is raised. Add a conclusion with `record_experiment_conclusion` after reviewing the evidence.

```python
from crypto_research.backtest import DeterministicRuleProvider
from crypto_research.experiments import (
    StrategyHypothesis,
    run_controlled_experiments,
)

hypothesis = StrategyHypothesis(
    name="baseline_control",
    hypothesis="Current trend and momentum rules serve as the control.",
    provider_name="DeterministicRuleProvider",
    provider_factory=DeterministicRuleProvider,
    configuration={"rule_set": "current_baseline"},
)
runs = run_controlled_experiments(
    candles,
    [hypothesis],
    db_path="research/experiments.sqlite",
    data_source="Verified public OHLCV CSV",
)
```

This is an experiment harness, not evidence of a profitable edge. It includes a small fixed comparison set but does not generate strategies or optimize parameters automatically. Review both development and validation results, and keep the holdout untouched until strategy selection is frozen.

### Run the included theory comparison

With the virtual environment active, run:

```sh
python -m crypto_research.experiments --symbol BTC/USDT --timeframe 1d --candle-limit 1826
```

The CLI fetches public, closed daily candles and compares four fixed hypotheses: the existing baseline, 30-candle time-series momentum, a prior-nine-candle Donchian breakout, and a 2% below-SMA50 mean-reversion entry. It uses the same risk limits, fees, and slippage for each, reports development and validation results, and saves experiment history under `research/experiments.sqlite`. It does not evaluate the final holdout. Use only hypotheses chosen before reviewing results; do not repeatedly adjust thresholds to chase the best validation score.

To use a local CSV instead, provide `--csv-file path/to/candles.csv`. It must contain valid, contiguous daily OHLCV candles, including at least 200 warm-up candles before 2022-06-01 and data through 2025-06-30. The included `data/btc_sample_1d.csv` is too short for the full development/validation experiment and will be rejected.

## Execution assumptions & limitations

1. **Next-Candle Execution**: A trade proposal generated on candle $T_k$ fills strictly at the **open price** of candle $T_{k+1}$ ($T_{k+1} > T_k$).
2. **Spot Long-Only**: Sells require existing holdings. Short-selling without inventory is strictly rejected.
3. **Fills & Liquidity**: Fills assume full execution at the adjusted open price; intrabar orderbook queue priority and market depth slippage curves are not simulated.
4. **State Isolation**: Backtests write to `account_id=f"backtest_{run_id}"`, preserving the normal paper-trading account intact.

5. **Trade P&L Cost Allocation**: Backtest trade statistics treat each sell execution as a realized exit tranche. Buy fees are allocated pro rata by quantity across exits from the symbol's pooled average-cost position; any fee portion for unsold quantity remains with open inventory. Buy slippage is already in the buy fill/cost basis, while sell slippage and sell fees are already in sell proceeds. These costs are not deducted again from portfolio equity, which remains cash plus marked position value.

6. **Evaluation Cutoff and Benchmarks**: The evaluation harness supports daily candles only. Candle timestamps label bar opens; close-derived signals are eligible only after that candle closes, at the next UTC midnight. An inclusive period end-date includes the candle with that UTC open-date, whose close is available at the following midnight. Signals without a following candle open remain unexecuted. BTC benchmarks are gross of transaction costs; the exposure-matched benchmark holds a constant target weight and rebalances daily.

7. **Historical Data Integrity**: OHLC relationships, finite nonnegative volume, unique increasing source timestamps, and contiguous candle spacing are required. A gap greater than 1.5 times the configured timeframe is rejected; invalid input never falls back to synthetic data.

## Tests

```powershell
python -m pytest -v
```

The test suite runs offline using synthetic fixtures; it makes no live exchange orders and does not require OpenAI calls.

## Safety notes

- Decisions only use information available up to the closed-candle timestamp.
- If a deterministic risk check fails, the trade is rejected regardless of agent proposals.
- Historical backtest results are for research auditing only; past performance does not predict future returns.
- Live trading is permanently disabled.
