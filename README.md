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

Requires **Python 3.11+** (developed against 3.13). Public market data only — **do not put exchange API keys in `.env`.**

```powershell
cd C:\Users\force\crypto-research-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` if you want a different public symbol or exchange. Set `OPENAI_API_KEY` only if you want Scout/Skeptic to run. `OPENAI_MODEL` defaults to `gpt-4o-mini`.

## Running the live research pipeline

```powershell
python -m crypto_research
```

Fetches the latest closed candles, executes any pending paper trades from prior closed candles, runs Scout and Skeptic, evaluates deterministic risk filters, logs candidates to SQLite, and prints virtual portfolio status.

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

All 85 tests run offline using synthetic fixtures with zero network dependencies or API costs.

## Safety notes

- Decisions only use information available up to the closed-candle timestamp.
- If a deterministic risk check fails, the trade is rejected regardless of agent proposals.
- Historical backtest results are for research auditing only; past performance does not predict future returns.
- Live trading is permanently disabled.
