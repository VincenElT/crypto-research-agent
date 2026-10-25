# AI-Assisted Crypto Trading Research Agent

Educational research system for historical crypto analysis and, later, paper trading. **It does not place live orders.** AI agents will never send exchange orders or override Python risk limits.

This is not financial advice. Indicator output is a market snapshot, not a profitable strategy. Do not treat Scout confidence (when it exists) as a calibrated probability of profit.

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
    paper_trading.py    # paper-trade candidate generation (non-live)
    storage.py          # SQLite decision-log and audit trail
    backtest.py         # placeholder (later)
  tests/
  .env.example
  requirements.txt
```

## Implementation plan

| Milestone | Status | Scope |
| --- | --- | --- |
| 1. Market snapshot | **Implemented** | Fetch public candles, compute indicators, print a summary |
| 2. Dual agents | **Implemented** | Scout + Skeptic via OpenAI, strict JSON schema validation |
| 3. Risk + paper trade | **Implemented** | Deterministic risk filters, paper-trade candidate, SQLite logs |
| 4. Backtest | Placeholder | Walk-forward test vs baselines, with fees and slippage |

## Implemented in this version

- Public historical OHLCV via CCXT (`fetch_ohlcv`)
- Drop in-progress candles so summaries use closed bars only
- Reject series with missing candles
- SMA 50, SMA 200 (windows in **candles**)
- 30-day momentum as percent change over the candles that fit in 30 calendar days
- Retry on rate limits and transient network errors
- Unit tests for indicators, config, candle handling, and JSON schemas
- Scout and Skeptic via the OpenAI Responses API (`responses.parse` + local schema checks)
- Agents skipped cleanly when `OPENAI_API_KEY` is unset
- Deterministic Python risk filters (position size, cooldown, max drawdown, bar volatility, Skeptic approval, and Scout confidence)
- Paper-trade candidate generation emitted only when Scout, Skeptic, and deterministic risk filters all agree
- SQLite audit logging persisting market snapshots, agent JSON, risk evaluations, paper trades, and audit events
- Automatic secret scrubbing guaranteeing API keys and credentials are never persisted to disk
- Live trading is strictly disabled across all components

## Not implemented yet

- Backtesting (walk-forward test vs baselines, with fees and slippage)
- Live trading (will not be added in this project’s first versions)

## Setup

Requires **Python 3.11+** (developed against 3.13). Public market data only — **do not put exchange API keys in `.env`.**

```powershell
cd C:\Users\force\crypto-research-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` if you want a different public symbol or exchange. Set `OPENAI_API_KEY` only if you want Scout/Skeptic to run. `OPENAI_MODEL` defaults to `gpt-4o-mini` (structured JSON output).

## Run the market summary (and agents)

```powershell
python -m crypto_research
```

Default: Binance public `BTC/USDT` daily candles. The command prints close, SMA 50, SMA 200, 30-day momentum, and a simple SMA trend label.

If `OPENAI_API_KEY` is set, it then sends **only that closed-candle payload** to Scout and Skeptic, validates both JSON objects, runs deterministic risk filters, and prints the proposal, critique, risk filter evaluation, and paper-trade candidate (if approved). All events are logged to the SQLite database. No live orders are ever sent.

If fetch fails with a TLS/certificate error, a local proxy or antivirus HTTPS scanner is often intercepting the request. Switch `EXCHANGE_ID` only if you want a different **public** venue; do not add exchange API keys, and do not disable certificate verification.

## Tests

```powershell
python -m pytest -q
```

Tests use synthetic candles and do not call exchanges or OpenAI (network-free).

## Safety notes

- Decisions may only use information available at the closed-candle timestamp.
- If a later risk check fails, the system must reject the trade regardless of the agents.
- Live trading is permanently disabled.
- Backtests (when added) must include fees, slippage, and out-of-sample evaluation before anyone talks about profitability.
