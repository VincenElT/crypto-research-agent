"""Historical backtesting engine for the crypto research agent.

Replays historical OHLCV candles chronologically, evaluating agent decisions
and deterministic risk filters on past closed bars only, and simulating executions
at next-candle open prices without look-ahead bias.

Reuses persistent portfolio accounting, slippage, fees, and deterministic risk checks.
Live trading is strictly disabled. No live orders can be placed.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from crypto_research.config import Settings, load_settings
from crypto_research.exceptions import (
    BacktestError,
    ConfigurationError,
    InsufficientHistoryError,
    MarketDataError,
    PaperTradingError,
)
from crypto_research.indicators import (
    IndicatorSnapshot,
    add_indicators,
    candles_for_calendar_days,
    latest_snapshot,
)
from crypto_research.market_context import build_market_context
from crypto_research.market_data import (
    create_public_exchange,
    fetch_ohlcv,
    find_missing_candle_gaps,
)
from crypto_research.paper_trading import (
    PaperExecution,
    PaperTradeCandidate,
    Position,
    VirtualAccount,
    calculate_portfolio_equity,
    calculate_unrealized_pnl,
    create_paper_trade,
    simulate_execution,
)
from crypto_research.research import DualAgentResult, run_dual_agents
from crypto_research.risk import (
    PortfolioState,
    RiskLimits,
    convert_directional_signal_to_action,
    convert_portfolio_action,
    evaluate_risk,
)
from crypto_research.schemas import ScoutProposal, SkepticReview
from crypto_research.storage import (
    get_all_positions,
    get_or_create_portfolio_account,
    get_paper_executions,
    get_portfolio_account,
    get_position,
    init_db,
    load_portfolio_state_from_db,
    log_paper_trade,
)


class DecisionProvider(Protocol):
    """Protocol for supplying Scout proposals and Skeptic critiques."""

    def __call__(
        self,
        market_context: dict[str, Any],
        snapshot: IndicatorSnapshot,
        symbol: str,
    ) -> DualAgentResult | None: ...


class DeterministicRuleProvider:
    """Deterministic rule-based agent decision provider for offline testing and replay.

    Generates BUY signals on strong trend / momentum alignment and SELL signals on
    downtrend / trend reversals, satisfying schema constraints without OpenAI network calls.
    """

    def __call__(
        self,
        market_context: dict[str, Any],
        snapshot: IndicatorSnapshot,
        symbol: str,
        has_open_position: bool | None = None,
    ) -> DualAgentResult:
        sma_fast = snapshot.sma_50
        sma_slow = snapshot.sma_200
        momentum = snapshot.momentum_30d

        if has_open_position is None:
            has_open_position = bool(market_context.get("has_open_position", False))

        if sma_fast > sma_slow and momentum > 0.01:
            proposal = ScoutProposal(
                action="BUY",
                confidence=0.85,
                thesis=f"Uptrend confirmed: SMA50 ({sma_fast:.2f}) > SMA200 ({sma_slow:.2f}) with momentum {momentum:.2%}",
                evidence=["SMA 50 above SMA 200", f"Positive 30d momentum {momentum:.2%}"],
                risks=["Potential short-term pullback", "Macro market volatility"],
                missing_data=[],
            )
            critique = SkepticReview(
                verdict="APPROVE",
                agrees_with_action=True,
                rationale="Technical alignment supported by moving averages and positive momentum.",
                missing_evidence=[],
                weaknesses=["Orderbook depth unavailable"],
                confidence=0.80,
            )
        elif sma_fast < sma_slow or momentum < -0.02:
            proposal = ScoutProposal(
                action="SELL",
                confidence=0.80,
                thesis=f"Downtrend or negative momentum: SMA50 ({sma_fast:.2f}), SMA200 ({sma_slow:.2f}), mom {momentum:.2%}",
                evidence=["Bearish trend or momentum divergence"],
                risks=["Oversold bounce risk"],
                missing_data=[],
            )
            critique = SkepticReview(
                verdict="APPROVE",
                agrees_with_action=True,
                rationale="Risk reduction / profit taking appropriate given bearish conditions.",
                missing_evidence=[],
                weaknesses=["Volume spike verification pending"],
                confidence=0.75,
            )
        else:
            proposal = ScoutProposal(
                action="HOLD",
                confidence=0.60,
                thesis="Neutral market conditions; no directional edge.",
                evidence=["Moving averages converging", "Momentum near zero"],
                risks=["Chop and range-bound trading"],
                missing_data=[],
            )
            critique = SkepticReview(
                verdict="REJECT",
                agrees_with_action=False,
                rationale="No tradable edge present in neutral market.",
                missing_evidence=["Clear trend continuation signal"],
                weaknesses=["Low conviction"],
                confidence=0.70,
            )

        raw_result = DualAgentResult(proposal=proposal, critique=critique)
        return convert_portfolio_action(raw_result, has_open_position=has_open_position)


class RecordedDecisionProvider:
    """Replays pre-recorded or mock agent decisions keyed by (symbol, as_of_utc)."""

    def __init__(self, decisions: dict[tuple[str, str], DualAgentResult]) -> None:
        self.decisions = decisions

    def __call__(
        self,
        market_context: dict[str, Any],
        snapshot: IndicatorSnapshot,
        symbol: str,
    ) -> DualAgentResult | None:
        key = (symbol, snapshot.as_of.isoformat())
        return self.decisions.get(key)


class LiveAgentProvider:
    """Calls OpenAI Scout and Skeptic with in-memory caching per (symbol, as_of_utc).

    Clearly separated from deterministic replay. Note: Live model calls are non-deterministic.
    """

    def __init__(self, model_name: str = "gpt-4o-mini") -> None:
        self.model_name = model_name
        self._cache: dict[tuple[str, str], DualAgentResult] = {}

    def __call__(
        self,
        market_context: dict[str, Any],
        snapshot: IndicatorSnapshot,
        symbol: str,
    ) -> DualAgentResult:
        key = (symbol, snapshot.as_of.isoformat())
        if key in self._cache:
            return self._cache[key]
        result = run_dual_agents(market_context, model_name=self.model_name)
        self._cache[key] = result
        return result


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for historical backtesting simulation."""

    run_id: str
    symbols: list[str]
    timeframe: str = "1d"
    initial_cash_usd: float = 10_000.0
    fee_rate: float = 0.001  # 0.10%
    slippage_pct: float = 0.0005  # 0.05%
    start_date: str | None = None
    end_date: str | None = None
    risk_limits: RiskLimits = field(default_factory=RiskLimits)
    sqlite_db_path: str = ":memory:"


@dataclass(frozen=True)
class BacktestReport:
    """Comprehensive performance and accounting report for a backtest run."""

    run_id: str
    config: BacktestConfig
    start_time_utc: str
    end_time_utc: str
    initial_equity: float
    final_equity: float
    total_net_profit_usd: float
    total_return_pct: float
    benchmark_return_pct: float
    max_drawdown_pct: float
    total_executions: int
    completed_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    gross_profit_usd: float
    gross_loss_usd: float
    net_realized_pnl_usd: float
    unrealized_pnl_usd: float  # Gross unrealized P&L (market value - purchase cost basis)
    gross_unrealized_pnl_usd: float
    net_unrealized_pnl_usd: float  # Net unrealized P&L after accounting for entry fees on open positions
    total_fees_paid_usd: float
    open_positions_fees_usd: float
    profit_factor: float
    signals_proposed: int
    signals_approved: int
    signals_rejected: int
    rejections_by_reason: dict[str, int]
    proposals_by_action: dict[str, int]
    unexecuted_orders: int
    equity_curve: list[dict[str, Any]]
    symbol_breakdown: dict[str, dict[str, Any]]
    execution_history: list[dict[str, Any]]
    completed_trade_records: list[dict[str, Any]]


def allocate_entry_fees_to_exits(
    execution_history: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Return sell-tranche P&L after allocating buy fees by quantity.

    Buys add quantity and their fees to a symbol-level average-cost pool. Each
    sell allocates the same fraction of that pool as the fraction of inventory
    sold. Buy slippage is already included in the buy fill price and hence in
    ``cost_basis_sold_usd``; sell slippage and sell fees are already included
    in the execution's realized P&L. This helper therefore deducts only the
    allocated entry fee and does not change cash or portfolio equity.

    Each sell execution is reported as one realized exit tranche. Remaining
    entry fees are returned by symbol for the still-open inventory.
    """
    open_quantity: dict[str, float] = {}
    open_entry_fees: dict[str, float] = {}
    net_exits: list[dict[str, Any]] = []

    for execution in execution_history:
        symbol = str(execution["symbol"])
        quantity = float(execution["quantity"])
        if execution["action"] == "BUY":
            open_quantity[symbol] = open_quantity.get(symbol, 0.0) + quantity
            open_entry_fees[symbol] = open_entry_fees.get(symbol, 0.0) + float(
                execution.get("fee_usd", 0.0)
            )
            continue

        if execution["action"] != "SELL":
            continue

        held_quantity = open_quantity.get(symbol, 0.0)
        if held_quantity <= 0.0 or quantity > held_quantity + 1e-9:
            raise BacktestError(
                f"Cannot allocate entry costs for {symbol} sell of {quantity} "
                f"against open quantity {held_quantity}."
            )
        allocated_fee = open_entry_fees.get(symbol, 0.0) * min(1.0, quantity / held_quantity)
        net_pnl = float(execution["realized_pnl_usd"]) - allocated_fee
        net_exits.append(
            {
                **execution,
                "allocated_entry_fee_usd": allocated_fee,
                "net_realized_pnl_usd": net_pnl,
                # Keep the historical field but make it reflect the all-in
                # net value used by the trade metrics.
                "realized_pnl_usd": net_pnl,
            }
        )
        remaining_fraction = max(0.0, (held_quantity - quantity) / held_quantity)
        open_quantity[symbol] = max(0.0, held_quantity - quantity)
        open_entry_fees[symbol] = open_entry_fees.get(symbol, 0.0) * remaining_fraction

    return net_exits, open_entry_fees


def validate_backtest_candles(
    candles: pd.DataFrame, symbol: str, timeframe_seconds: int
) -> pd.DataFrame:
    """Validate historical candles fail-closed for non-numeric, non-finite, or gapped data."""
    if candles.empty:
        raise InsufficientHistoryError(f"Candle data for {symbol} is empty.")

    required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
    if not required_cols.issubset(candles.columns):
        missing = required_cols - set(candles.columns)
        raise BacktestError(f"Candles for {symbol} missing required columns: {missing}")

    frame = candles.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)

    # Validate source order before doing any ordering. Sorting here would hide
    # a broken or shuffled input sequence and undermine reproducibility.
    if frame["timestamp"].isna().any():
        raise MarketDataError(f"Invalid timestamp in candle series for {symbol}.")
    if frame["timestamp"].duplicated().any():
        dup_ts = frame[frame["timestamp"].duplicated()]["timestamp"].iloc[0]
        raise MarketDataError(
            f"Duplicate timestamp {dup_ts.isoformat()} in candle series for {symbol}."
        )
    if not frame["timestamp"].is_monotonic_increasing:
        raise MarketDataError(f"Non-monotonic timestamps in candle series for {symbol}.")
    frame = frame.reset_index(drop=True)

    # Validate numeric integrity
    for col in ("open", "high", "low", "close", "volume"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
        if frame[col].isna().any():
            raise MarketDataError(f"Non-numeric values in {col} column for {symbol}.")

    if not frame["volume"].map(math.isfinite).all() or (frame["volume"] < 0).any():
        raise MarketDataError(f"Non-finite or negative volume found in candle data for {symbol}.")

    ohlc_cols = ["open", "high", "low", "close"]
    for col in ohlc_cols:
        series = frame[col]
        # Check non-finite without requiring external numpy
        if series.isin([float("inf"), float("-inf")]).any():
            raise MarketDataError(f"Non-finite OHLC values found in candle data for {symbol}.")
        if (series <= 0).any():
            raise MarketDataError(f"Non-positive OHLC prices found in candle data for {symbol}.")

    if (frame["high"] < frame["low"]).any():
        raise MarketDataError(f"Invalid candle with high < low found for {symbol}.")
    if (frame["high"] < frame[["open", "close"]].max(axis=1)).any():
        raise MarketDataError(f"Invalid candle with high below open or close for {symbol}.")
    if (frame["low"] > frame[["open", "close"]].min(axis=1)).any():
        raise MarketDataError(f"Invalid candle with low above open or close for {symbol}.")

    # Check for missing candle gaps
    gaps = find_missing_candle_gaps(frame, timeframe_seconds * 1000)
    if gaps:
        preview = "; ".join(gaps[:3])
        raise MarketDataError(
            f"Missing candle gaps detected for {symbol}: {preview}. Backtesting requires contiguous data."
        )

    return frame


def run_backtest(
    config: BacktestConfig,
    *,
    candles_by_symbol: dict[str, pd.DataFrame] | None = None,
    decision_provider: DecisionProvider | None = None,
    conn: sqlite3.Connection | None = None,
) -> BacktestReport:
    """Execute a chronological historical backtest.

    - Replays closed candles chronologically.
    - Evaluates signals using past closed bars only (no future data leakage).
    - Simulates executions at next-candle open prices with adverse slippage and fees.
    - Isolates virtual portfolio state using account_id f"backtest_{config.run_id}".
    - Maintains a true running peak of equity for accurate drawdown calculation.
    - Live trading is strictly disabled.
    """
    if not config.symbols:
        raise ConfigurationError("Backtest requires at least one symbol in config.symbols.")

    if config.initial_cash_usd <= 0:
        raise ConfigurationError(
            f"Initial cash must be positive, got {config.initial_cash_usd}."
        )

    provider = decision_provider or DeterministicRuleProvider()
    account_id = f"backtest_{config.run_id}"

    # Database initialization
    db_conn = conn or init_db(config.sqlite_db_path)
    get_or_create_portfolio_account(
        db_conn, account_id=account_id, initial_cash_usd=config.initial_cash_usd
    )

    # Parse timeframe seconds
    timeframe_seconds = 86_400
    if config.timeframe.endswith("d"):
        timeframe_seconds = int(config.timeframe[:-1]) * 86_400
    elif config.timeframe.endswith("h"):
        timeframe_seconds = int(config.timeframe[:-1]) * 3600
    elif config.timeframe.endswith("m"):
        timeframe_seconds = int(config.timeframe[:-1]) * 60

    # Validate and compute indicators for each symbol
    prepared_candles: dict[str, pd.DataFrame] = {}
    candles_by_symbol = candles_by_symbol or {}

    for symbol in config.symbols:
        if symbol not in candles_by_symbol:
            raise BacktestError(
                f"No candle data provided for symbol {symbol!r}. Supply data in candles_by_symbol."
            )
        valid_df = validate_backtest_candles(
            candles_by_symbol[symbol], symbol, timeframe_seconds
        )
        with_ind = add_indicators(valid_df, timeframe_seconds=timeframe_seconds)
        prepared_candles[symbol] = with_ind

    # Determine simulation timeline across all symbols
    all_timestamps: set[pd.Timestamp] = set()
    for symbol, df in prepared_candles.items():
        # Only bars where indicators are fully defined are eligible decision points
        valid_bars = df[
            df["sma_50"].notna()
            & df["sma_200"].notna()
            & df["momentum_30d"].notna()
        ]
        if valid_bars.empty:
            raise InsufficientHistoryError(
                f"Insufficient candle history to calculate SMA200 and momentum for {symbol}."
            )
        all_timestamps.update(valid_bars["timestamp"].tolist())

    timeline = sorted(list(all_timestamps))

    if config.start_date:
        start_ts = pd.Timestamp(config.start_date)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        timeline = [t for t in timeline if t >= start_ts]
    if config.end_date:
        end_ts = pd.Timestamp(config.end_date)
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        timeline = [t for t in timeline if t <= end_ts]

    if not timeline:
        raise BacktestError("No candle timestamps remain within the requested date range.")

    start_time_utc = timeline[0].isoformat()
    end_time_utc = timeline[-1].isoformat()

    # Track backtest state
    pending_candidates: dict[str, PaperTradeCandidate] = {}
    completed_trades: list[dict[str, Any]] = []
    equity_curve: list[dict[str, Any]] = []
    benchmark_start_prices: dict[str, float] = {}
    benchmark_end_prices: dict[str, float] = {}

    # Signal and risk tracking metrics
    signals_proposed = 0
    signals_approved = 0
    signals_rejected = 0
    proposals_by_action = {"BUY": 0, "SELL": 0, "HOLD": 0}
    rejections_by_reason: dict[str, int] = {}

    # Running peak equity tracker for true point-in-time drawdown calculation
    running_peak_equity = float(config.initial_cash_usd)

    settings_dummy = Settings(
        exchange_id="simulated",
        symbol=config.symbols[0],
        timeframe=config.timeframe,
        candle_limit=400,
        openai_model="gpt-4o-mini",
        openai_api_key_configured=False,
    )

    # Main Chronological Replay Loop
    for k, current_ts in enumerate(timeline):
        # 1. Execute any pending orders from the PREVIOUS bar at CURRENT bar's OPEN price
        for symbol in list(pending_candidates.keys()):
            cand = pending_candidates[symbol]
            df = prepared_candles[symbol]
            curr_bar_match = df[df["timestamp"] == current_ts]
            if curr_bar_match.empty:
                continue

            curr_bar = curr_bar_match.iloc[0]
            exec_price = float(curr_bar["open"])
            if not math.isfinite(exec_price) or exec_price <= 0:
                raise BacktestError(
                    f"Invalid execution price {exec_price} on bar {current_ts.isoformat()} for {symbol}."
                )

            # Strict anti-lookahead verification: exec timestamp must be strictly after candidate signal timestamp
            cand_ts = pd.Timestamp(cand.as_of_utc)
            if current_ts <= cand_ts:
                raise BacktestError(
                    f"Look-ahead violation: execution timestamp {current_ts} is not after signal timestamp {cand_ts}."
                )

            try:
                execution = simulate_execution(
                    db_conn,
                    candidate=cand,
                    execution_price=exec_price,
                    execution_timestamp=current_ts,
                    account_id=account_id,
                    initial_cash_usd=config.initial_cash_usd,
                    fee_rate=config.fee_rate,
                    slippage_pct=config.slippage_pct,
                )
                if execution.action == "SELL":
                    completed_trades.append(
                        {
                            "symbol": execution.symbol,
                            "execution_id": execution.execution_id,
                            "timestamp_utc": execution.execution_timestamp_utc,
                            "quantity": execution.quantity,
                            "execution_price": execution.execution_price,
                            "fee_usd": execution.fee_usd,
                            "cost_basis_sold_usd": execution.cost_basis_sold_usd,
                            "realized_pnl_usd": execution.realized_pnl_usd,
                        }
                    )
            except PaperTradingError:
                # E.g. insufficient cash or inventory guard rejection at fill time
                pass

            del pending_candidates[symbol]

        # 2. Evaluate Signals & Deterministic Risk on Closed Bar up to current_ts
        current_close_prices: dict[str, float] = {}
        for symbol in config.symbols:
            df = prepared_candles[symbol]
            aligned_candles = df[df["timestamp"] <= current_ts]
            if aligned_candles.empty or aligned_candles.iloc[-1]["timestamp"] != current_ts:
                continue

            # Record benchmark prices
            close_price = float(aligned_candles.iloc[-1]["close"])
            current_close_prices[symbol] = close_price
            if symbol not in benchmark_start_prices:
                benchmark_start_prices[symbol] = close_price
            benchmark_end_prices[symbol] = close_price

            # Extract snapshot on past closed bars only
            snap = latest_snapshot(aligned_candles)
            market_ctx = build_market_context(settings_dummy, snap, aligned_candles)

            # Load persistent portfolio state for deterministic risk checks and inventory awareness
            port_state = load_portfolio_state_from_db(
                db_conn,
                symbol=symbol,
                as_of=current_ts,
                timeframe_seconds=timeframe_seconds,
                account_id=account_id,
                current_price=close_price,
            )
            has_open_pos = port_state.current_position_usd > 0.0
            market_ctx["has_open_position"] = has_open_pos

            # Retrieve agent decision
            agent_result = provider(market_ctx, snap, symbol)
            if not agent_result:
                continue

            action = agent_result.proposal.action
            if action in proposals_by_action:
                proposals_by_action[action] += 1
            signals_proposed += 1

            if action not in ("BUY", "SELL"):
                continue

            risk_eval = evaluate_risk(
                proposal=agent_result.proposal,
                critique=agent_result.critique,
                snapshot=snap,
                recent_candles=aligned_candles,
                portfolio=port_state,
                limits=config.risk_limits,
            )

            if risk_eval.passed:
                signals_approved += 1
                cand = create_paper_trade(
                    symbol=symbol,
                    snapshot=snap,
                    proposal=agent_result.proposal,
                    critique=agent_result.critique,
                    risk_evaluation=risk_eval,
                )
                log_paper_trade(db_conn, cand)
                # Queue as pending for execution at NEXT candle open
                pending_candidates[symbol] = cand
            else:
                signals_rejected += 1
                for reason in risk_eval.rejection_reasons:
                    # Categorize reason for report breakdown
                    reason_key = (
                        "allocation_limit_reached"
                        if "allocation limit" in reason.lower()
                        else "below_min_trade_size"
                        if "below minimum trade size" in reason.lower()
                        else "cooldown_active"
                        if "cooldown" in reason.lower()
                        else "no_holdings_to_sell"
                        if "no open long position" in reason.lower()
                        else "insufficient_cash"
                        if "insufficient cash" in reason.lower()
                        else reason[:40]
                    )
                    rejections_by_reason[reason_key] = (
                        rejections_by_reason.get(reason_key, 0) + 1
                    )

        # 3. Record Point-in-Time Portfolio Valuation & Running Peak Equity
        acc = get_portfolio_account(db_conn, account_id)
        assert acc is not None
        positions = get_all_positions(db_conn, account_id)
        current_equity = calculate_portfolio_equity(acc, positions, current_close_prices)

        # Update running peak equity across daily mark-to-market
        if current_equity > running_peak_equity:
            running_peak_equity = current_equity
            # Persist updated peak equity to SQLite so load_portfolio_state_from_db evaluates risk accurately
            db_conn.execute(
                "UPDATE portfolio_accounts SET peak_equity_usd = ? WHERE account_id = ?",
                (running_peak_equity, account_id),
            )

        drawdown_pct = (
            max(0.0, (running_peak_equity - current_equity) / running_peak_equity)
            if running_peak_equity > 0
            else 0.0
        )
        position_value_usd = max(0.0, current_equity - acc.cash_usd)
        benchmark_price = current_close_prices.get(config.symbols[0], 0.0)
        equity_curve.append(
            {
                "timestamp_utc": current_ts.isoformat(),
                "cash_usd": acc.cash_usd,
                "equity_usd": current_equity,
                "position_value_usd": position_value_usd,
                "benchmark_price": benchmark_price,
                "peak_equity_usd": running_peak_equity,
                "drawdown_pct": drawdown_pct,
            }
        )

    # Final Accounting & Reporting
    final_account = get_portfolio_account(db_conn, account_id)
    assert final_account is not None
    final_positions = get_all_positions(db_conn, account_id)
    all_executions = list(reversed(get_paper_executions(db_conn, account_id=account_id, limit=10_000)))

    # Net exit-tranche P&L includes sell-side costs from the execution record
    # and a quantity-proportional allocation of pooled entry fees.
    completed_trades, remaining_entry_fees = allocate_entry_fees_to_exits(all_executions)

    # Compute gross and net unrealized P&L on remaining holdings
    gross_unrealized_pnl = sum(
        calculate_unrealized_pnl(pos, benchmark_end_prices.get(pos.symbol, pos.average_entry_price))
        for pos in final_positions
        if pos.quantity > 0
    )
    final_equity = calculate_portfolio_equity(final_account, final_positions, benchmark_end_prices)
    total_net_profit_usd = final_equity - config.initial_cash_usd
    total_return_pct = (total_net_profit_usd / config.initial_cash_usd) * 100.0

    # Separate entry fees on open positions vs closed trades
    open_positions_fees_usd = sum(remaining_entry_fees.values())
    net_realized_pnl = sum(float(t["net_realized_pnl_usd"]) for t in completed_trades)
    net_unrealized_pnl = gross_unrealized_pnl - open_positions_fees_usd

    # Buy-and-Hold Benchmark calculation
    bench_returns: list[float] = []
    symbol_breakdown: dict[str, dict[str, Any]] = {}
    for s in config.symbols:
        start_p = benchmark_start_prices.get(s, 1.0)
        end_p = benchmark_end_prices.get(s, start_p)
        b_ret = ((end_p - start_p) / start_p) * 100.0 if start_p > 0 else 0.0
        bench_returns.append(b_ret)

        s_execs = [e for e in all_executions if e["symbol"] == s]
        s_trades = [t for t in completed_trades if t["symbol"] == s]
        s_realized = sum(float(t["net_realized_pnl_usd"]) for t in s_trades)
        s_fees = sum(float(e["fee_usd"]) for e in s_execs)
        symbol_breakdown[s] = {
            "executions_count": len(s_execs),
            "completed_trades": len(s_trades),
            "net_realized_pnl_usd": s_realized,
            "total_fees_usd": s_fees,
            "benchmark_return_pct": b_ret,
        }

    benchmark_return_pct = (
        float(sum(bench_returns) / len(bench_returns)) if bench_returns else 0.0
    )
    max_drawdown_pct = max((pt["drawdown_pct"] for pt in equity_curve), default=0.0) * 100.0

    # Trade statistics
    winning_trades = [t for t in completed_trades if t["net_realized_pnl_usd"] > 0]
    losing_trades = [t for t in completed_trades if t["net_realized_pnl_usd"] < 0]
    gross_profit_usd = sum(t["net_realized_pnl_usd"] for t in winning_trades)
    gross_loss_usd = abs(sum(t["net_realized_pnl_usd"] for t in losing_trades))

    if gross_loss_usd > 0:
        profit_factor = gross_profit_usd / gross_loss_usd
    elif gross_profit_usd > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    total_completed = len(completed_trades)
    win_rate_pct = (len(winning_trades) / total_completed * 100.0) if total_completed > 0 else 0.0

    return BacktestReport(
        run_id=config.run_id,
        config=config,
        start_time_utc=start_time_utc,
        end_time_utc=end_time_utc,
        initial_equity=config.initial_cash_usd,
        final_equity=final_equity,
        total_net_profit_usd=total_net_profit_usd,
        total_return_pct=total_return_pct,
        benchmark_return_pct=benchmark_return_pct,
        max_drawdown_pct=max_drawdown_pct,
        total_executions=len(all_executions),
        completed_trades=total_completed,
        winning_trades=len(winning_trades),
        losing_trades=len(losing_trades),
        win_rate_pct=win_rate_pct,
        gross_profit_usd=gross_profit_usd,
        gross_loss_usd=gross_loss_usd,
        net_realized_pnl_usd=net_realized_pnl,
        unrealized_pnl_usd=gross_unrealized_pnl,
        gross_unrealized_pnl_usd=gross_unrealized_pnl,
        net_unrealized_pnl_usd=net_unrealized_pnl,
        total_fees_paid_usd=final_account.total_fees_paid_usd,
        open_positions_fees_usd=open_positions_fees_usd,
        profit_factor=profit_factor,
        signals_proposed=signals_proposed,
        signals_approved=signals_approved,
        signals_rejected=signals_rejected,
        rejections_by_reason=rejections_by_reason,
        proposals_by_action=proposals_by_action,
        unexecuted_orders=len(pending_candidates),
        equity_curve=equity_curve,
        symbol_breakdown=symbol_breakdown,
        execution_history=all_executions,
        completed_trade_records=completed_trades,
    )


def format_backtest_report(report: BacktestReport) -> str:
    """Render a human-readable performance, risk, and accounting audit report."""
    pf_str = (
        "inf (no losses)"
        if math.isinf(report.profit_factor)
        else f"{report.profit_factor:.2f}"
    )

    lines = [
        "=" * 70,
        f"HISTORICAL BACKTEST REPORT: {report.run_id}",
        "=" * 70,
        f"Simulation Window:  {report.start_time_utc} -> {report.end_time_utc}",
        f"Symbols Tested:     {', '.join(report.config.symbols)}",
        f"Timeframe:          {report.config.timeframe}",
        f"Fee Rate / Slip.:   {report.config.fee_rate:.4%} / {report.config.slippage_pct:.4%}",
        "",
        "PORTFOLIO PERFORMANCE & METRICS:",
        f"  Initial Equity:         ${report.initial_equity:,.2f} USD",
        f"  Final Equity:           ${report.final_equity:,.2f} USD",
        f"  Total Net Profit:       ${report.total_net_profit_usd:+,.2f} USD",
        f"  Total Return:           {report.total_return_pct:+.2f}%",
        f"  Buy & Hold Benchmark:   {report.benchmark_return_pct:+.2f}% (100% asset buy-and-hold)",
        f"  Max Drawdown:           {report.max_drawdown_pct:.2f}% (from running peak equity)",
        f"  Total Executions:       {report.total_executions}",
        f"  Realized Exit Tranches:  {report.completed_trades} (sell executions)",
        f"  Win Rate:               {report.win_rate_pct:.1f}% ({report.winning_trades} wins, {report.losing_trades} losses)",
        f"  Gross Profit:           ${report.gross_profit_usd:,.2f} USD",
        f"  Gross Loss:             ${report.gross_loss_usd:,.2f} USD",
        f"  Net Realized P&L:       ${report.net_realized_pnl_usd:+,.2f} USD",
        f"  Gross Unrealized P&L:   ${report.gross_unrealized_pnl_usd:+,.2f} USD (asset price gain on open positions)",
        f"  Net Unrealized P&L:     ${report.net_unrealized_pnl_usd:+,.2f} USD (after open position entry fees)",
        f"  Total Trading Fees:     ${report.total_fees_paid_usd:,.2f} USD",
        f"  Profit Factor:          {pf_str}",
        "",
        "SIGNAL & RISK FILTER ACTIVITY:",
        f"  Signals Proposed:       {report.signals_proposed} ({report.proposals_by_action.get('BUY', 0)} BUY, {report.proposals_by_action.get('SELL', 0)} SELL, {report.proposals_by_action.get('HOLD', 0)} HOLD)",
        f"  Approved by Risk:       {report.signals_approved}",
        f"  Rejected by Risk:       {report.signals_rejected}",
    ]

    if report.rejections_by_reason:
        lines.append("  Rejection Breakdown:")
        for r_name, count in sorted(report.rejections_by_reason.items(), key=lambda x: -x[1]):
            lines.append(f"    - {r_name}: {count}")

    lines.extend(
        [
            f"  Fills Executed:         {report.total_executions}",
            f"  Unexecuted (Final Bar): {report.unexecuted_orders}",
            "",
            "ACCOUNTING RECONCILIATION:",
            f"  Final Equity (${report.final_equity:,.2f}) = Initial (${report.initial_equity:,.2f}) + Net Realized (${report.net_realized_pnl_usd:+,.2f}) + Net Unrealized (${report.net_unrealized_pnl_usd:+,.2f})",
            f"  Net Unrealized (${report.net_unrealized_pnl_usd:+,.2f})  = Gross Unrealized (${report.gross_unrealized_pnl_usd:+,.2f}) - Open Entry Fees (${report.open_positions_fees_usd:,.2f})",
            "",
            "SYMBOL BREAKDOWN:",
        ]
    )

    for sym, stats in report.symbol_breakdown.items():
        lines.append(
            f"  - {sym:<10}: {stats['completed_trades']} trades | "
            f"Realized P&L: ${stats['net_realized_pnl_usd']:+,.2f} | "
            f"Fees: ${stats['total_fees_usd']:,.2f} | "
            f"B&H: {stats['benchmark_return_pct']:+.2f}%"
        )

    lines.extend(
        [
            "",
            "METHODOLOGY & CALCULATION NOTES:",
            "- Signals generated using ONLY past closed bars (zero look-ahead bias).",
            "- Executions simulated at the OPEN price of the bar following signal close.",
            "- Adverse slippage and trading fees are applied to every executed fill.",
            "- In spot long-only mode, shorting is strictly prohibited; sells require holdings.",
            "- Benchmark is equal-weighted 100% buy-and-hold across tested symbols from period start to end.",
            "- Max drawdown tracks running peak portfolio equity across every daily mark-to-market step.",
            "- Profit factor is defined as gross profit divided by gross loss (infinite if zero losses).",
            "",
            "DISCLAIMER:",
            "Historical backtest results are strictly educational and for research auditing only.",
            "Past performance does not predict future returns. Live trading remains disabled.",
            "=" * 70,
        ]
    )
    return "\n".join(lines)


def generate_backtest_candles(
    symbol: str = "BTC/USDT",
    periods: int = 300,
    start_price: float = 50_000.0,
    start_date: str = "2024-01-01",
    freq: str = "1D",
    trend: str = "uptrend",
) -> pd.DataFrame:
    """Generate deterministic synthetic OHLCV candles with known properties for network-free testing.

    Supported trends:
    - 'uptrend': monotonic positive price trend
    - 'downtrend': monotonic negative price trend
    - 'cycle': alternating bull and bear market waves (40 bars up, 20 bars down)
    - 'sideways': oscillating sine wave
    """
    timestamps = pd.date_range(start_date, periods=periods, freq=freq, tz="UTC")
    closes = [start_price]

    for i in range(1, periods):
        if trend == "uptrend":
            change = 50.0 + (i % 7) * 10.0 - (i % 5) * 5.0
        elif trend == "downtrend":
            change = -50.0 - (i % 7) * 10.0 + (i % 5) * 5.0
        elif trend == "cycle":
            cycle_phase = i % 60
            if cycle_phase < 40:
                change = 80.0 + (i % 5) * 15.0
            else:
                change = -140.0 - (i % 4) * 20.0
        else:  # sideways
            change = math.sin(i / 10.0) * 100.0
        closes.append(max(100.0, closes[-1] + change))

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": closes,
            "high": [c + 150.0 for c in closes],
            "low": [c - 150.0 for c in closes],
            "close": closes,
            "volume": [100.0] * periods,
        }
    )


def main() -> int:
    """CLI entry point for running historical backtests."""
    parser = argparse.ArgumentParser(
        description="Historical backtesting engine for the crypto research agent (Strictly non-live)."
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTC/USDT",
        help="Comma-separated symbols to test (default: BTC/USDT).",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1d",
        help="Timeframe to test (default: 1d).",
    )
    parser.add_argument(
        "--candle-limit",
        type=int,
        default=400,
        help="Number of historical closed candles to fetch if using exchange data (default: 400).",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=10_000.0,
        help="Initial virtual cash in USD (default: 10000.0).",
    )
    parser.add_argument(
        "--fee-rate",
        type=float,
        default=0.001,
        help="Trading fee rate (default: 0.001 = 0.10%).",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=0.0005,
        help="Slippage percentage (default: 0.0005 = 0.05%).",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional ISO start date filter (e.g. 2024-01-01).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional ISO end date filter (e.g. 2024-12-31).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run offline using deterministic synthetic candles (network-free, zero API costs).",
    )
    parser.add_argument(
        "--trend",
        type=str,
        default="uptrend",
        choices=["uptrend", "downtrend", "cycle", "sideways"],
        help="Market regime trend for synthetic data (default: uptrend).",
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default=None,
        help="Optional path to custom CSV OHLCV file.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=":memory:",
        help="SQLite path for storing backtest execution audit logs (default: :memory:).",
    )

    args = parser.parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    run_id = f"bt_{uuid.uuid4().hex[:10]}"

    config = BacktestConfig(
        run_id=run_id,
        symbols=symbols,
        timeframe=args.timeframe,
        initial_cash_usd=args.initial_cash,
        fee_rate=args.fee_rate,
        slippage_pct=args.slippage,
        start_date=args.start_date,
        end_date=args.end_date,
        sqlite_db_path=args.db_path,
    )

    print(f"Starting historical backtest {run_id} for {symbols}...")
    candles_by_symbol: dict[str, pd.DataFrame] = {}

    try:
        if args.synthetic:
            print(f"Using deterministic synthetic candle data (trend={args.trend}, network-free).")
            for sym in symbols:
                candles_by_symbol[sym] = generate_backtest_candles(
                    symbol=sym,
                    periods=max(300, args.candle_limit),
                    trend=args.trend,
                )
        elif args.csv_file:
            print(f"Loading candles from CSV: {args.csv_file}")
            df = pd.read_csv(args.csv_file)
            for sym in symbols:
                candles_by_symbol[sym] = df
        else:
            settings = load_settings()
            print(f"Fetching public historical candles for {symbols} via public exchange API ({settings.exchange_id})...")
            exchange = create_public_exchange(settings.exchange_id)
            for sym in symbols:
                sym_settings = Settings(
                    exchange_id=settings.exchange_id,
                    symbol=sym,
                    timeframe=args.timeframe,
                    candle_limit=args.candle_limit,
                    openai_model=settings.openai_model,
                    openai_api_key_configured=settings.openai_api_key_configured,
                )
                candles_by_symbol[sym] = fetch_ohlcv(sym_settings, exchange=exchange)

        report = run_backtest(
            config,
            candles_by_symbol=candles_by_symbol,
            decision_provider=DeterministicRuleProvider(),
        )

        print()
        print(format_backtest_report(report))
        return 0
    except Exception as exc:
        print(f"Backtest error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
