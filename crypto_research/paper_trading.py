"""Paper-trading simulation candidate generation and virtual execution.

Emits paper-trade candidates after Scout, Skeptic, and deterministic risk filters agree.
Simulates execution at next eligible candle price after signal candle closes (avoiding look-ahead bias).
Maintains persistent virtual portfolio state (cash, quantities, cost basis, realized & unrealized P&L, fees, slippage).
Live trading is strictly disabled. No live orders can be placed.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from crypto_research.exceptions import PaperTradingError
from crypto_research.indicators import IndicatorSnapshot
from crypto_research.risk import RiskEvaluation
from crypto_research.schemas import ScoutProposal, SkepticReview

if TYPE_CHECKING:
    import sqlite3


@dataclass(frozen=True)
class PaperTradeCandidate:
    """A simulated paper-trade candidate proposal. Strictly non-live."""

    candidate_id: str
    symbol: str
    action: Literal["BUY", "SELL"]
    reference_price: float
    size_usd: float
    size_asset: float
    as_of_utc: str
    scout_confidence: float
    scout_thesis: str
    skeptic_rationale: str
    is_live: bool = False
    status: str = "CANDIDATE"


@dataclass(frozen=True)
class VirtualAccount:
    """Virtual cash account for paper trading."""

    account_id: str
    initial_cash_usd: float
    cash_usd: float
    realized_pnl_usd: float
    total_fees_paid_usd: float
    peak_equity_usd: float
    updated_at_utc: str


@dataclass(frozen=True)
class Position:
    """Holding for a specific asset symbol in spot long-only mode."""

    account_id: str
    symbol: str
    quantity: float
    average_entry_price: float
    cost_basis_usd: float
    updated_at_utc: str


@dataclass(frozen=True)
class PaperExecution:
    """Simulated execution fill. Strictly non-live."""

    execution_id: str
    candidate_id: str
    account_id: str
    symbol: str
    action: Literal["BUY", "SELL"]
    signal_candle_utc: str
    execution_timestamp_utc: str
    raw_price: float
    slippage_pct: float
    execution_price: float
    quantity: float
    notional_usd: float
    fee_rate: float
    fee_usd: float
    net_cash_impact_usd: float
    realized_pnl_usd: float
    cost_basis_sold_usd: float
    post_cash_usd: float
    post_quantity: float
    post_avg_entry_price: float
    created_at_utc: str
    is_live: bool = False


@dataclass(frozen=True)
class PortfolioSnapshot:
    """Point-in-time valuation of the virtual portfolio."""

    account_id: str
    as_of_utc: str
    cash_usd: float
    positions_value_usd: float
    total_equity_usd: float
    unrealized_pnl_usd: float
    realized_pnl_usd: float
    peak_equity_usd: float
    drawdown_pct: float
    created_at_utc: str


def create_paper_trade(
    *,
    symbol: str,
    snapshot: IndicatorSnapshot,
    proposal: ScoutProposal,
    critique: SkepticReview,
    risk_evaluation: RiskEvaluation,
) -> PaperTradeCandidate:
    """Generate a paper-trade candidate when all filters and agents agree.

    Raises:
        PaperTradingError: If risk checks failed or proposed action is not tradable.
    """
    if not risk_evaluation.passed:
        reasons = "; ".join(risk_evaluation.rejection_reasons)
        raise PaperTradingError(
            f"Cannot create paper trade: risk evaluation rejected candidate ({reasons})."
        )

    if proposal.action not in ("BUY", "SELL"):
        raise PaperTradingError(
            f"Cannot create paper trade for non-tradable action {proposal.action!r}."
        )

    if snapshot.close <= 0:
        raise PaperTradingError(
            f"Invalid reference close price {snapshot.close}."
        )

    size_usd = risk_evaluation.approved_size_usd
    if size_usd <= 0:
        raise PaperTradingError(
            f"Approved position size must be positive, got {size_usd}."
        )

    size_asset = size_usd / snapshot.close
    candidate_id = f"pt_{uuid.uuid4().hex[:12]}"

    return PaperTradeCandidate(
        candidate_id=candidate_id,
        symbol=symbol,
        action=proposal.action,
        reference_price=snapshot.close,
        size_usd=size_usd,
        size_asset=size_asset,
        as_of_utc=snapshot.as_of.isoformat(),
        scout_confidence=proposal.confidence,
        scout_thesis=proposal.thesis,
        skeptic_rationale=critique.rationale,
        is_live=False,  # Hard invariant: live trading cannot be enabled
        status="CANDIDATE",
    )


def simulate_execution(
    conn: sqlite3.Connection,
    *,
    candidate: PaperTradeCandidate,
    execution_price: float,
    execution_timestamp: pd.Timestamp | datetime | str,
    account_id: str = "default",
    initial_cash_usd: float = 10_000.0,
    fee_rate: float = 0.001,
    slippage_pct: float = 0.0005,
    quantity: float | None = None,
) -> PaperExecution:
    """Simulate order execution at the next eligible candle price after signal closes.

    Strict look-ahead bias prevention: execution_timestamp must be strictly greater than
    candidate.as_of_utc (the signal candle close).

    Accounting:
    - In spot long-only mode:
      - BUY: execution_price = raw_price * (1 + slippage), notional = qty * execution_price,
        fee = notional * fee_rate, total_cost = notional + fee. Requires total_cost <= cash.
        Cash decreases by total_cost. Holdings and cost basis increase by notional.
      - SELL: execution_price = raw_price * (1 - slippage), gross_proceeds = qty * execution_price,
        fee = gross_proceeds * fee_rate, net_proceeds = gross_proceeds - fee.
        Requires qty <= position.quantity. Sells cannot create short positions.
        Cost basis sold = qty * average_entry_price.
        Realized P&L = net_proceeds - cost_basis_sold.
        Cash increases by net_proceeds. Holdings decrease by qty. Average entry price of remaining
        holdings is preserved.

    Updates database atomically inside a transaction.
    """
    from crypto_research.storage import (
        get_all_positions,
        get_or_create_portfolio_account,
        get_position,
        has_execution_for_candidate,
        has_execution_for_candle,
        save_execution_and_update_portfolio,
    )

    # 1. Anti-lookahead timestamp guard
    if isinstance(execution_timestamp, str):
        exec_ts = pd.Timestamp(execution_timestamp)
    elif isinstance(execution_timestamp, datetime):
        exec_ts = pd.Timestamp(execution_timestamp)
    else:
        exec_ts = execution_timestamp

    signal_ts = pd.Timestamp(candidate.as_of_utc)
    if exec_ts <= signal_ts:
        raise PaperTradingError(
            f"Execution timestamp {exec_ts.isoformat()} must be strictly after signal candle "
            f"timestamp {signal_ts.isoformat()} to prevent look-ahead bias."
        )

    # 2. Execution price validation (fail closed)
    if not math.isfinite(execution_price) or execution_price <= 0:
        raise PaperTradingError(
            f"Execution price ({execution_price}) must be positive and finite."
        )

    # 3. Parameters validation
    if not (0.0 <= fee_rate < 1.0):
        raise PaperTradingError(f"Fee rate {fee_rate} must be in [0, 1).")
    if not (0.0 <= slippage_pct < 1.0):
        raise PaperTradingError(f"Slippage pct {slippage_pct} must be in [0, 1).")

    # 4. Duplicate execution prevention
    if has_execution_for_candidate(conn, candidate.candidate_id):
        raise PaperTradingError(
            f"Candidate {candidate.candidate_id} has already been executed."
        )
    if has_execution_for_candle(conn, account_id, candidate.symbol, candidate.as_of_utc):
        raise PaperTradingError(
            f"An execution already exists for account {account_id!r}, symbol {candidate.symbol!r}, "
            f"and signal candle {candidate.as_of_utc}."
        )

    # 5. Load current persistent account & position
    account = get_or_create_portfolio_account(conn, account_id, initial_cash_usd)
    position = get_position(conn, account_id, candidate.symbol)

    # 6. Trade quantity determination
    if quantity is not None:
        trade_qty = float(quantity)
    else:
        trade_qty = float(candidate.size_asset)

    if not math.isfinite(trade_qty) or trade_qty <= 0:
        raise PaperTradingError(
            f"Trade quantity ({trade_qty}) must be positive and finite."
        )

    now_utc = datetime.now(timezone.utc).isoformat()

    # 7. Execution simulation & accounting
    if candidate.action == "BUY":
        fill_price = execution_price * (1.0 + slippage_pct)
        notional = trade_qty * fill_price
        fee_usd = notional * fee_rate
        total_cost = notional + fee_usd

        if total_cost > account.cash_usd:
            raise PaperTradingError(
                f"Insufficient virtual cash for BUY: required {total_cost:.2f} USD "
                f"(notional {notional:.2f} + fee {fee_usd:.2f}), available {account.cash_usd:.2f} USD."
            )

        post_cash = account.cash_usd - total_cost
        post_qty = position.quantity + trade_qty
        post_cost_basis = position.cost_basis_usd + notional
        post_avg_entry_price = post_cost_basis / post_qty if post_qty > 0 else 0.0

        realized_pnl = 0.0
        cost_basis_sold = 0.0
        net_cash_impact = -total_cost

    elif candidate.action == "SELL":
        if position.quantity <= 0.0:
            raise PaperTradingError(
                f"Cannot execute SELL: no open long position exists for {candidate.symbol} in spot mode."
            )

        # Allow slight floating point tolerance when selling entire holdings
        if trade_qty > position.quantity:
            if (trade_qty - position.quantity) <= 1e-7:
                trade_qty = position.quantity
            else:
                raise PaperTradingError(
                    f"Cannot execute SELL: requested quantity {trade_qty} exceeds existing "
                    f"holdings {position.quantity} in spot mode."
                )

        fill_price = execution_price * (1.0 - slippage_pct)
        notional = trade_qty * fill_price
        fee_usd = notional * fee_rate
        gross_proceeds = notional
        net_proceeds = gross_proceeds - fee_usd
        cost_basis_sold = trade_qty * position.average_entry_price
        realized_pnl = net_proceeds - cost_basis_sold

        post_cash = account.cash_usd + net_proceeds
        post_qty = max(0.0, position.quantity - trade_qty)
        if post_qty < 1e-9:
            post_qty = 0.0
            post_cost_basis = 0.0
            post_avg_entry_price = 0.0
        else:
            post_cost_basis = post_qty * position.average_entry_price
            post_avg_entry_price = position.average_entry_price

        net_cash_impact = net_proceeds

    else:
        raise PaperTradingError(f"Unsupported action {candidate.action!r} for execution.")

    # 8. Portfolio valuation & peak equity / drawdown
    post_realized_pnl = account.realized_pnl_usd + realized_pnl
    post_total_fees = account.total_fees_paid_usd + fee_usd

    all_positions = get_all_positions(conn, account_id)
    other_positions_value = sum(
        p.quantity * p.average_entry_price
        for p in all_positions
        if p.symbol != candidate.symbol
    )
    current_symbol_value = post_qty * execution_price
    total_positions_value = other_positions_value + current_symbol_value
    post_equity = post_cash + total_positions_value
    post_peak_equity = max(account.peak_equity_usd, post_equity)
    drawdown = (
        max(0.0, (post_peak_equity - post_equity) / post_peak_equity)
        if post_peak_equity > 0
        else 0.0
    )

    new_account = VirtualAccount(
        account_id=account.account_id,
        initial_cash_usd=account.initial_cash_usd,
        cash_usd=post_cash,
        realized_pnl_usd=post_realized_pnl,
        total_fees_paid_usd=post_total_fees,
        peak_equity_usd=post_peak_equity,
        updated_at_utc=now_utc,
    )

    new_position = Position(
        account_id=account.account_id,
        symbol=candidate.symbol,
        quantity=post_qty,
        average_entry_price=post_avg_entry_price,
        cost_basis_usd=post_cost_basis,
        updated_at_utc=now_utc,
    )

    execution = PaperExecution(
        execution_id=f"exec_{uuid.uuid4().hex[:12]}",
        candidate_id=candidate.candidate_id,
        account_id=account.account_id,
        symbol=candidate.symbol,
        action=candidate.action,
        signal_candle_utc=candidate.as_of_utc,
        execution_timestamp_utc=exec_ts.isoformat(),
        raw_price=execution_price,
        slippage_pct=slippage_pct,
        execution_price=fill_price,
        quantity=trade_qty,
        notional_usd=notional,
        fee_rate=fee_rate,
        fee_usd=fee_usd,
        net_cash_impact_usd=net_cash_impact,
        realized_pnl_usd=realized_pnl,
        cost_basis_sold_usd=cost_basis_sold,
        post_cash_usd=post_cash,
        post_quantity=post_qty,
        post_avg_entry_price=post_avg_entry_price,
        created_at_utc=now_utc,
        is_live=False,  # Hard invariant: live trading cannot be enabled
    )

    unrealized_pnl = (current_symbol_value - post_cost_basis) + sum(
        (p.quantity * p.average_entry_price) - p.cost_basis_usd
        for p in all_positions
        if p.symbol != candidate.symbol
    )

    snapshot = PortfolioSnapshot(
        account_id=account.account_id,
        as_of_utc=exec_ts.isoformat(),
        cash_usd=post_cash,
        positions_value_usd=total_positions_value,
        total_equity_usd=post_equity,
        unrealized_pnl_usd=unrealized_pnl,
        realized_pnl_usd=post_realized_pnl,
        peak_equity_usd=post_peak_equity,
        drawdown_pct=drawdown,
        created_at_utc=now_utc,
    )

    # 9. Atomic SQLite commit
    save_execution_and_update_portfolio(conn, execution, new_account, new_position, snapshot)
    return execution


def calculate_unrealized_pnl(position: Position, current_price: float) -> float:
    """Calculate unrealized P&L for a position at current market price."""
    if position.quantity <= 0.0 or not math.isfinite(current_price) or current_price <= 0:
        return 0.0
    current_market_value = position.quantity * current_price
    return current_market_value - position.cost_basis_usd


def calculate_portfolio_equity(
    account: VirtualAccount,
    positions: list[Position],
    current_prices: dict[str, float] | None = None,
) -> float:
    """Calculate total virtual portfolio equity given current market prices."""
    current_prices = current_prices or {}
    positions_value = sum(
        pos.quantity * current_prices.get(pos.symbol, pos.average_entry_price)
        for pos in positions
        if pos.quantity > 0.0
    )
    return account.cash_usd + positions_value
