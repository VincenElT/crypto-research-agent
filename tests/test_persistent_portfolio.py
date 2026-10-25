"""Network-free regression tests for persistent virtual paper trading.

Covers:
- Buy, sell, and partial sell accounting
- Fees and slippage calculations
- Insufficient cash and insufficient holdings rejection
- Realized and unrealized P&L
- Equity and drawdown calculations
- Restart persistence across SQLite connections
- Duplicate execution prevention across restarts
- Transaction rollback and failure recovery
- Next-candle execution without look-ahead bias
- Repeated runs producing consistent accounting results
- Fail-closed behavior on invalid prices and quantities
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from crypto_research.exceptions import PaperTradingError, StorageError
from crypto_research.indicators import IndicatorSnapshot
from crypto_research.paper_trading import (
    PaperExecution,
    PaperTradeCandidate,
    PortfolioSnapshot,
    Position,
    VirtualAccount,
    calculate_portfolio_equity,
    calculate_unrealized_pnl,
    create_paper_trade,
    simulate_execution,
)
from crypto_research.risk import RiskEvaluation, evaluate_risk
from crypto_research.schemas import ScoutProposal, SkepticReview
from crypto_research.storage import (
    get_all_positions,
    get_latest_execution,
    get_latest_portfolio_snapshot,
    get_or_create_portfolio_account,
    get_paper_executions,
    get_pending_paper_trades,
    get_position,
    init_db,
    load_portfolio_state_from_db,
    log_paper_trade,
    save_execution_and_update_portfolio,
)


def _mem_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    return conn


def _make_candidate(
    *,
    candidate_id: str = "pt_cand_1",
    symbol: str = "BTC/USDT",
    action: str = "BUY",
    reference_price: float = 50_000.0,
    size_usd: float = 1000.0,
    size_asset: float = 0.02,
    as_of_utc: str = "2024-06-01T00:00:00Z",
) -> PaperTradeCandidate:
    return PaperTradeCandidate(
        candidate_id=candidate_id,
        symbol=symbol,
        action=action,  # type: ignore[arg-type]
        reference_price=reference_price,
        size_usd=size_usd,
        size_asset=size_asset,
        as_of_utc=as_of_utc,
        scout_confidence=0.85,
        scout_thesis="Bullish momentum confirmation",
        skeptic_rationale="Thesis supported by indicators",
        is_live=False,
        status="CANDIDATE",
    )


# =========================================================================
# 1. Buy, Sell, and Partial Sell Accounting & Fees/Slippage
# =========================================================================

def test_buy_accounting_with_fee_and_slippage() -> None:
    conn = _mem_db()
    initial_cash = 10_000.0
    account = get_or_create_portfolio_account(conn, "default", initial_cash)
    assert account.cash_usd == 10_000.0

    candidate = _make_candidate(
        candidate_id="pt_buy_1",
        action="BUY",
        reference_price=50_000.0,
        size_asset=0.02,
        as_of_utc="2024-06-01T00:00:00Z",
    )
    log_paper_trade(conn, candidate)

    # Next candle execution: raw price 50,000, slippage 0.05% (0.0005), fee 0.10% (0.001)
    execution = simulate_execution(
        conn,
        candidate=candidate,
        execution_price=50_000.0,
        execution_timestamp="2024-06-02T00:00:00Z",
        slippage_pct=0.0005,
        fee_rate=0.001,
    )

    expected_fill_price = 50_000.0 * 1.0005  # 50,025.0
    expected_notional = 0.02 * expected_fill_price  # 1000.50
    expected_fee = expected_notional * 0.001  # 1.0005
    expected_total_cost = expected_notional + expected_fee  # 1001.5005
    expected_post_cash = initial_cash - expected_total_cost

    assert execution.execution_price == pytest.approx(expected_fill_price)
    assert execution.notional_usd == pytest.approx(expected_notional)
    assert execution.fee_usd == pytest.approx(expected_fee)
    assert execution.net_cash_impact_usd == pytest.approx(-expected_total_cost)
    assert execution.realized_pnl_usd == 0.0
    assert execution.post_cash_usd == pytest.approx(expected_post_cash)
    assert execution.post_quantity == pytest.approx(0.02)
    assert execution.post_avg_entry_price == pytest.approx(expected_fill_price)
    assert execution.is_live is False

    # Check persistent DB state
    pos = get_position(conn, "default", "BTC/USDT")
    assert pos.quantity == pytest.approx(0.02)
    assert pos.average_entry_price == pytest.approx(expected_fill_price)
    assert pos.cost_basis_usd == pytest.approx(expected_notional)


def test_partial_sell_and_full_sell_accounting() -> None:
    conn = _mem_db()
    get_or_create_portfolio_account(conn, "default", 10_000.0)

    # 1. BUY 0.02 BTC
    cand_buy = _make_candidate(
        candidate_id="c_buy",
        action="BUY",
        size_asset=0.02,
        as_of_utc="2024-06-01T00:00:00Z",
    )
    log_paper_trade(conn, cand_buy)
    exec_buy = simulate_execution(
        conn,
        candidate=cand_buy,
        execution_price=50_000.0,
        execution_timestamp="2024-06-02T00:00:00Z",
        slippage_pct=0.0005,
        fee_rate=0.001,
    )
    buy_avg_price = exec_buy.execution_price  # 50,025.0
    cash_after_buy = exec_buy.post_cash_usd

    # 2. PARTIAL SELL: sell 0.01 BTC (half) at raw price 60,000 on candle 3
    cand_sell_1 = _make_candidate(
        candidate_id="c_sell_1",
        action="SELL",
        size_asset=0.01,
        as_of_utc="2024-06-02T12:00:00Z",
    )
    log_paper_trade(conn, cand_sell_1)
    exec_sell_1 = simulate_execution(
        conn,
        candidate=cand_sell_1,
        execution_price=60_000.0,
        execution_timestamp="2024-06-03T00:00:00Z",
        slippage_pct=0.0005,
        fee_rate=0.001,
    )

    expected_sell_1_fill = 60_000.0 * (1 - 0.0005)  # 59,970.0
    expected_gross_1 = 0.01 * expected_sell_1_fill  # 599.70
    expected_fee_1 = expected_gross_1 * 0.001  # 0.5997
    expected_net_1 = expected_gross_1 - expected_fee_1  # 599.1003
    expected_cost_basis_sold_1 = 0.01 * buy_avg_price  # 500.25
    expected_realized_pnl_1 = expected_net_1 - expected_cost_basis_sold_1  # 98.8503

    assert exec_sell_1.execution_price == pytest.approx(expected_sell_1_fill)
    assert exec_sell_1.fee_usd == pytest.approx(expected_fee_1)
    assert exec_sell_1.net_cash_impact_usd == pytest.approx(expected_net_1)
    assert exec_sell_1.cost_basis_sold_usd == pytest.approx(expected_cost_basis_sold_1)
    assert exec_sell_1.realized_pnl_usd == pytest.approx(expected_realized_pnl_1)
    assert exec_sell_1.post_cash_usd == pytest.approx(cash_after_buy + expected_net_1)
    assert exec_sell_1.post_quantity == pytest.approx(0.01)
    # Remaining units retain the exact same average entry price
    assert exec_sell_1.post_avg_entry_price == pytest.approx(buy_avg_price)

    # 3. FULL SELL: sell remaining 0.01 BTC at raw price 40,000 on candle 4
    cand_sell_2 = _make_candidate(
        candidate_id="c_sell_2",
        action="SELL",
        size_asset=0.01,
        as_of_utc="2024-06-03T12:00:00Z",
    )
    log_paper_trade(conn, cand_sell_2)
    exec_sell_2 = simulate_execution(
        conn,
        candidate=cand_sell_2,
        execution_price=40_000.0,
        execution_timestamp="2024-06-04T00:00:00Z",
        slippage_pct=0.0005,
        fee_rate=0.001,
    )

    expected_sell_2_fill = 40_000.0 * (1 - 0.0005)  # 39,980.0
    expected_gross_2 = 0.01 * expected_sell_2_fill  # 399.80
    expected_fee_2 = expected_gross_2 * 0.001  # 0.3998
    expected_net_2 = expected_gross_2 - expected_fee_2  # 399.4002
    expected_cost_basis_sold_2 = 0.01 * buy_avg_price  # 500.25
    expected_realized_pnl_2 = expected_net_2 - expected_cost_basis_sold_2  # -100.8498

    assert exec_sell_2.realized_pnl_usd == pytest.approx(expected_realized_pnl_2)
    assert exec_sell_2.post_quantity == 0.0
    assert exec_sell_2.post_avg_entry_price == 0.0

    pos_final = get_position(conn, "default", "BTC/USDT")
    assert pos_final.quantity == 0.0
    assert pos_final.cost_basis_usd == 0.0

    acc_final = get_or_create_portfolio_account(conn, "default")
    assert acc_final.realized_pnl_usd == pytest.approx(expected_realized_pnl_1 + expected_realized_pnl_2)


# =========================================================================
# 2. Insufficient Cash & Insufficient Holdings (Spot Long-Only Guards)
# =========================================================================

def test_insufficient_cash_rejects_buy() -> None:
    conn = _mem_db()
    get_or_create_portfolio_account(conn, "default", initial_cash_usd=500.0)

    # Attempt to buy $1000 of asset with only $500 cash
    candidate = _make_candidate(
        candidate_id="pt_big_buy",
        action="BUY",
        size_asset=0.02,  # at 50,000 raw price = ~1000 USD
        as_of_utc="2024-06-01T00:00:00Z",
    )
    log_paper_trade(conn, candidate)

    with pytest.raises(PaperTradingError, match="Insufficient virtual cash for BUY"):
        simulate_execution(
            conn,
            candidate=candidate,
            execution_price=50_000.0,
            execution_timestamp="2024-06-02T00:00:00Z",
        )

    # Cash and positions untouched
    acc = get_or_create_portfolio_account(conn, "default")
    assert acc.cash_usd == 500.0
    pos = get_position(conn, "default", "BTC/USDT")
    assert pos.quantity == 0.0


def test_insufficient_holdings_rejects_sell() -> None:
    conn = _mem_db()
    get_or_create_portfolio_account(conn, "default", initial_cash_usd=10_000.0)

    # Case A: Sell with zero holdings
    cand_empty = _make_candidate(
        candidate_id="pt_sell_empty",
        action="SELL",
        size_asset=0.01,
        as_of_utc="2024-06-01T00:00:00Z",
    )
    with pytest.raises(PaperTradingError, match="no open long position exists"):
        simulate_execution(
            conn,
            candidate=cand_empty,
            execution_price=50_000.0,
            execution_timestamp="2024-06-02T00:00:00Z",
        )

    # Case B: Buy 0.01 BTC, then try to sell 0.02 BTC
    cand_buy = _make_candidate(
        candidate_id="pt_buy_half",
        action="BUY",
        size_asset=0.01,
        as_of_utc="2024-06-01T00:00:00Z",
    )
    simulate_execution(
        conn,
        candidate=cand_buy,
        execution_price=50_000.0,
        execution_timestamp="2024-06-02T00:00:00Z",
    )

    cand_excess_sell = _make_candidate(
        candidate_id="pt_excess_sell",
        action="SELL",
        size_asset=0.02,
        as_of_utc="2024-06-02T12:00:00Z",
    )
    with pytest.raises(PaperTradingError, match="exceeds existing holdings"):
        simulate_execution(
            conn,
            candidate=cand_excess_sell,
            execution_price=50_000.0,
            execution_timestamp="2024-06-03T00:00:00Z",
        )

    # Holdings remain 0.01
    pos = get_position(conn, "default", "BTC/USDT")
    assert pos.quantity == pytest.approx(0.01)


# =========================================================================
# 3. Realized & Unrealized P&L, Equity, and Drawdown
# =========================================================================

def test_realized_and_unrealized_pnl_and_equity_calculations() -> None:
    conn = _mem_db()
    account = get_or_create_portfolio_account(conn, "default", 10_000.0)

    # Buy 0.02 BTC at raw price 50,000 (fill price = 50,025)
    cand = _make_candidate(
        candidate_id="c_unreal",
        action="BUY",
        size_asset=0.02,
        as_of_utc="2024-06-01T00:00:00Z",
    )
    simulate_execution(
        conn,
        candidate=cand,
        execution_price=50_000.0,
        execution_timestamp="2024-06-02T00:00:00Z",
        slippage_pct=0.0005,
    )
    account = get_or_create_portfolio_account(conn, "default")
    pos = get_position(conn, "default", "BTC/USDT")

    # When market price is 55,000:
    # Cost basis = 0.02 * 50025 = 1000.50
    # Current value = 0.02 * 55000 = 1100.00
    # Unrealized P&L = 1100.00 - 1000.50 = +99.50
    unreal_gain = calculate_unrealized_pnl(pos, 55_000.0)
    assert unreal_gain == pytest.approx(99.50)

    equity_gain = calculate_portfolio_equity(account, [pos], {"BTC/USDT": 55_000.0})
    assert equity_gain == pytest.approx(account.cash_usd + 1100.0)

    # When market drops to 45,000:
    # Current value = 0.02 * 45000 = 900.00
    # Unrealized P&L = 900.00 - 1000.50 = -100.50
    unreal_loss = calculate_unrealized_pnl(pos, 45_000.0)
    assert unreal_loss == pytest.approx(-100.50)

    equity_loss = calculate_portfolio_equity(account, [pos], {"BTC/USDT": 45_000.0})
    assert equity_loss == pytest.approx(account.cash_usd + 900.0)


def test_peak_equity_and_drawdown_tracking() -> None:
    conn = _mem_db()
    account = get_or_create_portfolio_account(conn, "default", 10_000.0)
    assert account.peak_equity_usd == 10_000.0

    # 1. Buy 0.1 BTC at 50,000
    cand_buy = _make_candidate(
        candidate_id="c_draw_1",
        action="BUY",
        size_asset=0.1,
        as_of_utc="2024-06-01T00:00:00Z",
    )
    simulate_execution(
        conn,
        candidate=cand_buy,
        execution_price=50_000.0,
        execution_timestamp="2024-06-02T00:00:00Z",
    )

    # 2. Sell at 70,000 (huge gain) -> peak equity rises above 10,000
    cand_sell = _make_candidate(
        candidate_id="c_draw_2",
        action="SELL",
        size_asset=0.1,
        as_of_utc="2024-06-02T12:00:00Z",
    )
    exec_gain = simulate_execution(
        conn,
        candidate=cand_sell,
        execution_price=70_000.0,
        execution_timestamp="2024-06-03T00:00:00Z",
    )
    assert exec_gain.post_cash_usd > 11_000.0
    account_gain = get_or_create_portfolio_account(conn, "default")
    assert account_gain.peak_equity_usd == pytest.approx(exec_gain.post_cash_usd)

    snap_gain = get_latest_portfolio_snapshot(conn, "default")
    assert snap_gain is not None
    assert snap_gain["drawdown_pct"] == 0.0

    # 3. Next trade: buy 0.1 BTC at 70,000 and sell at 50,000 (loss) -> drawdown rises
    cand_buy_2 = _make_candidate(
        candidate_id="c_draw_3",
        action="BUY",
        size_asset=0.1,
        as_of_utc="2024-06-04T00:00:00Z",
    )
    simulate_execution(
        conn,
        candidate=cand_buy_2,
        execution_price=70_000.0,
        execution_timestamp="2024-06-05T00:00:00Z",
    )

    cand_sell_loss = _make_candidate(
        candidate_id="c_draw_4",
        action="SELL",
        size_asset=0.1,
        as_of_utc="2024-06-05T12:00:00Z",
    )
    simulate_execution(
        conn,
        candidate=cand_sell_loss,
        execution_price=50_000.0,
        execution_timestamp="2024-06-06T00:00:00Z",
    )

    account_loss = get_or_create_portfolio_account(conn, "default")
    assert account_loss.peak_equity_usd == pytest.approx(account_gain.peak_equity_usd)
    assert account_loss.cash_usd < account_loss.peak_equity_usd

    snap_loss = get_latest_portfolio_snapshot(conn, "default")
    assert snap_loss is not None
    assert snap_loss["drawdown_pct"] > 0.05  # Significant drawdown recorded


# =========================================================================
# 4. Restart Persistence Across SQLite Connections
# =========================================================================

def test_restart_persistence_across_connections(tmp_path: Path) -> None:
    db_file = tmp_path / "paper_test.db"

    # Session 1: init db, create account, execute buy
    conn1 = init_db(db_file)
    get_or_create_portfolio_account(conn1, "default", 10_000.0)
    cand_buy = _make_candidate(
        candidate_id="restart_cand_1",
        action="BUY",
        size_asset=0.05,
        as_of_utc="2024-06-01T00:00:00Z",
    )
    log_paper_trade(conn1, cand_buy)
    exec_1 = simulate_execution(
        conn1,
        candidate=cand_buy,
        execution_price=60_000.0,
        execution_timestamp="2024-06-02T00:00:00Z",
    )
    saved_cash = exec_1.post_cash_usd
    conn1.close()  # Application shutdown simulation

    # Session 2: reconnect from new connection
    conn2 = init_db(db_file)
    acc2 = get_or_create_portfolio_account(conn2, "default")
    pos2 = get_position(conn2, "default", "BTC/USDT")

    assert acc2.cash_usd == pytest.approx(saved_cash)
    assert pos2.quantity == pytest.approx(0.05)
    assert pos2.average_entry_price == pytest.approx(exec_1.execution_price)

    # Execute SELL on the reloaded state
    cand_sell = _make_candidate(
        candidate_id="restart_cand_2",
        action="SELL",
        size_asset=0.05,
        as_of_utc="2024-06-02T12:00:00Z",
    )
    log_paper_trade(conn2, cand_sell)
    exec_2 = simulate_execution(
        conn2,
        candidate=cand_sell,
        execution_price=65_000.0,
        execution_timestamp="2024-06-03T00:00:00Z",
    )
    assert exec_2.post_quantity == 0.0
    assert exec_2.realized_pnl_usd > 0.0

    conn2.close()


# =========================================================================
# 5. Duplicate Execution Prevention
# =========================================================================

def test_duplicate_execution_prevention() -> None:
    conn = _mem_db()
    get_or_create_portfolio_account(conn, "default", 10_000.0)

    cand = _make_candidate(
        candidate_id="dup_exec_1",
        action="BUY",
        size_asset=0.01,
        as_of_utc="2024-06-01T00:00:00Z",
    )
    log_paper_trade(conn, cand)

    # First execution succeeds
    simulate_execution(
        conn,
        candidate=cand,
        execution_price=50_000.0,
        execution_timestamp="2024-06-02T00:00:00Z",
    )

    # Attempting to execute the exact same candidate again fails
    with pytest.raises(PaperTradingError, match="has already been executed"):
        simulate_execution(
            conn,
            candidate=cand,
            execution_price=50_000.0,
            execution_timestamp="2024-06-02T00:00:00Z",
        )

    # Attempting to execute another candidate for the same signal candle and symbol fails
    cand_same_candle = _make_candidate(
        candidate_id="dup_exec_2",  # Different candidate ID
        action="BUY",
        size_asset=0.01,
        as_of_utc="2024-06-01T00:00:00Z",  # Same signal candle
    )
    with pytest.raises(PaperTradingError, match="An execution already exists"):
        simulate_execution(
            conn,
            candidate=cand_same_candle,
            execution_price=50_000.0,
            execution_timestamp="2024-06-02T00:00:00Z",
        )


# =========================================================================
# 6. Transaction Rollback & Failure Recovery
# =========================================================================

def test_transaction_rollback_on_database_error() -> None:
    conn = _mem_db()
    get_or_create_portfolio_account(conn, "default", 10_000.0)

    # Create dummy candidate
    cand = _make_candidate(
        candidate_id="rollback_cand",
        action="BUY",
        size_asset=0.01,
        as_of_utc="2024-06-01T00:00:00Z",
    )
    log_paper_trade(conn, cand)

    # First execution
    exec_valid = simulate_execution(
        conn,
        candidate=cand,
        execution_price=50_000.0,
        execution_timestamp="2024-06-02T00:00:00Z",
    )
    cash_before = exec_valid.post_cash_usd
    qty_before = exec_valid.post_quantity

    # Attempt an atomic update that violates DB constraint directly
    duplicate_execution = PaperExecution(
        execution_id="exec_conflict",
        candidate_id=cand.candidate_id,  # Duplicate candidate_id violates UNIQUE
        account_id="default",
        symbol="BTC/USDT",
        action="BUY",
        signal_candle_utc="2024-06-03T00:00:00Z",
        execution_timestamp_utc="2024-06-04T00:00:00Z",
        raw_price=50_000.0,
        slippage_pct=0.0005,
        execution_price=50025.0,
        quantity=0.01,
        notional_usd=500.25,
        fee_rate=0.001,
        fee_usd=0.50,
        net_cash_impact_usd=-500.75,
        realized_pnl_usd=0.0,
        cost_basis_sold_usd=0.0,
        post_cash_usd=cash_before - 500.75,
        post_quantity=qty_before + 0.01,
        post_avg_entry_price=50025.0,
        created_at_utc="2024-06-04T00:00:00Z",
        is_live=False,
    )
    new_acc = VirtualAccount(
        account_id="default",
        initial_cash_usd=10_000.0,
        cash_usd=cash_before - 500.75,
        realized_pnl_usd=0.0,
        total_fees_paid_usd=1.0,
        peak_equity_usd=10_000.0,
        updated_at_utc="2024-06-04T00:00:00Z",
    )
    new_pos = Position(
        account_id="default",
        symbol="BTC/USDT",
        quantity=qty_before + 0.01,
        average_entry_price=50025.0,
        cost_basis_usd=1000.50,
        updated_at_utc="2024-06-04T00:00:00Z",
    )

    with pytest.raises(StorageError, match="Duplicate or conflicting"):
        save_execution_and_update_portfolio(conn, duplicate_execution, new_acc, new_pos)

    # Verify atomic rollback: account cash and position were NOT changed
    acc_check = get_or_create_portfolio_account(conn, "default")
    assert acc_check.cash_usd == pytest.approx(cash_before)
    pos_check = get_position(conn, "default", "BTC/USDT")
    assert pos_check.quantity == pytest.approx(qty_before)


# =========================================================================
# 7. Next-Candle Execution Without Look-Ahead Bias
# =========================================================================

def test_next_candle_execution_without_look_ahead_bias() -> None:
    conn = _mem_db()
    get_or_create_portfolio_account(conn, "default", 10_000.0)

    cand = _make_candidate(
        candidate_id="c_time",
        action="BUY",
        size_asset=0.01,
        as_of_utc="2024-06-01T00:00:00Z",
    )

    # 1. Execution timestamp equal to signal candle timestamp fails
    with pytest.raises(PaperTradingError, match="strictly after signal candle timestamp"):
        simulate_execution(
            conn,
            candidate=cand,
            execution_price=50_000.0,
            execution_timestamp="2024-06-01T00:00:00Z",
        )

    # 2. Execution timestamp prior to signal candle timestamp fails
    with pytest.raises(PaperTradingError, match="strictly after signal candle timestamp"):
        simulate_execution(
            conn,
            candidate=cand,
            execution_price=50_000.0,
            execution_timestamp="2024-05-31T23:59:59Z",
        )

    # 3. Execution timestamp strictly after signal candle succeeds
    exec_valid = simulate_execution(
        conn,
        candidate=cand,
        execution_price=50_000.0,
        execution_timestamp="2024-06-02T00:00:00Z",
    )
    assert pd.Timestamp(exec_valid.execution_timestamp_utc) == pd.Timestamp("2024-06-02T00:00:00Z")


# =========================================================================
# 8. Fail-Closed on Invalid Prices and Quantities
# =========================================================================

def test_fail_closed_on_invalid_or_missing_prices() -> None:
    conn = _mem_db()
    get_or_create_portfolio_account(conn, "default", 10_000.0)

    cand = _make_candidate(candidate_id="c_bad_price", as_of_utc="2024-06-01T00:00:00Z")

    for bad_price in [0.0, -100.0, float("nan"), float("inf")]:
        with pytest.raises(PaperTradingError, match="must be positive and finite"):
            simulate_execution(
                conn,
                candidate=cand,
                execution_price=bad_price,
                execution_timestamp="2024-06-02T00:00:00Z",
            )

    for bad_qty in [0.0, -0.5, float("nan"), float("inf")]:
        with pytest.raises(PaperTradingError, match="must be positive and finite"):
            simulate_execution(
                conn,
                candidate=cand,
                execution_price=50_000.0,
                execution_timestamp="2024-06-02T00:00:00Z",
                quantity=bad_qty,
            )


# =========================================================================
# 9. Repeated Runs Producing Consistent Accounting Results
# =========================================================================

def test_repeated_runs_deterministic_consistency() -> None:
    def _run_sequence() -> tuple[float, float, float]:
        conn = _mem_db()
        get_or_create_portfolio_account(conn, "default", 10_000.0)

        # 1. Buy
        c1 = _make_candidate(candidate_id="seq1", action="BUY", size_asset=0.02, as_of_utc="2024-06-01T00:00:00Z")
        log_paper_trade(conn, c1)
        simulate_execution(conn, candidate=c1, execution_price=50_000.0, execution_timestamp="2024-06-02T00:00:00Z")

        # 2. Sell half
        c2 = _make_candidate(candidate_id="seq2", action="SELL", size_asset=0.01, as_of_utc="2024-06-02T12:00:00Z")
        log_paper_trade(conn, c2)
        simulate_execution(conn, candidate=c2, execution_price=55_000.0, execution_timestamp="2024-06-03T00:00:00Z")

        acc = get_or_create_portfolio_account(conn, "default")
        pos = get_position(conn, "default", "BTC/USDT")
        return acc.cash_usd, acc.realized_pnl_usd, pos.quantity

    res1 = _run_sequence()
    res2 = _run_sequence()

    assert res1[0] == pytest.approx(res2[0])
    assert res1[1] == pytest.approx(res2[1])
    assert res1[2] == pytest.approx(res2[2])


# =========================================================================
# 10. Integration: load_portfolio_state_from_db Uses Persistent Tables
# =========================================================================

def test_load_portfolio_state_uses_persistent_tables() -> None:
    conn = _mem_db()
    get_or_create_portfolio_account(conn, "default", initial_cash_usd=10_000.0)

    # Buy 0.02 BTC
    cand = _make_candidate(candidate_id="cand_state", action="BUY", size_asset=0.02, as_of_utc="2024-06-01T00:00:00Z")
    log_paper_trade(conn, cand)
    simulate_execution(
        conn,
        candidate=cand,
        execution_price=50_000.0,
        execution_timestamp="2024-06-02T00:00:00Z",
    )

    # Load portfolio state at June 4 (2 days after June 2 execution)
    p_state = load_portfolio_state_from_db(
        conn,
        symbol="BTC/USDT",
        as_of=pd.Timestamp("2024-06-04T00:00:00Z"),
        timeframe_seconds=86_400,
        account_id="default",
    )

    assert p_state.cash_usd < 10_000.0
    assert p_state.current_position_usd == pytest.approx(1000.50)
    assert p_state.bars_since_last_trade == 2
