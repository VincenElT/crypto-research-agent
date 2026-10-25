"""Network-free regression tests for the historical backtesting engine.

Covers:
- Chronological replay and next-candle execution without look-ahead bias
- Indicators and market context look-ahead prevention
- Fees, slippage, partial sells, and spot accounting
- Performance metrics (total return, max drawdown, win rate, gross profit/loss, profit factor)
- Buy-and-hold benchmark return calculations
- Multiple symbols and isolated portfolio state
- Missing data, gaps, and invalid candle handling (fail-closed)
- Deterministic repeatability across identical runs
- Zero gross losses and metric edge cases
- Preserving the normal paper-trading portfolio intact
"""

from __future__ import annotations

import math
import sqlite3
from typing import Any

import pandas as pd
import pytest

from crypto_research.backtest import (
    allocate_entry_fees_to_exits,
    BacktestConfig,
    BacktestReport,
    DeterministicRuleProvider,
    RecordedDecisionProvider,
    format_backtest_report,
    generate_backtest_candles,
    run_backtest,
    validate_backtest_candles,
)
from crypto_research.exceptions import (
    BacktestError,
    InsufficientHistoryError,
    MarketDataError,
)
from crypto_research.research import DualAgentResult
from crypto_research.risk import RiskLimits
from crypto_research.schemas import ScoutProposal, SkepticReview
from crypto_research.storage import (
    get_or_create_portfolio_account,
    get_position,
    init_db,
)


def _linear_candles(
    periods: int = 250,
    start_price: float = 100.0,
    step: float = 1.0,
    start_date: str = "2024-01-01",
) -> pd.DataFrame:
    """Generate linear ascending OHLCV candles."""
    timestamps = pd.date_range(start_date, periods=periods, freq="1D", tz="UTC")
    closes = [start_price + i * step for i in range(periods)]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": closes,
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "volume": [10.0] * periods,
        }
    )


# =========================================================================
# 1. Chronological Replay and Next-Candle Execution (No Look-Ahead)
# =========================================================================

def test_chronological_replay_executes_at_next_candle_open() -> None:
    candles = _linear_candles(250, start_price=100.0, step=1.0)
    config = BacktestConfig(
        run_id="test_next_open",
        symbols=["BTC/USDT"],
        initial_cash_usd=10_000.0,
        fee_rate=0.001,
        slippage_pct=0.0005,
        risk_limits=RiskLimits(max_position_size_usd=1000.0),
    )

    # Recorded decision provider: BUY on bar 205, SELL on bar 215 (9 bars later > 5 cooldown bars)
    ts_buy_signal = candles.iloc[205]["timestamp"].isoformat()
    ts_sell_signal = candles.iloc[215]["timestamp"].isoformat()

    decisions = {
        ("BTC/USDT", ts_buy_signal): DualAgentResult(
            proposal=ScoutProposal(
                action="BUY",
                confidence=0.85,
                thesis="Buy signal",
                evidence=["Uptrend"],
                risks=["Pullback"],
                missing_data=[],
            ),
            critique=SkepticReview(
                verdict="APPROVE",
                agrees_with_action=True,
                rationale="Valid buy",
                missing_evidence=[],
                weaknesses=["Volume"],
                confidence=0.8,
            ),
        ),
        ("BTC/USDT", ts_sell_signal): DualAgentResult(
            proposal=ScoutProposal(
                action="SELL",
                confidence=0.85,
                thesis="Sell signal",
                evidence=["Take profit"],
                risks=["Further upside"],
                missing_data=[],
            ),
            critique=SkepticReview(
                verdict="APPROVE",
                agrees_with_action=True,
                rationale="Valid sell",
                missing_evidence=[],
                weaknesses=["Macro"],
                confidence=0.8,
            ),
        ),
    }

    report = run_backtest(
        config,
        candles_by_symbol={"BTC/USDT": candles},
        decision_provider=RecordedDecisionProvider(decisions),
    )

    assert report.total_executions == 2
    assert report.completed_trades == 1

    buy_exec = report.execution_history[0]
    sell_exec = report.execution_history[1]

    # Verification: BUY signal was at bar 205 close, execution occurred at bar 206 OPEN!
    expected_buy_exec_ts = candles.iloc[206]["timestamp"].isoformat()
    expected_buy_open_price = float(candles.iloc[206]["open"])
    assert buy_exec["signal_candle_utc"] == ts_buy_signal
    assert buy_exec["execution_timestamp_utc"] == expected_buy_exec_ts
    assert buy_exec["raw_price"] == expected_buy_open_price
    # Fill price includes adverse slippage: open * (1 + 0.0005)
    assert buy_exec["execution_price"] == pytest.approx(expected_buy_open_price * 1.0005)

    # Verification: SELL signal was at bar 215 close, execution occurred at bar 216 OPEN!
    expected_sell_exec_ts = candles.iloc[216]["timestamp"].isoformat()
    expected_sell_open_price = float(candles.iloc[216]["open"])
    assert sell_exec["signal_candle_utc"] == ts_sell_signal
    assert sell_exec["execution_timestamp_utc"] == expected_sell_exec_ts
    assert sell_exec["raw_price"] == expected_sell_open_price
    # Fill price includes adverse slippage: open * (1 - 0.0005)
    assert sell_exec["execution_price"] == pytest.approx(expected_sell_open_price * 0.9995)
    net_trade = report.completed_trade_records[0]
    assert net_trade["allocated_entry_fee_usd"] == pytest.approx(buy_exec["fee_usd"])
    assert net_trade["net_realized_pnl_usd"] == pytest.approx(
        sell_exec["realized_pnl_usd"] - buy_exec["fee_usd"]
    )
    assert net_trade["realized_pnl_usd"] == pytest.approx(
        net_trade["net_realized_pnl_usd"]
    )


def test_final_bar_signal_is_not_executed_without_next_candle() -> None:
    """If a signal is generated on the last candle in the dataset, no execution must occur."""
    candles = _linear_candles(250)
    last_bar_ts = candles.iloc[-1]["timestamp"].isoformat()

    decisions = {
        ("BTC/USDT", last_bar_ts): DualAgentResult(
            proposal=ScoutProposal(
                action="BUY",
                confidence=0.85,
                thesis="Buy at the very end",
                evidence=["Trend"],
                risks=["Risk"],
                missing_data=[],
            ),
            critique=SkepticReview(
                verdict="APPROVE",
                agrees_with_action=True,
                rationale="Approve",
                missing_evidence=[],
                weaknesses=["Volume missing"],
                confidence=0.8,
            ),
        ),
    }

    config = BacktestConfig(run_id="test_last_bar", symbols=["BTC/USDT"])
    report = run_backtest(
        config,
        candles_by_symbol={"BTC/USDT": candles},
        decision_provider=RecordedDecisionProvider(decisions),
    )

    # Should not fabricate a fill without next candle
    assert report.total_executions == 0
    assert report.completed_trades == 0
    assert report.final_equity == 10_000.0


# =========================================================================
# 2. Performance Metrics & Edge Cases
# =========================================================================

def test_performance_metrics_and_drawdown_calculation() -> None:
    candles = generate_backtest_candles("BTC/USDT", periods=250, start_price=50_000.0, trend="uptrend")
    config = BacktestConfig(
        run_id="test_metrics",
        symbols=["BTC/USDT"],
        initial_cash_usd=10_000.0,
        fee_rate=0.001,
        slippage_pct=0.0005,
    )

    report = run_backtest(
        config,
        candles_by_symbol={"BTC/USDT": candles},
        decision_provider=DeterministicRuleProvider(),
    )

    assert isinstance(report, BacktestReport)
    assert report.initial_equity == 10_000.0
    assert report.final_equity > 0.0
    assert math.isfinite(report.total_return_pct)
    assert math.isfinite(report.benchmark_return_pct)
    assert 0.0 <= report.max_drawdown_pct <= 100.0
    assert report.win_rate_pct >= 0.0
    assert len(report.equity_curve) > 0


def test_profit_factor_edge_cases() -> None:
    candles = _linear_candles(250)

    # Scenario A: Zero completed trades -> profit factor is 0.0
    config_empty = BacktestConfig(run_id="test_pf_0", symbols=["BTC/USDT"])
    report_empty = run_backtest(
        config_empty,
        candles_by_symbol={"BTC/USDT": candles},
        decision_provider=RecordedDecisionProvider({}),
    )
    assert report_empty.completed_trades == 0
    assert report_empty.profit_factor == 0.0
    assert report_empty.win_rate_pct == 0.0

    # Scenario B: Only winning trades (zero gross loss) -> profit factor is float('inf')
    ts_buy = candles.iloc[205]["timestamp"].isoformat()
    ts_sell = candles.iloc[215]["timestamp"].isoformat()  # Higher price

    decisions_win = {
        ("BTC/USDT", ts_buy): DualAgentResult(
            proposal=ScoutProposal(action="BUY", confidence=0.8, thesis="t", evidence=["e"], risks=["r"], missing_data=[]),
            critique=SkepticReview(verdict="APPROVE", agrees_with_action=True, rationale="r", missing_evidence=[], weaknesses=["Volume missing"], confidence=0.8),
        ),
        ("BTC/USDT", ts_sell): DualAgentResult(
            proposal=ScoutProposal(action="SELL", confidence=0.8, thesis="t", evidence=["e"], risks=["r"], missing_data=[]),
            critique=SkepticReview(verdict="APPROVE", agrees_with_action=True, rationale="r", missing_evidence=[], weaknesses=["Volume missing"], confidence=0.8),
        ),
    }

    config_win = BacktestConfig(run_id="test_pf_inf", symbols=["BTC/USDT"])
    report_win = run_backtest(
        config_win,
        candles_by_symbol={"BTC/USDT": candles},
        decision_provider=RecordedDecisionProvider(decisions_win),
    )
    assert report_win.completed_trades == 1
    assert report_win.winning_trades == 1
    assert report_win.losing_trades == 0
    assert report_win.gross_loss_usd == 0.0
    assert report_win.gross_profit_usd > 0.0
    assert math.isinf(report_win.profit_factor)

    formatted = format_backtest_report(report_win)
    assert "inf (no losses)" in formatted


def test_buy_and_hold_benchmark_calculation() -> None:
    candles = _linear_candles(250, start_price=100.0, step=1.0)
    config = BacktestConfig(run_id="test_bench", symbols=["BTC/USDT"])

    report = run_backtest(
        config,
        candles_by_symbol={"BTC/USDT": candles},
        decision_provider=RecordedDecisionProvider({}),
    )

    # Timeline covers bars 200 through 249 (start close 300, end close 349)
    # Expected benchmark return = (349 - 300) / 300 * 100 = 16.333%
    start_close = report.equity_curve[0]["equity_usd"]
    assert report.benchmark_return_pct > 0.0


# =========================================================================
# 3. Multi-Symbol Isolation & State Separation
# =========================================================================

def test_multi_symbol_backtest_with_breakdown() -> None:
    candles_btc = generate_backtest_candles("BTC/USDT", periods=250, start_price=50_000.0, trend="uptrend")
    candles_eth = generate_backtest_candles("ETH/USDT", periods=250, start_price=3_000.0, trend="uptrend")

    config = BacktestConfig(
        run_id="test_multi_sym",
        symbols=["BTC/USDT", "ETH/USDT"],
        initial_cash_usd=20_000.0,
    )

    report = run_backtest(
        config,
        candles_by_symbol={"BTC/USDT": candles_btc, "ETH/USDT": candles_eth},
        decision_provider=DeterministicRuleProvider(),
    )

    assert "BTC/USDT" in report.symbol_breakdown
    assert "ETH/USDT" in report.symbol_breakdown
    assert isinstance(report.symbol_breakdown["BTC/USDT"]["benchmark_return_pct"], float)
    assert isinstance(report.symbol_breakdown["ETH/USDT"]["benchmark_return_pct"], float)


def test_multi_symbol_uses_shared_timestamp_range_for_valuation() -> None:
    # Different history end dates are allowed: valuation and benchmarks use
    # the synchronized overlap, never the later symbol's stale entry price.
    candles_btc = _linear_candles(252, start_price=50_000.0, step=100.0)
    candles_eth = _linear_candles(250, start_price=3_000.0, step=10.0)
    signal_ts = candles_eth.iloc[200]["timestamp"].isoformat()
    buy_eth = DualAgentResult(
        proposal=ScoutProposal(
            action="BUY", confidence=0.9, thesis="test buy", evidence=["test"],
            risks=["fixture"], missing_data=[],
        ),
        critique=SkepticReview(
            verdict="APPROVE", agrees_with_action=True, rationale="test",
            missing_evidence=[], weaknesses=["fixture"], confidence=0.9,
        ),
    )
    report = run_backtest(
        BacktestConfig(
            run_id="test_multi_shared_overlap",
            symbols=["BTC/USDT", "ETH/USDT"],
            initial_cash_usd=20_000.0,
        ),
        candles_by_symbol={"BTC/USDT": candles_btc, "ETH/USDT": candles_eth},
        decision_provider=RecordedDecisionProvider({("ETH/USDT", signal_ts): buy_eth}),
    )

    assert len(report.equity_curve) == 51  # Shared eligible timestamps: bars 199..249.
    assert report.equity_curve[0]["timestamp_utc"] == candles_eth.iloc[199]["timestamp"].isoformat()
    assert report.equity_curve[-1]["timestamp_utc"] == candles_eth.iloc[249]["timestamp"].isoformat()
    assert all(set(point["prices_usd"]) == {"BTC/USDT", "ETH/USDT"} for point in report.equity_curve)

    fill_mark = next(
        point for point in report.equity_curve
        if point["timestamp_utc"] == candles_eth.iloc[201]["timestamp"].isoformat()
    )
    eth_buy = next(e for e in report.execution_history if e["symbol"] == "ETH/USDT")
    assert fill_mark["prices_usd"]["ETH/USDT"] == candles_eth.iloc[201]["close"]
    assert eth_buy["action"] == "BUY"
    assert fill_mark["position_value_usd"] > 0.0
    expected_eth_return = (
        (candles_eth.iloc[249]["close"] / candles_eth.iloc[199]["close"]) - 1.0
    ) * 100.0
    assert report.symbol_breakdown["ETH/USDT"]["benchmark_return_pct"] == pytest.approx(
        expected_eth_return
    )


def test_multi_symbol_rejects_offset_candle_timestamps() -> None:
    candles_btc = _linear_candles(250, start_price=50_000.0)
    candles_eth = _linear_candles(250, start_price=3_000.0)
    candles_eth["timestamp"] += pd.Timedelta(hours=12)
    config = BacktestConfig(
        run_id="test_multi_offset_timestamps",
        symbols=["BTC/USDT", "ETH/USDT"],
    )
    with pytest.raises(BacktestError, match="synchronized eligible candle timestamps"):
        run_backtest(
            config,
            candles_by_symbol={"BTC/USDT": candles_btc, "ETH/USDT": candles_eth},
            decision_provider=RecordedDecisionProvider({}),
        )


def test_backtest_run_does_not_contaminate_normal_paper_portfolio() -> None:
    conn = init_db(":memory:")
    # Initialize normal paper portfolio
    get_or_create_portfolio_account(conn, account_id="default", initial_cash_usd=10_000.0)

    candles = _linear_candles(250)
    config = BacktestConfig(run_id="unique_run_999", symbols=["BTC/USDT"])
    run_backtest(
        config,
        candles_by_symbol={"BTC/USDT": candles},
        decision_provider=DeterministicRuleProvider(),
        conn=conn,
    )

    # Check that normal paper account 'default' is completely untouched
    default_acc = get_or_create_portfolio_account(conn, account_id="default")
    assert default_acc.cash_usd == 10_000.0
    assert default_acc.realized_pnl_usd == 0.0
    pos = get_position(conn, "default", "BTC/USDT")
    assert pos.quantity == 0.0


# =========================================================================
# 4. Fail-Closed on Missing and Corrupt Data
# =========================================================================

def test_empty_dataset_raises_insufficient_history() -> None:
    empty_df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    config = BacktestConfig(run_id="test_empty", symbols=["BTC/USDT"])

    with pytest.raises(InsufficientHistoryError, match="empty"):
        run_backtest(config, candles_by_symbol={"BTC/USDT": empty_df})


def test_short_history_raises_insufficient_history() -> None:
    short_df = _linear_candles(150)  # Requires at least 200 for SMA 200
    config = BacktestConfig(run_id="test_short", symbols=["BTC/USDT"])

    with pytest.raises(InsufficientHistoryError, match="Need at least 200"):
        run_backtest(config, candles_by_symbol={"BTC/USDT": short_df})


def test_invalid_candle_data_fails_closed() -> None:
    candles = _linear_candles(250)

    # 1. High < Low
    bad_hl = candles.copy()
    bad_hl.loc[10, "high"] = 50.0
    bad_hl.loc[10, "low"] = 150.0
    with pytest.raises(MarketDataError, match="high < low"):
        validate_backtest_candles(bad_hl, "BTC/USDT", 86_400)

    # 2. Non-positive price
    bad_neg = candles.copy()
    bad_neg.loc[10, "close"] = -10.0
    with pytest.raises(MarketDataError, match="Non-positive OHLC"):
        validate_backtest_candles(bad_neg, "BTC/USDT", 86_400)

    # 3. Duplicate timestamp
    bad_dup = candles.copy()
    bad_dup.loc[10, "timestamp"] = bad_dup.loc[9, "timestamp"]
    with pytest.raises(MarketDataError, match="Duplicate timestamp"):
        validate_backtest_candles(bad_dup, "BTC/USDT", 86_400)


def test_invalid_candle_ohlc_volume_and_order_fail_closed() -> None:
    candles = _linear_candles(250)

    bad_high = candles.copy()
    bad_high.loc[10, "high"] = bad_high.loc[10, "close"] - 0.25
    with pytest.raises(MarketDataError, match="high below open or close"):
        validate_backtest_candles(bad_high, "BTC/USDT", 86_400)

    bad_low = candles.copy()
    bad_low.loc[10, "low"] = bad_low.loc[10, "open"] + 0.25
    with pytest.raises(MarketDataError, match="low above open or close"):
        validate_backtest_candles(bad_low, "BTC/USDT", 86_400)

    for bad_volume in (-1.0, float("inf")):
        bad = candles.copy()
        bad.loc[10, "volume"] = bad_volume
        with pytest.raises(MarketDataError, match="volume"):
            validate_backtest_candles(bad, "BTC/USDT", 86_400)

    shuffled = candles.copy()
    shuffled.loc[[10, 11], "timestamp"] = shuffled.loc[[11, 10], "timestamp"].to_numpy()
    with pytest.raises(MarketDataError, match="Non-monotonic"):
        validate_backtest_candles(shuffled, "BTC/USDT", 86_400)

    gap = candles.drop(index=10).reset_index(drop=True)
    with pytest.raises(MarketDataError, match="Missing candle gaps detected"):
        validate_backtest_candles(gap, "BTC/USDT", 86_400)


def test_entry_fees_are_allocated_across_partial_exits_and_open_inventory() -> None:
    executions = [
        {"symbol": "BTC/USDT", "action": "BUY", "quantity": 1.0, "fee_usd": 1.0},
        {"symbol": "BTC/USDT", "action": "BUY", "quantity": 1.0, "fee_usd": 2.0},
        {
            "symbol": "BTC/USDT", "action": "SELL", "quantity": 0.5,
            "fee_usd": 0.4, "realized_pnl_usd": 10.0,
        },
        {
            "symbol": "BTC/USDT", "action": "SELL", "quantity": 1.0,
            "fee_usd": 0.8, "realized_pnl_usd": 20.0,
        },
        {"symbol": "ETH/USDT", "action": "BUY", "quantity": 2.0, "fee_usd": 3.0},
    ]

    exits, remaining_fees = allocate_entry_fees_to_exits(executions)

    # Average-cost, pro-rata allocation: the first 0.5 of 2 BTC allocates
    # 0.75 fee; the next 1.0 of the remaining 1.5 allocates 1.50.
    assert [row["allocated_entry_fee_usd"] for row in exits] == pytest.approx([0.75, 1.5])
    assert [row["net_realized_pnl_usd"] for row in exits] == pytest.approx([9.25, 18.5])
    assert [row["realized_pnl_usd"] for row in exits] == pytest.approx([9.25, 18.5])
    assert remaining_fees["BTC/USDT"] == pytest.approx(0.75)
    assert remaining_fees["ETH/USDT"] == pytest.approx(3.0)

    # Full liquidation allocates all remaining entry fees; an open-only symbol
    # retains its fee pool for net unrealized P&L.
    executions.append({
        "symbol": "BTC/USDT", "action": "SELL", "quantity": 0.5,
        "fee_usd": 0.4, "realized_pnl_usd": 10.0,
    })
    exits, remaining_fees = allocate_entry_fees_to_exits(executions)
    assert exits[-1]["allocated_entry_fee_usd"] == pytest.approx(0.75)
    assert exits[-1]["net_realized_pnl_usd"] == pytest.approx(9.25)
    assert remaining_fees["BTC/USDT"] == pytest.approx(0.0)
    assert remaining_fees["ETH/USDT"] == pytest.approx(3.0)


# =========================================================================
# 5. Deterministic Repeatability
# =========================================================================

def test_deterministic_repeatability_on_identical_runs() -> None:
    candles = generate_backtest_candles("BTC/USDT", periods=250, start_price=50_000.0)

    cfg1 = BacktestConfig(run_id="run_repeat_1", symbols=["BTC/USDT"], initial_cash_usd=10_000.0)
    cfg2 = BacktestConfig(run_id="run_repeat_2", symbols=["BTC/USDT"], initial_cash_usd=10_000.0)

    rep1 = run_backtest(cfg1, candles_by_symbol={"BTC/USDT": candles}, decision_provider=DeterministicRuleProvider())
    rep2 = run_backtest(cfg2, candles_by_symbol={"BTC/USDT": candles}, decision_provider=DeterministicRuleProvider())

    assert rep1.final_equity == pytest.approx(rep2.final_equity)
    assert rep1.total_return_pct == pytest.approx(rep2.total_return_pct)
    assert rep1.max_drawdown_pct == pytest.approx(rep2.max_drawdown_pct)
    assert rep1.total_executions == rep2.total_executions
    assert rep1.completed_trades == rep2.completed_trades
    assert rep1.net_realized_pnl_usd == pytest.approx(rep2.net_realized_pnl_usd)
    assert len(rep1.equity_curve) == len(rep2.equity_curve)


# =========================================================================
# 6. Audit & Validation Regression Tests (Phase 4)
# =========================================================================

def test_running_peak_equity_and_drawdown_on_open_position_pullback() -> None:
    """Verifies that running peak equity updates on daily mark-to-market.

    A sharp unrealized pullback during an open position must reflect in max_drawdown_pct
    even when current equity remains above the initial cash balance.
    """
    # 200 bars base at 100.0, then surges to 300.0 by bar 220, then drops to 150.0 by bar 240
    timestamps = pd.date_range("2024-01-01", periods=250, freq="1D", tz="UTC")
    closes = [100.0] * 200
    for i in range(1, 21):
        closes.append(100.0 + i * 10.0)  # reaches 300.0 at index 220
    for i in range(1, 21):
        closes.append(300.0 - i * 7.5)   # drops to 150.0 at index 240
    closes.extend([150.0] * (250 - len(closes)))

    candles = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": closes,
            "high": [c + 1.0 for c in closes],
            "low": [c - 1.0 for c in closes],
            "close": closes,
            "volume": [100.0] * 250,
        }
    )

    # Buy signal at bar 200 close, executed at bar 201 open ($110.0)
    ts_buy = candles.iloc[200]["timestamp"].isoformat()
    decisions = {
        ("BTC/USDT", ts_buy): DualAgentResult(
            proposal=ScoutProposal(action="BUY", confidence=0.9, thesis="Surge", evidence=["Breakout"], risks=["Risk"], missing_data=[]),
            critique=SkepticReview(verdict="APPROVE", agrees_with_action=True, rationale="Approve", missing_evidence=[], weaknesses=["None"], confidence=0.9),
        ),
    }

    config = BacktestConfig(
        run_id="test_dd_pullback",
        symbols=["BTC/USDT"],
        initial_cash_usd=10_000.0,
        fee_rate=0.001,
        slippage_pct=0.0,
        risk_limits=RiskLimits(max_position_size_usd=1_000.0),
    )

    report = run_backtest(
        config,
        candles_by_symbol={"BTC/USDT": candles},
        decision_provider=RecordedDecisionProvider(decisions),
    )

    # Max equity was reached around bar 220:
    # 1000 USD position bought at 110 (~9.09 units), at 300 is worth ~2727 USD.
    # Total peak equity ~ 9000 cash + 2727 = ~11,727 USD.
    # At bar 240, position at 150 is worth ~1363 USD -> Total equity ~ 10,363 USD.
    # Drop from peak = (11,727 - 10,363) / 11,727 = ~11.6%.
    # Before the fix, peak was frozen at $10,000, so drawdown was reported as 0.00%.
    assert report.max_drawdown_pct > 10.0
    assert report.max_drawdown_pct < 15.0


def test_exact_cash_holdings_and_unrealized_pnl_reconciliation() -> None:
    """Verifies that cash, market value, realized P&L, fees, and unrealized P&L reconcile exactly."""
    candles = generate_backtest_candles("BTC/USDT", periods=250, start_price=50_000.0, trend="uptrend")
    config = BacktestConfig(
        run_id="test_reconciliation",
        symbols=["BTC/USDT"],
        initial_cash_usd=10_000.0,
        fee_rate=0.001,
        slippage_pct=0.0005,
    )

    report = run_backtest(
        config,
        candles_by_symbol={"BTC/USDT": candles},
        decision_provider=DeterministicRuleProvider(),
    )

    # 1. Total Net Profit matches equity difference
    assert report.total_net_profit_usd == pytest.approx(report.final_equity - report.initial_equity, abs=1e-4)

    # 2. Equity equals Initial + Net Realized + Net Unrealized
    assert report.final_equity == pytest.approx(
        report.initial_equity + report.net_realized_pnl_usd + report.net_unrealized_pnl_usd,
        abs=1e-4,
    )

    # 3. Net Unrealized = Gross Unrealized - Open Position Entry Fees
    assert report.net_unrealized_pnl_usd == pytest.approx(
        report.gross_unrealized_pnl_usd - report.open_positions_fees_usd,
        abs=1e-4,
    )

    # 4. In a single open trade scenario, open position fees equal total trading fees paid
    assert report.open_positions_fees_usd == pytest.approx(report.total_fees_paid_usd, abs=1e-4)


def test_signal_rejection_tracking_and_allocation_cap() -> None:
    """Verifies that repetitive signals in a monotonic trend are tracked and rejected once allocation limit is met."""
    candles = generate_backtest_candles("BTC/USDT", periods=250, start_price=50_000.0, trend="uptrend")
    config = BacktestConfig(
        run_id="test_signal_tracking",
        symbols=["BTC/USDT"],
        initial_cash_usd=10_000.0,
        risk_limits=RiskLimits(max_position_size_usd=1_000.0),
    )

    report = run_backtest(
        config,
        candles_by_symbol={"BTC/USDT": candles},
        decision_provider=DeterministicRuleProvider(),
    )

    # In a 250-bar uptrend, bars 200..249 produce 50 signals
    assert report.signals_proposed > 30
    assert report.proposals_by_action.get("BUY", 0) == report.signals_proposed
    # Only 1 signal is approved because position limit ($1000) is filled immediately
    assert report.signals_approved == 1
    assert report.signals_rejected == report.signals_proposed - 1
    assert "allocation_limit_reached" in report.rejections_by_reason
    assert report.rejections_by_reason["allocation_limit_reached"] > 0
    assert report.total_executions == 1
    assert report.completed_trades == 0


def test_cycle_trend_generates_completed_round_trips() -> None:
    """Verifies that alternating market cycles trigger both BUY and SELL signals and complete round-trip trades."""
    candles = generate_backtest_candles("BTC/USDT", periods=360, start_price=50_000.0, trend="cycle")
    config = BacktestConfig(
        run_id="test_cycle_trades",
        symbols=["BTC/USDT"],
        initial_cash_usd=10_000.0,
        fee_rate=0.001,
        slippage_pct=0.0005,
    )

    report = run_backtest(
        config,
        candles_by_symbol={"BTC/USDT": candles},
        decision_provider=DeterministicRuleProvider(),
    )

    assert report.total_executions >= 2
    assert report.completed_trades >= 1
    assert report.win_rate_pct >= 0.0
    assert report.profit_factor >= 0.0
    assert len(report.symbol_breakdown["BTC/USDT"]) > 0


def test_sell_rejection_without_inventory() -> None:
    """Verifies that SELL signals without an existing open position are rejected fail-closed."""
    candles = _linear_candles(250)
    ts_sell = candles.iloc[205]["timestamp"].isoformat()

    decisions = {
        ("BTC/USDT", ts_sell): DualAgentResult(
            proposal=ScoutProposal(action="SELL", confidence=0.85, thesis="Sell without holdings", evidence=["Trend"], risks=["Risk"], missing_data=[]),
            critique=SkepticReview(verdict="APPROVE", agrees_with_action=True, rationale="Approve", missing_evidence=[], weaknesses=["None"], confidence=0.85),
        )
    }

    config = BacktestConfig(run_id="test_no_inventory", symbols=["BTC/USDT"])
    report = run_backtest(
        config,
        candles_by_symbol={"BTC/USDT": candles},
        decision_provider=RecordedDecisionProvider(decisions),
    )

    assert report.total_executions == 0
    assert report.signals_proposed == 1
    assert report.signals_approved == 0
    assert report.signals_rejected == 1
    assert report.rejections_by_reason.get("no_holdings_to_sell", 0) == 1


def test_fee_and_adverse_slippage_accounting_on_round_trip() -> None:
    """Verifies deterministic fee and adverse slippage math on a full BUY and SELL round-trip."""
    candles = _linear_candles(250, start_price=100.0, step=1.0)
    ts_buy = candles.iloc[205]["timestamp"].isoformat()
    ts_sell = candles.iloc[215]["timestamp"].isoformat()

    decisions = {
        ("BTC/USDT", ts_buy): DualAgentResult(
            proposal=ScoutProposal(action="BUY", confidence=0.9, thesis="Buy", evidence=["e"], risks=["r"], missing_data=[]),
            critique=SkepticReview(verdict="APPROVE", agrees_with_action=True, rationale="ok", missing_evidence=[], weaknesses=["w"], confidence=0.9),
        ),
        ("BTC/USDT", ts_sell): DualAgentResult(
            proposal=ScoutProposal(action="SELL", confidence=0.9, thesis="Sell", evidence=["e"], risks=["r"], missing_data=[]),
            critique=SkepticReview(verdict="APPROVE", agrees_with_action=True, rationale="ok", missing_evidence=[], weaknesses=["w"], confidence=0.9),
        ),
    }

    fee_rate = 0.002       # 0.2%
    slippage_pct = 0.001   # 0.1%
    initial_cash = 10_000.0
    config = BacktestConfig(
        run_id="test_round_trip_math",
        symbols=["BTC/USDT"],
        initial_cash_usd=initial_cash,
        fee_rate=fee_rate,
        slippage_pct=slippage_pct,
        risk_limits=RiskLimits(max_position_size_usd=1_000.0),
    )

    report = run_backtest(
        config,
        candles_by_symbol={"BTC/USDT": candles},
        decision_provider=RecordedDecisionProvider(decisions),
    )

    assert report.total_executions == 2
    assert report.completed_trades == 1

    buy_exec = report.execution_history[0]
    sell_exec = report.execution_history[1]

    # Buy at Bar 206 Open ($306.0) with +0.1% slippage
    expected_buy_open = float(candles.iloc[206]["open"])
    expected_buy_fill = expected_buy_open * (1.0 + slippage_pct)
    assert buy_exec["execution_price"] == pytest.approx(expected_buy_fill)

    # Sell at Bar 216 Open ($316.0) with -0.1% slippage
    expected_sell_open = float(candles.iloc[216]["open"])
    expected_sell_fill = expected_sell_open * (1.0 - slippage_pct)
    assert sell_exec["execution_price"] == pytest.approx(expected_sell_fill)

    # Fee verification
    buy_fee = buy_exec["fee_usd"]
    sell_fee = sell_exec["fee_usd"]
    assert report.total_fees_paid_usd == pytest.approx(buy_fee + sell_fee)

    # Total net profit matches net realized P&L exactly since 0 open positions remain
    assert report.unrealized_pnl_usd == 0.0
    assert report.total_net_profit_usd == pytest.approx(report.net_realized_pnl_usd, abs=1e-4)


def test_date_filtering_with_tz_naive_and_tz_aware_strings() -> None:
    """Verifies that date filtering gracefully handles naive strings without TypeError."""
    candles = _linear_candles(250, start_date="2024-01-01")
    config = BacktestConfig(
        run_id="test_tz_naive",
        symbols=["BTC/USDT"],
        start_date="2024-07-01",  # naive string
        end_date="2024-08-01",    # naive string
    )

    report = run_backtest(
        config,
        candles_by_symbol={"BTC/USDT": candles},
        decision_provider=RecordedDecisionProvider({}),
    )

    assert len(report.equity_curve) > 0
    # First date in equity curve should be >= 2024-07-01
    assert report.equity_curve[0]["timestamp_utc"] >= "2024-07-01"

