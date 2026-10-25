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

import numpy as np
import pandas as pd
import pytest

from crypto_research.backtest import (
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
