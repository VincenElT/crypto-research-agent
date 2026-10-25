"""Unit and regression tests for the historical evaluation harness.

Covers:
- Date boundaries and indicator warm-up protection
- Incomplete candle dropping
- Benchmark calculations (Cash, 100% BTC Buy-and-Hold, Exposure-Matched)
- Quantitative metric calculations (Sharpe with sqrt(365), win rate, expectancy, profit factor, exposure)
- Transaction cost sensitivity analysis
- Explicit fail-closed behavior on data errors (no silent synthetic fallback)
- Artifact serialization (JSON and Markdown)
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from crypto_research.backtest import generate_backtest_candles
from crypto_research.evaluation import (
    DEFAULT_PERIODS,
    BenchmarkMetrics,
    EvaluationHarnessResult,
    EvaluationPeriod,
    PeriodEvaluationResult,
    SensitivityScenario,
    StrategyMetrics,
    _closed_candles_through_period_end,
    calculate_benchmarks,
    calculate_exposure_metrics,
    calculate_friction_metrics,
    calculate_sharpe_ratio,
    calculate_trade_metrics,
    format_evaluation_markdown,
    run_historical_evaluation,
    run_period_evaluation,
    run_sensitivity_analysis,
    save_evaluation_artifacts,
)
from crypto_research.exceptions import (
    BacktestError,
    ConfigurationError,
    InsufficientHistoryError,
    MarketDataError,
)
from crypto_research.market_data import drop_incomplete_candle


def test_date_boundaries_no_overlap_and_warmup() -> None:
    """Verify that DEFAULT_PERIODS partitions are strictly chronological and non-overlapping."""
    dev = next(p for p in DEFAULT_PERIODS if p.name == "development")
    val = next(p for p in DEFAULT_PERIODS if p.name == "validation")
    oos = next(p for p in DEFAULT_PERIODS if p.name == "out_of_sample")

    # Chronological strict ordering
    assert dev.start_date < dev.end_date
    assert dev.end_date < val.start_date
    assert val.start_date < val.end_date
    assert val.end_date < oos.start_date
    assert oos.start_date < oos.end_date

    # Exact expected boundaries
    assert dev.start_date == "2022-06-01"
    assert dev.end_date == "2023-12-31"
    assert val.start_date == "2024-01-01"
    assert val.end_date == "2025-06-30"
    assert oos.start_date == "2025-07-01"
    assert oos.end_date == "2026-10-24"


def test_period_cutoff_includes_only_bars_closed_by_end_boundary() -> None:
    candles = generate_backtest_candles(
        periods=5, start_date="2024-06-29", trend="cycle"
    )
    period = EvaluationPeriod("cutoff", "2024-06-30", "2024-06-30", "cutoff")
    eligible = _closed_candles_through_period_end(candles, period)
    # June 30's daily candle is available at July 1 00:00 UTC; a July 1 candle
    # cannot be used in this period.
    assert eligible["timestamp"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-06-29", "2024-06-30"
    ]


def test_evaluation_rejects_unsupported_timeframe() -> None:
    candles = generate_backtest_candles(periods=250, start_date="2022-01-01")
    period = EvaluationPeriod("hourly", "2022-08-01", "2022-08-10", "hourly")
    with pytest.raises(ConfigurationError, match="only timeframe='1d'"):
        run_period_evaluation(period, candles, timeframe="1h")


def test_warmup_candles_requirement_enforced() -> None:
    """Verify that evaluation fails closed if fewer than 200 warmup candles precede the first period."""
    # Create dataset with only 100 days before 2022-06-01
    short_candles = generate_backtest_candles(
        periods=150,
        start_date="2022-04-01",
        freq="1D",
        trend="cycle",
    )
    with patch("crypto_research.evaluation.load_settings"), \
         patch("crypto_research.evaluation.create_public_exchange"), \
         patch("crypto_research.evaluation.fetch_ohlcv", return_value=short_candles):
        with pytest.raises(InsufficientHistoryError, match="at least 200 warm-up candles"):
            run_historical_evaluation(candle_limit=150)


def test_incomplete_candle_dropped_before_evaluation() -> None:
    """Verify that drop_incomplete_candle removes an in-progress open candle."""
    now_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    # Candle that started 1 hour ago for a 1d timeframe (86,400,000 ms) is incomplete
    incomplete_open = now_ts - 3600 * 1000
    df = pd.DataFrame(
        {
            "timestamp": [
                pd.to_datetime(incomplete_open - 86_400_000, unit="ms", utc=True),
                pd.to_datetime(incomplete_open, unit="ms", utc=True),
            ],
            "open": [50000.0, 51000.0],
            "high": [52000.0, 53000.0],
            "low": [49000.0, 50000.0],
            "close": [51000.0, 52000.0],
            "volume": [100.0, 10.0],
        }
    )
    closed = drop_incomplete_candle(df, timeframe_ms=86_400_000)
    assert len(closed) == 1
    assert closed.iloc[0]["open"] == 50000.0


def test_cash_benchmark_calculations() -> None:
    """Verify Cash-only benchmark returns exactly 0% return, 0% max drawdown, 0.0 Sharpe."""
    eq_curve = [
        {"equity_usd": 10000.0, "cash_usd": 10000.0, "benchmark_price": 50000.0},
        {"equity_usd": 10000.0, "cash_usd": 10000.0, "benchmark_price": 55000.0},
        {"equity_usd": 10000.0, "cash_usd": 10000.0, "benchmark_price": 45000.0},
    ]
    cash_b, _, _ = calculate_benchmarks(eq_curve, initial_equity=10000.0, strategy_avg_exposure_pct=0.0)
    assert cash_b.name == "cash_only"
    assert cash_b.initial_equity == 10000.0
    assert cash_b.final_equity == 10000.0
    assert cash_b.total_net_profit_usd == 0.0
    assert cash_b.total_return_pct == 0.0
    assert cash_b.max_drawdown_pct == 0.0
    assert cash_b.sharpe_ratio == 0.0
    assert cash_b.avg_exposure_pct == 0.0
    assert cash_b.pct_time_invested == 0.0


def test_btc_buy_and_hold_benchmark() -> None:
    """Verify 100% BTC Buy-and-Hold benchmark return, drawdown, and Sharpe."""
    eq_curve = [
        {"equity_usd": 10000.0, "cash_usd": 10000.0, "benchmark_price": 100.0},
        {"equity_usd": 10000.0, "cash_usd": 10000.0, "benchmark_price": 150.0},
        {"equity_usd": 10000.0, "cash_usd": 10000.0, "benchmark_price": 120.0},
        {"equity_usd": 10000.0, "cash_usd": 10000.0, "benchmark_price": 200.0},
    ]
    _, bnh_b, _ = calculate_benchmarks(eq_curve, initial_equity=10000.0, strategy_avg_exposure_pct=10.0)
    assert bnh_b.name == "btc_buy_and_hold"
    # Price went from 100 to 200 -> +100% return
    assert bnh_b.total_return_pct == pytest.approx(100.0)
    assert bnh_b.final_equity == pytest.approx(20000.0)
    # Peak at 150 dropped to 120 -> drawdown = (150 - 120) / 150 = 20.0%
    assert bnh_b.max_drawdown_pct == pytest.approx(20.0)
    assert bnh_b.avg_exposure_pct == 100.0
    assert bnh_b.pct_time_invested == 100.0
    assert bnh_b.sharpe_ratio > 0.0


@pytest.mark.parametrize("invalid_price", [None, 0.0, -1.0, float("nan"), float("inf")])
def test_benchmarks_fail_closed_on_missing_or_invalid_asset_prices(invalid_price: float | None) -> None:
    curve = [
        {"equity_usd": 10_000.0, "cash_usd": 10_000.0, "benchmark_price": 100.0},
        {"equity_usd": 10_000.0, "cash_usd": 10_000.0, "benchmark_price": invalid_price},
    ]
    with pytest.raises(BacktestError, match="finite, positive benchmark prices"):
        calculate_benchmarks(curve, initial_equity=10_000.0, strategy_avg_exposure_pct=0.0)


def test_exposure_matched_benchmark() -> None:
    """Verify Exposure-Matched benchmark scales daily asset return by average exposure."""
    eq_curve = [
        {"equity_usd": 10000.0, "cash_usd": 9000.0, "position_value_usd": 1000.0, "benchmark_price": 100.0},
        {"equity_usd": 10000.0, "cash_usd": 9000.0, "position_value_usd": 1000.0, "benchmark_price": 110.0},
        {"equity_usd": 10000.0, "cash_usd": 9000.0, "position_value_usd": 1000.0, "benchmark_price": 120.0},
    ]
    # Strategy exposure = 10% (1000 / 10000)
    _, _, exp_b = calculate_benchmarks(eq_curve, initial_equity=10000.0, strategy_avg_exposure_pct=10.0)
    assert exp_b.name == "exposure_matched"
    assert exp_b.avg_exposure_pct == pytest.approx(10.0)
    # Day 1: BTC +10% -> port +1%
    # Day 2: BTC +9.09% (120/110 - 1) -> port +0.909%
    # Compound return approx: 1.01 * 1.00909 - 1 = +1.918%
    assert 1.8 < exp_b.total_return_pct < 2.0
    assert exp_b.pct_time_invested == 100.0


def test_sharpe_ratio_annualization_math() -> None:
    """Verify Sharpe ratio calculates daily returns and annualizes with sqrt(365)."""
    # Constant equity -> zero variance -> Sharpe = 0.0
    flat_curve = [{"equity_usd": 10000.0} for _ in range(10)]
    assert calculate_sharpe_ratio(flat_curve) == 0.0

    # Insufficient length -> Sharpe = 0.0
    assert calculate_sharpe_ratio([{"equity_usd": 10000.0}]) == 0.0
    assert calculate_sharpe_ratio([]) == 0.0

    # Deterministic daily returns: alternating +1% and -0.5%
    test_curve = [{"equity_usd": 10000.0}]
    curr = 10000.0
    for i in range(100):
        ret = 0.01 if i % 2 == 0 else -0.005
        curr *= (1.0 + ret)
        test_curve.append({"equity_usd": curr})

    sharpe = calculate_sharpe_ratio(test_curve, annualization_factor=math.sqrt(365))
    # Daily returns: mean ~ 0.0025, std ~ 0.0075 -> daily SR ~ 0.33 -> Annualized ~ 0.33 * 19.1 = ~6.3
    assert 5.0 < sharpe < 8.0


def test_sharpe_hand_calculated_daily_returns() -> None:
    # Daily simple returns: +10%, -10%, +10%; sample standard deviation.
    curve = [
        {"equity_usd": 100.0},
        {"equity_usd": 110.0},
        {"equity_usd": 99.0},
        {"equity_usd": 108.9},
    ]
    returns = [0.10, -0.10, 0.10]
    mean = sum(returns) / 3
    sample_std = math.sqrt(sum((value - mean) ** 2 for value in returns) / 2)
    expected = math.sqrt(365) * mean / sample_std
    assert calculate_sharpe_ratio(curve) == pytest.approx(expected)


def test_trade_metrics_calculation() -> None:
    """Verify win rate, average win/loss, expectancy, and profit factor."""
    trades = [
        {"realized_pnl_usd": 150.0},
        {"realized_pnl_usd": -50.0},
        {"realized_pnl_usd": 300.0},
        {"realized_pnl_usd": -100.0},
    ]
    (
        completed,
        wins,
        losses,
        win_rate,
        avg_win,
        avg_loss,
        wl_ratio,
        expectancy,
        pf,
    ) = calculate_trade_metrics(trades)

    assert completed == 4
    assert wins == 2
    assert losses == 2
    assert win_rate == 50.0
    assert avg_win == 225.0  # (150 + 300) / 2
    assert avg_loss == 75.0  # (50 + 100) / 2
    assert wl_ratio == 3.0   # 225 / 75
    assert expectancy == 75.0  # (150 - 50 + 300 - 100) / 4 = 300 / 4
    assert pf == 3.0         # 450 / 150


def test_trade_metrics_zero_trades() -> None:
    """Verify trade metrics behave safely when no completed trades exist."""
    res = calculate_trade_metrics([])
    assert res == (0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_exposure_and_time_invested() -> None:
    """Verify percent time invested and average exposure calculation."""
    eq_curve = [
        {"equity_usd": 10000.0, "cash_usd": 10000.0, "position_value_usd": 0.0},    # Flat (0%)
        {"equity_usd": 10000.0, "cash_usd": 9000.0, "position_value_usd": 1000.0},  # 10%
        {"equity_usd": 10000.0, "cash_usd": 9000.0, "position_value_usd": 1000.0},  # 10%
        {"equity_usd": 10000.0, "cash_usd": 10000.0, "position_value_usd": 0.0},    # Flat (0%)
    ]
    pct_time, avg_exp = calculate_exposure_metrics(eq_curve)
    # Invested 2 out of 4 days = 50.0%
    assert pct_time == 50.0
    # Exposure: (0 + 0.10 + 0.10 + 0) / 4 = 0.05 = 5.0%
    assert avg_exp == pytest.approx(5.0)


def test_friction_metrics_calculation() -> None:
    """Verify total fees and slippage calculations."""
    executions = [
        {"fee_usd": 1.0, "quantity": 0.02, "raw_price": 50000.0, "slippage_pct": 0.0005},
        {"fee_usd": 1.2, "quantity": 0.02, "raw_price": 60000.0, "slippage_pct": 0.0005},
    ]
    fees, slip, total = calculate_friction_metrics(executions)
    assert fees == pytest.approx(2.2)
    # Exec 1 slippage: 0.02 * 50000 * 0.0005 = 0.50
    # Exec 2 slippage: 0.02 * 60000 * 0.0005 = 0.60
    assert slip == pytest.approx(1.10)
    assert total == pytest.approx(3.30)


def test_sensitivity_analysis_monotonic_friction() -> None:
    """Verify that sensitivity stress tests produce increasing friction and decreasing profit."""
    candles = generate_backtest_candles(
        periods=350,
        start_date="2022-01-01",
        trend="cycle",
    )
    period = EvaluationPeriod(
        name="test_period",
        start_date="2022-08-01",
        end_date="2022-12-15",
        description="Test cycle period",
    )
    scenarios = run_sensitivity_analysis(
        candles=candles,
        period=period,
        symbol="BTC/USDT",
    )
    assert len(scenarios) == 4
    # Check monotonically increasing friction
    for i in range(1, len(scenarios)):
        prev = scenarios[i - 1]
        curr = scenarios[i]
        assert curr.total_friction_usd >= prev.total_friction_usd
        assert curr.total_return_pct <= prev.total_return_pct


def test_fail_closed_on_network_or_api_error() -> None:
    """Verify that evaluation fails closed with MarketDataError on network errors."""
    with patch("crypto_research.evaluation.load_settings"), \
         patch("crypto_research.evaluation.create_public_exchange"), \
         patch("crypto_research.evaluation.fetch_ohlcv", side_effect=MarketDataError("Connection failed")):
        with pytest.raises(MarketDataError, match="Connection failed"):
            run_historical_evaluation()


def test_save_evaluation_artifacts(tmp_path: Path) -> None:
    """Verify that JSON and Markdown artifacts are properly serialized to disk."""
    candles = generate_backtest_candles(
        periods=300,
        start_date="2022-01-01",
        trend="cycle",
    )
    period = EvaluationPeriod(
        name="dev_test",
        start_date="2022-08-01",
        end_date="2022-10-31",
        description="Short test period",
    )
    period_res = run_period_evaluation(
        period=period,
        full_candles=candles,
        symbol="BTC/USDT",
    )

    result = EvaluationHarnessResult(
        run_id="test_run_123",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        symbol="BTC/USDT",
        timeframe="1d",
        data_source="Test Synthetic Source",
        total_candles_fetched=300,
        dataset_start_utc="2022-01-01T00:00:00+00:00",
        dataset_end_utc="2022-10-31T00:00:00+00:00",
        warmup_candles_count=212,
        warmup_start_utc="2022-01-01T00:00:00+00:00",
        warmup_end_utc="2022-07-31T00:00:00+00:00",
        period_results={
            "development": period_res,
            "validation": period_res,
            "out_of_sample": period_res,
            "full": period_res,
        },
        sensitivity_results=[
            SensitivityScenario(
                scenario_name="Base",
                fee_rate=0.001,
                slippage_pct=0.0005,
                total_return_pct=1.5,
                net_profit_usd=150.0,
                max_drawdown_pct=0.8,
                sharpe_ratio=1.2,
                profit_factor=2.0,
                win_rate_pct=50.0,
                total_fees_usd=2.0,
                total_slippage_usd=1.0,
                total_friction_usd=3.0,
            )
        ],
        trade_records=period_res.raw_report.execution_history,
        dataset_sha256="abc123",
        code_revision="deadbeef",
        code_fingerprint_sha256="feedface",
        dependency_versions={"python": "3.13.0"},
        experiment_config={"fee_rate": 0.001, "benchmark_costs": "excluded"},
    )

    md_file, json_file = save_evaluation_artifacts(result, output_dir=str(tmp_path))
    assert md_file.exists()
    assert json_file.exists()

    md_text = md_file.read_text(encoding="utf-8")
    assert "# Historical Strategy Evaluation Report: BTC/USDT (test_run_123)" in md_text
    assert "Multi-Period Walk-Forward Performance Comparison" in md_text
    assert "Sensitivity Analysis" in md_text

    json_data = json.loads(json_file.read_text(encoding="utf-8"))
    assert json_data["run_id"] == "test_run_123"
    assert "period_results" in json_data
    assert "sensitivity_results" in json_data
    assert "trade_records" in json_data
    assert json_data["dataset_sha256"] == "abc123"
    assert json_data["code_revision"] == "deadbeef"
    assert json_data["experiment_config"]["benchmark_costs"] == "excluded"
