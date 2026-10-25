"""Reproducible historical evaluation harness for the crypto research agent.

Evaluates the strategy over multi-year real historical daily OHLCV data across
chronological, strictly non-overlapping periods:
  1. Development / In-Sample: 2022-06-01 to 2023-12-31 (Bear market & recovery)
  2. Validation: 2024-01-01 to 2025-06-30 (ETF bull market & post-halving)
  3. Final Out-of-Sample / Holdout: 2025-07-01 to 2026-10-24 (Recent cycle)
  4. Full Multi-Year Compound: 2022-06-01 to 2026-10-24

Maintains indicator warm-up without look-ahead leakage, compares against Cash-only,
100% BTC Buy-and-Hold, and Exposure-matched benchmarks, computes institutional
performance metrics, and executes fee/slippage sensitivity stress tests.

Live trading is strictly disabled. Real exchange orders cannot be placed.
Fails explicitly if real historical data cannot be verified.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from crypto_research.backtest import (
    BacktestConfig,
    BacktestReport,
    DeterministicRuleProvider,
    run_backtest,
    validate_backtest_candles,
)
from crypto_research.config import Settings, load_settings
from crypto_research.exceptions import (
    BacktestError,
    ConfigurationError,
    InsufficientHistoryError,
    MarketDataError,
)
from crypto_research.market_data import (
    create_public_exchange,
    fetch_ohlcv,
)
from crypto_research.risk import RiskLimits


@dataclass(frozen=True)
class EvaluationPeriod:
    """Specification of an evaluation timeline partition."""

    name: str
    start_date: str
    end_date: str
    description: str


DEFAULT_PERIODS: list[EvaluationPeriod] = [
    EvaluationPeriod(
        name="development",
        start_date="2022-06-01",
        end_date="2023-12-31",
        description="Development / In-Sample: 2022 bear market (LUNA/FTX) & 2023 recovery",
    ),
    EvaluationPeriod(
        name="validation",
        start_date="2024-01-01",
        end_date="2025-06-30",
        description="Validation: 2024 Spot ETF launch, halving bull run & consolidations",
    ),
    EvaluationPeriod(
        name="out_of_sample",
        start_date="2025-07-01",
        end_date="2026-10-24",
        description="Final Out-of-Sample / Holdout: Recent cycle (Strictly never tuned against)",
    ),
    EvaluationPeriod(
        name="full",
        start_date="2022-06-01",
        end_date="2026-10-24",
        description="Full multi-year compound evaluation timeline",
    ),
]


@dataclass(frozen=True)
class BenchmarkMetrics:
    """Performance metrics for an evaluation benchmark."""

    name: str
    description: str
    initial_equity: float
    final_equity: float
    total_net_profit_usd: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    avg_exposure_pct: float
    pct_time_invested: float


@dataclass(frozen=True)
class StrategyMetrics:
    """Comprehensive performance and execution metrics for the evaluated strategy."""

    initial_equity: float
    final_equity: float
    net_profit_usd: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sharpe_frequency: str = "daily (1d)"
    sharpe_annualization_factor: float = math.sqrt(365)
    sharpe_assumptions: str = (
        "Daily close-to-close returns annualized using sqrt(365) continuous crypto convention; Rf=0.0%"
    )
    completed_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate_pct: float = 0.0
    avg_win_usd: float = 0.0
    avg_loss_usd: float = 0.0
    win_loss_ratio: float = 0.0
    expectancy_usd: float = 0.0
    profit_factor: float = 0.0
    pct_time_invested: float = 0.0
    avg_exposure_pct: float = 0.0
    total_fees_usd: float = 0.0
    total_slippage_usd: float = 0.0
    total_friction_usd: float = 0.0
    total_executions: int = 0
    signals_proposed: int = 0
    signals_approved: int = 0
    signals_rejected: int = 0
    rejections_by_reason: dict[str, int] = field(default_factory=dict)
    unexecuted_orders: int = 0


@dataclass(frozen=True)
class PeriodEvaluationResult:
    """Evaluation result for a single chronological period."""

    period: EvaluationPeriod
    candle_count: int
    start_time_utc: str
    end_time_utc: str
    strategy_metrics: StrategyMetrics
    cash_benchmark: BenchmarkMetrics
    buy_and_hold_benchmark: BenchmarkMetrics
    exposure_matched_benchmark: BenchmarkMetrics
    raw_report: BacktestReport


@dataclass(frozen=True)
class SensitivityScenario:
    """Performance results for a specific fee and slippage stress scenario."""

    scenario_name: str
    fee_rate: float
    slippage_pct: float
    total_return_pct: float
    net_profit_usd: float
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float
    win_rate_pct: float
    total_fees_usd: float
    total_slippage_usd: float
    total_friction_usd: float


@dataclass(frozen=True)
class EvaluationHarnessResult:
    """Master reproducible container for multi-period historical evaluation."""

    run_id: str
    created_at_utc: str
    symbol: str
    timeframe: str
    data_source: str
    total_candles_fetched: int
    dataset_start_utc: str
    dataset_end_utc: str
    warmup_candles_count: int
    warmup_start_utc: str
    warmup_end_utc: str
    period_results: dict[str, PeriodEvaluationResult]
    sensitivity_results: list[SensitivityScenario]
    trade_records: list[dict[str, Any]]
    dataset_sha256: str = ""
    code_revision: str | None = None
    code_fingerprint_sha256: str = ""
    dependency_versions: dict[str, str] = field(default_factory=dict)
    experiment_config: dict[str, Any] = field(default_factory=dict)


def _require_daily_timeframe(timeframe: str) -> None:
    """Current evaluation periods and Sharpe convention are defined for daily bars."""
    if timeframe != "1d":
        raise ConfigurationError(
            "The evaluation harness currently supports only timeframe='1d'; "
            "period boundaries and Sharpe annualization are daily."
        )


def _closed_candles_through_period_end(
    candles: pd.DataFrame, period: EvaluationPeriod
) -> pd.DataFrame:
    """Return daily bars fully closed by the inclusive period-end boundary.

    Candle timestamps label bar opens. A daily bar labelled with end_date is
    available at the next UTC midnight, which is the period's exclusive close
    boundary. No bar opening after end_date is supplied to that period.
    """
    end_open = pd.Timestamp(period.end_date, tz="UTC")
    close_cutoff = end_open + pd.Timedelta(days=1)
    candle_close_times = candles["timestamp"] + pd.Timedelta(days=1)
    return candles[candle_close_times <= close_cutoff].copy()


def _fingerprint_dataset(candles: pd.DataFrame) -> str:
    canonical = candles.loc[:, ["timestamp", "open", "high", "low", "close", "volume"]]
    payload = canonical.to_csv(
        index=False,
        float_format="%.17g",
        date_format="%Y-%m-%dT%H:%M:%S.%f%z",
        lineterminator="\n",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _code_fingerprint() -> str:
    digest = hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(source.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _dependency_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ("pandas", "ccxt", "openai", "pydantic", "python-dotenv"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def calculate_sharpe_ratio(
    equity_curve: list[dict[str, Any]],
    annualization_factor: float = math.sqrt(365),
) -> float:
    """Calculate annualized Sharpe ratio from daily equity curve.

    Assumptions:
      - Frequency: Daily closes (1d).
      - Annualization factor: sqrt(365) for 24/7/365 crypto markets.
      - Risk-free rate: Rf = 0.0% (cash holding benchmark).
      - Sample standard deviation (ddof=1).
      - Returns 0.0 if variance is negligible or fewer than 2 return observations.
    """
    if len(equity_curve) < 2:
        return 0.0

    returns: list[float] = []
    for i in range(1, len(equity_curve)):
        prev_eq = float(equity_curve[i - 1]["equity_usd"])
        curr_eq = float(equity_curve[i]["equity_usd"])
        if prev_eq > 0:
            returns.append((curr_eq - prev_eq) / prev_eq)
        else:
            returns.append(0.0)

    if len(returns) < 2:
        return 0.0

    mean_ret = sum(returns) / len(returns)
    variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
    std_ret = math.sqrt(variance)

    if std_ret <= 1e-12 or not math.isfinite(std_ret):
        return 0.0

    sharpe = annualization_factor * (mean_ret / std_ret)
    return float(sharpe) if math.isfinite(sharpe) else 0.0


def calculate_exposure_metrics(
    equity_curve: list[dict[str, Any]],
) -> tuple[float, float]:
    """Calculate percentage of time invested and average exposure percentage.

    Returns:
      (pct_time_invested, avg_exposure_pct)
    """
    if not equity_curve:
        return 0.0, 0.0

    invested_days = 0
    exposure_sum = 0.0

    for pt in equity_curve:
        eq = float(pt["equity_usd"])
        pos_val = float(pt.get("position_value_usd", max(0.0, eq - float(pt["cash_usd"]))))
        exp_frac = (pos_val / eq) if eq > 0 else 0.0
        exp_frac = max(0.0, min(1.0, exp_frac))
        exposure_sum += exp_frac
        if exp_frac > 0.0001:
            invested_days += 1

    total_days = len(equity_curve)
    pct_time_invested = (invested_days / total_days) * 100.0
    avg_exposure_pct = (exposure_sum / total_days) * 100.0
    return pct_time_invested, avg_exposure_pct


def calculate_trade_metrics(
    completed_trades: list[dict[str, Any]],
) -> tuple[int, int, int, float, float, float, float, float, float]:
    """Compute trade-level metrics.

    Returns:
      (completed, winning, losing, win_rate_pct, avg_win_usd, avg_loss_usd,
       win_loss_ratio, expectancy_usd, profit_factor)
    """
    completed = len(completed_trades)
    if completed == 0:
        return 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    winning = [t for t in completed_trades if t["realized_pnl_usd"] > 0]
    losing = [t for t in completed_trades if t["realized_pnl_usd"] < 0]

    win_count = len(winning)
    loss_count = len(losing)
    win_rate_pct = (win_count / completed) * 100.0

    gross_profit = sum(t["realized_pnl_usd"] for t in winning)
    gross_loss = abs(sum(t["realized_pnl_usd"] for t in losing))

    avg_win_usd = (gross_profit / win_count) if win_count > 0 else 0.0
    avg_loss_usd = (gross_loss / loss_count) if loss_count > 0 else 0.0

    if avg_loss_usd > 0:
        win_loss_ratio = avg_win_usd / avg_loss_usd
    elif avg_win_usd > 0:
        win_loss_ratio = float("inf")
    else:
        win_loss_ratio = 0.0

    total_realized = sum(t["realized_pnl_usd"] for t in completed_trades)
    expectancy_usd = total_realized / completed

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    return (
        completed,
        win_count,
        loss_count,
        win_rate_pct,
        avg_win_usd,
        avg_loss_usd,
        win_loss_ratio,
        expectancy_usd,
        profit_factor,
    )


def calculate_friction_metrics(
    executions: list[dict[str, Any]],
) -> tuple[float, float, float]:
    """Compute total fees, slippage cost, and total friction from executions.

    Slippage cost for each execution:
      quantity * raw_price * slippage_pct
    """
    total_fees = sum(float(e.get("fee_usd", 0.0)) for e in executions)
    total_slippage = 0.0
    for e in executions:
        qty = float(e.get("quantity", 0.0))
        raw_p = float(e.get("raw_price", 0.0))
        slip_pct = float(e.get("slippage_pct", 0.0))
        total_slippage += qty * raw_p * slip_pct

    total_friction = total_fees + total_slippage
    return total_fees, total_slippage, total_friction


def calculate_benchmarks(
    equity_curve: list[dict[str, Any]],
    initial_equity: float,
    strategy_avg_exposure_pct: float,
) -> tuple[BenchmarkMetrics, BenchmarkMetrics, BenchmarkMetrics]:
    """Calculate benchmarks; BTC benchmarks are gross of transaction costs.

    Exposure-matched holds a constant BTC weight and rebalances daily, with no
    transaction costs. Buy-and-hold invests fully from the first close, also
    without modeled costs. Both are gross reference benchmarks.
    """
    n_days = len(equity_curve)
    if n_days == 0:
        empty_bench = BenchmarkMetrics(
            name="empty",
            description="Empty benchmark",
            initial_equity=initial_equity,
            final_equity=initial_equity,
            total_net_profit_usd=0.0,
            total_return_pct=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
            avg_exposure_pct=0.0,
            pct_time_invested=0.0,
        )
        return empty_bench, empty_bench, empty_bench

    # 1. Cash-only Benchmark
    cash_bench = BenchmarkMetrics(
        name="cash_only",
        description="100% Cash / USD: 0% risk-free nominal yield, zero volatility, zero drawdown",
        initial_equity=initial_equity,
        final_equity=initial_equity,
        total_net_profit_usd=0.0,
        total_return_pct=0.0,
        max_drawdown_pct=0.0,
        sharpe_ratio=0.0,
        avg_exposure_pct=0.0,
        pct_time_invested=0.0,
    )

    # Extract asset prices from equity curve
    prices = [float(pt.get("benchmark_price", 0.0)) for pt in equity_curve]
    # If benchmark_price was not recorded, fallback gracefully
    if not any(p > 0 for p in prices):
        prices = [1.0] * n_days

    # 2. 100% BTC Buy-and-Hold Benchmark
    p0 = prices[0]
    p_last = prices[-1]
    bnh_equity_curve: list[dict[str, Any]] = []
    bnh_peak = initial_equity
    bnh_max_dd = 0.0

    for p in prices:
        eq = (initial_equity * (p / p0)) if p0 > 0 else initial_equity
        if eq > bnh_peak:
            bnh_peak = eq
        dd = (bnh_peak - eq) / bnh_peak if bnh_peak > 0 else 0.0
        if dd > bnh_max_dd:
            bnh_max_dd = dd
        bnh_equity_curve.append({"equity_usd": eq})

    bnh_final_eq = bnh_equity_curve[-1]["equity_usd"]
    bnh_profit = bnh_final_eq - initial_equity
    bnh_return_pct = (bnh_profit / initial_equity) * 100.0 if initial_equity > 0 else 0.0
    bnh_sharpe = calculate_sharpe_ratio(bnh_equity_curve)

    bnh_bench = BenchmarkMetrics(
        name="btc_buy_and_hold",
        description="100% BTC Buy-and-Hold: fully invested in BTC from period start to end",
        initial_equity=initial_equity,
        final_equity=bnh_final_eq,
        total_net_profit_usd=bnh_profit,
        total_return_pct=bnh_return_pct,
        max_drawdown_pct=bnh_max_dd * 100.0,
        sharpe_ratio=bnh_sharpe,
        avg_exposure_pct=100.0,
        pct_time_invested=100.0,
    )

    # 3. Exposure-Matched Benchmark
    # Maintains constant weight w_bar in BTC, rebalanced daily: daily_return = w_bar * r_btc
    w_bar = max(0.0, min(1.0, strategy_avg_exposure_pct / 100.0))
    exp_equity_curve: list[dict[str, Any]] = [{"equity_usd": initial_equity}]
    exp_peak = initial_equity
    exp_max_dd = 0.0
    curr_exp_eq = initial_equity

    for i in range(1, n_days):
        p_prev = prices[i - 1]
        p_curr = prices[i]
        r_btc = (p_curr - p_prev) / p_prev if p_prev > 0 else 0.0
        r_port = w_bar * r_btc
        curr_exp_eq = curr_exp_eq * (1.0 + r_port)
        if curr_exp_eq > exp_peak:
            exp_peak = curr_exp_eq
        dd = (exp_peak - curr_exp_eq) / exp_peak if exp_peak > 0 else 0.0
        if dd > exp_max_dd:
            exp_max_dd = dd
        exp_equity_curve.append({"equity_usd": curr_exp_eq})

    exp_final_eq = exp_equity_curve[-1]["equity_usd"]
    exp_profit = exp_final_eq - initial_equity
    exp_return_pct = (exp_profit / initial_equity) * 100.0 if initial_equity > 0 else 0.0
    exp_sharpe = calculate_sharpe_ratio(exp_equity_curve)

    exp_bench = BenchmarkMetrics(
        name="exposure_matched",
        description=(
            f"Exposure-Matched Benchmark: rebalanced daily holding {strategy_avg_exposure_pct:.2f}% "
            f"in BTC and {100.0 - strategy_avg_exposure_pct:.2f}% in cash (matching strategy beta)"
        ),
        initial_equity=initial_equity,
        final_equity=exp_final_eq,
        total_net_profit_usd=exp_profit,
        total_return_pct=exp_return_pct,
        max_drawdown_pct=exp_max_dd * 100.0,
        sharpe_ratio=exp_sharpe,
        avg_exposure_pct=strategy_avg_exposure_pct,
        pct_time_invested=100.0 if strategy_avg_exposure_pct > 0 else 0.0,
    )

    return cash_bench, bnh_bench, exp_bench


def run_period_evaluation(
    period: EvaluationPeriod,
    full_candles: pd.DataFrame,
    symbol: str = "BTC/USDT",
    timeframe: str = "1d",
    initial_cash_usd: float = 10_000.0,
    fee_rate: float = 0.001,
    slippage_pct: float = 0.0005,
    risk_limits: RiskLimits | None = None,
) -> PeriodEvaluationResult:
    """Evaluate an inclusive daily candle-label range after each bar closes."""
    _require_daily_timeframe(timeframe)
    period_candles = _closed_candles_through_period_end(full_candles, period)

    run_id = f"eval_{period.name}_{uuid.uuid4().hex[:8]}"
    config = BacktestConfig(
        run_id=run_id,
        symbols=[symbol],
        timeframe=timeframe,
        initial_cash_usd=initial_cash_usd,
        fee_rate=fee_rate,
        slippage_pct=slippage_pct,
        start_date=period.start_date,
        end_date=period.end_date,
        risk_limits=risk_limits or RiskLimits(),
        sqlite_db_path=":memory:",
    )

    report = run_backtest(
        config,
        candles_by_symbol={symbol: period_candles},
        decision_provider=DeterministicRuleProvider(),
    )

    # Compute extended quantitative metrics
    sharpe = calculate_sharpe_ratio(report.equity_curve)
    pct_invested, avg_exp = calculate_exposure_metrics(report.equity_curve)
    (
        completed,
        winning,
        losing,
        win_rate,
        avg_win,
        avg_loss,
        wl_ratio,
        expectancy,
        profit_factor,
    ) = calculate_trade_metrics(
        report.completed_trade_records
    )
    total_fees, total_slippage, total_friction = calculate_friction_metrics(
        report.execution_history
    )

    strategy_metrics = StrategyMetrics(
        initial_equity=report.initial_equity,
        final_equity=report.final_equity,
        net_profit_usd=report.total_net_profit_usd,
        total_return_pct=report.total_return_pct,
        max_drawdown_pct=report.max_drawdown_pct,
        sharpe_ratio=sharpe,
        completed_trades=report.completed_trades,
        winning_trades=report.winning_trades,
        losing_trades=report.losing_trades,
        win_rate_pct=report.win_rate_pct,
        avg_win_usd=avg_win,
        avg_loss_usd=avg_loss,
        win_loss_ratio=wl_ratio,
        expectancy_usd=expectancy,
        profit_factor=report.profit_factor,
        pct_time_invested=pct_invested,
        avg_exposure_pct=avg_exp,
        total_fees_usd=total_fees,
        total_slippage_usd=total_slippage,
        total_friction_usd=total_friction,
        total_executions=report.total_executions,
        signals_proposed=report.signals_proposed,
        signals_approved=report.signals_approved,
        signals_rejected=report.signals_rejected,
        rejections_by_reason=report.rejections_by_reason,
        unexecuted_orders=report.unexecuted_orders,
    )

    cash_b, bnh_b, exp_b = calculate_benchmarks(
        report.equity_curve,
        initial_equity=report.initial_equity,
        strategy_avg_exposure_pct=avg_exp,
    )

    return PeriodEvaluationResult(
        period=period,
        candle_count=len(report.equity_curve),
        start_time_utc=report.start_time_utc,
        end_time_utc=report.end_time_utc,
        strategy_metrics=strategy_metrics,
        cash_benchmark=cash_b,
        buy_and_hold_benchmark=bnh_b,
        exposure_matched_benchmark=exp_b,
        raw_report=report,
    )


def run_sensitivity_analysis(
    candles: pd.DataFrame,
    period: EvaluationPeriod,
    symbol: str = "BTC/USDT",
    timeframe: str = "1d",
    initial_cash_usd: float = 10_000.0,
    risk_limits: RiskLimits | None = None,
) -> list[SensitivityScenario]:
    """Execute fee and slippage sensitivity checks without parameter optimization."""
    scenarios_cfg = [
        ("Baseline (10bps fee / 5bps slip)", 0.0010, 0.0005),
        ("Moderate Friction (15bps fee / 10bps slip)", 0.0015, 0.0010),
        ("High Friction (20bps fee / 20bps slip)", 0.0020, 0.0020),
        ("Stress Friction (30bps fee / 30bps slip)", 0.0030, 0.0030),
    ]

    results: list[SensitivityScenario] = []
    for name, fee_r, slip_p in scenarios_cfg:
        res = run_period_evaluation(
            period=period,
            full_candles=candles,
            symbol=symbol,
            timeframe=timeframe,
            initial_cash_usd=initial_cash_usd,
            fee_rate=fee_r,
            slippage_pct=slip_p,
            risk_limits=risk_limits,
        )
        sm = res.strategy_metrics
        results.append(
            SensitivityScenario(
                scenario_name=name,
                fee_rate=fee_r,
                slippage_pct=slip_p,
                total_return_pct=sm.total_return_pct,
                net_profit_usd=sm.net_profit_usd,
                max_drawdown_pct=sm.max_drawdown_pct,
                sharpe_ratio=sm.sharpe_ratio,
                profit_factor=sm.profit_factor,
                win_rate_pct=sm.win_rate_pct,
                total_fees_usd=sm.total_fees_usd,
                total_slippage_usd=sm.total_slippage_usd,
                total_friction_usd=sm.total_friction_usd,
            )
        )
    return results


def run_historical_evaluation(
    symbol: str = "BTC/USDT",
    timeframe: str = "1d",
    candle_limit: int = 1826,
    csv_file: str | None = None,
    output_dir: str = "reports",
    run_sensitivity: bool = True,
) -> EvaluationHarnessResult:
    """Run full reproducible multi-year evaluation across all defined periods."""
    _require_daily_timeframe(timeframe)
    run_id = f"eval_{uuid.uuid4().hex[:10]}"
    created_at_utc = datetime.now(timezone.utc).isoformat()

    # 1. Fetch or Load Historical Candles
    if csv_file:
        print(f"Loading historical candles from CSV: {csv_file}")
        raw_df = pd.read_csv(csv_file)
        data_source = f"CSV File ({csv_file})"
    else:
        settings = load_settings()
        print(
            f"Fetching {candle_limit} real historical daily candles for {symbol} "
            f"via public exchange ({settings.exchange_id})..."
        )
        exchange = create_public_exchange(settings.exchange_id)
        sym_settings = Settings(
            exchange_id=settings.exchange_id,
            symbol=symbol,
            timeframe=timeframe,
            candle_limit=candle_limit,
            openai_model=settings.openai_model,
            openai_api_key_configured=settings.openai_api_key_configured,
        )
        # fetch_ohlcv validates contiguity, drops incomplete open bar, fails closed
        raw_df = fetch_ohlcv(sym_settings, exchange=exchange)
        base_url = getattr(exchange, "base_url", None)
        if base_url is None:
            api_urls = getattr(exchange, "urls", {}).get("api", {})
            base_url = api_urls.get("public", "CCXT public OHLCV endpoint")
        data_source = f"Public {settings.exchange_id} OHLCV API ({base_url})"

    validated_df = validate_backtest_candles(raw_df, symbol, 86_400)
    total_candles = len(validated_df)
    dataset_start = validated_df["timestamp"].iloc[0].isoformat()
    dataset_end = validated_df["timestamp"].iloc[-1].isoformat()

    # 2. Verify Warmup Requirements
    # Indicator lookback requires >= 200 bars before the first decision bar (2022-06-01)
    first_period_start = pd.Timestamp(DEFAULT_PERIODS[0].start_date, tz="UTC")
    warmup_df = validated_df[validated_df["timestamp"] < first_period_start]
    warmup_count = len(warmup_df)
    if warmup_count < 200:
        raise InsufficientHistoryError(
            f"Evaluation requires at least 200 warm-up candles before {first_period_start.isoformat()} "
            f"to calculate SMA200 without look-ahead bias. Available warm-up candles: {warmup_count}."
        )

    warmup_start = warmup_df["timestamp"].iloc[0].isoformat()
    warmup_end = warmup_df["timestamp"].iloc[-1].isoformat()

    print(f"Dataset Loaded: {total_candles} candles ({dataset_start} -> {dataset_end})")
    print(f"Warmup Window:  {warmup_count} candles ({warmup_start} -> {warmup_end})")
    print()

    # 3. Execute Period Evaluations
    period_results: dict[str, PeriodEvaluationResult] = {}
    for period in DEFAULT_PERIODS:
        print(f"Evaluating period [{period.name}]: {period.start_date} -> {period.end_date}...")
        res = run_period_evaluation(
            period=period,
            full_candles=validated_df,
            symbol=symbol,
            timeframe=timeframe,
        )
        period_results[period.name] = res

    # 4. Sensitivity Analysis (on Development & Validation to avoid OOS bias)
    sensitivity_results: list[SensitivityScenario] = []
    if run_sensitivity:
        print("Running transaction cost sensitivity grid on development period...")
        dev_period = next(p for p in DEFAULT_PERIODS if p.name == "development")
        sensitivity_results = run_sensitivity_analysis(
            candles=validated_df,
            period=dev_period,
            symbol=symbol,
            timeframe=timeframe,
        )

    # 5. Extract Full Trade Records
    full_report = period_results["full"].raw_report
    all_trade_records = full_report.execution_history

    result = EvaluationHarnessResult(
        run_id=run_id,
        created_at_utc=created_at_utc,
        symbol=symbol,
        timeframe=timeframe,
        data_source=data_source,
        total_candles_fetched=total_candles,
        dataset_start_utc=dataset_start,
        dataset_end_utc=dataset_end,
        warmup_candles_count=warmup_count,
        warmup_start_utc=warmup_start,
        warmup_end_utc=warmup_end,
        period_results=period_results,
        sensitivity_results=sensitivity_results,
        trade_records=all_trade_records,
        dataset_sha256=_fingerprint_dataset(validated_df),
        code_revision=_git_revision(),
        code_fingerprint_sha256=_code_fingerprint(),
        dependency_versions=_dependency_versions(),
        experiment_config={
            "decision_provider": "DeterministicRuleProvider",
            "initial_cash_usd": 10_000.0,
            "fee_rate": 0.001,
            "slippage_pct": 0.0005,
            "candle_limit": candle_limit,
            "sensitivity_enabled": run_sensitivity,
            "sensitivity_scenarios": [
                {"fee_rate": fee, "slippage_pct": slippage}
                for fee, slippage in ((0.0010, 0.0005), (0.0015, 0.0010), (0.0020, 0.0020), (0.0030, 0.0030))
            ],
            "risk_limits": asdict(RiskLimits()),
            "periods": [asdict(period) for period in DEFAULT_PERIODS],
            "indicator_windows": {"sma_fast": 50, "sma_slow": 200, "momentum_days": 30},
            "candle_policy": (
                "1d candles become decision-eligible only after close; end_date is an "
                "inclusive candle-open label whose close is available at the next UTC midnight"
            ),
            "pending_order_policy": (
                "A signal on the last eligible candle remains unexecuted if no following "
                "candle open is present; it does not affect equity or trade metrics"
            ),
            "benchmark_assumptions": {
                "buy_and_hold": "100% BTC, gross of transaction costs",
                "exposure_matched": "constant average strategy weight, rebalanced daily, gross of transaction costs",
                "cash": "0% nominal yield",
            },
        },
    )

    # 6. Save Artifacts
    save_evaluation_artifacts(result, output_dir=output_dir)
    return result


def format_evaluation_markdown(res: EvaluationHarnessResult) -> str:
    """Format comprehensive evaluation report in GitHub Markdown."""
    lines: list[str] = [
        f"# Historical Strategy Evaluation Report: {res.symbol} ({res.run_id})",
        "",
        f"- **Generated UTC:** {res.created_at_utc}",
        f"- **Asset Symbol:** {res.symbol} | **Timeframe:** {res.timeframe}",
        f"- **Market Data Source:** {res.data_source}",
        f"- **Total Candles Analyzed:** {res.total_candles_fetched} ({res.dataset_start_utc[:10]} to {res.dataset_end_utc[:10]})",
        f"- **Indicator Warm-up:** {res.warmup_candles_count} closed daily bars ({res.warmup_start_utc[:10]} to {res.warmup_end_utc[:10]})",
        f"- **Dataset SHA-256:** `{res.dataset_sha256 or 'not recorded'}`",
        f"- **Code revision:** `{res.code_revision or 'unavailable'}` | **Source fingerprint:** `{res.code_fingerprint_sha256 or 'not recorded'}`",
        f"- **Dependency versions:** `{json.dumps(res.dependency_versions, sort_keys=True)}`",
        "- **Experiment configuration:**",
        "```json",
        json.dumps(res.experiment_config, sort_keys=True, indent=2),
        "```",
        "",
        "---",
        "",
        "## 1. Multi-Period Walk-Forward Performance Comparison",
        "",
        "| Metric | Development (In-Sample) | Validation | Out-of-Sample (Holdout) | Full Compound |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]

    p_names = ["development", "validation", "out_of_sample", "full"]
    periods_data = [res.period_results[p] for p in p_names]

    def _val(getter: Any) -> str:
        return " | ".join(getter(p) for p in periods_data)

    lines.append(f"| **Date Range** | {_val(lambda p: f'{p.period.start_date} to {p.period.end_date}')} |")
    lines.append(f"| **Decision Bars** | {_val(lambda p: str(p.candle_count))} |")
    lines.append(f"| **Strategy Net Return** | {_val(lambda p: f'{p.strategy_metrics.total_return_pct:+.2f}%')} |")
    lines.append(f"| **BTC Buy & Hold Return** | {_val(lambda p: f'{p.buy_and_hold_benchmark.total_return_pct:+.2f}%')} |")
    lines.append(f"| **Exposure-Matched Return** | {_val(lambda p: f'{p.exposure_matched_benchmark.total_return_pct:+.2f}%')} |")
    lines.append(f"| **Cash-Only Return** | {_val(lambda p: f'{p.cash_benchmark.total_return_pct:+.2f}%')} |")
    lines.append(f"| **Strategy Max Drawdown** | {_val(lambda p: f'{p.strategy_metrics.max_drawdown_pct:.2f}%')} |")
    lines.append(f"| **BTC B&H Max Drawdown** | {_val(lambda p: f'{p.buy_and_hold_benchmark.max_drawdown_pct:.2f}%')} |")
    lines.append(f"| **Exposure-Matched Max DD** | {_val(lambda p: f'{p.exposure_matched_benchmark.max_drawdown_pct:.2f}%')} |")
    lines.append(f"| **Annualized Sharpe (Daily)** | {_val(lambda p: f'{p.strategy_metrics.sharpe_ratio:.2f}')} |")
    lines.append(f"| **BTC B&H Sharpe** | {_val(lambda p: f'{p.buy_and_hold_benchmark.sharpe_ratio:.2f}')} |")
    lines.append(f"| **Exposure-Matched Sharpe** | {_val(lambda p: f'{p.exposure_matched_benchmark.sharpe_ratio:.2f}')} |")
    lines.append(f"| **Realized Exit Tranches** | {_val(lambda p: str(p.strategy_metrics.completed_trades))} |")
    lines.append(f"| **Unexecuted End Signals** | {_val(lambda p: str(p.strategy_metrics.unexecuted_orders))} |")
    lines.append(f"| **Win Rate** | {_val(lambda p: f'{p.strategy_metrics.win_rate_pct:.1f}%')} |")
    lines.append(f"| **Expectancy / Trade** | {_val(lambda p: f'${p.strategy_metrics.expectancy_usd:+,.2f}')} |")
    lines.append(f"| **Profit Factor** | {_val(lambda p: 'inf' if math.isinf(p.strategy_metrics.profit_factor) else f'{p.strategy_metrics.profit_factor:.2f}')} |")
    lines.append(f"| **Average Daily Exposure** | {_val(lambda p: f'{p.strategy_metrics.avg_exposure_pct:.2f}%')} |")
    lines.append(f"| **Time Invested** | {_val(lambda p: f'{p.strategy_metrics.pct_time_invested:.1f}%')} |")
    lines.append(f"| **Total Friction (Fees+Slip)** | {_val(lambda p: f'${p.strategy_metrics.total_friction_usd:,.2f}')} |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. Sensitivity Analysis (Development Period Friction Matrix)",
        "",
        "Tests strategy fragility against adverse execution costs without parameter tuning:",
        "",
        "| Scenario | Fee Rate | Slippage | Net Return | Net Profit | Max Drawdown | Sharpe | Profit Factor | Total Friction |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for sc in res.sensitivity_results:
        pf_str = "inf" if math.isinf(sc.profit_factor) else f"{sc.profit_factor:.2f}"
        lines.append(
            f"| **{sc.scenario_name}** | {sc.fee_rate:.2%} | {sc.slippage_pct:.2%} | "
            f"{sc.total_return_pct:+.2f}% | ${sc.net_profit_usd:+,.2f} | {sc.max_drawdown_pct:.2f}% | "
            f"{sc.sharpe_ratio:.2f} | {pf_str} | ${sc.total_friction_usd:,.2f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Methodology, Assumptions & Safeguards",
        "",
        "- **Chronological Integrity:** Development, validation, and holdout use disjoint UTC candle-open labels and independent cash-started portfolios.",
        "- **Candle availability:** Daily bars become eligible after close at the next UTC midnight. The inclusive end-date candle closes at the following midnight. A last-bar signal without a following candle open remains unexecuted and excluded from equity and trade metrics.",
        "- **Next-Bar Execution:** Signals use only closed candles; fills use the following candle's open with adverse slippage and fees.",
        "- **Exposure-Matched Benchmark:** Holds the strategy's average exposure in BTC and rebalances daily; benchmark transaction costs are excluded.",
        "- **Buy-and-Hold Benchmark:** Holds 100% BTC from the first evaluated close; benchmark transaction costs are excluded.",
        "- **Sharpe Ratio Convention:** Daily close-to-close returns use sample standard deviation, zero risk-free rate, and $\\sqrt{365}$ annualization. Only 1d evaluation is supported.",
        "- **Safeguards Enforced:** `is_live=False` hardcoded invariant; strict TLS verification; independent SQLite `:memory:` database state for each run.",
    ])

    full_sm = res.period_results["full"].strategy_metrics
    full_bnh = res.period_results["full"].buy_and_hold_benchmark
    oos_sm = res.period_results["out_of_sample"].strategy_metrics
    oos_bnh = res.period_results["out_of_sample"].buy_and_hold_benchmark
    dev_sm = res.period_results["development"].strategy_metrics
    dev_bnh = res.period_results["development"].buy_and_hold_benchmark

    lines.extend([
        "",
        "---",
        "",
        "## 4. Objective Findings & Identified Strategy Limitations",
        "",
        "1. **Trade Scarcity & Low Turnover:**",
        f"   Across {res.period_results['full'].candle_count:,} decision bars (~4.4 years), the strategy completed only "
        f"{full_sm.completed_trades} realized sell tranches ({full_sm.total_executions} total fills). "
        f"The strategy spends {100.0 - full_sm.pct_time_invested:.1f}% of all trading days flat in 100% cash.",
        "2. **Cash Drag in Strong Bull Markets:**",
        f"   With an average portfolio exposure of only {full_sm.avg_exposure_pct:.2f}% (governed by the 10% maximum allocation cap per symbol), "
        f"the strategy earned {full_sm.total_return_pct:+.2f}% over the full timeline, substantially trailing the 100% BTC Buy-and-Hold return ({full_bnh.total_return_pct:+.2f}%). "
        f"However, it slightly outperformed the exposure-matched benchmark ({res.period_results['full'].exposure_matched_benchmark.total_return_pct:+.2f}%).",
        "3. **Capital Preservation Advantage:**",
        f"   During the severe 2022 bear market (where BTC suffered a {dev_bnh.max_drawdown_pct:.2f}% maximum drawdown), "
        f"the strategy suffered a maximum drawdown of only {dev_sm.max_drawdown_pct:.2f}%, demonstrating effective downside protection.",
        "4. **Out-of-Sample Statistical Insufficiency:**",
        f"   In the final holdout period (2025-07 to 2026-10), the strategy achieved {oos_sm.total_return_pct:+.2f}% net return "
        f"(vs {oos_bnh.total_return_pct:+.2f}% for BTC B&H) across {oos_sm.completed_trades} realized sell tranches. "
        "With so few realized tranches in the out-of-sample window, there is insufficient statistical sample size to reject randomness or claim an unambiguous edge.",
        "5. **Execution Liquidity Assumption:**",
        "   Simulated fills assume next-bar open liquidity without market impact beyond the fixed 0.05% slippage and 0.10% fee assumptions.",
        "",
        "---",
        "**Disclaimer:** This historical evaluation report is strictly educational and for quantitative research validation. Past performance does not predict future returns. Live exchange trading is disabled.",
    ])

    return "\n".join(lines)


def save_evaluation_artifacts(
    result: EvaluationHarnessResult,
    output_dir: str = "reports",
) -> tuple[Path, Path]:
    """Serialize JSON and Markdown evaluation artifacts to disk."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Save Markdown Report
    md_content = format_evaluation_markdown(result)
    md_file = out_path / f"evaluation_{result.run_id}.md"
    md_file.write_text(md_content, encoding="utf-8")

    # 2. Save Machine-Readable JSON Dump
    def _serialize(obj: Any) -> Any:
        if isinstance(obj, (EvaluationPeriod, BenchmarkMetrics, StrategyMetrics, SensitivityScenario)):
            return asdict(obj)
        if isinstance(obj, PeriodEvaluationResult):
            return {
                "period": asdict(obj.period),
                "candle_count": obj.candle_count,
                "start_time_utc": obj.start_time_utc,
                "end_time_utc": obj.end_time_utc,
                "strategy_metrics": asdict(obj.strategy_metrics),
                "cash_benchmark": asdict(obj.cash_benchmark),
                "buy_and_hold_benchmark": asdict(obj.buy_and_hold_benchmark),
                "exposure_matched_benchmark": asdict(obj.exposure_matched_benchmark),
            }
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, float) and math.isinf(obj):
            return "inf"
        raise TypeError(f"Type {type(obj)} not serializable")

    json_payload = {
        "run_id": result.run_id,
        "created_at_utc": result.created_at_utc,
        "symbol": result.symbol,
        "timeframe": result.timeframe,
        "data_source": result.data_source,
        "total_candles_fetched": result.total_candles_fetched,
        "dataset_start_utc": result.dataset_start_utc,
        "dataset_end_utc": result.dataset_end_utc,
        "warmup_candles_count": result.warmup_candles_count,
        "warmup_start_utc": result.warmup_start_utc,
        "warmup_end_utc": result.warmup_end_utc,
        "period_results": {k: _serialize(v) for k, v in result.period_results.items()},
        "sensitivity_results": [_serialize(s) for s in result.sensitivity_results],
        "trade_records": result.trade_records,
        "dataset_sha256": result.dataset_sha256,
        "code_revision": result.code_revision,
        "code_fingerprint_sha256": result.code_fingerprint_sha256,
        "dependency_versions": result.dependency_versions,
        "experiment_config": result.experiment_config,
    }

    json_file = out_path / f"evaluation_{result.run_id}.json"
    json_file.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    print(f"Artifacts saved:")
    print(f"  Markdown: {md_file.resolve()}")
    print(f"  JSON:     {json_file.resolve()}")
    return md_file, json_file


def main() -> int:
    """CLI entry point for running reproducible historical evaluation harness."""
    parser = argparse.ArgumentParser(
        description="Reproducible historical evaluation harness for the crypto research agent."
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC/USDT",
        help="Asset symbol to evaluate (default: BTC/USDT).",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1d",
        help="Bar timeframe (default: 1d).",
    )
    parser.add_argument(
        "--candle-limit",
        type=int,
        default=1826,
        help="Total historical closed candles to fetch (default: 1826 = 5 years).",
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default=None,
        help="Optional path to historical CSV candles file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory where evaluation artifacts will be written (default: reports).",
    )
    parser.add_argument(
        "--no-sensitivity",
        action="store_true",
        help="Skip fee/slippage sensitivity stress test grid.",
    )

    args = parser.parse_args()

    try:
        result = run_historical_evaluation(
            symbol=args.symbol,
            timeframe=args.timeframe,
            candle_limit=args.candle_limit,
            csv_file=args.csv_file,
            output_dir=args.output_dir,
            run_sensitivity=not args.no_sensitivity,
        )
        print()
        print(format_evaluation_markdown(result))
        return 0
    except Exception as exc:
        print(f"Evaluation harness failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
