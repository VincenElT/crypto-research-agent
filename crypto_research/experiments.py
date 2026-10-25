"""Controlled, reproducible strategy hypothesis experiments.

This runner compares caller-supplied decision providers on development and
validation periods only. It intentionally offers no holdout evaluation path;
the final holdout remains reserved for a separately approved final report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from crypto_research.backtest import DecisionProvider, DeterministicRuleProvider
from crypto_research.config import Settings, load_settings
from crypto_research.evaluation import (
    DEFAULT_PERIODS,
    PeriodEvaluationResult,
    _code_fingerprint,
    _dependency_versions,
    _fingerprint_dataset,
    _git_revision,
    run_period_evaluation,
)
from crypto_research.exceptions import ConfigurationError, StorageError
from crypto_research.indicators import IndicatorSnapshot
from crypto_research.market_data import create_public_exchange, fetch_ohlcv
from crypto_research.research import DualAgentResult
from crypto_research.risk import convert_portfolio_action
from crypto_research.risk import RiskLimits
from crypto_research.schemas import ScoutProposal, SkepticReview
from crypto_research.storage import scrub_secrets


@dataclass(frozen=True)
class StrategyHypothesis:
    """One explicitly named strategy hypothesis and its reproducible setup."""

    name: str
    hypothesis: str
    provider_name: str
    provider_factory: Callable[[], DecisionProvider]
    configuration: dict[str, Any]
    parent_experiment_id: str | None = None


class _TheoryRuleProvider:
    """Small, fixed-rule examples for predeclared theory comparisons."""

    def __init__(self, theory: str) -> None:
        self.theory = theory

    def __call__(
        self,
        market_context: dict[str, Any],
        snapshot: IndicatorSnapshot,
        symbol: str,
    ) -> DualAgentResult:
        close = snapshot.close
        if self.theory == "time_series_momentum":
            action = (
                "BUY" if snapshot.momentum_30d > 0
                else "SELL" if snapshot.momentum_30d < 0
                else "HOLD"
            )
            rationale = "Fixed 30-candle time-series momentum sign rule."
        elif self.theory == "donchian_9":
            bars = market_context.get("recent_closed_bars", [])
            previous = bars[:-1]
            if not previous:
                action = "HOLD"
            elif close > max(float(bar["high"]) for bar in previous):
                action = "BUY"
            elif close < min(float(bar["low"]) for bar in previous):
                action = "SELL"
            else:
                action = "HOLD"
            rationale = "Fixed 9-prior-bar Donchian close breakout; current bar excluded from range."
        elif self.theory == "sma50_mean_reversion":
            if close < snapshot.sma_50 * 0.98:
                action = "BUY"
            elif close >= snapshot.sma_50:
                action = "SELL"
            else:
                action = "HOLD"
            rationale = "Fixed 2% below-SMA50 dip entry, exit on recovery to SMA50."
        else:
            raise ConfigurationError(f"Unknown theory provider: {self.theory}")

        scout = ScoutProposal(
            action=action,
            confidence=0.8 if action != "HOLD" else 0.6,
            thesis=rationale,
            evidence=[rationale],
            risks=["Research hypothesis only; rule is not validated as predictive."],
            missing_data=[],
        )
        critique = SkepticReview(
            verdict="APPROVE" if action != "HOLD" else "REJECT",
            agrees_with_action=action != "HOLD",
            rationale=rationale,
            missing_evidence=[],
            weaknesses=["Fixed illustrative rule; market regime may invalidate it."],
            confidence=0.8 if action != "HOLD" else 0.6,
        )
        converted = convert_portfolio_action(
            DualAgentResult(proposal=scout, critique=critique),
            has_open_position=bool(market_context.get("has_open_position", False)),
        )
        return converted


def standard_hypotheses() -> list[StrategyHypothesis]:
    """Return a small, predeclared theory set; this does not tune parameters."""
    return [
        StrategyHypothesis(
            name="current_baseline",
            hypothesis="Control: existing SMA50/SMA200 trend plus 30-day momentum rules.",
            provider_name="DeterministicRuleProvider",
            provider_factory=DeterministicRuleProvider,
            configuration={"rule_set": "current_baseline"},
        ),
        StrategyHypothesis(
            name="time_series_momentum_30",
            hypothesis="A positive 30-candle return predicts continuation; negative return exits.",
            provider_name="TimeSeriesMomentum30Provider",
            provider_factory=lambda: _TheoryRuleProvider("time_series_momentum"),
            configuration={"signal": "30-candle momentum sign", "shorting": False},
        ),
        StrategyHypothesis(
            name="donchian_9_breakout",
            hypothesis="A close outside the prior nine-candle range captures short-term breakouts.",
            provider_name="Donchian9Provider",
            provider_factory=lambda: _TheoryRuleProvider("donchian_9"),
            configuration={"lookback_prior_candles": 9, "entry_exit": "close outside range"},
        ),
        StrategyHypothesis(
            name="sma50_mean_reversion",
            hypothesis="A 2% close below SMA50 mean-reverts toward its average; exit on recovery.",
            provider_name="SMA50MeanReversionProvider",
            provider_factory=lambda: _TheoryRuleProvider("sma50_mean_reversion"),
            configuration={
                "reference": "SMA50",
                "entry_deviation_pct": -2.0,
                "exit": "close >= SMA50",
            },
        ),
    ]


@dataclass(frozen=True)
class ExperimentPeriodSummary:
    period_name: str
    start_date: str
    end_date: str
    candle_count: int
    strategy_metrics: dict[str, Any]
    buy_and_hold_metrics: dict[str, Any]
    exposure_matched_metrics: dict[str, Any]


@dataclass(frozen=True)
class ExperimentRun:
    run_id: str
    experiment_id: str
    created_at_utc: str
    hypothesis_name: str
    hypothesis: str
    parent_experiment_id: str | None
    symbol: str
    timeframe: str
    data_source: str
    dataset_sha256: str
    dataset_start_utc: str
    dataset_end_utc: str
    code_revision: str | None
    code_fingerprint_sha256: str
    dependency_versions: dict[str, str]
    configuration: dict[str, Any]
    periods: list[ExperimentPeriodSummary]
    status: str = "completed"
    failure: str | None = None


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _period_summary(result: PeriodEvaluationResult) -> ExperimentPeriodSummary:
    return ExperimentPeriodSummary(
        period_name=result.period.name,
        start_date=result.period.start_date,
        end_date=result.period.end_date,
        candle_count=result.candle_count,
        strategy_metrics=_json_safe(asdict(result.strategy_metrics)),
        buy_and_hold_metrics=_json_safe(asdict(result.buy_and_hold_benchmark)),
        exposure_matched_metrics=_json_safe(asdict(result.exposure_matched_benchmark)),
    )


def _initialize_store(conn: sqlite3.Connection) -> None:
    try:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS research_experiment_runs (
                run_id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                created_at_utc TEXT NOT NULL,
                hypothesis_name TEXT NOT NULL,
                hypothesis TEXT NOT NULL,
                parent_experiment_id TEXT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                data_source TEXT NOT NULL,
                dataset_sha256 TEXT NOT NULL,
                dataset_start_utc TEXT NOT NULL,
                dataset_end_utc TEXT NOT NULL,
                code_revision TEXT,
                code_fingerprint_sha256 TEXT NOT NULL,
                dependency_versions_json TEXT NOT NULL,
                configuration_json TEXT NOT NULL,
                period_summaries_json TEXT NOT NULL,
                status TEXT NOT NULL,
                failure TEXT,
                conclusion TEXT NOT NULL DEFAULT ''
            )"""
        )
        conn.commit()
    except sqlite3.Error as exc:
        raise StorageError(f"Failed to initialize research experiment store: {exc}") from exc


def run_controlled_experiments(
    candles: pd.DataFrame,
    hypotheses: list[StrategyHypothesis],
    *,
    db_path: str | Path,
    symbol: str = "BTC/USDT",
    timeframe: str = "1d",
    data_source: str = "caller-provided dataframe",
    initial_cash_usd: float = 10_000.0,
    fee_rate: float = 0.001,
    slippage_pct: float = 0.0005,
    risk_limits: RiskLimits | None = None,
) -> list[ExperimentRun]:
    """Run each supplied hypothesis on development and validation only.

    The persisted experiment fingerprint excludes the random run ID and
    creation timestamp, and includes data, code/config identity, and split
    names. Providers are constructed afresh for each period.
    """
    if not hypotheses:
        raise ConfigurationError("At least one strategy hypothesis is required.")
    if timeframe != "1d":
        raise ConfigurationError("Controlled research currently supports daily bars only.")
    if len({hyp.name for hyp in hypotheses}) != len(hypotheses):
        raise ConfigurationError("Strategy hypothesis names must be unique within a run.")

    # Hard-code the allowed splits rather than accept arbitrary period names.
    periods = [
        period for period in DEFAULT_PERIODS
        if period.name in {"development", "validation"}
    ]
    if [period.name for period in periods] != ["development", "validation"]:
        raise ConfigurationError("Development and validation periods must be configured.")

    from crypto_research.backtest import validate_backtest_candles

    validated = validate_backtest_candles(candles, symbol, 86_400)
    dataset_hash = _fingerprint_dataset(validated)
    dataset_start = validated["timestamp"].iloc[0].isoformat()
    dataset_end = validated["timestamp"].iloc[-1].isoformat()
    limits = risk_limits or RiskLimits()
    shared_configuration = {
        "symbol": symbol,
        "timeframe": timeframe,
        "data_source": data_source,
        "initial_cash_usd": initial_cash_usd,
        "fee_rate": fee_rate,
        "slippage_pct": slippage_pct,
        "risk_limits": asdict(limits),
        "periods": [asdict(period) for period in periods],
    }

    out_path = Path(db_path)
    if str(db_path) != ":memory:":
        out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        conn = sqlite3.connect(str(db_path))
    except sqlite3.Error as exc:
        raise StorageError(f"Failed to open research experiment store: {exc}") from exc

    runs: list[ExperimentRun] = []
    try:
        _initialize_store(conn)
        for hypothesis in hypotheses:
            experiment_config = scrub_secrets(
                {
                    "hypothesis_name": hypothesis.name,
                    "hypothesis": hypothesis.hypothesis,
                    "provider_name": hypothesis.provider_name,
                    "parent_experiment_id": hypothesis.parent_experiment_id,
                    "configuration": hypothesis.configuration,
                    "shared_configuration": shared_configuration,
                    "dataset_sha256": dataset_hash,
                    "code_revision": _git_revision(),
                    "code_fingerprint_sha256": _code_fingerprint(),
                    "dependency_versions": _dependency_versions(),
                }
            )
            experiment_id = _canonical_hash(experiment_config)
            summaries: list[ExperimentPeriodSummary] = []
            failure: Exception | None = None
            try:
                for period in periods:
                    result = run_period_evaluation(
                        period=period,
                        full_candles=validated,
                        symbol=symbol,
                        timeframe=timeframe,
                        initial_cash_usd=initial_cash_usd,
                        fee_rate=fee_rate,
                        slippage_pct=slippage_pct,
                        risk_limits=limits,
                        decision_provider=hypothesis.provider_factory(),
                    )
                    summaries.append(_period_summary(result))
            except Exception as exc:
                failure = exc

            run = ExperimentRun(
                run_id=f"research_{uuid.uuid4().hex[:12]}",
                experiment_id=experiment_id,
                created_at_utc=datetime.now(timezone.utc).isoformat(),
                hypothesis_name=hypothesis.name,
                hypothesis=scrub_secrets(hypothesis.hypothesis),
                parent_experiment_id=hypothesis.parent_experiment_id,
                symbol=symbol,
                timeframe=timeframe,
                data_source=scrub_secrets(data_source),
                dataset_sha256=dataset_hash,
                dataset_start_utc=dataset_start,
                dataset_end_utc=dataset_end,
                code_revision=experiment_config["code_revision"],
                code_fingerprint_sha256=experiment_config["code_fingerprint_sha256"],
                dependency_versions=experiment_config["dependency_versions"],
                configuration=_json_safe(experiment_config),
                periods=summaries,
                status="failed" if failure else "completed",
                failure=(
                    scrub_secrets(f"{type(failure).__name__}: {failure}") if failure else None
                ),
            )
            conn.execute(
                """INSERT INTO research_experiment_runs (
                    run_id, experiment_id, created_at_utc, hypothesis_name, hypothesis,
                    parent_experiment_id, symbol, timeframe, data_source, dataset_sha256,
                    dataset_start_utc, dataset_end_utc, code_revision,
                    code_fingerprint_sha256, dependency_versions_json, configuration_json,
                    period_summaries_json, status, failure
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run.run_id,
                    run.experiment_id,
                    run.created_at_utc,
                    run.hypothesis_name,
                    run.hypothesis,
                    run.parent_experiment_id,
                    run.symbol,
                    run.timeframe,
                    run.data_source,
                    run.dataset_sha256,
                    run.dataset_start_utc,
                    run.dataset_end_utc,
                    run.code_revision,
                    run.code_fingerprint_sha256,
                    json.dumps(run.dependency_versions, sort_keys=True, allow_nan=False),
                    json.dumps(run.configuration, sort_keys=True, allow_nan=False),
                    json.dumps([asdict(item) for item in run.periods], sort_keys=True, allow_nan=False),
                    run.status,
                    run.failure,
                ),
            )
            conn.commit()
            runs.append(run)
            if failure:
                raise failure
    except sqlite3.Error as exc:
        conn.rollback()
        raise StorageError(f"Failed to persist research experiment: {exc}") from exc
    finally:
        conn.close()
    return runs


def list_experiment_runs(db_path: str | Path) -> list[dict[str, Any]]:
    """Retrieve persisted experiment summaries in creation order."""
    try:
        with sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute(
                """SELECT run_id, experiment_id, created_at_utc, hypothesis_name,
                          hypothesis, parent_experiment_id, symbol, timeframe,
                          data_source, dataset_sha256, dataset_start_utc, dataset_end_utc,
                          code_revision, code_fingerprint_sha256, dependency_versions_json,
                          configuration_json, period_summaries_json, status, failure,
                          conclusion
                   FROM research_experiment_runs ORDER BY created_at_utc, run_id"""
            ).fetchall()
    except sqlite3.Error as exc:
        raise StorageError(f"Failed to retrieve research experiments: {exc}") from exc
    columns = (
        "run_id", "experiment_id", "created_at_utc", "hypothesis_name", "hypothesis",
        "parent_experiment_id", "symbol", "timeframe", "data_source", "dataset_sha256",
        "dataset_start_utc", "dataset_end_utc", "code_revision",
        "code_fingerprint_sha256", "dependency_versions", "configuration", "periods",
        "status", "failure", "conclusion",
    )
    output: list[dict[str, Any]] = []
    for row in rows:
        item = dict(zip(columns, row))
        item["configuration"] = json.loads(item["configuration"])
        item["dependency_versions"] = json.loads(item["dependency_versions"])
        item["periods"] = json.loads(item["periods"])
        output.append(item)
    return output


def record_experiment_conclusion(
    db_path: str | Path,
    run_id: str,
    conclusion: str,
) -> None:
    """Attach a human- or agent-written conclusion to a persisted experiment."""
    if not conclusion.strip():
        raise ConfigurationError("Experiment conclusion must not be empty.")
    try:
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.execute(
                "UPDATE research_experiment_runs SET conclusion = ? WHERE run_id = ?",
                (conclusion.strip(), run_id),
            )
            if cursor.rowcount != 1:
                raise StorageError(f"Unknown research experiment run: {run_id}")
    except sqlite3.Error as exc:
        raise StorageError(f"Failed to record research experiment conclusion: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    """CLI for a predefined, no-network-credentials research comparison."""
    parser = argparse.ArgumentParser(
        description="Compare predeclared strategy hypotheses on development and validation only."
    )
    parser.add_argument(
        "--csv-file",
        help="Optional real OHLCV CSV; otherwise fetch public exchange candles.",
    )
    parser.add_argument("--symbol", default=None, help="Market symbol (default from .env).")
    parser.add_argument("--timeframe", default="1d", help="Evaluation supports 1d only.")
    parser.add_argument("--candle-limit", type=int, default=1826)
    parser.add_argument("--db-path", default="research/experiments.sqlite")
    args = parser.parse_args(argv)

    if args.timeframe != "1d":
        parser.error("controlled experiments currently support --timeframe 1d only")
    if args.csv_file:
        candles = pd.read_csv(args.csv_file)
        data_source = f"CSV file: {Path(args.csv_file).resolve()}"
        settings = load_settings()
        symbol = args.symbol or settings.symbol
        exchange_id = settings.exchange_id
        risk_limits = settings.to_risk_limits()
        fee_rate = settings.paper_fee_rate
        slippage_pct = settings.paper_slippage_pct
        initial_cash_usd = settings.paper_initial_cash_usd
    else:
        settings = load_settings()
        symbol = args.symbol or settings.symbol
        exchange_id = settings.exchange_id
        exchange = create_public_exchange(exchange_id)
        fetch_settings = Settings(
            exchange_id=exchange_id,
            symbol=symbol,
            timeframe=args.timeframe,
            candle_limit=args.candle_limit,
            openai_model=settings.openai_model,
            openai_api_key_configured=settings.openai_api_key_configured,
        )
        candles = fetch_ohlcv(fetch_settings, exchange=exchange)
        api_urls = getattr(exchange, "urls", {}).get("api", {})
        endpoint = api_urls.get("public", "CCXT public OHLCV endpoint")
        data_source = f"Public {exchange_id} OHLCV API ({endpoint})"
        risk_limits = settings.to_risk_limits()
        fee_rate = settings.paper_fee_rate
        slippage_pct = settings.paper_slippage_pct
        initial_cash_usd = settings.paper_initial_cash_usd

    runs = run_controlled_experiments(
        candles,
        standard_hypotheses(),
        db_path=args.db_path,
        symbol=symbol,
        timeframe=args.timeframe,
        data_source=data_source,
        initial_cash_usd=initial_cash_usd,
        fee_rate=fee_rate,
        slippage_pct=slippage_pct,
        risk_limits=risk_limits,
    )
    for run in runs:
        print(f"\n{run.hypothesis_name} ({run.run_id}) [{run.status}]")
        print(f"  Experiment fingerprint: {run.experiment_id}")
        print(f"  Dataset SHA-256: {run.dataset_sha256}")
        for period in run.periods:
            metrics = period.strategy_metrics
            buy_hold = period.buy_and_hold_metrics
            exposure_matched = period.exposure_matched_metrics
            print(
                f"  {period.period_name}: return={metrics['total_return_pct']:+.2f}% "
                f"Sharpe={metrics['sharpe_ratio']:.2f} "
                f"max_drawdown={metrics['max_drawdown_pct']:.2f}% "
                f"exposure={metrics['avg_exposure_pct']:.2f}% | "
                f"BTC B&H={buy_hold['total_return_pct']:+.2f}% gross | "
                f"exposure-matched={exposure_matched['total_return_pct']:+.2f}% gross"
            )
    print("\nFinal holdout was not evaluated. No exchange orders were placed.")
    print(f"Experiment history: {Path(args.db_path).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
