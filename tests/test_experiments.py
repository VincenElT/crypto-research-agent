"""Tests for the bounded development/validation hypothesis runner."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from crypto_research.backtest import DeterministicRuleProvider, generate_backtest_candles
from crypto_research.exceptions import ConfigurationError
from crypto_research.experiments import (
    StrategyHypothesis,
    list_experiment_runs,
    record_experiment_conclusion,
    run_controlled_experiments,
    standard_hypotheses,
)


def _hypothesis() -> StrategyHypothesis:
    return StrategyHypothesis(
        name="baseline_rules",
        hypothesis="The current trend and momentum baseline is the control.",
        provider_name="DeterministicRuleProvider",
        provider_factory=DeterministicRuleProvider,
        configuration={
            "rule_set": "current_baseline",
            "api_key": "sk-proj-1234567890abcdef1234567890",
        },
    )


def test_controlled_runner_persists_reproducible_development_validation_runs(
    tmp_path: Path,
) -> None:
    candles = generate_backtest_candles(
        periods=1826, start_date="2021-10-25", trend="cycle"
    )
    db_path = tmp_path / "experiments.sqlite"

    first = run_controlled_experiments(candles, [_hypothesis()], db_path=db_path)
    second = run_controlled_experiments(candles, [_hypothesis()], db_path=db_path)
    saved = list_experiment_runs(db_path)

    assert [period.period_name for period in first[0].periods] == [
        "development", "validation"
    ]
    assert {period.period_name for period in first[0].periods}.isdisjoint(
        {"out_of_sample", "full"}
    )
    assert first[0].run_id != second[0].run_id
    assert first[0].experiment_id == second[0].experiment_id
    assert len(saved) == 2
    assert saved[0]["experiment_id"] == first[0].experiment_id
    assert saved[0]["configuration"]["dataset_sha256"] == first[0].dataset_sha256
    assert saved[0]["configuration"]["configuration"]["api_key"] == "[REDACTED]"
    assert saved[0]["parent_experiment_id"] is None
    assert saved[0]["status"] == "completed"
    assert saved[0]["code_fingerprint_sha256"]
    record_experiment_conclusion(db_path, first[0].run_id, "Baseline retained as the control.")
    assert list_experiment_runs(db_path)[0]["conclusion"] == "Baseline retained as the control."


def test_controlled_runner_persists_failed_experiment_before_reraising(
    tmp_path: Path,
) -> None:
    candles = generate_backtest_candles(
        periods=1826, start_date="2021-10-25", trend="cycle"
    )

    def failing_provider_factory():
        def fail(*args, **kwargs):
            raise RuntimeError("provider failure fixture")

        return fail

    hypothesis = StrategyHypothesis(
        name="failing_fixture",
        hypothesis="Verify failures are retained.",
        provider_name="FailingFixture",
        provider_factory=failing_provider_factory,
        configuration={"fixture": True},
    )
    db_path = tmp_path / "failed-experiments.sqlite"

    with pytest.raises(RuntimeError, match="provider failure fixture"):
        run_controlled_experiments(candles, [hypothesis], db_path=db_path)

    saved = list_experiment_runs(db_path)
    assert len(saved) == 1
    assert saved[0]["status"] == "failed"
    assert "provider failure fixture" in saved[0]["failure"]
    assert saved[0]["periods"] == []


def test_controlled_runner_rejects_duplicate_hypothesis_names(tmp_path: Path) -> None:
    candles = generate_backtest_candles(
        periods=300, start_date="2021-10-25", trend="cycle"
    )
    with pytest.raises(ConfigurationError, match="names must be unique"):
        run_controlled_experiments(
            candles, [_hypothesis(), _hypothesis()], db_path=tmp_path / "experiments.sqlite"
        )


def test_controlled_runner_requires_an_explicit_hypothesis(tmp_path: Path) -> None:
    candles = generate_backtest_candles(
        periods=300, start_date="2021-10-25", trend="cycle"
    )
    with pytest.raises(ConfigurationError, match="At least one strategy hypothesis"):
        run_controlled_experiments(candles, [], db_path=tmp_path / "experiments.sqlite")


def test_standard_hypotheses_are_named_and_cover_distinct_theories() -> None:
    candidates = standard_hypotheses()
    assert [candidate.name for candidate in candidates] == [
        "current_baseline",
        "time_series_momentum_30",
        "donchian_9_breakout",
        "sma50_mean_reversion",
    ]
    assert len({candidate.provider_name for candidate in candidates}) == len(candidates)
    assert all(candidate.hypothesis and candidate.configuration for candidate in candidates)


def test_standard_theory_set_backtests_on_development_and_validation_only(
    tmp_path: Path,
) -> None:
    candles = generate_backtest_candles(
        periods=1826, start_date="2021-10-25", trend="cycle"
    )
    runs = run_controlled_experiments(
        candles,
        standard_hypotheses(),
        db_path=tmp_path / "theory-comparison.sqlite",
    )
    assert {run.hypothesis_name for run in runs} == {
        "current_baseline",
        "time_series_momentum_30",
        "donchian_9_breakout",
        "sma50_mean_reversion",
    }
    for run in runs:
        assert run.status == "completed"
        assert [period.period_name for period in run.periods] == [
            "development", "validation"
        ]
        assert {period.start_date for period in run.periods} == {
            "2022-06-01", "2024-01-01"
        }


def test_predeclared_theory_providers_use_only_current_or_prior_closed_inputs() -> None:
    candidates = {candidate.name: candidate for candidate in standard_hypotheses()}
    snapshot = SimpleNamespace(
        close=101.0,
        sma_50=100.0,
        sma_200=98.0,
        momentum_30d=0.05,
    )
    prior_bars = [
        {"high": 100.0 + idx * 0.01, "low": 95.0, "close": 99.0, "open": 98.0, "volume": 1.0}
        for idx in range(9)
    ]
    current_bar = {"high": 102.0, "low": 100.0, "close": 101.0, "open": 99.0, "volume": 1.0}
    context = {"recent_closed_bars": prior_bars + [current_bar], "has_open_position": False}

    momentum = candidates["time_series_momentum_30"].provider_factory()(context, snapshot, "BTC/USDT")
    breakout = candidates["donchian_9_breakout"].provider_factory()(context, snapshot, "BTC/USDT")
    reversion = candidates["sma50_mean_reversion"].provider_factory()(
        context,
        SimpleNamespace(close=97.0, sma_50=100.0, sma_200=98.0, momentum_30d=-0.03),
        "BTC/USDT",
    )
    assert momentum.proposal.action == "BUY"
    # The current high (102) is excluded from the previous range; the close breaks prior highs.
    assert breakout.proposal.action == "BUY"
    assert reversion.proposal.action == "BUY"
