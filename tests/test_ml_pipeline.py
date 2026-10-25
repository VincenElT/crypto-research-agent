from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd

from crypto_research.backtest import generate_backtest_candles
from crypto_research.ml_pipeline import (
    HORIZON_BARS,
    build_feature_dataset,
    chronological_split,
    train_models_on_candles,
)


def hourly_candles(periods: int = 1_600) -> pd.DataFrame:
    candles = generate_backtest_candles(
        symbol="BTC/USDT", periods=periods, freq="1h", trend="cycle"
    )
    # A small deterministic oscillation in close creates both target classes.
    wave = np.sin(np.arange(periods) / 7.0) * 25.0
    candles["close"] += wave
    candles["open"] = candles["close"]
    candles["high"] = candles["close"] + 150
    candles["low"] = candles["close"] - 150
    return candles


def test_features_are_causal_and_target_uses_four_future_closes() -> None:
    candles = hourly_candles()
    features, target, forward_return, _ = build_feature_dataset(candles, symbol="BTC/USDT")
    timestamp = features.index[500]
    source_index = candles.index[candles.timestamp == timestamp][0]
    assert target.loc[timestamp] == int(
        candles.loc[source_index + HORIZON_BARS, "close"] > candles.loc[source_index, "close"]
    )
    future_changed = candles.copy()
    future_changed.loc[source_index + 1 :, "close"] *= 1.4
    future_changed.loc[source_index + 1 :, "open"] = future_changed.loc[source_index + 1 :, "close"]
    future_changed.loc[source_index + 1 :, "high"] = future_changed.loc[source_index + 1 :, "close"] + 150
    future_changed.loc[source_index + 1 :, "low"] = future_changed.loc[source_index + 1 :, "close"] - 150
    altered, _, _, _ = build_feature_dataset(future_changed, symbol="BTC/USDT")
    pd.testing.assert_frame_equal(features.loc[:timestamp].iloc[:-1], altered.loc[:timestamp].iloc[:-1])
    assert np.isfinite(forward_return.loc[timestamp])


def test_chronological_split_purges_overlapping_labels() -> None:
    features, _, _, _ = build_feature_dataset(hourly_candles(), symbol="BTC/USDT")
    split = chronological_split(features.index)
    assert split.train.is_monotonic_increasing
    assert split.validation.is_monotonic_increasing
    assert split.out_of_sample.is_monotonic_increasing
    assert split.train[-1] < split.validation[0]
    assert split.validation[-1] < split.out_of_sample[0]
    assert (split.validation[0] - split.train[-1]).total_seconds() >= (HORIZON_BARS + 1) * 3600
    assert (split.out_of_sample[0] - split.validation[-1]).total_seconds() >= (HORIZON_BARS + 1) * 3600
    for fold in split.walk_forward_folds:
        assert fold.train[-1] < fold.validation[0]
        assert (fold.validation[0] - fold.train[-1]).total_seconds() >= (HORIZON_BARS + 1) * 3600


def test_training_persists_artifacts_metrics_and_oos_predictions(tmp_path) -> None:
    candles = hourly_candles(1_700)
    results = train_models_on_candles(
        candles,
        symbol="BTC/USDT",
        data_source="deterministic synthetic test fixture",
        db_path=tmp_path / "runs.sqlite",
        artifact_root=tmp_path / "artifacts",
    )
    assert {r["model_name"] for r in results} == {
        "logistic_regression", "hist_gradient_boosting"
    }
    assert results[0]["metadata"]["experiment_config_sha256"] != results[1]["metadata"]["experiment_config_sha256"]
    conn = sqlite3.connect(tmp_path / "runs.sqlite")
    try:
        assert conn.execute("SELECT COUNT(*) FROM model_runs WHERE status='completed'").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM model_predictions WHERE split_name='out_of_sample'").fetchone()[0] > 0
    finally:
        conn.close()
    for result in results:
        artifact = tmp_path / "artifacts" / "models" / result["model_version"]
        metadata = json.loads((artifact / "metadata.json").read_text(encoding="utf-8"))
        assert (artifact / "model.joblib").is_file()
        assert (artifact / "out_of_sample_predictions.csv").is_file()
        assert metadata["dataset_sha256"]
        assert metadata["code_revision"] is None or isinstance(metadata["code_revision"], str)
        assert result["metrics"]["classification"]["out_of_sample"]["brier_score"] is not None
