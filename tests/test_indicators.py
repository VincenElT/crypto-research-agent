"""Unit tests for SMA and 30-day momentum calculations."""

from __future__ import annotations

import pandas as pd
import pytest

from crypto_research.exceptions import InsufficientHistoryError
from crypto_research.indicators import (
    add_indicators,
    candles_for_calendar_days,
    latest_snapshot,
)


def _linear_candles(length: int, start: float = 100.0, step: float = 1.0) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=length, freq="D", tz="UTC")
    closes = [start + i * step for i in range(length)]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1.0] * length,
        }
    )


def test_daily_momentum_uses_30_candles() -> None:
    assert candles_for_calendar_days(86_400) == 30


def test_hourly_momentum_uses_720_candles() -> None:
    assert candles_for_calendar_days(3_600) == 720


def test_sma_and_momentum_on_known_series() -> None:
    candles = _linear_candles(220, start=100.0, step=1.0)
    result = add_indicators(candles, timeframe_seconds=86_400)
    last = result.iloc[-1]

    expected_sma_50 = sum(range(270, 320)) / 50  # last 50 closes: 270..319
    expected_sma_200 = sum(range(120, 320)) / 200  # last 200 closes: 120..319
    expected_momentum = (319.0 / 289.0) - 1.0

    assert last["close"] == 319.0
    assert last["sma_50"] == pytest.approx(expected_sma_50)
    assert last["sma_200"] == pytest.approx(expected_sma_200)
    assert last["momentum_30d"] == pytest.approx(expected_momentum)


def test_insufficient_history_raises() -> None:
    candles = _linear_candles(50)
    with pytest.raises(InsufficientHistoryError, match="Need at least"):
        add_indicators(candles, timeframe_seconds=86_400)


def test_latest_snapshot_trend_label() -> None:
    candles = _linear_candles(220, start=100.0, step=1.0)
    result = add_indicators(candles, timeframe_seconds=86_400)
    snapshot = latest_snapshot(result)
    assert "uptrend" in snapshot.trend
    assert snapshot.close == 319.0
