"""Unit tests for candle conversion, incomplete-bar handling, and gaps."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from crypto_research.config import Settings
from crypto_research.exceptions import MarketDataError
from crypto_research.market_data import (
    drop_incomplete_candle,
    fetch_ohlcv,
    find_missing_candle_gaps,
    ohlcv_to_dataframe,
)


def _mock_exchange(ohlcv_rows: list[list[Any]]) -> MagicMock:
    exchange = MagicMock()
    exchange.timeframes = {"1d": "1d"}
    exchange.parse_timeframe.return_value = 86_400
    exchange.fetch_ohlcv.return_value = ohlcv_rows
    return exchange


def _settings() -> Settings:
    return Settings("binance", "BTC/USDT", "1d", 400, "gpt-4o-mini", True)


def test_ohlcv_to_dataframe_parses_utc_timestamps() -> None:
    open_ms = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    frame = ohlcv_to_dataframe([[open_ms, 1, 2, 0.5, 1.5, 10]])
    assert frame.iloc[0]["close"] == 1.5
    assert str(frame.iloc[0]["timestamp"].tz) == "UTC"


def test_drop_incomplete_candle_removes_open_bar() -> None:
    open_ms = int(datetime(2024, 1, 2, tzinfo=timezone.utc).timestamp() * 1000)
    prior_ms = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    frame = ohlcv_to_dataframe(
        [
            [prior_ms, 1, 1, 1, 1, 1],
            [open_ms, 2, 2, 2, 2, 2],
        ]
    )
    now = datetime(2024, 1, 2, 12, tzinfo=timezone.utc)
    closed = drop_incomplete_candle(frame, timeframe_ms=86_400_000, now=now)
    assert len(closed) == 1
    assert closed.iloc[0]["close"] == 1


def test_detects_missing_daily_candle() -> None:
    first = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    third = int(datetime(2024, 1, 3, tzinfo=timezone.utc).timestamp() * 1000)
    frame = ohlcv_to_dataframe(
        [
            [first, 1, 1, 1, 1, 1],
            [third, 2, 2, 2, 2, 2],
        ]
    )
    gaps = find_missing_candle_gaps(frame, 86_400_000)
    assert len(gaps) == 1


def test_fetch_ohlcv_rejects_non_positive_price() -> None:
    first = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    exchange = _mock_exchange([[first, 100.0, 105.0, 95.0, -10.0, 50.0]])
    with pytest.raises(MarketDataError, match="non-positive OHLC prices"):
        fetch_ohlcv(_settings(), exchange=exchange)


def test_fetch_ohlcv_rejects_infinite_price() -> None:
    first = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    exchange = _mock_exchange([[first, 100.0, float("inf"), 95.0, 100.0, 50.0]])
    with pytest.raises(MarketDataError, match="non-finite OHLC prices"):
        fetch_ohlcv(_settings(), exchange=exchange)


def test_fetch_ohlcv_rejects_inverted_high_low() -> None:
    first = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    exchange = _mock_exchange([[first, 100.0, 90.0, 110.0, 100.0, 50.0]])  # high < low
    with pytest.raises(MarketDataError, match="high was lower than low"):
        fetch_ohlcv(_settings(), exchange=exchange)
