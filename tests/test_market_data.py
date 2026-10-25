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


# =========================================================================
# Tests for BinanceVisionClient (Response parsing, pagination, errors, TLS)
# =========================================================================

import io
import json
import urllib.error
from unittest.mock import patch
from crypto_research.market_data import BinanceVisionClient, create_ccxt_binance_exchange


def _mock_http_response(content: bytes) -> MagicMock:
    resp = MagicMock()
    resp.read.return_value = content
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = None
    return resp


def test_binance_vision_response_parsing() -> None:
    raw = [
        [1700000000000, "50000.0", "51000.0", "49000.0", "50500.0", "123.45", 1700086399999, "6234125.0", 100, "50.0", "2500000.0", "0"],
        [1700086400000, "50500.0", "52000.0", "50000.0", "51500.0", "234.56", 1700172799999, "12000000.0", 200, "100.0", "5150000.0", "0"],
    ]
    client = BinanceVisionClient()

    with patch("urllib.request.urlopen", return_value=_mock_http_response(json.dumps(raw).encode("utf-8"))):
        bars = client.fetch_ohlcv("BTC/USDT", timeframe="1d", limit=1)

    assert len(bars) == 2
    assert bars[0] == [1700000000000, 50000.0, 51000.0, 49000.0, 50500.0, 123.45]
    assert bars[1] == [1700086400000, 50500.0, 52000.0, 50000.0, 51500.0, 234.56]


def test_binance_vision_symbol_normalization() -> None:
    client = BinanceVisionClient()
    assert client.normalize_symbol("BTC/USDT") == "BTCUSDT"
    assert client.normalize_symbol("eth-usdt") == "ETHUSDT"
    assert client.normalize_symbol("sol_usdt") == "SOLUSDT"


def test_binance_vision_pagination_backward() -> None:
    # Simulates limit=1200 requiring 2 calls (1000 + 201)
    # Chunk 1: open times 1000..1999
    chunk1 = [[t * 86_400_000, "100", "110", "90", "105", "10"] for t in range(1000, 2000)]
    # Chunk 2: open times 799..999
    chunk2 = [[t * 86_400_000, "100", "110", "90", "105", "10"] for t in range(799, 1000)]

    call_count = 0

    def mock_urlopen(req: Any, **kwargs: Any) -> MagicMock:
        nonlocal call_count
        call_count += 1
        url = req.full_url
        if "endTime" not in url:
            # First request: returns chunk 1
            return _mock_http_response(json.dumps(chunk1).encode("utf-8"))
        else:
            # Second request with endTime: returns chunk 2
            return _mock_http_response(json.dumps(chunk2).encode("utf-8"))

    client = BinanceVisionClient()
    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        bars = client.fetch_ohlcv("BTC/USDT", timeframe="1d", limit=1200)

    assert call_count == 2
    assert len(bars) == 1201  # limit + 1
    # Check chronological ordering and continuity
    for i in range(1, len(bars)):
        assert bars[i][0] > bars[i - 1][0]


def test_binance_vision_malformed_json_raises_market_data_error() -> None:
    client = BinanceVisionClient()
    with patch("urllib.request.urlopen", return_value=_mock_http_response(b"<html>Bad Gateway</html>")):
        with pytest.raises(MarketDataError, match="Malformed JSON"):
            client.fetch_ohlcv("BTC/USDT", timeframe="1d")


def test_binance_vision_malformed_candle_row() -> None:
    client = BinanceVisionClient()
    # Row with non-numeric price
    bad_data = [[1700000000000, "NaN_PRICE", "51000", "49000", "50000", "10"]]
    with patch("urllib.request.urlopen", return_value=_mock_http_response(json.dumps(bad_data).encode("utf-8"))):
        with pytest.raises(MarketDataError, match="Non-numeric values"):
            client.fetch_ohlcv("BTC/USDT", timeframe="1d")

    # Truncated row
    truncated = [[1700000000000, "50000"]]
    with patch("urllib.request.urlopen", return_value=_mock_http_response(json.dumps(truncated).encode("utf-8"))):
        with pytest.raises(MarketDataError, match="Malformed candle row"):
            client.fetch_ohlcv("BTC/USDT", timeframe="1d")


def test_binance_vision_http_429_rate_limit() -> None:
    client = BinanceVisionClient()
    err_body = io.BytesIO(b'{"code": -1003, "msg": "Way too many requests"}')
    http_err = urllib.error.HTTPError(
        url="https://data-api.binance.vision/api/v3/klines",
        code=429,
        msg="Too Many Requests",
        hdrs=None,  # type: ignore
        fp=err_body,
    )
    with patch("urllib.request.urlopen", side_effect=http_err):
        with pytest.raises(MarketDataError, match="HTTP 429"):
            client.fetch_ohlcv("BTC/USDT", timeframe="1d")


def test_binance_vision_tls_error() -> None:
    client = BinanceVisionClient()
    url_err = urllib.error.URLError(reason="[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
    with patch("urllib.request.urlopen", side_effect=url_err):
        with pytest.raises(MarketDataError, match="TLS verification failed"):
            client.fetch_ohlcv("BTC/USDT", timeframe="1d")


def test_binance_vision_network_timeout() -> None:
    client = BinanceVisionClient()
    with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
        with pytest.raises(MarketDataError, match="Network timeout"):
            client.fetch_ohlcv("BTC/USDT", timeframe="1d")


def test_ccxt_binance_configured_with_vision_url() -> None:
    exchange = create_ccxt_binance_exchange()
    assert exchange.urls["api"]["public"] == "https://data-api.binance.vision/api/v3"
    assert exchange.options.get("fetchMarkets") == ["spot"]

