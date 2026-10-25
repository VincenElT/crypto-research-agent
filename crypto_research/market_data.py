"""Fetch historical OHLCV candles from a public CCXT market-data API or Binance Vision.

This module never authenticates to an exchange and never places orders.
Decisions later in the pipeline must use only closed candles so incomplete
bars cannot leak future information (look-ahead bias).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import ssl
import time
from typing import Any
import urllib.error
import urllib.request

import ccxt
import numpy as np
import pandas as pd
from ccxt.base.errors import (
    BaseError,
    ExchangeNotAvailable,
    NetworkError,
    RateLimitExceeded,
)

from crypto_research.config import Settings
from crypto_research.exceptions import ConfigurationError, MarketDataError

OHLCV_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
MAX_FETCH_ATTEMPTS = 4


class BinanceVisionClient:
    """Isolated public market data adapter for Binance Vision endpoints.

    Uses Python's standard urllib HTTPS client with TLS certificate verification
    strictly enabled. Never authenticates, never places orders, and queries only
    public market data at https://data-api.binance.vision/api/v3.
    """

    BASE_URL: str = "https://data-api.binance.vision/api/v3"
    timeframes: dict[str, str] = {
        "1m": "1m",
        "3m": "3m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "6h": "6h",
        "8h": "8h",
        "12h": "12h",
        "1d": "1d",
        "3d": "3d",
        "1w": "1w",
        "1M": "1M",
    }
    rateLimit: int = 100

    def __init__(self, base_url: str | None = None, timeout_seconds: float = 15.0) -> None:
        self.base_url = (base_url or self.BASE_URL).rstrip("/")
        self.timeout_seconds = timeout_seconds

    def parse_timeframe(self, timeframe: str) -> int:
        """Parse timeframe string into seconds."""
        units = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800, "M": 2592000}
        num_str = timeframe[:-1]
        unit = timeframe[-1]
        if not num_str.isdigit() or unit not in units:
            raise ConfigurationError(f"Unsupported or invalid timeframe format: {timeframe!r}")
        return int(num_str) * units[unit]

    def sleep(self, milliseconds: int) -> None:
        time.sleep(milliseconds / 1000.0)

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """Convert 'BTC/USDT' or 'BTC-USDT' to Binance market format 'BTCUSDT'."""
        return symbol.replace("/", "").replace("-", "").replace("_", "").upper()

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        since: int | None = None,
        limit: int = 500,
        params: dict[str, Any] | None = None,
    ) -> list[list[Any]]:
        """Fetch historical OHLCV candles with backward pagination and TLS verification."""
        if timeframe not in self.timeframes:
            supported = ", ".join(sorted(self.timeframes))
            raise ConfigurationError(
                f"Timeframe {timeframe!r} is not supported by Binance Vision. Supported: {supported}."
            )

        norm_sym = self.normalize_symbol(symbol)
        ctx = ssl.create_default_context()
        all_bars: list[list[Any]] = []
        # Request limit + 1 so that after drop_incomplete_candle drops the in-progress bar,
        # exactly limit closed candles remain.
        target_count = max(1, limit + 1)
        remaining = target_count
        end_time: int | None = None

        while remaining > 0:
            fetch_count = min(1000, remaining)
            url = (
                f"{self.base_url}/klines?"
                f"symbol={norm_sym}&interval={timeframe}&limit={fetch_count}"
            )
            if end_time is not None:
                url += f"&endTime={end_time}"
            if since is not None and end_time is None:
                url += f"&startTime={since}"

            req = urllib.request.Request(
                url,
                headers={"User-Agent": "crypto-research-agent/1.0"},
            )

            try:
                with urllib.request.urlopen(req, context=ctx, timeout=self.timeout_seconds) as resp:
                    raw_data = resp.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                detail = f"HTTP {exc.code} {exc.reason}"
                try:
                    err_json = json.loads(body)
                    if isinstance(err_json, dict) and "msg" in err_json:
                        detail = f"{detail}: {err_json['msg']}"
                except Exception:
                    pass
                raise MarketDataError(
                    f"Binance Vision API HTTP error fetching {symbol}: {detail}"
                ) from exc
            except urllib.error.URLError as exc:
                detail = str(exc.reason)
                if "certificate" in detail.lower() or "ssl" in detail.lower():
                    raise MarketDataError(
                        f"TLS verification failed while fetching {symbol} from Binance Vision: {detail}"
                    ) from exc
                raise MarketDataError(
                    f"Network connection failed while fetching {symbol} from Binance Vision: {detail}"
                ) from exc
            except TimeoutError as exc:
                raise MarketDataError(
                    f"Network timeout ({self.timeout_seconds}s) while fetching {symbol} from Binance Vision."
                ) from exc

            try:
                data = json.loads(raw_data)
            except json.JSONDecodeError as exc:
                raise MarketDataError(
                    f"Malformed JSON response from Binance Vision for {symbol}: {exc}"
                ) from exc

            if not isinstance(data, list):
                raise MarketDataError(
                    f"Unexpected response structure from Binance Vision for {symbol}: expected list of candles."
                )

            if not data:
                break

            parsed_chunk: list[list[Any]] = []
            for item in data:
                if not isinstance(item, (list, tuple)) or len(item) < 6:
                    raise MarketDataError(
                        f"Malformed candle row from Binance Vision for {symbol}: {item}"
                    )
                try:
                    t_ms = int(item[0])
                    o = float(item[1])
                    h = float(item[2])
                    l = float(item[3])
                    c = float(item[4])
                    v = float(item[5])
                except (ValueError, TypeError) as exc:
                    raise MarketDataError(
                        f"Non-numeric values in candle from Binance Vision for {symbol}: {exc}"
                    ) from exc
                parsed_chunk.append([t_ms, o, h, l, c, v])

            all_bars = parsed_chunk + all_bars
            remaining -= len(parsed_chunk)
            earliest_open = parsed_chunk[0][0]
            end_time = earliest_open - 1

            if len(parsed_chunk) < fetch_count:
                break

        seen_ts: set[int] = set()
        deduped: list[list[Any]] = []
        for bar in all_bars:
            if bar[0] not in seen_ts:
                seen_ts.add(bar[0])
                deduped.append(bar)

        deduped.sort(key=lambda x: x[0])
        return deduped[-target_count:]


def create_ccxt_binance_exchange() -> ccxt.binance:
    """Create a CCXT Binance exchange configured to use data-api.binance.vision."""
    return ccxt.binance(
        {
            "enableRateLimit": True,
            "options": {"fetchMarkets": ["spot"]},
            "urls": {
                "api": {
                    "public": "https://data-api.binance.vision/api/v3",
                }
            },
        }
    )


def _exception_chain_text(exc: BaseException) -> str:
    """Flatten CCXT/requests exception causes so TLS failures are visible."""

    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(str(current))
        current = current.__cause__ or current.__context__
    return " | ".join(parts)


def create_public_exchange(exchange_id: str) -> ccxt.Exchange | BinanceVisionClient:
    """Instantiate a public market data client.

    For 'binance', returns BinanceVisionClient using the verified public
    data-api.binance.vision endpoint with strict TLS verification and pagination.
    For other supported exchanges, instantiates CCXT for public market data.
    """
    if exchange_id.lower() == "binance":
        return BinanceVisionClient()

    try:
        exchange_class = getattr(ccxt, exchange_id)
    except AttributeError as exc:
        raise ConfigurationError(
            f"Unknown CCXT exchange id {exchange_id!r}. Use a supported public "
            "exchange such as 'binance'."
        ) from exc

    exchange = exchange_class({"enableRateLimit": True})
    if not exchange.has.get("fetchOHLCV"):
        raise ConfigurationError(
            f"Exchange {exchange_id!r} does not support fetchOHLCV in CCXT."
        )
    return exchange


def ohlcv_to_dataframe(ohlcv: list[list[Any]]) -> pd.DataFrame:
    """Convert CCXT OHLCV rows to a UTC-indexed DataFrame."""

    if not ohlcv:
        return pd.DataFrame(columns=list(OHLCV_COLUMNS))

    frame = pd.DataFrame(ohlcv, columns=list(OHLCV_COLUMNS))
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("timestamp").reset_index(drop=True)


def drop_incomplete_candle(
    frame: pd.DataFrame, timeframe_ms: int, *, now: datetime | None = None
) -> pd.DataFrame:
    """Drop the last bar if it has not closed yet."""

    if frame.empty:
        return frame
    now = now or datetime.now(timezone.utc)
    last_open = frame.iloc[-1]["timestamp"].to_pydatetime()
    if last_open.tzinfo is None:
        last_open = last_open.replace(tzinfo=timezone.utc)
    close_time_ms = int(last_open.timestamp() * 1000) + timeframe_ms
    now_ms = int(now.timestamp() * 1000)
    if close_time_ms > now_ms:
        return frame.iloc[:-1].reset_index(drop=True)
    return frame


def find_missing_candle_gaps(frame: pd.DataFrame, timeframe_ms: int) -> list[str]:
    """Return human-readable gap descriptions when candles are missing."""

    if len(frame) < 2:
        return []
    expected = pd.Timedelta(milliseconds=timeframe_ms)
    deltas = frame["timestamp"].diff().iloc[1:]
    gap_mask = deltas > expected * 1.5
    gaps: list[str] = []
    for idx in deltas[gap_mask].index:
        previous = frame.loc[idx - 1, "timestamp"]
        current = frame.loc[idx, "timestamp"]
        gaps.append(f"{previous.isoformat()} -> {current.isoformat()}")
    return gaps


def fetch_ohlcv(
    settings: Settings,
    exchange: ccxt.Exchange | BinanceVisionClient | None = None,
) -> pd.DataFrame:
    """Fetch closed historical candles for the configured market.

    Args:
        settings: Exchange, symbol, timeframe, and candle limit.
        exchange: Optional pre-built exchange client (CCXT or BinanceVisionClient).

    Returns:
        DataFrame of closed OHLCV candles in UTC.

    Raises:
        MarketDataError: On API failures, empty results, or unusable candles.
        ConfigurationError: If the timeframe is not supported by the exchange.
    """

    client = exchange or create_public_exchange(settings.exchange_id)
    if settings.timeframe not in client.timeframes:
        supported = ", ".join(sorted(client.timeframes))
        raise ConfigurationError(
            f"Timeframe {settings.timeframe!r} is not supported by "
            f"{settings.exchange_id}. Supported: {supported}."
        )

    timeframe_ms = int(client.parse_timeframe(settings.timeframe) * 1000)
    raw = _fetch_ohlcv_with_retries(client, settings)
    frame = ohlcv_to_dataframe(raw)
    if frame.empty:
        raise MarketDataError(
            f"No OHLCV data returned for {settings.symbol} on {settings.exchange_id}."
        )
    if frame[["open", "high", "low", "close"]].isna().any().any():
        raise MarketDataError(
            "Market data contained non-numeric OHLC values and cannot be used."
        )
    numeric_ohlc = frame[["open", "high", "low", "close"]].to_numpy()
    if np.isinf(numeric_ohlc).any():
        raise MarketDataError(
            "Market data contained non-finite OHLC prices and cannot be used."
        )
    if (numeric_ohlc <= 0).any():
        raise MarketDataError(
            "Market data contained non-positive OHLC prices and cannot be used."
        )
    if (frame["high"] < frame["low"]).any():
        raise MarketDataError(
            "Market data contained invalid candles where high was lower than low."
        )

    closed = drop_incomplete_candle(frame, timeframe_ms)
    if closed.empty:
        raise MarketDataError(
            "Only an in-progress candle was returned; wait for a closed bar."
        )

    # If extra candles were fetched to guarantee candle_limit after dropping the open bar, slice to target
    if len(closed) > settings.candle_limit:
        closed = closed.tail(settings.candle_limit).reset_index(drop=True)

    gaps = find_missing_candle_gaps(closed, timeframe_ms)
    if gaps:
        preview = "; ".join(gaps[:5])
        extra = f" ({len(gaps) - 5} more)" if len(gaps) > 5 else ""
        raise MarketDataError(
            f"Missing candles detected for {settings.symbol} {settings.timeframe}: "
            f"{preview}{extra}. Refusing to compute indicators on a gapped series."
        )
    return closed


def _fetch_ohlcv_with_retries(
    exchange: ccxt.Exchange | BinanceVisionClient, settings: Settings
) -> list[list[Any]]:
    last_error: Exception | None = None
    for attempt in range(1, MAX_FETCH_ATTEMPTS + 1):
        try:
            return exchange.fetch_ohlcv(
                settings.symbol,
                timeframe=settings.timeframe,
                limit=settings.candle_limit,
            )
        except MarketDataError:
            # Re-raise explicit MarketDataErrors immediately without CCXT wraps
            raise
        except RateLimitExceeded as exc:
            last_error = exc
            if attempt == MAX_FETCH_ATTEMPTS:
                break
            exchange.sleep(exchange.rateLimit * attempt)
        except (NetworkError, ExchangeNotAvailable) as exc:
            last_error = exc
            detail = _exception_chain_text(exc).lower()
            if "certificate" in detail or "ssl" in detail:
                raise MarketDataError(
                    f"TLS failed while fetching {settings.symbol} from "
                    f"{settings.exchange_id}: {_exception_chain_text(exc)}. "
                    "A local proxy or antivirus HTTPS scanner can cause this. "
                    "Public OHLCV still does not need exchange API keys; do not "
                    "disable certificate checks."
                ) from exc
            if attempt == MAX_FETCH_ATTEMPTS:
                break
            exchange.sleep(1000 * attempt)
        except BaseError as exc:
            raise MarketDataError(
                f"CCXT failed to fetch {settings.symbol} from {settings.exchange_id}: {exc}"
            ) from exc
    detail = _exception_chain_text(last_error) if last_error else "unknown error"
    raise MarketDataError(
        f"Exhausted {MAX_FETCH_ATTEMPTS} attempts fetching {settings.symbol} "
        f"from {settings.exchange_id}: {detail}"
    ) from last_error
