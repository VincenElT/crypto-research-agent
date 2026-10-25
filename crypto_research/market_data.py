"""Fetch historical OHLCV candles from a public CCXT market-data API.

This module never authenticates to an exchange and never places orders.
Decisions later in the pipeline must use only closed candles so incomplete
bars cannot leak future information (look-ahead bias).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

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


def create_public_exchange(exchange_id: str) -> ccxt.Exchange:
    """Instantiate a CCXT exchange client for public market data only."""

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


def fetch_ohlcv(settings: Settings, exchange: ccxt.Exchange | None = None) -> pd.DataFrame:
    """Fetch closed historical candles for the configured market.

    Args:
        settings: Exchange, symbol, timeframe, and candle limit.
        exchange: Optional pre-built CCXT client (useful in tests).

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
    exchange: ccxt.Exchange, settings: Settings
) -> list[list[Any]]:
    last_error: Exception | None = None
    for attempt in range(1, MAX_FETCH_ATTEMPTS + 1):
        try:
            return exchange.fetch_ohlcv(
                settings.symbol,
                timeframe=settings.timeframe,
                limit=settings.candle_limit,
            )
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
