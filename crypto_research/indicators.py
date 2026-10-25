"""Deterministic technical indicators computed with pandas.

Windows are measured in candles. SMA 50 and SMA 200 therefore mean 50 and
200 closed bars. 30-day momentum is percent change over the number of
candles that fit in 30 calendar days for the active timeframe.

All values are computed from past and current closed bars only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from crypto_research.exceptions import InsufficientHistoryError

SMA_FAST_WINDOW = 50
SMA_SLOW_WINDOW = 200
MOMENTUM_DAYS = 30
SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class IndicatorSnapshot:
    """Indicator values as of a single closed candle."""

    as_of: pd.Timestamp
    close: float
    sma_50: float
    sma_200: float
    momentum_30d: float
    trend: str


def candles_for_calendar_days(timeframe_seconds: int, days: int = MOMENTUM_DAYS) -> int:
    """Convert a calendar-day lookback into a whole number of candles."""

    if timeframe_seconds <= 0:
        raise ValueError("timeframe_seconds must be positive.")
    periods = int((days * SECONDS_PER_DAY) / timeframe_seconds)
    if periods < 1:
        raise ValueError(
            f"Timeframe {timeframe_seconds}s is longer than {days} days; "
            "cannot compute 30-day momentum."
        )
    return periods


def add_indicators(
    candles: pd.DataFrame,
    *,
    timeframe_seconds: int,
    sma_fast: int = SMA_FAST_WINDOW,
    sma_slow: int = SMA_SLOW_WINDOW,
    momentum_days: int = MOMENTUM_DAYS,
) -> pd.DataFrame:
    """Return a copy of `candles` with SMA and momentum columns.

    Raises:
        InsufficientHistoryError: If there are not enough closed candles.
    """

    if candles.empty:
        raise InsufficientHistoryError("No closed candles were provided.")

    momentum_periods = candles_for_calendar_days(timeframe_seconds, momentum_days)
    required = max(sma_slow, momentum_periods + 1)
    if len(candles) < required:
        raise InsufficientHistoryError(
            f"Need at least {required} closed candles to compute SMA {sma_slow} "
            f"and {momentum_days}-day momentum; received {len(candles)}."
        )

    frame = candles.copy()
    frame["sma_50"] = frame["close"].rolling(window=sma_fast, min_periods=sma_fast).mean()
    frame["sma_200"] = frame["close"].rolling(window=sma_slow, min_periods=sma_slow).mean()
    frame["momentum_30d"] = frame["close"].pct_change(periods=momentum_periods)
    return frame


def latest_snapshot(frame: pd.DataFrame) -> IndicatorSnapshot:
    """Extract indicator values from the most recent closed candle.

    Raises:
        InsufficientHistoryError: If frame is empty or indicators are not populated
            on the latest closed candle.
    """

    if frame.empty:
        raise InsufficientHistoryError("No closed candles were provided.")

    row = frame.iloc[-1]
    for column in ("sma_50", "sma_200", "momentum_30d", "close"):
        val = row.get(column)
        if val is None or pd.isna(val) or not math.isfinite(float(val)):
            raise InsufficientHistoryError(
                f"Indicators are not defined on the latest closed candle "
                f"(missing or non-finite {column!r} as of {row['timestamp']})."
            )

    sma_50 = float(row["sma_50"])
    sma_200 = float(row["sma_200"])
    if sma_50 > sma_200:
        trend = "uptrend (SMA 50 above SMA 200)"
    elif sma_50 < sma_200:
        trend = "downtrend (SMA 50 below SMA 200)"
    else:
        trend = "flat (SMA 50 equals SMA 200)"
    return IndicatorSnapshot(
        as_of=pd.Timestamp(row["timestamp"]),
        close=float(row["close"]),
        sma_50=sma_50,
        sma_200=sma_200,
        momentum_30d=float(row["momentum_30d"]),
        trend=trend,
    )
