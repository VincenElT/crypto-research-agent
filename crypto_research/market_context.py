"""Build the structured market payload sent to Scout and Skeptic.

Only closed-candle fields are included so agents cannot see future bars.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from crypto_research.config import Settings
from crypto_research.indicators import IndicatorSnapshot

RECENT_CLOSED_BARS = 10


def build_market_context(
    settings: Settings,
    snapshot: IndicatorSnapshot,
    candles: pd.DataFrame,
    *,
    recent_bars: int = RECENT_CLOSED_BARS,
) -> dict[str, Any]:
    """Return JSON-serializable market facts as of the last closed candle."""

    if candles.empty:
        raise ValueError("Candles DataFrame is empty.")

    # Strictly align candles up to snapshot.as_of to prevent any future bar leakage
    aligned = candles[candles["timestamp"] <= snapshot.as_of]
    if aligned.empty or aligned.iloc[-1]["timestamp"] != snapshot.as_of:
        raise ValueError(
            f"Candles do not contain a closed bar matching snapshot timestamp {snapshot.as_of.isoformat()}."
        )

    recent = aligned.tail(recent_bars)
    rows: list[dict[str, Any]] = []
    for _, row in recent.iterrows():
        rows.append(
            {
                "timestamp_utc": pd.Timestamp(row["timestamp"]).isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
        )
    return {
        "exchange": settings.exchange_id,
        "symbol": settings.symbol,
        "timeframe": settings.timeframe,
        "as_of_utc": snapshot.as_of.isoformat(),
        "close": snapshot.close,
        "sma_50": snapshot.sma_50,
        "sma_200": snapshot.sma_200,
        "momentum_30d": snapshot.momentum_30d,
        "trend": snapshot.trend,
        "recent_closed_bars": rows,
        "constraints": [
            "Use only these fields. Do not invent prices, news, or future candles.",
            "The as_of_utc timestamp is the last closed bar; later information is forbidden.",
            "Confidence is not a calibrated probability of profit.",
            "You cannot place, cancel, or modify exchange orders.",
        ],
    }
