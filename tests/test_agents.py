"""Unit tests for Scout/Skeptic orchestration with a fake completer."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from crypto_research.config import Settings
from crypto_research.indicators import IndicatorSnapshot
from crypto_research.market_context import build_market_context
from crypto_research.exceptions import AgentResponseError
from crypto_research.research import run_dual_agents
from crypto_research.schemas import ScoutProposal, SkepticReview


def _settings() -> Settings:
    return Settings(
        exchange_id="binance",
        symbol="BTC/USDT",
        timeframe="1d",
        candle_limit=400,
        openai_model="gpt-4o-mini",
        openai_api_key_configured=True,
    )


def _snapshot() -> IndicatorSnapshot:
    return IndicatorSnapshot(
        as_of=pd.Timestamp("2024-06-01T00:00:00Z"),
        close=100.0,
        sma_50=90.0,
        sma_200=80.0,
        momentum_30d=0.1,
        trend="uptrend (SMA 50 above SMA 200)",
    )


def _candles() -> pd.DataFrame:
    timestamps = pd.date_range("2024-05-23", periods=10, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0] * 10,
            "high": [101.0] * 10,
            "low": [99.0] * 10,
            "close": [100.0] * 10,
            "volume": [1.0] * 10,
        }
    )


def test_market_context_uses_closed_snapshot_only() -> None:
    context = build_market_context(_settings(), _snapshot(), _candles())
    assert context["as_of_utc"].startswith("2024-06-01")
    assert context["close"] == 100.0
    assert len(context["recent_closed_bars"]) == 10
    assert context["recent_closed_bars"][-1]["timestamp_utc"].startswith("2024-06-01")


def test_dual_agents_validate_fake_model_output() -> None:
    calls: list[str] = []

    def fake_complete(*, schema: type, **_: Any) -> ScoutProposal | SkepticReview:
        calls.append(schema.__name__)
        if schema is ScoutProposal:
            return ScoutProposal(
                action="HOLD",
                confidence=0.3,
                thesis="Need more than a short uptrend snapshot.",
                evidence=["only 10 bars of context were provided"],
                risks=["sample is too small for a directional call"],
                missing_data=["full SMA history"],
            )
        return SkepticReview(
            verdict="APPROVE",
            agrees_with_action=True,
            rationale="HOLD is appropriate given limited context.",
            missing_evidence=["no volatility"],
            weaknesses=["Scout restated the payload without new structure"],
            confidence=0.6,
        )

    result = run_dual_agents(
        build_market_context(_settings(), _snapshot(), _candles()),
        model_name="gpt-4o-mini",
        complete=fake_complete,
    )
    assert calls == ["ScoutProposal", "SkepticReview"]
    assert result.proposal.action == "HOLD"
    assert result.critique.verdict == "APPROVE"


def test_dual_agents_reject_invalid_fake_output() -> None:
    def fake_complete(*, schema: type, **_: Any) -> dict[str, Any]:
        if schema is ScoutProposal:
            return {
                "action": "BUY",
                "confidence": 0.9,
                "thesis": "x",
                "evidence": ["sma"],
                "risks": ["r"],
                "missing_data": [],
            }
        return {"verdict": "NOPE"}

    with pytest.raises(AgentResponseError):
        run_dual_agents(
            build_market_context(_settings(), _snapshot(), _candles()),
            model_name="gpt-4o-mini",
            complete=fake_complete,
        )
