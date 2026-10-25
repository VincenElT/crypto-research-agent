"""Unit tests for paper-trading candidate creation."""

from __future__ import annotations

import pandas as pd
import pytest

from crypto_research.exceptions import PaperTradingError
from crypto_research.indicators import IndicatorSnapshot
from crypto_research.paper_trading import create_paper_trade
from crypto_research.risk import RiskEvaluation
from crypto_research.schemas import ScoutProposal, SkepticReview


def _snapshot(close: float = 50_000.0) -> IndicatorSnapshot:
    return IndicatorSnapshot(
        as_of=pd.Timestamp("2024-06-01T00:00:00Z"),
        close=close,
        sma_50=48_000.0,
        sma_200=45_000.0,
        momentum_30d=0.08,
        trend="uptrend (SMA 50 above SMA 200)",
    )


def _proposal(action: str = "BUY") -> ScoutProposal:
    return ScoutProposal(
        action=action,  # type: ignore[arg-type]
        confidence=0.8,
        thesis="Breakout above SMA 50",
        evidence=["SMA 50 cross"],
        risks=["macro volatility"],
        missing_data=[],
    )


def _critique() -> SkepticReview:
    return SkepticReview(
        verdict="APPROVE",
        agrees_with_action=True,
        rationale="Matches trend",
        missing_evidence=[],
        weaknesses=["sample does not include order book liquidity"],
        confidence=0.75,
    )


def test_create_paper_trade_success_buy() -> None:
    risk_eval = RiskEvaluation(
        passed=True,
        proposed_action="BUY",
        approved_size_usd=1000.0,
    )
    candidate = create_paper_trade(
        symbol="BTC/USDT",
        snapshot=_snapshot(50_000.0),
        proposal=_proposal("BUY"),
        critique=_critique(),
        risk_evaluation=risk_eval,
    )
    assert candidate.symbol == "BTC/USDT"
    assert candidate.action == "BUY"
    assert candidate.reference_price == 50_000.0
    assert candidate.size_usd == 1000.0
    assert candidate.size_asset == pytest.approx(0.02)
    assert candidate.is_live is False
    assert candidate.status == "CANDIDATE"
    assert candidate.candidate_id.startswith("pt_")


def test_create_paper_trade_success_sell() -> None:
    risk_eval = RiskEvaluation(
        passed=True,
        proposed_action="SELL",
        approved_size_usd=500.0,
    )
    candidate = create_paper_trade(
        symbol="ETH/USDT",
        snapshot=_snapshot(2500.0),
        proposal=_proposal("SELL"),
        critique=_critique(),
        risk_evaluation=risk_eval,
    )
    assert candidate.action == "SELL"
    assert candidate.size_usd == 500.0
    assert candidate.size_asset == pytest.approx(0.2)
    assert candidate.is_live is False


def test_create_paper_trade_fails_if_risk_rejected() -> None:
    risk_eval = RiskEvaluation(
        passed=False,
        proposed_action="BUY",
        approved_size_usd=0.0,
        rejection_reasons=["Cooldown active"],
    )
    with pytest.raises(PaperTradingError, match="risk evaluation rejected"):
        create_paper_trade(
            symbol="BTC/USDT",
            snapshot=_snapshot(),
            proposal=_proposal("BUY"),
            critique=_critique(),
            risk_evaluation=risk_eval,
        )


def test_create_paper_trade_fails_if_action_is_hold() -> None:
    risk_eval = RiskEvaluation(
        passed=True,  # hypothetical bad state
        proposed_action="HOLD",
        approved_size_usd=100.0,
    )
    with pytest.raises(PaperTradingError, match="non-tradable action"):
        create_paper_trade(
            symbol="BTC/USDT",
            snapshot=_snapshot(),
            proposal=_proposal("HOLD"),
            critique=_critique(),
            risk_evaluation=risk_eval,
        )


def test_create_paper_trade_fails_on_zero_close_price() -> None:
    risk_eval = RiskEvaluation(
        passed=True,
        proposed_action="BUY",
        approved_size_usd=100.0,
    )
    with pytest.raises(PaperTradingError, match="Invalid reference close price"):
        create_paper_trade(
            symbol="BTC/USDT",
            snapshot=_snapshot(close=0.0),
            proposal=_proposal("BUY"),
            critique=_critique(),
            risk_evaluation=risk_eval,
        )
