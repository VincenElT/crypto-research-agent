"""Unit tests for deterministic Python risk filters."""

from __future__ import annotations

import pandas as pd
import pytest

from crypto_research.exceptions import RiskFilterError
from crypto_research.indicators import IndicatorSnapshot
from crypto_research.risk import (
    PortfolioState,
    RiskLimits,
    convert_directional_signal_to_action,
    convert_portfolio_action,
    evaluate_risk,
)
from crypto_research.schemas import ScoutProposal, SkepticReview


def _valid_scout(action: str = "BUY", confidence: float = 0.75) -> ScoutProposal:
    return ScoutProposal(
        action=action,  # type: ignore[arg-type]
        confidence=confidence,
        thesis="Strong trend supported by indicators.",
        evidence=["SMA 50 > SMA 200", "positive momentum"],
        risks=["volatility"],
        missing_data=[],
    )


def _valid_skeptic(verdict: str = "APPROVE", agrees: bool = True) -> SkepticReview:
    return SkepticReview(
        verdict=verdict,  # type: ignore[arg-type]
        agrees_with_action=agrees,
        rationale="Setup aligns with closed candles.",
        missing_evidence=[],
        weaknesses=["no orderbook"],
        confidence=0.7,
    )


def _snapshot(close: float = 100.0) -> IndicatorSnapshot:
    return IndicatorSnapshot(
        as_of=pd.Timestamp("2024-06-01T00:00:00Z"),
        close=close,
        sma_50=90.0,
        sma_200=80.0,
        momentum_30d=0.1,
        trend="uptrend (SMA 50 above SMA 200)",
    )


def _candles(high: float = 102.0, low: float = 98.0, close: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2024-06-01T00:00:00Z"),
                "open": 99.0,
                "high": high,
                "low": low,
                "close": close,
                "volume": 10.0,
            }
        ]
    )


def test_risk_passes_when_all_conditions_met() -> None:
    eval_result = evaluate_risk(
        proposal=_valid_scout("BUY", confidence=0.8),
        critique=_valid_skeptic("APPROVE", agrees=True),
        snapshot=_snapshot(100.0),
        recent_candles=_candles(high=102.0, low=99.0, close=100.0),
        portfolio=PortfolioState(cash_usd=5000.0, current_equity_usd=5000.0, peak_equity_usd=5000.0, bars_since_last_trade=10),
        limits=RiskLimits(max_position_size_usd=500.0, max_portfolio_allocation_pct=0.10, cooldown_bars=5),
    )
    assert eval_result.passed is True
    assert eval_result.approved_size_usd == 500.0
    assert eval_result.rejection_reasons == []
    assert eval_result.checks["cooldown_passed"] is True
    assert eval_result.checks["volatility_within_limit"] is True


def test_risk_rejects_hold_action() -> None:
    eval_result = evaluate_risk(
        proposal=_valid_scout("HOLD", confidence=0.8),
        critique=_valid_skeptic("APPROVE", agrees=True),
        snapshot=_snapshot(),
        recent_candles=_candles(),
    )
    assert eval_result.passed is False
    assert eval_result.approved_size_usd == 0.0
    assert any("HOLD" in r for r in eval_result.rejection_reasons)


def test_risk_rejects_when_skeptic_rejects() -> None:
    eval_result = evaluate_risk(
        proposal=_valid_scout("BUY", confidence=0.9),
        critique=_valid_skeptic("REJECT", agrees=False),
        snapshot=_snapshot(),
        recent_candles=_candles(),
    )
    assert eval_result.passed is False
    assert any("Skeptic" in r for r in eval_result.rejection_reasons)


def test_risk_rejects_when_skeptic_disagrees() -> None:
    eval_result = evaluate_risk(
        proposal=_valid_scout("BUY", confidence=0.9),
        critique=_valid_skeptic("APPROVE", agrees=False),
        snapshot=_snapshot(),
        recent_candles=_candles(),
    )
    assert eval_result.passed is False
    assert any("Skeptic" in r for r in eval_result.rejection_reasons)


def test_risk_rejects_low_scout_confidence() -> None:
    eval_result = evaluate_risk(
        proposal=_valid_scout("BUY", confidence=0.4),
        critique=_valid_skeptic("APPROVE", agrees=True),
        snapshot=_snapshot(),
        recent_candles=_candles(),
        limits=RiskLimits(min_scout_confidence=0.6),
    )
    assert eval_result.passed is False
    assert any("confidence" in r for r in eval_result.rejection_reasons)


def test_risk_rejects_active_cooldown() -> None:
    eval_result = evaluate_risk(
        proposal=_valid_scout("BUY", confidence=0.8),
        critique=_valid_skeptic("APPROVE", agrees=True),
        snapshot=_snapshot(),
        recent_candles=_candles(),
        portfolio=PortfolioState(bars_since_last_trade=2),
        limits=RiskLimits(cooldown_bars=5),
    )
    assert eval_result.passed is False
    assert any("Cooldown active" in r for r in eval_result.rejection_reasons)


def test_risk_rejects_drawdown_exceeded() -> None:
    eval_result = evaluate_risk(
        proposal=_valid_scout("BUY", confidence=0.8),
        critique=_valid_skeptic("APPROVE", agrees=True),
        snapshot=_snapshot(),
        recent_candles=_candles(),
        portfolio=PortfolioState(peak_equity_usd=10_000.0, current_equity_usd=8000.0),  # 20% drawdown
        limits=RiskLimits(max_drawdown_pct=0.15),
    )
    assert eval_result.passed is False
    assert any("Drawdown" in r for r in eval_result.rejection_reasons)


def test_risk_rejects_high_volatility() -> None:
    # 15% range on a 100 close bar
    eval_result = evaluate_risk(
        proposal=_valid_scout("BUY", confidence=0.8),
        critique=_valid_skeptic("APPROVE", agrees=True),
        snapshot=_snapshot(100.0),
        recent_candles=_candles(high=115.0, low=100.0, close=100.0),
        limits=RiskLimits(max_volatility_pct=0.08),
    )
    assert eval_result.passed is False
    assert any("volatility" in r.lower() for r in eval_result.rejection_reasons)


def test_risk_rejects_insufficient_cash() -> None:
    eval_result = evaluate_risk(
        proposal=_valid_scout("BUY", confidence=0.8),
        critique=_valid_skeptic("APPROVE", agrees=True),
        snapshot=_snapshot(100.0),
        recent_candles=_candles(),
        portfolio=PortfolioState(cash_usd=50.0, current_equity_usd=10_000.0),
        limits=RiskLimits(max_position_size_usd=1000.0, max_portfolio_allocation_pct=0.10),
    )
    assert eval_result.passed is False
    assert any("Insufficient cash" in r for r in eval_result.rejection_reasons)


def test_risk_raises_on_empty_candles() -> None:
    with pytest.raises(RiskFilterError, match="candle data is empty"):
        evaluate_risk(
            proposal=_valid_scout("BUY"),
            critique=_valid_skeptic("APPROVE"),
            snapshot=_snapshot(),
            recent_candles=pd.DataFrame(),
        )


def test_risk_rejects_sell_when_no_holdings() -> None:
    eval_result = evaluate_risk(
        proposal=_valid_scout("SELL", confidence=0.8),
        critique=_valid_skeptic("APPROVE", agrees=True),
        snapshot=_snapshot(100.0),
        recent_candles=_candles(),
        portfolio=PortfolioState(cash_usd=10_000.0, current_position_usd=0.0, current_equity_usd=10_000.0),
    )
    assert eval_result.passed is False
    assert eval_result.checks["has_position_to_sell"] is False
    assert any("no open long position" in r for r in eval_result.rejection_reasons)


def test_risk_rejects_buy_when_allocation_limit_reached() -> None:
    eval_result = evaluate_risk(
        proposal=_valid_scout("BUY", confidence=0.8),
        critique=_valid_skeptic("APPROVE", agrees=True),
        snapshot=_snapshot(100.0),
        recent_candles=_candles(),
        portfolio=PortfolioState(
            cash_usd=9000.0,
            current_position_usd=1000.0,
            current_equity_usd=10_000.0,
        ),
        limits=RiskLimits(max_portfolio_allocation_pct=0.10),
    )
    assert eval_result.passed is False
    assert eval_result.checks["allocation_capacity_available"] is False
    assert any("Position allocation limit reached" in r for r in eval_result.rejection_reasons)


def test_risk_approves_full_position_size_on_sell_even_if_above_max_position() -> None:
    # Position has grown to $1500, above the initial $1000 position limit
    eval_result = evaluate_risk(
        proposal=_valid_scout("SELL", confidence=0.85),
        critique=_valid_skeptic("APPROVE", agrees=True),
        snapshot=_snapshot(150.0),
        recent_candles=_candles(close=150.0),
        portfolio=PortfolioState(
            cash_usd=8500.0,
            current_position_usd=1500.0,
            current_equity_usd=10_000.0,
            bars_since_last_trade=10,
        ),
        limits=RiskLimits(max_position_size_usd=1000.0),
    )
    assert eval_result.passed is True
    assert eval_result.approved_size_usd == 1500.0
    assert eval_result.checks["has_position_to_sell"] is True


# =========================================================================
# Tests for Minimum Order Notional (Part A)
# =========================================================================

def test_risk_rejects_buy_below_min_trade_size() -> None:
    # 10% of 10,000 = 1,000. Position is 997.0 -> remaining capacity is 3.0 USD < 10.0 USD
    eval_result = evaluate_risk(
        proposal=_valid_scout("BUY", confidence=0.8),
        critique=_valid_skeptic("APPROVE", agrees=True),
        snapshot=_snapshot(100.0),
        recent_candles=_candles(),
        portfolio=PortfolioState(
            cash_usd=9000.0,
            current_position_usd=997.0,
            current_equity_usd=10_000.0,
            bars_since_last_trade=10,
        ),
        limits=RiskLimits(max_portfolio_allocation_pct=0.10, min_trade_size_usd=10.0),
    )
    assert eval_result.passed is False
    assert eval_result.checks["meets_min_trade_size"] is False
    assert any("below minimum trade size threshold" in r for r in eval_result.rejection_reasons)


def test_risk_accepts_buy_exactly_at_min_trade_size() -> None:
    # Remaining capacity = 1000 - 990 = 10.0 USD == min_trade_size_usd
    eval_result = evaluate_risk(
        proposal=_valid_scout("BUY", confidence=0.8),
        critique=_valid_skeptic("APPROVE", agrees=True),
        snapshot=_snapshot(100.0),
        recent_candles=_candles(),
        portfolio=PortfolioState(
            cash_usd=9000.0,
            current_position_usd=990.0,
            current_equity_usd=10_000.0,
            bars_since_last_trade=10,
        ),
        limits=RiskLimits(max_portfolio_allocation_pct=0.10, min_trade_size_usd=10.0),
    )
    assert eval_result.passed is True
    assert eval_result.approved_size_usd == 10.0
    assert eval_result.checks["meets_min_trade_size"] is True


def test_risk_accepts_normal_buy() -> None:
    eval_result = evaluate_risk(
        proposal=_valid_scout("BUY", confidence=0.85),
        critique=_valid_skeptic("APPROVE", agrees=True),
        snapshot=_snapshot(100.0),
        recent_candles=_candles(),
        portfolio=PortfolioState(
            cash_usd=10_000.0,
            current_position_usd=0.0,
            current_equity_usd=10_000.0,
            bars_since_last_trade=10,
        ),
        limits=RiskLimits(max_position_size_usd=1000.0, min_trade_size_usd=10.0),
    )
    assert eval_result.passed is True
    assert eval_result.approved_size_usd == 1000.0
    assert eval_result.checks["meets_min_trade_size"] is True


def test_risk_allows_full_liquidation_of_position_below_min_trade_size() -> None:
    # Residual position worth only 4.50 USD (< 10.0 USD threshold)
    eval_result = evaluate_risk(
        proposal=_valid_scout("SELL", confidence=0.85),
        critique=_valid_skeptic("APPROVE", agrees=True),
        snapshot=_snapshot(100.0),
        recent_candles=_candles(),
        portfolio=PortfolioState(
            cash_usd=9995.50,
            current_position_usd=4.50,
            current_equity_usd=10_000.0,
            bars_since_last_trade=10,
        ),
        limits=RiskLimits(min_trade_size_usd=10.0),
    )
    assert eval_result.passed is True
    assert eval_result.approved_size_usd == 4.50
    assert eval_result.checks["has_position_to_sell"] is True


# =========================================================================
# Tests for Portfolio-Aware Action Conversion (Part B)
# =========================================================================

def test_action_conversion_flat_bearish() -> None:
    assert convert_directional_signal_to_action("SELL", has_open_position=False) == "HOLD"
    prop = _valid_scout("SELL")
    converted = convert_portfolio_action(prop, has_open_position=False)
    assert converted.action == "HOLD"
    assert "maintaining cash" in converted.thesis


def test_action_conversion_long_bearish() -> None:
    assert convert_directional_signal_to_action("SELL", has_open_position=True) == "SELL"
    prop = _valid_scout("SELL")
    converted = convert_portfolio_action(prop, has_open_position=True)
    assert converted.action == "SELL"


def test_action_conversion_flat_bullish() -> None:
    assert convert_directional_signal_to_action("BUY", has_open_position=False) == "BUY"
    prop = _valid_scout("BUY")
    converted = convert_portfolio_action(prop, has_open_position=False)
    assert converted.action == "BUY"


def test_action_conversion_long_bullish() -> None:
    assert convert_directional_signal_to_action("BUY", has_open_position=True) == "BUY"
    prop = _valid_scout("BUY")
    converted = convert_portfolio_action(prop, has_open_position=True)
    assert converted.action == "BUY"


