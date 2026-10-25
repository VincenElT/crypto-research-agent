"""Audit regression tests.

Verifies that confirmed audit bugs are resolved and fail-closed behavior is enforced.
"""

from __future__ import annotations

import sqlite3
import pandas as pd
import pytest

from crypto_research.exceptions import InsufficientHistoryError, StorageError
from crypto_research.indicators import IndicatorSnapshot, add_indicators, latest_snapshot
from crypto_research.paper_trading import PaperTradeCandidate
from crypto_research.risk import PortfolioState, RiskLimits, evaluate_risk
from crypto_research.schemas import ScoutProposal, SkepticReview
from crypto_research.storage import (
    init_db,
    log_paper_trade,
    scrub_secrets,
)


def _scout(action: str = "BUY", confidence: float = 0.8) -> ScoutProposal:
    return ScoutProposal(
        action=action,  # type: ignore[arg-type]
        confidence=confidence,
        thesis="Trend thesis",
        evidence=["Indicator evidence"],
        risks=["Market risks"],
        missing_data=[],
    )


def _skeptic(verdict: str = "APPROVE", agrees: bool = True) -> SkepticReview:
    return SkepticReview(
        verdict=verdict,  # type: ignore[arg-type]
        agrees_with_action=agrees,
        rationale="Critique rationale",
        missing_evidence=[],
        weaknesses=["Liquidity weakness"],
        confidence=0.8,
    )


def _snapshot(close: float = 100.0) -> IndicatorSnapshot:
    return IndicatorSnapshot(
        as_of=pd.Timestamp("2024-06-01T00:00:00Z"),
        close=close,
        sma_50=90.0,
        sma_200=80.0,
        momentum_30d=0.05,
        trend="uptrend (SMA 50 above SMA 200)",
    )


def _candle_df(close: float = 100.0, high: float = 101.0, low: float = 99.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2024-06-01T00:00:00Z"),
                "open": 100.0,
                "high": high,
                "low": low,
                "close": close,
                "volume": 10.0,
            }
        ]
    )


# 1. Audit Fix: SELL without holdings
def test_audit_gap_sell_requires_open_position() -> None:
    portfolio = PortfolioState(cash_usd=10_000.0, current_position_usd=0.0)
    result = evaluate_risk(
        proposal=_scout("SELL"),
        critique=_skeptic("APPROVE", agrees=True),
        snapshot=_snapshot(),
        recent_candles=_candle_df(),
        portfolio=portfolio,
    )
    assert result.passed is False
    assert any("position" in r.lower() for r in result.rejection_reasons)


# 2. Audit Fix: Position allocation limit stacking
def test_audit_gap_allocation_guard_respects_existing_position() -> None:
    # 10% limit of $10,000 equity = $1,000. Existing position is already $1,000.
    portfolio = PortfolioState(cash_usd=9000.0, current_position_usd=1000.0, current_equity_usd=10_000.0)
    limits = RiskLimits(max_position_size_usd=1000.0, max_portfolio_allocation_pct=0.10)
    result = evaluate_risk(
        proposal=_scout("BUY"),
        critique=_skeptic("APPROVE", agrees=True),
        snapshot=_snapshot(),
        recent_candles=_candle_df(),
        portfolio=portfolio,
        limits=limits,
    )
    assert result.passed is False
    assert any("allocation" in r.lower() or "position" in r.lower() for r in result.rejection_reasons)


# 3. Audit Fix: Negative equity approval
def test_audit_gap_negative_equity_fails_closed() -> None:
    portfolio = PortfolioState(cash_usd=100.0, current_equity_usd=-500.0, peak_equity_usd=-500.0)
    result = evaluate_risk(
        proposal=_scout("BUY"),
        critique=_skeptic("APPROVE", agrees=True),
        snapshot=_snapshot(),
        recent_candles=_candle_df(),
        portfolio=portfolio,
    )
    assert result.passed is False
    assert any("equity" in r.lower() for r in result.rejection_reasons)


# 4. Audit Fix: Negative candle close passing volatility
def test_audit_gap_negative_close_fails_closed() -> None:
    result = evaluate_risk(
        proposal=_scout("BUY"),
        critique=_skeptic("APPROVE", agrees=True),
        snapshot=_snapshot(-100.0),
        recent_candles=_candle_df(close=-100.0),
    )
    assert result.passed is False
    assert any("price" in r.lower() or "close" in r.lower() for r in result.rejection_reasons)


# 5. Audit Fix: Secret scrubbing embedded in strings
def test_audit_gap_scrub_secrets_embedded_in_strings() -> None:
    embedded = "Bearer sk-proj-1234567890abcdef1234567890"
    scrubbed = scrub_secrets(embedded)
    assert "sk-proj" not in scrubbed
    assert "[REDACTED_API_KEY]" in scrubbed

    url = "https://api.openai.com/v1?api_key=sk-abcdef1234567890"
    assert "sk-abcdef" not in scrub_secrets(url)

    err = {"detail": "Failed with token sk-112233445566778899"}
    assert "sk-112233" not in scrub_secrets(err)["detail"]


# 6. Audit Fix: Stale indicator look-ahead bias
def test_audit_gap_latest_snapshot_fails_if_latest_bar_has_nan() -> None:
    timestamps = pd.date_range("2024-01-01", periods=220, freq="D", tz="UTC")
    closes = [100.0 + i for i in range(220)]
    candles = pd.DataFrame(
        {"timestamp": timestamps, "open": closes, "high": closes, "low": closes, "close": closes, "volume": [10.0] * 220}
    )
    with_ind = add_indicators(candles, timeframe_seconds=86400)
    # Simulate corrupted indicator on latest closed bar
    with_ind.loc[with_ind.index[-1], "momentum_30d"] = float("nan")

    with pytest.raises(InsufficientHistoryError, match="Indicators are not defined on the latest closed candle"):
        latest_snapshot(with_ind)


# 7. Audit Fix: Duplicate paper trades on same candle
def test_audit_gap_rejects_duplicate_paper_trade_for_same_candle() -> None:
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    c1 = PaperTradeCandidate(
        candidate_id="pt_1",
        symbol="BTC/USDT",
        action="BUY",
        reference_price=50000.0,
        size_usd=1000.0,
        size_asset=0.02,
        as_of_utc="2024-06-01T00:00:00Z",
        scout_confidence=0.8,
        scout_thesis="",
        skeptic_rationale="",
    )
    c2 = PaperTradeCandidate(
        candidate_id="pt_2",
        symbol="BTC/USDT",
        action="BUY",
        reference_price=50000.0,
        size_usd=1000.0,
        size_asset=0.02,
        as_of_utc="2024-06-01T00:00:00Z",
        scout_confidence=0.8,
        scout_thesis="",
        skeptic_rationale="",
    )
    log_paper_trade(conn, c1)
    with pytest.raises(StorageError, match="Duplicate paper-trade candidate"):
        log_paper_trade(conn, c2)
