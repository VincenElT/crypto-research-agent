"""Unit tests for SQLite audit logging and decision trail storage."""

from __future__ import annotations

import sqlite3
import pandas as pd

from crypto_research.config import Settings
from crypto_research.indicators import IndicatorSnapshot
from crypto_research.paper_trading import PaperTradeCandidate
from crypto_research.research import DualAgentResult
from crypto_research.risk import RiskEvaluation
from crypto_research.schemas import ScoutProposal, SkepticReview
from crypto_research.storage import (
    get_audit_events,
    get_paper_trades,
    get_risk_evaluations,
    has_paper_trade,
    init_db,
    load_portfolio_state_from_db,
    log_agent_decision,
    log_audit_event,
    log_market_snapshot,
    log_paper_trade,
    log_risk_evaluation,
    scrub_secrets,
)


def _mem_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    return conn


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
        close=60_000.0,
        sma_50=58_000.0,
        sma_200=52_000.0,
        momentum_30d=0.05,
        trend="uptrend (SMA 50 above SMA 200)",
    )


def test_init_db_creates_expected_tables() -> None:
    conn = _mem_db()
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    assert "market_snapshots" in tables
    assert "agent_decisions" in tables
    assert "risk_evaluations" in tables
    assert "paper_trades" in tables
    assert "audit_events" in tables


def test_log_market_snapshot() -> None:
    conn = _mem_db()
    row_id = log_market_snapshot(conn, _settings(), _snapshot())
    assert row_id == 1

    cursor = conn.execute("SELECT symbol, close, sma_50 FROM market_snapshots WHERE id = 1")
    row = cursor.fetchone()
    assert row["symbol"] == "BTC/USDT"
    assert row["close"] == 60_000.0
    assert row["sma_50"] == 58_000.0


def test_log_agent_decision() -> None:
    conn = _mem_db()
    result = DualAgentResult(
        proposal=ScoutProposal(
            action="BUY",
            confidence=0.85,
            thesis="Clear uptrend",
            evidence=["SMA golden cross"],
            risks=["CPI print"],
            missing_data=[],
        ),
        critique=SkepticReview(
            verdict="APPROVE",
            agrees_with_action=True,
            rationale="Solid momentum",
            missing_evidence=[],
            weaknesses=["ignores liquidity shifts"],
            confidence=0.75,
        ),
    )
    row_id = log_agent_decision(conn, "BTC/USDT", "2024-06-01T00:00:00Z", result)
    assert row_id == 1

    cursor = conn.execute("SELECT scout_action, skeptic_verdict FROM agent_decisions WHERE id = 1")
    row = cursor.fetchone()
    assert row["scout_action"] == "BUY"
    assert row["skeptic_verdict"] == "APPROVE"


def test_log_and_query_risk_evaluation() -> None:
    conn = _mem_db()
    evaluation = RiskEvaluation(
        passed=True,
        proposed_action="BUY",
        approved_size_usd=1000.0,
        checks={"cooldown_passed": True, "volatility_within_limit": True},
        rejection_reasons=[],
    )
    row_id = log_risk_evaluation(conn, "BTC/USDT", "2024-06-01T00:00:00Z", evaluation)
    assert row_id == 1

    rows = get_risk_evaluations(conn)
    assert len(rows) == 1
    assert rows[0]["passed"] == 1
    assert rows[0]["approved_size_usd"] == 1000.0


def test_log_and_query_paper_trade() -> None:
    conn = _mem_db()
    candidate = PaperTradeCandidate(
        candidate_id="pt_test123",
        symbol="BTC/USDT",
        action="BUY",
        reference_price=60_000.0,
        size_usd=1000.0,
        size_asset=0.01666667,
        as_of_utc="2024-06-01T00:00:00Z",
        scout_confidence=0.85,
        scout_thesis="Thesis text",
        skeptic_rationale="Rationale text",
        is_live=False,
        status="CANDIDATE",
    )
    row_id = log_paper_trade(conn, candidate)
    assert row_id == 1

    trades = get_paper_trades(conn)
    assert len(trades) == 1
    assert trades[0]["candidate_id"] == "pt_test123"
    assert trades[0]["is_live"] == 0  # Invariant: non-live


def test_log_audit_event_and_scrub_secrets() -> None:
    conn = _mem_db()
    dirty_payload = {
        "event": "run_completed",
        "api_key": "sk-proj-1234567890abcdef",
        "nested": {
            "password": "supersecretpassword",
            "safe_param": 42,
        },
    }
    log_audit_event(conn, "SYSTEM_EVENT", "2024-06-01T00:00:00Z", dirty_payload)

    events = get_audit_events(conn, event_type="SYSTEM_EVENT")
    assert len(events) == 1
    payload_str = events[0]["payload_json"]
    assert "sk-proj" not in payload_str
    assert "supersecretpassword" not in payload_str
    assert "[REDACTED]" in payload_str
    assert '"safe_param": 42' in payload_str


def test_scrub_secrets_function_directly() -> None:
    assert scrub_secrets("sk-abcdefghijk12345") == "[REDACTED_API_KEY]"
    assert scrub_secrets({"OPENAI_API_KEY": "secret_val"}) == {"OPENAI_API_KEY": "[REDACTED]"}
    assert scrub_secrets({"safe": "hello"}) == {"safe": "hello"}
    # Embedded key in bearer header
    assert scrub_secrets("Bearer sk-proj-1234567890abcdef123456") == "Bearer [REDACTED_API_KEY]"
    # Embedded key in error message
    err = {"msg": "Error at sk-12345678901234567890"}
    assert "sk-" not in scrub_secrets(err)["msg"]


def test_has_paper_trade_and_load_portfolio_state() -> None:
    conn = _mem_db()
    assert has_paper_trade(conn, "BTC/USDT", "2024-06-01T00:00:00Z") is False

    candidate1 = PaperTradeCandidate(
        candidate_id="pt_1",
        symbol="BTC/USDT",
        action="BUY",
        reference_price=50_000.0,
        size_usd=1000.0,
        size_asset=0.02,
        as_of_utc="2024-06-01T00:00:00Z",
        scout_confidence=0.8,
        scout_thesis="Thesis",
        skeptic_rationale="Rationale",
    )
    log_paper_trade(conn, candidate1)
    assert has_paper_trade(conn, "BTC/USDT", "2024-06-01T00:00:00Z") is True

    # Reconstruct portfolio state after trade 1
    p_state = load_portfolio_state_from_db(
        conn,
        "BTC/USDT",
        as_of=pd.Timestamp("2024-06-03T00:00:00Z"),
        timeframe_seconds=86_400,
        initial_cash_usd=10_000.0,
    )
    assert p_state.cash_usd == 9000.0
    assert p_state.current_position_usd == 1000.0
    assert p_state.current_equity_usd == 10_000.0
    assert p_state.bars_since_last_trade == 2  # June 1 to June 3 = 2 days
