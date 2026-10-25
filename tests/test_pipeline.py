"""End-to-end integration test for the research, risk, and paper-trade pipeline."""

from __future__ import annotations

import sqlite3
from typing import Any

import pandas as pd
import pytest

from crypto_research.config import Settings
from crypto_research.indicators import IndicatorSnapshot, add_indicators, latest_snapshot
from crypto_research.market_context import build_market_context
from crypto_research.paper_trading import PaperTradeCandidate, create_paper_trade
from crypto_research.research import run_dual_agents
from crypto_research.risk import PortfolioState, RiskEvaluation, RiskLimits, evaluate_risk
from crypto_research.schemas import ScoutProposal, SkepticReview
from crypto_research.storage import (
    get_paper_trades,
    get_risk_evaluations,
    init_db,
    load_portfolio_state_from_db,
    log_agent_decision,
    log_market_snapshot,
    log_paper_trade,
    log_risk_evaluation,
)


def _linear_candles(length: int = 220, start: float = 100.0, step: float = 1.0) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=length, freq="D", tz="UTC")
    closes = [start + i * step for i in range(length)]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": closes,
            "high": [c + 1.0 for c in closes],
            "low": [c - 1.0 for c in closes],
            "close": closes,
            "volume": [10.0] * length,
        }
    )


def test_full_pipeline_synthetic_execution() -> None:
    settings = Settings(
        exchange_id="binance",
        symbol="BTC/USDT",
        timeframe="1d",
        candle_limit=400,
        openai_model="gpt-4o-mini",
        openai_api_key_configured=True,
    )

    # 1. Market indicators
    candles = _linear_candles(220)
    with_indicators = add_indicators(candles, timeframe_seconds=86_400)
    snapshot = latest_snapshot(with_indicators)
    assert snapshot.close == 319.0
    assert "uptrend" in snapshot.trend

    # 2. Dual agent mock
    def fake_complete(*, schema: type, **_: Any) -> ScoutProposal | SkepticReview:
        if schema is ScoutProposal:
            return ScoutProposal(
                action="BUY",
                confidence=0.85,
                thesis="SMA 50 > SMA 200 with positive momentum.",
                evidence=["SMA 50 is above SMA 200", "momentum is positive"],
                risks=["volatility spike"],
                missing_data=[],
            )
        return SkepticReview(
            verdict="APPROVE",
            agrees_with_action=True,
            rationale="Downtrend absent; uptrend aligns with evidence.",
            missing_evidence=[],
            weaknesses=["orderbook missing"],
            confidence=0.8,
        )

    context = build_market_context(settings, snapshot, candles)
    agent_result = run_dual_agents(context, model_name=settings.openai_model, complete=fake_complete)
    assert agent_result.proposal.action == "BUY"
    assert agent_result.critique.verdict == "APPROVE"

    # 3. Deterministic risk evaluation
    portfolio = PortfolioState(
        cash_usd=10_000.0,
        current_position_usd=0.0,
        current_equity_usd=10_000.0,
        peak_equity_usd=10_000.0,
        bars_since_last_trade=20,
    )
    limits = RiskLimits(
        max_position_size_usd=1000.0,
        max_portfolio_allocation_pct=0.10,
        cooldown_bars=5,
        max_drawdown_pct=0.15,
        max_volatility_pct=0.08,
    )
    risk_evaluation = evaluate_risk(
        proposal=agent_result.proposal,
        critique=agent_result.critique,
        snapshot=snapshot,
        recent_candles=candles,
        portfolio=portfolio,
        limits=limits,
    )
    assert risk_evaluation.passed is True
    assert risk_evaluation.approved_size_usd == 1000.0

    # 4. Paper-trade candidate creation
    candidate = create_paper_trade(
        symbol=settings.symbol,
        snapshot=snapshot,
        proposal=agent_result.proposal,
        critique=agent_result.critique,
        risk_evaluation=risk_evaluation,
    )
    assert candidate.symbol == "BTC/USDT"
    assert candidate.action == "BUY"
    assert candidate.reference_price == 319.0
    assert candidate.size_usd == 1000.0
    assert candidate.is_live is False
    assert candidate.status == "CANDIDATE"

    # 5. SQLite persistence & audit
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    log_market_snapshot(conn, settings, snapshot)
    log_agent_decision(conn, settings.symbol, snapshot.as_of.isoformat(), agent_result)
    log_risk_evaluation(conn, settings.symbol, snapshot.as_of.isoformat(), risk_evaluation)
    log_paper_trade(conn, candidate)

    saved_trades = get_paper_trades(conn)
    saved_evals = get_risk_evaluations(conn)
    assert len(saved_trades) == 1
    assert len(saved_evals) == 1
    assert saved_trades[0]["is_live"] == 0
    assert saved_evals[0]["passed"] == 1


def test_full_pipeline_buy_then_sell_sequence() -> None:
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    symbol = "BTC/USDT"

    # Initial state: 0 position
    p0 = load_portfolio_state_from_db(conn, symbol, pd.Timestamp("2024-06-01T00:00:00Z"), 86_400)
    assert p0.current_position_usd == 0.0

    # First trade: BUY $1000
    c_buy = PaperTradeCandidate(
        candidate_id="pt_buy_1",
        symbol=symbol,
        action="BUY",
        reference_price=100.0,
        size_usd=1000.0,
        size_asset=10.0,
        as_of_utc="2024-06-01T00:00:00Z",
        scout_confidence=0.8,
        scout_thesis="Thesis",
        skeptic_rationale="Rationale",
    )
    log_paper_trade(conn, c_buy)

    # After BUY: position is $1000
    p1 = load_portfolio_state_from_db(conn, symbol, pd.Timestamp("2024-06-10T00:00:00Z"), 86_400)
    assert p1.current_position_usd == 1000.0
    assert p1.cash_usd == 9000.0
    assert p1.bars_since_last_trade == 9

    # Now evaluate SELL proposal: should pass and approve $1000
    candles = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2024-06-10T00:00:00Z"),
                "open": 110.0,
                "high": 111.0,
                "low": 109.0,
                "close": 110.0,
                "volume": 10.0,
            }
        ]
    )
    snap = IndicatorSnapshot(
        as_of=pd.Timestamp("2024-06-10T00:00:00Z"),
        close=110.0,
        sma_50=105.0,
        sma_200=95.0,
        momentum_30d=0.08,
        trend="uptrend",
    )
    scout_sell = ScoutProposal(
        action="SELL",
        confidence=0.8,
        thesis="Take profit",
        evidence=["Overextended"],
        risks=["Continued rally"],
        missing_data=[],
    )
    skeptic_approve = SkepticReview(
        verdict="APPROVE",
        agrees_with_action=True,
        rationale="Take profit valid",
        missing_evidence=[],
        weaknesses=["Orderbook"],
        confidence=0.8,
    )
    risk_sell = evaluate_risk(
        proposal=scout_sell,
        critique=skeptic_approve,
        snapshot=snap,
        recent_candles=candles,
        portfolio=p1,
    )
    assert risk_sell.passed is True
    assert risk_sell.approved_size_usd == 1000.0

    # Execute SELL in paper trading
    c_sell = create_paper_trade(
        symbol=symbol,
        snapshot=snap,
        proposal=scout_sell,
        critique=skeptic_approve,
        risk_evaluation=risk_sell,
    )
    log_paper_trade(conn, c_sell)

    # After SELL: position is 0
    p2 = load_portfolio_state_from_db(conn, symbol, pd.Timestamp("2024-06-15T00:00:00Z"), 86_400)
    assert p2.current_position_usd == 0.0
    assert p2.cash_usd == 10_000.0


def test_full_pipeline_with_simulated_execution_and_persistent_portfolio() -> None:
    from crypto_research.paper_trading import simulate_execution
    from crypto_research.storage import (
        get_all_positions,
        get_or_create_portfolio_account,
        get_paper_executions,
        get_position,
    )

    conn = sqlite3.connect(":memory:")
    init_db(conn)
    symbol = "BTC/USDT"
    account = get_or_create_portfolio_account(conn, "default", 10_000.0)
    assert account.cash_usd == 10_000.0

    # Candle 0 (Signal candle): 2024-06-01
    snap0 = IndicatorSnapshot(
        as_of=pd.Timestamp("2024-06-01T00:00:00Z"),
        close=100.0,
        sma_50=95.0,
        sma_200=90.0,
        momentum_30d=0.10,
        trend="uptrend",
    )
    p0 = load_portfolio_state_from_db(
        conn, symbol, snap0.as_of, 86_400, account_id="default"
    )
    assert p0.current_position_usd == 0.0

    # Risk evaluation approves BUY $1000
    risk_eval_buy = RiskEvaluation(
        passed=True,
        proposed_action="BUY",
        approved_size_usd=1000.0,
    )
    scout_buy = ScoutProposal(
        action="BUY",
        confidence=0.85,
        thesis="Breakout",
        evidence=["Momentum"],
        risks=["Pullback"],
        missing_data=[],
    )
    skeptic_buy = SkepticReview(
        verdict="APPROVE",
        agrees_with_action=True,
        rationale="Valid breakout",
        missing_evidence=[],
        weaknesses=["Volume data"],
        confidence=0.8,
    )
    cand_buy = create_paper_trade(
        symbol=symbol,
        snapshot=snap0,
        proposal=scout_buy,
        critique=skeptic_buy,
        risk_evaluation=risk_eval_buy,
    )
    log_paper_trade(conn, cand_buy)

    # Execution on Candle 1 open: 2024-06-02T00:00:00Z at price 102.0
    exec_buy = simulate_execution(
        conn,
        candidate=cand_buy,
        execution_price=102.0,
        execution_timestamp=pd.Timestamp("2024-06-02T00:00:00Z"),
        account_id="default",
        fee_rate=0.001,
        slippage_pct=0.0005,
    )
    assert exec_buy.is_live is False
    assert exec_buy.action == "BUY"
    assert exec_buy.quantity == pytest.approx(cand_buy.size_asset)

    # Verify portfolio state uses persistent tables
    p1 = load_portfolio_state_from_db(
        conn, symbol, pd.Timestamp("2024-06-02T00:00:00Z"), 86_400, account_id="default"
    )
    assert p1.current_position_usd == pytest.approx(exec_buy.notional_usd)
    assert p1.cash_usd == pytest.approx(exec_buy.post_cash_usd)

    # Now Candle 1 generates SELL signal at 2024-06-02T12:00:00Z
    snap1 = IndicatorSnapshot(
        as_of=pd.Timestamp("2024-06-02T12:00:00Z"),
        close=110.0,
        sma_50=96.0,
        sma_200=91.0,
        momentum_30d=0.12,
        trend="uptrend",
    )
    risk_eval_sell = RiskEvaluation(
        passed=True,
        proposed_action="SELL",
        approved_size_usd=1100.0,
    )
    scout_sell = ScoutProposal(
        action="SELL",
        confidence=0.80,
        thesis="Target reached",
        evidence=["Overextended"],
        risks=["Further upside"],
        missing_data=[],
    )
    skeptic_sell = SkepticReview(
        verdict="APPROVE",
        agrees_with_action=True,
        rationale="Profit taking reasonable",
        missing_evidence=[],
        weaknesses=["Macro trend"],
        confidence=0.8,
    )
    cand_sell = create_paper_trade(
        symbol=symbol,
        snapshot=snap1,
        proposal=scout_sell,
        critique=skeptic_sell,
        risk_evaluation=risk_eval_sell,
    )
    log_paper_trade(conn, cand_sell)

    # Execution on Candle 2 open: 2024-06-03T00:00:00Z at price 112.0
    exec_sell = simulate_execution(
        conn,
        candidate=cand_sell,
        execution_price=112.0,
        execution_timestamp=pd.Timestamp("2024-06-03T00:00:00Z"),
        account_id="default",
        fee_rate=0.001,
        slippage_pct=0.0005,
    )
    assert exec_sell.action == "SELL"
    assert exec_sell.realized_pnl_usd > 0.0  # Profitable round trip
    assert exec_sell.post_quantity == 0.0

    # Final state: 0 position, increased cash
    p_final = load_portfolio_state_from_db(
        conn, symbol, pd.Timestamp("2024-06-04T00:00:00Z"), 86_400, account_id="default"
    )
    assert p_final.current_position_usd == 0.0
    assert p_final.cash_usd > 10_000.0
    assert p_final.bars_since_last_trade == 1

    # Verify execution audit history
    history = get_paper_executions(conn, "default")
    assert len(history) == 2

