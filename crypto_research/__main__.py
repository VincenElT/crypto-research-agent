"""CLI entry: market summary, dual agents, deterministic risk, and persistent paper trading."""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone

from crypto_research.config import load_settings
from crypto_research.exceptions import CryptoResearchError, PaperTradingError
from crypto_research.indicators import IndicatorSnapshot, add_indicators, latest_snapshot
from crypto_research.market_context import build_market_context
from crypto_research.market_data import create_public_exchange, fetch_ohlcv
from crypto_research.paper_trading import (
    PaperTradeCandidate,
    create_paper_trade,
    simulate_execution,
)
from crypto_research.research import run_dual_agents
from crypto_research.risk import evaluate_risk
from crypto_research.storage import (
    get_all_positions,
    get_or_create_portfolio_account,
    get_pending_paper_trades,
    get_portfolio_account,
    has_paper_trade,
    init_db,
    load_portfolio_state_from_db,
    log_agent_decision,
    log_audit_event,
    log_market_snapshot,
    log_paper_trade,
    log_risk_evaluation,
)
from crypto_research.summary import (
    format_dual_agent_result,
    format_market_summary,
    format_paper_execution,
    format_paper_trade_candidate,
    format_portfolio_status,
    format_risk_evaluation,
)


def main() -> int:
    """Fetch closed candles, execute pending orders, run agents, evaluate risk, and log audit trail."""

    conn: sqlite3.Connection | None = None
    snapshot: IndicatorSnapshot | None = None

    try:
        settings = load_settings()
        exchange = create_public_exchange(settings.exchange_id)
        candles = fetch_ohlcv(settings, exchange=exchange)
        timeframe_seconds = int(exchange.parse_timeframe(settings.timeframe))
        with_indicators = add_indicators(candles, timeframe_seconds=timeframe_seconds)
        snapshot = latest_snapshot(with_indicators)
        print(format_market_summary(settings, snapshot, len(candles)))

        # SQLite audit logging and persistent portfolio setup
        conn = init_db(settings.sqlite_db_path)
        log_market_snapshot(conn, settings, snapshot)
        account = get_or_create_portfolio_account(
            conn,
            account_id=settings.paper_account_id,
            initial_cash_usd=settings.paper_initial_cash_usd,
        )

        # Execute any pending paper-trade proposals whose signal candle closed prior to current bar
        pending_candidates = get_pending_paper_trades(conn, symbol=settings.symbol)
        for pending in pending_candidates:
            if pending["as_of_utc"] < snapshot.as_of.isoformat():
                exec_price = float(candles.iloc[-1]["open"])
                candidate_obj = PaperTradeCandidate(
                    candidate_id=pending["candidate_id"],
                    symbol=pending["symbol"],
                    action=pending["action"],
                    reference_price=float(pending["reference_price"]),
                    size_usd=float(pending["size_usd"]),
                    size_asset=float(pending["size_asset"]),
                    as_of_utc=pending["as_of_utc"],
                    scout_confidence=float(pending["scout_confidence"]),
                    scout_thesis=pending["scout_thesis"],
                    skeptic_rationale=pending["skeptic_rationale"],
                    is_live=False,
                    status=pending["status"],
                )
                try:
                    execution = simulate_execution(
                        conn,
                        candidate=candidate_obj,
                        execution_price=exec_price,
                        execution_timestamp=snapshot.as_of,
                        account_id=settings.paper_account_id,
                        initial_cash_usd=settings.paper_initial_cash_usd,
                        fee_rate=settings.paper_fee_rate,
                        slippage_pct=settings.paper_slippage_pct,
                    )
                    print()
                    print(format_paper_execution(execution))
                except PaperTradingError as pte:
                    print(f"\nPending paper-trade execution rejected: {pte}")
                    log_audit_event(
                        conn,
                        event_type="EXECUTION_REJECTED",
                        as_of_utc=snapshot.as_of.isoformat(),
                        payload={
                            "candidate_id": pending["candidate_id"],
                            "reason": str(pte),
                        },
                    )

        if not settings.openai_api_key_configured:
            print(
                "\nScout/Skeptic skipped: OPENAI_API_KEY is not set. "
                "Market summary above used public data only."
            )
            log_audit_event(
                conn,
                event_type="AGENTS_SKIPPED",
                as_of_utc=snapshot.as_of.isoformat(),
                payload={"reason": "OPENAI_API_KEY is not set"},
            )
            current_account = get_portfolio_account(conn, settings.paper_account_id) or account
            all_positions = get_all_positions(conn, settings.paper_account_id)
            print()
            print(format_portfolio_status(current_account, all_positions))
            return 0

        context = build_market_context(settings, snapshot, candles)
        result = run_dual_agents(context, model_name=settings.openai_model)
        print()
        print(format_dual_agent_result(result))
        log_agent_decision(conn, settings.symbol, snapshot.as_of.isoformat(), result)

        # Reconstruct portfolio state from SQLite history
        portfolio = load_portfolio_state_from_db(
            conn,
            settings.symbol,
            snapshot.as_of,
            timeframe_seconds,
            initial_cash_usd=settings.paper_initial_cash_usd,
            account_id=settings.paper_account_id,
            current_price=snapshot.close,
        )

        # Deterministic risk evaluation using configured limits
        risk_limits = settings.to_risk_limits()
        risk_evaluation = evaluate_risk(
            proposal=result.proposal,
            critique=result.critique,
            snapshot=snapshot,
            recent_candles=candles,
            portfolio=portfolio,
            limits=risk_limits,
        )
        print()
        print(format_risk_evaluation(risk_evaluation))
        log_risk_evaluation(conn, settings.symbol, snapshot.as_of.isoformat(), risk_evaluation)

        # Paper-trade candidate generation (live trading strictly disabled)
        if risk_evaluation.passed and result.proposal.action in ("BUY", "SELL"):
            if has_paper_trade(conn, settings.symbol, snapshot.as_of.isoformat()):
                print("\nPaper-trade candidate skipped: a candidate already exists for this closed candle.")
                log_audit_event(
                    conn,
                    event_type="CANDIDATE_SKIPPED_DUPLICATE",
                    as_of_utc=snapshot.as_of.isoformat(),
                    payload={
                        "symbol": settings.symbol,
                        "action": result.proposal.action,
                        "reason": "Paper-trade candidate already logged for this candle",
                    },
                )
            else:
                candidate = create_paper_trade(
                    symbol=settings.symbol,
                    snapshot=snapshot,
                    proposal=result.proposal,
                    critique=result.critique,
                    risk_evaluation=risk_evaluation,
                )
                print()
                print(format_paper_trade_candidate(candidate))
                log_paper_trade(conn, candidate)
        elif not risk_evaluation.passed and result.proposal.action in ("BUY", "SELL"):
            print("\nPaper-trade candidate skipped: deterministic risk checks rejected the trade.")

        # Display current persistent portfolio status
        current_account = get_portfolio_account(conn, settings.paper_account_id) or account
        all_positions = get_all_positions(conn, settings.paper_account_id)
        print()
        print(format_portfolio_status(current_account, all_positions))

        return 0
    except CryptoResearchError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if conn is not None:
            try:
                as_of_str = (
                    snapshot.as_of.isoformat()
                    if snapshot is not None
                    else datetime.now(timezone.utc).isoformat()
                )
                log_audit_event(
                    conn,
                    event_type="PIPELINE_ERROR",
                    as_of_utc=as_of_str,
                    payload={"error_type": type(exc).__name__, "error_message": str(exc)},
                )
            except Exception:
                pass
        return 1
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
