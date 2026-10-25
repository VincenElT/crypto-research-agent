"""SQLite audit logging and decision-trail persistence.

Persists market snapshots, agent JSON, risk evaluations, paper-trade candidates,
virtual accounts, portfolio positions, executed paper trades, portfolio snapshots,
and system audit events. API keys and secrets are never stored.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from crypto_research.config import Settings
from crypto_research.exceptions import StorageError
from crypto_research.indicators import IndicatorSnapshot
from crypto_research.paper_trading import (
    PaperExecution,
    PaperTradeCandidate,
    PortfolioSnapshot,
    Position,
    VirtualAccount,
)
from crypto_research.research import DualAgentResult
from crypto_research.risk import RiskEvaluation

if TYPE_CHECKING:
    from crypto_research.risk import PortfolioState

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS market_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    as_of_utc TEXT NOT NULL,
    exchange TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    close REAL NOT NULL,
    sma_50 REAL NOT NULL,
    sma_200 REAL NOT NULL,
    momentum_30d REAL NOT NULL,
    trend TEXT NOT NULL,
    created_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    as_of_utc TEXT NOT NULL,
    symbol TEXT NOT NULL,
    scout_action TEXT NOT NULL,
    scout_confidence REAL NOT NULL,
    scout_thesis TEXT NOT NULL,
    scout_evidence_json TEXT NOT NULL,
    scout_risks_json TEXT NOT NULL,
    scout_missing_data_json TEXT NOT NULL,
    skeptic_verdict TEXT NOT NULL,
    skeptic_agrees INTEGER NOT NULL,
    skeptic_confidence REAL NOT NULL,
    skeptic_rationale TEXT NOT NULL,
    skeptic_weaknesses_json TEXT NOT NULL,
    skeptic_missing_evidence_json TEXT NOT NULL,
    created_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS risk_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    as_of_utc TEXT NOT NULL,
    symbol TEXT NOT NULL,
    action TEXT NOT NULL,
    passed INTEGER NOT NULL,
    approved_size_usd REAL NOT NULL,
    checks_json TEXT NOT NULL,
    rejection_reasons_json TEXT NOT NULL,
    details_json TEXT NOT NULL,
    created_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT UNIQUE NOT NULL,
    as_of_utc TEXT NOT NULL,
    symbol TEXT NOT NULL,
    action TEXT NOT NULL,
    reference_price REAL NOT NULL,
    size_usd REAL NOT NULL,
    size_asset REAL NOT NULL,
    scout_confidence REAL NOT NULL,
    scout_thesis TEXT NOT NULL,
    skeptic_rationale TEXT NOT NULL,
    is_live INTEGER NOT NULL,
    status TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    UNIQUE(symbol, as_of_utc)
);

CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    as_of_utc TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS portfolio_accounts (
    account_id TEXT PRIMARY KEY,
    initial_cash_usd REAL NOT NULL,
    cash_usd REAL NOT NULL,
    realized_pnl_usd REAL NOT NULL DEFAULT 0.0,
    total_fees_paid_usd REAL NOT NULL DEFAULT 0.0,
    peak_equity_usd REAL NOT NULL,
    updated_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS portfolio_positions (
    account_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    quantity REAL NOT NULL,
    average_entry_price REAL NOT NULL,
    cost_basis_usd REAL NOT NULL,
    updated_at_utc TEXT NOT NULL,
    PRIMARY KEY (account_id, symbol)
);

CREATE TABLE IF NOT EXISTS paper_executions (
    execution_id TEXT PRIMARY KEY,
    candidate_id TEXT UNIQUE NOT NULL,
    account_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    action TEXT NOT NULL,
    signal_candle_utc TEXT NOT NULL,
    execution_timestamp_utc TEXT NOT NULL,
    raw_price REAL NOT NULL,
    slippage_pct REAL NOT NULL,
    execution_price REAL NOT NULL,
    quantity REAL NOT NULL,
    notional_usd REAL NOT NULL,
    fee_rate REAL NOT NULL,
    fee_usd REAL NOT NULL,
    net_cash_impact_usd REAL NOT NULL,
    realized_pnl_usd REAL NOT NULL,
    cost_basis_sold_usd REAL NOT NULL DEFAULT 0.0,
    post_cash_usd REAL NOT NULL,
    post_quantity REAL NOT NULL,
    post_avg_entry_price REAL NOT NULL,
    created_at_utc TEXT NOT NULL,
    is_live INTEGER NOT NULL DEFAULT 0,
    UNIQUE(account_id, symbol, signal_candle_utc)
);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id TEXT NOT NULL,
    as_of_utc TEXT NOT NULL,
    cash_usd REAL NOT NULL,
    positions_value_usd REAL NOT NULL,
    total_equity_usd REAL NOT NULL,
    unrealized_pnl_usd REAL NOT NULL,
    realized_pnl_usd REAL NOT NULL,
    peak_equity_usd REAL NOT NULL,
    drawdown_pct REAL NOT NULL,
    created_at_utc TEXT NOT NULL
);
"""

FORBIDDEN_KEY_FRAGMENTS = ("key", "secret", "password", "token", "auth", "credential")
API_KEY_REGEX = re.compile(r"sk-(?:proj-)?[A-Za-z0-9_\-]{15,}")


def scrub_secrets(data: Any) -> Any:
    """Recursively scrub any sensitive keys or OpenAI API key patterns in strings."""
    if isinstance(data, dict):
        scrubbed: dict[str, Any] = {}
        for k, v in data.items():
            k_lower = str(k).lower()
            if any(fragment in k_lower for fragment in FORBIDDEN_KEY_FRAGMENTS):
                scrubbed[k] = "[REDACTED]"
            else:
                scrubbed[k] = scrub_secrets(v)
        return scrubbed
    if isinstance(data, list):
        return [scrub_secrets(item) for item in data]
    if isinstance(data, str):
        return API_KEY_REGEX.sub("[REDACTED_API_KEY]", data)
    return data


def init_db(db_target: str | Path | sqlite3.Connection) -> sqlite3.Connection:
    """Initialize SQLite tables for audit logging.

    Args:
        db_target: File path, ':memory:', or an existing sqlite3.Connection.
    """
    try:
        if isinstance(db_target, sqlite3.Connection):
            conn = db_target
        else:
            path = Path(db_target)
            if path.parent and not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        with conn:
            conn.executescript(SCHEMA_SQL)
        return conn
    except sqlite3.Error as exc:
        raise StorageError(f"Failed to initialize SQLite database: {exc}") from exc


def log_market_snapshot(
    conn: sqlite3.Connection,
    settings: Settings,
    snapshot: IndicatorSnapshot,
) -> int:
    """Record indicator snapshot. Never logs secrets."""
    now_utc = datetime.now(timezone.utc).isoformat()
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO market_snapshots (
                    as_of_utc, exchange, symbol, timeframe, close,
                    sma_50, sma_200, momentum_30d, trend, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.as_of.isoformat(),
                    settings.exchange_id,
                    settings.symbol,
                    settings.timeframe,
                    snapshot.close,
                    snapshot.sma_50,
                    snapshot.sma_200,
                    snapshot.momentum_30d,
                    snapshot.trend,
                    now_utc,
                ),
            )
            return int(cursor.lastrowid)
    except sqlite3.Error as exc:
        raise StorageError(f"Failed to log market snapshot: {exc}") from exc


def log_agent_decision(
    conn: sqlite3.Connection,
    symbol: str,
    as_of_utc: str,
    result: DualAgentResult,
) -> int:
    """Record validated Scout proposal and Skeptic review."""
    now_utc = datetime.now(timezone.utc).isoformat()
    p = result.proposal
    c = result.critique
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO agent_decisions (
                    as_of_utc, symbol, scout_action, scout_confidence,
                    scout_thesis, scout_evidence_json, scout_risks_json,
                    scout_missing_data_json, skeptic_verdict, skeptic_agrees,
                    skeptic_confidence, skeptic_rationale, skeptic_weaknesses_json,
                    skeptic_missing_evidence_json, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    as_of_utc,
                    symbol,
                    p.action,
                    p.confidence,
                    p.thesis,
                    json.dumps(p.evidence),
                    json.dumps(p.risks),
                    json.dumps(p.missing_data),
                    c.verdict,
                    1 if c.agrees_with_action else 0,
                    c.confidence,
                    c.rationale,
                    json.dumps(c.weaknesses),
                    json.dumps(c.missing_evidence),
                    now_utc,
                ),
            )
            return int(cursor.lastrowid)
    except sqlite3.Error as exc:
        raise StorageError(f"Failed to log agent decision: {exc}") from exc


def log_risk_evaluation(
    conn: sqlite3.Connection,
    symbol: str,
    as_of_utc: str,
    evaluation: RiskEvaluation,
) -> int:
    """Record deterministic risk evaluation and rejection reasons."""
    now_utc = datetime.now(timezone.utc).isoformat()
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO risk_evaluations (
                    as_of_utc, symbol, action, passed, approved_size_usd,
                    checks_json, rejection_reasons_json, details_json, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    as_of_utc,
                    symbol,
                    evaluation.proposed_action,
                    1 if evaluation.passed else 0,
                    evaluation.approved_size_usd,
                    json.dumps(evaluation.checks),
                    json.dumps(evaluation.rejection_reasons),
                    json.dumps(evaluation.details),
                    now_utc,
                ),
            )
            return int(cursor.lastrowid)
    except sqlite3.Error as exc:
        raise StorageError(f"Failed to log risk evaluation: {exc}") from exc


def log_paper_trade(
    conn: sqlite3.Connection,
    candidate: PaperTradeCandidate,
) -> int:
    """Record paper-trade candidate. Confirms live trading is disabled."""
    now_utc = datetime.now(timezone.utc).isoformat()
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO paper_trades (
                    candidate_id, as_of_utc, symbol, action, reference_price,
                    size_usd, size_asset, scout_confidence, scout_thesis,
                    skeptic_rationale, is_live, status, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate.candidate_id,
                    candidate.as_of_utc,
                    candidate.symbol,
                    candidate.action,
                    candidate.reference_price,
                    candidate.size_usd,
                    candidate.size_asset,
                    candidate.scout_confidence,
                    candidate.scout_thesis,
                    candidate.skeptic_rationale,
                    1 if candidate.is_live else 0,
                    candidate.status,
                    now_utc,
                ),
            )
            return int(cursor.lastrowid)
    except sqlite3.IntegrityError as exc:
        raise StorageError(
            f"Duplicate paper-trade candidate for {candidate.symbol} as of {candidate.as_of_utc}: {exc}"
        ) from exc
    except sqlite3.Error as exc:
        raise StorageError(f"Failed to log paper trade: {exc}") from exc


def log_audit_event(
    conn: sqlite3.Connection,
    event_type: str,
    as_of_utc: str,
    payload: dict[str, Any],
) -> int:
    """Record general audit event with scrubbed payload."""
    now_utc = datetime.now(timezone.utc).isoformat()
    clean_payload = scrub_secrets(payload)
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO audit_events (
                    event_type, as_of_utc, payload_json, created_at_utc
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    event_type,
                    as_of_utc,
                    json.dumps(clean_payload),
                    now_utc,
                ),
            )
            return int(cursor.lastrowid)
    except sqlite3.Error as exc:
        raise StorageError(f"Failed to log audit event: {exc}") from exc


def has_paper_trade(conn: sqlite3.Connection, symbol: str, as_of_utc: str) -> bool:
    """Check if a paper-trade candidate already exists for this symbol and timestamp."""
    try:
        cursor = conn.execute(
            "SELECT 1 FROM paper_trades WHERE symbol = ? AND as_of_utc = ? LIMIT 1",
            (symbol, as_of_utc),
        )
        return cursor.fetchone() is not None
    except sqlite3.Error as exc:
        raise StorageError(f"Failed to check existing paper trade: {exc}") from exc


def get_latest_paper_trade(conn: sqlite3.Connection, symbol: str) -> dict[str, Any] | None:
    """Retrieve the most recent paper trade for a symbol."""
    try:
        cursor = conn.execute(
            "SELECT * FROM paper_trades WHERE symbol = ? ORDER BY id DESC LIMIT 1",
            (symbol,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    except sqlite3.Error as exc:
        raise StorageError(f"Failed to query latest paper trade: {exc}") from exc


def get_pending_paper_trades(
    conn: sqlite3.Connection, symbol: str | None = None
) -> list[dict[str, Any]]:
    """Retrieve paper-trade candidates that have not yet been executed."""
    try:
        if symbol:
            cursor = conn.execute(
                """
                SELECT * FROM paper_trades
                WHERE symbol = ?
                  AND status IN ('CANDIDATE', 'PENDING')
                  AND candidate_id NOT IN (SELECT candidate_id FROM paper_executions)
                ORDER BY id ASC
                """,
                (symbol,),
            )
        else:
            cursor = conn.execute(
                """
                SELECT * FROM paper_trades
                WHERE status IN ('CANDIDATE', 'PENDING')
                  AND candidate_id NOT IN (SELECT candidate_id FROM paper_executions)
                ORDER BY id ASC
                """
            )
        return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as exc:
        raise StorageError(f"Failed to query pending paper trades: {exc}") from exc


def get_or_create_portfolio_account(
    conn: sqlite3.Connection,
    account_id: str = "default",
    initial_cash_usd: float = 10_000.0,
) -> VirtualAccount:
    """Retrieve existing virtual account or create a new one with initial cash."""
    now_utc = datetime.now(timezone.utc).isoformat()
    try:
        with conn:
            cursor = conn.execute(
                "SELECT * FROM portfolio_accounts WHERE account_id = ?",
                (account_id,),
            )
            row = cursor.fetchone()
            if row:
                return VirtualAccount(
                    account_id=row["account_id"],
                    initial_cash_usd=float(row["initial_cash_usd"]),
                    cash_usd=float(row["cash_usd"]),
                    realized_pnl_usd=float(row["realized_pnl_usd"]),
                    total_fees_paid_usd=float(row["total_fees_paid_usd"]),
                    peak_equity_usd=float(row["peak_equity_usd"]),
                    updated_at_utc=row["updated_at_utc"],
                )

            conn.execute(
                """
                INSERT INTO portfolio_accounts (
                    account_id, initial_cash_usd, cash_usd, realized_pnl_usd,
                    total_fees_paid_usd, peak_equity_usd, updated_at_utc
                ) VALUES (?, ?, ?, 0.0, 0.0, ?, ?)
                """,
                (account_id, initial_cash_usd, initial_cash_usd, initial_cash_usd, now_utc),
            )
            return VirtualAccount(
                account_id=account_id,
                initial_cash_usd=initial_cash_usd,
                cash_usd=initial_cash_usd,
                realized_pnl_usd=0.0,
                total_fees_paid_usd=0.0,
                peak_equity_usd=initial_cash_usd,
                updated_at_utc=now_utc,
            )
    except sqlite3.Error as exc:
        raise StorageError(f"Failed to get or create portfolio account {account_id}: {exc}") from exc


def get_portfolio_account(
    conn: sqlite3.Connection,
    account_id: str = "default",
) -> VirtualAccount | None:
    """Retrieve existing virtual account if it exists."""
    try:
        cursor = conn.execute(
            "SELECT * FROM portfolio_accounts WHERE account_id = ?",
            (account_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return VirtualAccount(
            account_id=row["account_id"],
            initial_cash_usd=float(row["initial_cash_usd"]),
            cash_usd=float(row["cash_usd"]),
            realized_pnl_usd=float(row["realized_pnl_usd"]),
            total_fees_paid_usd=float(row["total_fees_paid_usd"]),
            peak_equity_usd=float(row["peak_equity_usd"]),
            updated_at_utc=row["updated_at_utc"],
        )
    except sqlite3.Error as exc:
        raise StorageError(f"Failed to get portfolio account {account_id}: {exc}") from exc


def get_position(
    conn: sqlite3.Connection,
    account_id: str,
    symbol: str,
) -> Position:
    """Retrieve position for a symbol or return a zero-position."""
    now_utc = datetime.now(timezone.utc).isoformat()
    try:
        cursor = conn.execute(
            "SELECT * FROM portfolio_positions WHERE account_id = ? AND symbol = ?",
            (account_id, symbol),
        )
        row = cursor.fetchone()
        if row:
            return Position(
                account_id=row["account_id"],
                symbol=row["symbol"],
                quantity=float(row["quantity"]),
                average_entry_price=float(row["average_entry_price"]),
                cost_basis_usd=float(row["cost_basis_usd"]),
                updated_at_utc=row["updated_at_utc"],
            )
        return Position(
            account_id=account_id,
            symbol=symbol,
            quantity=0.0,
            average_entry_price=0.0,
            cost_basis_usd=0.0,
            updated_at_utc=now_utc,
        )
    except sqlite3.Error as exc:
        raise StorageError(f"Failed to get position for {symbol}: {exc}") from exc


def get_all_positions(
    conn: sqlite3.Connection,
    account_id: str = "default",
) -> list[Position]:
    """Retrieve all positions for an account."""
    try:
        cursor = conn.execute(
            "SELECT * FROM portfolio_positions WHERE account_id = ?",
            (account_id,),
        )
        positions: list[Position] = []
        for row in cursor.fetchall():
            positions.append(
                Position(
                    account_id=row["account_id"],
                    symbol=row["symbol"],
                    quantity=float(row["quantity"]),
                    average_entry_price=float(row["average_entry_price"]),
                    cost_basis_usd=float(row["cost_basis_usd"]),
                    updated_at_utc=row["updated_at_utc"],
                )
            )
        return positions
    except sqlite3.Error as exc:
        raise StorageError(f"Failed to query positions: {exc}") from exc


def has_execution_for_candidate(conn: sqlite3.Connection, candidate_id: str) -> bool:
    """Check if candidate has already been executed."""
    try:
        cursor = conn.execute(
            "SELECT 1 FROM paper_executions WHERE candidate_id = ? LIMIT 1",
            (candidate_id,),
        )
        return cursor.fetchone() is not None
    except sqlite3.Error as exc:
        raise StorageError(f"Failed to check execution for candidate {candidate_id}: {exc}") from exc


def has_execution_for_candle(
    conn: sqlite3.Connection,
    account_id: str,
    symbol: str,
    signal_candle_utc: str,
) -> bool:
    """Check if execution already exists for account, symbol, and signal candle."""
    try:
        cursor = conn.execute(
            """
            SELECT 1 FROM paper_executions
            WHERE account_id = ? AND symbol = ? AND signal_candle_utc = ?
            LIMIT 1
            """,
            (account_id, symbol, signal_candle_utc),
        )
        return cursor.fetchone() is not None
    except sqlite3.Error as exc:
        raise StorageError(f"Failed to check candle execution: {exc}") from exc


def save_execution_and_update_portfolio(
    conn: sqlite3.Connection,
    execution: PaperExecution,
    account: VirtualAccount,
    position: Position,
    snapshot: PortfolioSnapshot | None = None,
) -> None:
    """Record execution, update account and position balances, and record snapshot atomically."""
    try:
        with conn:
            # 1. Insert paper execution
            conn.execute(
                """
                INSERT INTO paper_executions (
                    execution_id, candidate_id, account_id, symbol, action,
                    signal_candle_utc, execution_timestamp_utc, raw_price,
                    slippage_pct, execution_price, quantity, notional_usd,
                    fee_rate, fee_usd, net_cash_impact_usd, realized_pnl_usd,
                    cost_basis_sold_usd, post_cash_usd, post_quantity,
                    post_avg_entry_price, created_at_utc, is_live
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    execution.execution_id,
                    execution.candidate_id,
                    execution.account_id,
                    execution.symbol,
                    execution.action,
                    execution.signal_candle_utc,
                    execution.execution_timestamp_utc,
                    execution.raw_price,
                    execution.slippage_pct,
                    execution.execution_price,
                    execution.quantity,
                    execution.notional_usd,
                    execution.fee_rate,
                    execution.fee_usd,
                    execution.net_cash_impact_usd,
                    execution.realized_pnl_usd,
                    execution.cost_basis_sold_usd,
                    execution.post_cash_usd,
                    execution.post_quantity,
                    execution.post_avg_entry_price,
                    execution.created_at_utc,
                    1 if execution.is_live else 0,
                ),
            )

            # 2. Upsert portfolio account
            conn.execute(
                """
                INSERT INTO portfolio_accounts (
                    account_id, initial_cash_usd, cash_usd, realized_pnl_usd,
                    total_fees_paid_usd, peak_equity_usd, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(account_id) DO UPDATE SET
                    cash_usd = excluded.cash_usd,
                    realized_pnl_usd = excluded.realized_pnl_usd,
                    total_fees_paid_usd = excluded.total_fees_paid_usd,
                    peak_equity_usd = excluded.peak_equity_usd,
                    updated_at_utc = excluded.updated_at_utc
                """,
                (
                    account.account_id,
                    account.initial_cash_usd,
                    account.cash_usd,
                    account.realized_pnl_usd,
                    account.total_fees_paid_usd,
                    account.peak_equity_usd,
                    account.updated_at_utc,
                ),
            )

            # 3. Upsert portfolio position
            conn.execute(
                """
                INSERT INTO portfolio_positions (
                    account_id, symbol, quantity, average_entry_price,
                    cost_basis_usd, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(account_id, symbol) DO UPDATE SET
                    quantity = excluded.quantity,
                    average_entry_price = excluded.average_entry_price,
                    cost_basis_usd = excluded.cost_basis_usd,
                    updated_at_utc = excluded.updated_at_utc
                """,
                (
                    position.account_id,
                    position.symbol,
                    position.quantity,
                    position.average_entry_price,
                    position.cost_basis_usd,
                    position.updated_at_utc,
                ),
            )

            # 4. Update paper_trades status to FILLED if row exists
            conn.execute(
                "UPDATE paper_trades SET status = 'FILLED' WHERE candidate_id = ?",
                (execution.candidate_id,),
            )

            # 5. Insert snapshot if provided
            if snapshot is not None:
                conn.execute(
                    """
                    INSERT INTO portfolio_snapshots (
                        account_id, as_of_utc, cash_usd, positions_value_usd,
                        total_equity_usd, unrealized_pnl_usd, realized_pnl_usd,
                        peak_equity_usd, drawdown_pct, created_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot.account_id,
                        snapshot.as_of_utc,
                        snapshot.cash_usd,
                        snapshot.positions_value_usd,
                        snapshot.total_equity_usd,
                        snapshot.unrealized_pnl_usd,
                        snapshot.realized_pnl_usd,
                        snapshot.peak_equity_usd,
                        snapshot.drawdown_pct,
                        snapshot.created_at_utc,
                    ),
                )
    except sqlite3.IntegrityError as exc:
        raise StorageError(f"Duplicate or conflicting paper execution: {exc}") from exc
    except sqlite3.Error as exc:
        raise StorageError(f"Failed to save execution: {exc}") from exc


def get_paper_executions(
    conn: sqlite3.Connection,
    account_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Retrieve executed paper trades for auditing."""
    try:
        if account_id:
            cursor = conn.execute(
                "SELECT * FROM paper_executions WHERE account_id = ? ORDER BY rowid DESC LIMIT ?",
                (account_id, limit),
            )
        else:
            cursor = conn.execute(
                "SELECT * FROM paper_executions ORDER BY rowid DESC LIMIT ?",
                (limit,),
            )
        return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as exc:
        raise StorageError(f"Failed to query paper executions: {exc}") from exc


def get_latest_execution(
    conn: sqlite3.Connection,
    symbol: str,
    account_id: str = "default",
) -> dict[str, Any] | None:
    """Retrieve the most recent execution for an account and symbol."""
    try:
        cursor = conn.execute(
            """
            SELECT * FROM paper_executions
            WHERE account_id = ? AND symbol = ?
            ORDER BY rowid DESC LIMIT 1
            """,
            (account_id, symbol),
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    except sqlite3.Error as exc:
        raise StorageError(f"Failed to query latest execution: {exc}") from exc


def record_portfolio_snapshot(
    conn: sqlite3.Connection,
    snapshot: PortfolioSnapshot,
) -> int:
    """Record a portfolio valuation snapshot."""
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO portfolio_snapshots (
                    account_id, as_of_utc, cash_usd, positions_value_usd,
                    total_equity_usd, unrealized_pnl_usd, realized_pnl_usd,
                    peak_equity_usd, drawdown_pct, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.account_id,
                    snapshot.as_of_utc,
                    snapshot.cash_usd,
                    snapshot.positions_value_usd,
                    snapshot.total_equity_usd,
                    snapshot.unrealized_pnl_usd,
                    snapshot.realized_pnl_usd,
                    snapshot.peak_equity_usd,
                    snapshot.drawdown_pct,
                    snapshot.created_at_utc,
                ),
            )
            return int(cursor.lastrowid)
    except sqlite3.Error as exc:
        raise StorageError(f"Failed to record portfolio snapshot: {exc}") from exc


def get_latest_portfolio_snapshot(
    conn: sqlite3.Connection,
    account_id: str = "default",
) -> dict[str, Any] | None:
    """Retrieve the latest portfolio snapshot for an account."""
    try:
        cursor = conn.execute(
            "SELECT * FROM portfolio_snapshots WHERE account_id = ? ORDER BY id DESC LIMIT 1",
            (account_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    except sqlite3.Error as exc:
        raise StorageError(f"Failed to query latest portfolio snapshot: {exc}") from exc


def load_portfolio_state_from_db(
    conn: sqlite3.Connection,
    symbol: str,
    as_of: pd.Timestamp,
    timeframe_seconds: int,
    initial_cash_usd: float = 10_000.0,
    account_id: str = "default",
    current_price: float | None = None,
) -> PortfolioState:
    """Load portfolio state from persistent tables or fallback to legacy candidate history."""
    from crypto_research.risk import PortfolioState

    try:
        # 1. Check if persistent account exists
        cursor = conn.execute(
            "SELECT cash_usd, peak_equity_usd FROM portfolio_accounts WHERE account_id = ?",
            (account_id,),
        )
        acc_row = cursor.fetchone()

        if acc_row is not None:
            cash = float(acc_row["cash_usd"])
            peak_equity = float(acc_row["peak_equity_usd"])

            cursor = conn.execute(
                "SELECT quantity, cost_basis_usd FROM portfolio_positions WHERE account_id = ? AND symbol = ?",
                (account_id, symbol),
            )
            pos_row = cursor.fetchone()
            if pos_row:
                qty = float(pos_row["quantity"])
                if current_price is not None and math.isfinite(current_price) and current_price > 0:
                    position_usd = qty * current_price
                else:
                    position_usd = float(pos_row["cost_basis_usd"])
            else:
                position_usd = 0.0

            current_equity = cash + position_usd
            peak_equity = max(peak_equity, current_equity)

            cursor = conn.execute(
                """
                SELECT execution_timestamp_utc FROM paper_executions
                WHERE account_id = ? AND symbol = ?
                ORDER BY execution_timestamp_utc DESC LIMIT 1
                """,
                (account_id, symbol),
            )
            exec_row = cursor.fetchone()
            if exec_row:
                last_trade_time = pd.Timestamp(exec_row["execution_timestamp_utc"])
                elapsed_sec = (as_of - last_trade_time).total_seconds()
                bars_since = max(0, int(elapsed_sec / timeframe_seconds))
            else:
                bars_since = 999

            return PortfolioState(
                cash_usd=cash,
                current_position_usd=position_usd,
                peak_equity_usd=peak_equity,
                current_equity_usd=current_equity,
                bars_since_last_trade=bars_since,
            )

        # 2. Fallback to candidate reconstruction if no persistent account was initialized
        cursor = conn.execute(
            "SELECT action, size_usd, as_of_utc FROM paper_trades WHERE symbol = ? ORDER BY id ASC",
            (symbol,),
        )
        rows = cursor.fetchall()
        if not rows:
            return PortfolioState(
                cash_usd=initial_cash_usd,
                current_position_usd=0.0,
                peak_equity_usd=initial_cash_usd,
                current_equity_usd=initial_cash_usd,
                bars_since_last_trade=999,
            )

        cash = initial_cash_usd
        position_usd = 0.0
        peak_equity = initial_cash_usd
        last_trade_time: pd.Timestamp | None = None

        for row in rows:
            action = row["action"]
            size = float(row["size_usd"])
            last_trade_time = pd.Timestamp(row["as_of_utc"])
            if action == "BUY":
                cash -= size
                position_usd += size
            elif action == "SELL":
                cash += size
                position_usd = max(0.0, position_usd - size)
            equity = cash + position_usd
            if equity > peak_equity:
                peak_equity = equity

        current_equity = cash + position_usd
        if last_trade_time is not None:
            elapsed_sec = (as_of - last_trade_time).total_seconds()
            bars_since = max(0, int(elapsed_sec / timeframe_seconds))
        else:
            bars_since = 999

        return PortfolioState(
            cash_usd=cash,
            current_position_usd=position_usd,
            peak_equity_usd=peak_equity,
            current_equity_usd=current_equity,
            bars_since_last_trade=bars_since,
        )
    except sqlite3.Error as exc:
        raise StorageError(f"Failed to load portfolio state: {exc}") from exc


def get_paper_trades(conn: sqlite3.Connection, limit: int = 50) -> list[dict[str, Any]]:
    """Retrieve logged paper-trade candidates for auditing."""
    try:
        cursor = conn.execute(
            "SELECT * FROM paper_trades ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as exc:
        raise StorageError(f"Failed to query paper trades: {exc}") from exc


def get_risk_evaluations(conn: sqlite3.Connection, limit: int = 50) -> list[dict[str, Any]]:
    """Retrieve logged risk evaluations for auditing."""
    try:
        cursor = conn.execute(
            "SELECT * FROM risk_evaluations ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as exc:
        raise StorageError(f"Failed to query risk evaluations: {exc}") from exc


def get_audit_events(
    conn: sqlite3.Connection,
    event_type: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Retrieve logged audit events."""
    try:
        if event_type:
            cursor = conn.execute(
                "SELECT * FROM audit_events WHERE event_type = ? ORDER BY id DESC LIMIT ?",
                (event_type, limit),
            )
        else:
            cursor = conn.execute(
                "SELECT * FROM audit_events ORDER BY id DESC LIMIT ?",
                (limit,),
            )
        return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as exc:
        raise StorageError(f"Failed to query audit events: {exc}") from exc
