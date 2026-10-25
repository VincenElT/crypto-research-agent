"""Local, read-only research dashboard API with append-only decision reviews.

The server binds to loopback by default. It exposes only selected research
records and fixed training actions; it never exposes environment variables or
arbitrary command execution, and it has no exchange-order route.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from crypto_research.config import load_settings
from crypto_research.exceptions import StorageError
from crypto_research.ml_pipeline import fetch_and_train
from crypto_research.storage import init_db, save_decision_review


STATIC_DIR = Path(__file__).parent / "dashboard_static"
ALLOWED_SYMBOLS = {"BTC/USDT", "ETH/USDT"}
MAX_ACTIVITY_PAGE_SIZE = 100
logger = logging.getLogger("crypto_research.dashboard")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path: str | Path) -> sqlite3.Connection:
    conn = init_db(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _json_safe(value: Any) -> Any:
    if isinstance(value, sqlite3.Row):
        return {key: _json_safe(value[key]) for key in value.keys()}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return value


def _decoded(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


class DashboardService:
    def __init__(
        self,
        db_path: str | Path,
        artifact_root: str | Path = "artifacts",
        paper_account_id: str = "default",
    ) -> None:
        self.db_path = str(db_path)
        self.artifact_root = Path(artifact_root)
        self.paper_account_id = paper_account_id
        self._job_lock = threading.Lock()
        with _connect(self.db_path) as conn:
            # Jobs left active by a stopped local process are failures, not successful runs.
            conn.execute(
                """UPDATE training_jobs SET status='failed', error_message=?, updated_at_utc=?
                   WHERE status IN ('queued', 'running')""",
                ("Dashboard server restarted before the job completed.", _utc_now()),
            )

    def overview(self) -> dict[str, Any]:
        with _connect(self.db_path) as conn:
            decision_count = conn.execute("SELECT COUNT(*) FROM agent_decisions").fetchone()[0]
            latest_decision = conn.execute(
                "SELECT * FROM agent_decisions ORDER BY id DESC LIMIT 1"
            ).fetchone()
            latest_snapshot = conn.execute(
                "SELECT * FROM market_snapshots ORDER BY as_of_utc DESC, id DESC LIMIT 1"
            ).fetchone()
            account = conn.execute(
                "SELECT * FROM portfolio_accounts WHERE account_id=?",
                (self.paper_account_id,),
            ).fetchone()
            positions_rows = conn.execute(
                """SELECT p.symbol, p.quantity, p.average_entry_price, p.cost_basis_usd,
                          (SELECT ms.close FROM market_snapshots ms
                           WHERE ms.symbol=p.symbol ORDER BY ms.as_of_utc DESC, ms.id DESC LIMIT 1) AS last_close,
                          (SELECT ms.as_of_utc FROM market_snapshots ms
                           WHERE ms.symbol=p.symbol ORDER BY ms.as_of_utc DESC, ms.id DESC LIMIT 1) AS marked_at_utc
                   FROM portfolio_positions p WHERE p.account_id=? AND p.quantity != 0""",
                (self.paper_account_id,),
            ).fetchall()
            model = conn.execute(
                "SELECT * FROM model_runs WHERE status='completed' ORDER BY completed_at_utc DESC LIMIT 1"
            ).fetchone()
            active_job = conn.execute(
                "SELECT * FROM training_jobs WHERE status IN ('queued','running') ORDER BY created_at_utc DESC LIMIT 1"
            ).fetchone()

        cash = float(account["cash_usd"]) if account else None
        position_value = 0.0
        unrealized = 0.0
        marks_fresh = True
        for position in positions_rows:
            if position["last_close"] is None:
                marks_fresh = False
                continue
            mark = float(position["last_close"])
            qty = float(position["quantity"])
            position_value += qty * mark
            unrealized += qty * (mark - float(position["average_entry_price"]))
        equity = cash + position_value if cash is not None and marks_fresh else None
        return {
            "status": "ready",
            "as_of_utc": _utc_now(),
            "market_data": {
                "latest_snapshot": _json_safe(latest_snapshot),
                "source": latest_snapshot["exchange"] if latest_snapshot else None,
                "freshness": latest_snapshot["as_of_utc"] if latest_snapshot else None,
            },
            "agents": {
                "decision_count": decision_count,
                "latest_decision": _json_safe(latest_decision),
                "scout_status": "recorded" if latest_decision else "not_run",
                "skeptic_status": "recorded" if latest_decision else "not_run",
                "predictor_status": "available" if model else "not_trained",
            },
            "paper_portfolio": {
                "status": "simulated_only",
                "account_id": self.paper_account_id,
                "cash_usd": cash,
                "position_value_usd": position_value if marks_fresh else None,
                "equity_usd": equity,
                "unrealized_pnl_usd": unrealized if marks_fresh else None,
                "realized_pnl_usd": float(account["realized_pnl_usd"]) if account else None,
                "peak_equity_usd": float(account["peak_equity_usd"]) if account else None,
                "positions": _json_safe(positions_rows),
                "mark_quality": "latest recorded market snapshots" if marks_fresh else "missing mark; equity withheld",
            },
            "active_model": self._format_model(model) if model else None,
            "active_training_job": _json_safe(active_job),
            "live_trading": "disabled",
        }

    @staticmethod
    def _format_model(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        model = dict(row)
        model["metadata"] = _decoded(model.pop("metadata_json", None), {})
        model["metrics"] = _decoded(model.pop("metrics_json", None), {})
        return model

    def list_models(self) -> list[dict[str, Any]]:
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM model_runs ORDER BY created_at_utc DESC LIMIT 100"
            ).fetchall()
        return [self._format_model(row) for row in rows]

    def list_jobs(self) -> list[dict[str, Any]]:
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM training_jobs ORDER BY created_at_utc DESC LIMIT 50"
            ).fetchall()
        output = []
        for row in rows:
            item = dict(row)
            item["symbols"] = _decoded(item.pop("symbols_json"), [])
            item["progress"] = _decoded(item.pop("progress_json"), {})
            output.append(item)
        return output

    def activity(self, params: dict[str, list[str]]) -> dict[str, Any]:
        limit = self._bounded_int(params, "limit", 25, 1, MAX_ACTIVITY_PAGE_SIZE)
        offset = self._bounded_int(params, "offset", 0, 0, 1_000_000)
        clauses: list[str] = []
        values: list[Any] = []
        for key, column in (("symbol", "d.symbol"), ("action", "d.scout_action")):
            requested = params.get(key, [""])[0]
            if requested:
                if key == "symbol" and requested not in ALLOWED_SYMBOLS:
                    raise ValueError("Unsupported symbol filter.")
                if key == "action" and requested not in {"BUY", "SELL", "HOLD"}:
                    raise ValueError("Unsupported action filter.")
                clauses.append(f"{column}=?")
                values.append(requested)
        agent = params.get("agent", [""])[0]
        if agent:
            if agent not in {"Scout", "Skeptic"}:
                raise ValueError("Agent must be Scout or Skeptic.")
        status_filter = params.get("status", [""])[0]
        if status_filter:
            allowed_statuses = {"executed_paper", "rejected_by_risk", "approved_no_candidate", "proposal_only", "pending", "rejected", "approved"}
            if status_filter not in allowed_statuses:
                raise ValueError("Unsupported decision status filter.")
        model_version = params.get("model_version", [""])[0].strip()
        if model_version:
            clauses.append("d.model_version = ?")
            values.append(model_version[:200])
        review_label = params.get("review_label", [""])[0]
        if review_label:
            from crypto_research.storage import DECISION_REVIEW_LABELS
            if review_label not in DECISION_REVIEW_LABELS:
                raise ValueError("Unsupported review label filter.")
            clauses.append("(SELECT dr.label FROM decision_reviews dr WHERE dr.decision_id=d.id ORDER BY dr.review_id DESC LIMIT 1)=?")
            values.append(review_label)
        date_from = params.get("date_from", [""])[0]
        date_to = params.get("date_to", [""])[0]
        if date_from:
            if len(date_from) == 10:
                date_from = f"{date_from}T00:00:00+00:00"
            clauses.append("d.as_of_utc >= ?")
            values.append(date_from)
        if date_to:
            if len(date_to) == 10:
                date_to = f"{date_to}T23:59:59.999999+00:00"
            clauses.append("d.as_of_utc <= ?")
            values.append(date_to)
        min_conf = params.get("confidence_min", [""])[0]
        if min_conf:
            try:
                conf = float(min_conf)
            except ValueError as exc:
                raise ValueError("Confidence minimum must be numeric.") from exc
            if not 0 <= conf <= 1:
                raise ValueError("Confidence minimum must be between 0 and 1.")
            clauses.append("d.scout_confidence >= ?")
            values.append(conf)
        max_conf = params.get("confidence_max", [""])[0]
        if max_conf:
            try:
                conf = float(max_conf)
            except ValueError as exc:
                raise ValueError("Confidence maximum must be numeric.") from exc
            if not 0 <= conf <= 1:
                raise ValueError("Confidence maximum must be between 0 and 1.")
            clauses.append("d.scout_confidence <= ?")
            values.append(conf)
        text_query = params.get("q", [""])[0].strip()
        if text_query:
            clauses.append("(d.scout_thesis LIKE ? OR d.skeptic_rationale LIKE ?)")
            values.extend([f"%{text_query[:100]}%"] * 2)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                f"""SELECT d.id, d.as_of_utc, d.created_at_utc, d.symbol,
                           d.scout_action, d.scout_confidence, d.scout_thesis,
                           d.scout_evidence_json, d.scout_risks_json,
                           d.scout_missing_data_json, d.skeptic_verdict,
                           d.skeptic_agrees, d.skeptic_confidence,
                           d.skeptic_rationale, d.skeptic_weaknesses_json,
                           d.skeptic_missing_evidence_json, d.model_version, d.prediction_horizon,
                           (SELECT r.passed FROM risk_evaluations r WHERE r.symbol=d.symbol
                             AND r.as_of_utc=d.as_of_utc ORDER BY r.id DESC LIMIT 1) AS risk_passed,
                           (SELECT r.rejection_reasons_json FROM risk_evaluations r WHERE r.symbol=d.symbol
                             AND r.as_of_utc=d.as_of_utc ORDER BY r.id DESC LIMIT 1) AS rejection_reasons_json,
                           (SELECT p.status FROM paper_trades p WHERE p.symbol=d.symbol
                             AND p.as_of_utc=d.as_of_utc ORDER BY p.id DESC LIMIT 1) AS paper_status,
                           (SELECT p.candidate_id FROM paper_trades p WHERE p.symbol=d.symbol
                             AND p.as_of_utc=d.as_of_utc ORDER BY p.id DESC LIMIT 1) AS candidate_id,
                           (SELECT e.execution_timestamp_utc FROM paper_executions e WHERE e.candidate_id=(
                             SELECT p.candidate_id FROM paper_trades p WHERE p.symbol=d.symbol
                             AND p.as_of_utc=d.as_of_utc ORDER BY p.id DESC LIMIT 1)
                             ORDER BY e.created_at_utc DESC LIMIT 1) AS paper_executed_at_utc,
                           (SELECT e.realized_pnl_usd FROM paper_executions e WHERE e.candidate_id=(
                             SELECT p.candidate_id FROM paper_trades p WHERE p.symbol=d.symbol
                             AND p.as_of_utc=d.as_of_utc ORDER BY p.id DESC LIMIT 1)
                             ORDER BY e.created_at_utc DESC LIMIT 1) AS realized_pnl_usd,
                           (SELECT dr.label FROM decision_reviews dr WHERE dr.decision_id=d.id
                             ORDER BY dr.review_id DESC LIMIT 1) AS latest_review_label
                    FROM agent_decisions d {where}
                    ORDER BY d.as_of_utc DESC, d.id DESC LIMIT ? OFFSET ?""",
                (*values, limit, offset),
            ).fetchall()
            total = conn.execute(
                f"SELECT COUNT(*) FROM agent_decisions d {where}", values
            ).fetchone()[0]
        items = []
        for row in rows:
            item = dict(row)
            for column in (
                "scout_evidence_json", "scout_risks_json", "scout_missing_data_json",
                "skeptic_weaknesses_json", "skeptic_missing_evidence_json", "rejection_reasons_json",
            ):
                source_key = column
                target_key = column.removesuffix("_json")
                item[target_key] = _decoded(item.pop(source_key), [])
            item["status"] = self._decision_status(item)
            if agent == "Scout" and item["scout_action"] not in {"BUY", "SELL", "HOLD"}:
                continue
            if agent == "Skeptic" and not item.get("skeptic_verdict"):
                continue
            if status_filter and item["status"] != status_filter:
                continue
            item["scout_skeptic_flow"] = ["Market data", "Scout", "Skeptic", "Risk checks", item["status"]]
            items.append(item)
        # Agent/status filters are applied after the correlated status query, so totals
        # are computed from the returned filtered page only when those filters are used.
        if agent or status_filter:
            filtered = []
            cursor = 0
            while True:
                page = self.activity({**params, "offset": [str(cursor)], "limit": [str(MAX_ACTIVITY_PAGE_SIZE)], "agent": [""] , "status": [""]})
                filtered.extend(item for item in page["items"] if (not agent or (item["scout_action"] in {"BUY", "SELL", "HOLD"} if agent == "Scout" else bool(item.get("skeptic_verdict")))) and (not status_filter or item["status"] == status_filter))
                cursor += MAX_ACTIVITY_PAGE_SIZE
                if cursor >= page["total"]:
                    break
            total = len(filtered)
            items = filtered[offset:offset + limit]
        return {"items": items, "total": total, "limit": limit, "offset": offset}

    @staticmethod
    def _decision_status(item: dict[str, Any]) -> str:
        if item.get("paper_executed_at_utc"):
            return "executed_paper"
        if item.get("risk_passed") == 0:
            return "rejected_by_risk"
        if item.get("paper_status"):
            return str(item["paper_status"]).lower()
        if item.get("risk_passed") == 1:
            return "approved_no_candidate"
        return "proposal_only"

    @staticmethod
    def _bounded_int(
        params: dict[str, list[str]], key: str, default: int, minimum: int, maximum: int
    ) -> int:
        value = params.get(key, [str(default)])[0]
        try:
            result = int(value)
        except ValueError as exc:
            raise ValueError(f"{key} must be an integer.") from exc
        if result < minimum or result > maximum:
            raise ValueError(f"{key} must be between {minimum} and {maximum}.")
        return result

    def review_history(self, decision_id: int) -> list[dict[str, Any]]:
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM decision_reviews WHERE decision_id=? ORDER BY review_id ASC",
                (decision_id,),
            ).fetchall()
        return [_json_safe(row) for row in rows]

    def save_review(self, decision_id: int, body: dict[str, Any]) -> dict[str, Any]:
        label = body.get("label")
        comment = body.get("comment", "")
        reviewer = body.get("reviewer", "local_user")
        if not isinstance(comment, str) or not isinstance(reviewer, str):
            raise ValueError("Review comment and reviewer must be text.")
        with _connect(self.db_path) as conn:
            review_id = save_decision_review(
                conn,
                decision_id=decision_id,
                label=label if isinstance(label, str) else "",
                comment=comment,
                reviewer=reviewer,
            )
        return self.review_history(decision_id)[-1] | {"review_id": review_id}

    def export_reviews_csv(self) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            ["decision_id", "as_of_utc", "symbol", "action", "scout_confidence", "review_label",
             "comment", "reviewer", "reviewed_at_utc", "realized_paper_pnl_usd", "usage_note"]
        )
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT d.id, d.as_of_utc, d.symbol, d.scout_action, d.scout_confidence,
                          dr.label, dr.comment, dr.reviewer, dr.reviewed_at_utc,
                          (SELECT e.realized_pnl_usd FROM paper_executions e WHERE e.candidate_id=(
                            SELECT p.candidate_id FROM paper_trades p WHERE p.symbol=d.symbol
                            AND p.as_of_utc=d.as_of_utc ORDER BY p.id DESC LIMIT 1)
                            ORDER BY e.created_at_utc DESC LIMIT 1) AS realized_paper_pnl_usd
                   FROM decision_reviews dr JOIN agent_decisions d ON d.id=dr.decision_id
                   WHERE dr.review_id=(SELECT MAX(dr2.review_id) FROM decision_reviews dr2
                                       WHERE dr2.decision_id=dr.decision_id)
                   ORDER BY d.as_of_utc DESC"""
            ).fetchall()
        for row in rows:
            writer.writerow([*row, "Annotation is not a market-outcome or supervised target label."])
        return output.getvalue()

    def market(self, symbol: str) -> dict[str, Any]:
        if symbol not in ALLOWED_SYMBOLS:
            raise ValueError("Unsupported market symbol.")
        symbol_slug = symbol.replace("/", "").replace("-", "").lower()
        dataset_files = sorted(
            (self.artifact_root / "datasets").glob(f"ohlcv_{symbol_slug}_*.csv"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        ) if (self.artifact_root / "datasets").exists() else []
        candles: list[dict[str, Any]] = []
        source = "no persisted OHLCV dataset"
        if dataset_files:
            dataset = pd_read_candles(dataset_files[0])
            candles = dataset.tail(2_000).to_dict(orient="records")
            source = "historical public OHLCV used by a training run"
        with _connect(self.db_path) as conn:
            snapshots = conn.execute(
                """SELECT as_of_utc AS timestamp, close, sma_50, sma_200, momentum_30d, trend
                   FROM market_snapshots WHERE symbol=? ORDER BY as_of_utc DESC LIMIT 1000""",
                (symbol,),
            ).fetchall()
            predictions = conn.execute(
                """SELECT p.as_of_utc, p.probability_up, p.predicted_direction,
                          p.observed_direction, p.model_version
                   FROM model_predictions p JOIN model_runs m ON m.model_version=p.model_version
                   WHERE p.symbol=? AND p.split_name='out_of_sample' AND m.status='completed'
                   ORDER BY p.as_of_utc DESC LIMIT 1000""",
                (symbol,),
            ).fetchall()
        candles.reverse()
        snapshots = list(reversed(snapshots))
        predictions = list(reversed(predictions))
        if candles:
            candles = _json_safe(candles)
        return {
            "symbol": symbol,
            "timeframe": "1h" if candles else "recorded snapshot timeframe",
            "source": source,
            "data_kind": "historical observation" if candles else "recorded agent market snapshots",
            "candles": candles,
            "snapshots": _json_safe(snapshots),
            "predictions": _json_safe(predictions),
        }

    def journal(self) -> dict[str, Any]:
        with _connect(self.db_path) as conn:
            positions = conn.execute(
                """SELECT p.*, (SELECT ms.close FROM market_snapshots ms WHERE ms.symbol=p.symbol
                           ORDER BY ms.as_of_utc DESC LIMIT 1) AS last_close
                   FROM portfolio_positions p WHERE p.account_id=? AND p.quantity != 0""",
                (self.paper_account_id,),
            ).fetchall()
            executions = conn.execute(
                """SELECT * FROM paper_executions WHERE is_live=0 ORDER BY execution_timestamp_utc DESC LIMIT 500"""
            ).fetchall()
            rejected = conn.execute(
                """SELECT * FROM risk_evaluations WHERE passed=0 ORDER BY as_of_utc DESC LIMIT 200"""
            ).fetchall()
            snapshots = conn.execute(
                """SELECT * FROM portfolio_snapshots WHERE account_id=? ORDER BY as_of_utc ASC LIMIT 2000""",
                (self.paper_account_id,),
            ).fetchall()
            account = conn.execute(
                "SELECT * FROM portfolio_accounts WHERE account_id=?", (self.paper_account_id,)
            ).fetchone()
        return {
            "account": _json_safe(account) if account else None,
            "positions": _json_safe(positions),
            "executions": _json_safe(executions),
            "rejections": _json_safe(rejected),
            "equity_snapshots": _json_safe(snapshots),
            "status": "simulated paper account; live execution disabled",
        }

    def start_training(self, body: dict[str, Any]) -> dict[str, Any]:
        symbols = body.get("symbols", ["BTC/USDT", "ETH/USDT"])
        if not isinstance(symbols, list) or not symbols or len(symbols) > 2:
            raise ValueError("Choose one or two supported symbols.")
        symbols = list(dict.fromkeys(symbols))
        if any(symbol not in ALLOWED_SYMBOLS for symbol in symbols):
            raise ValueError("Only BTC/USDT and ETH/USDT are currently supported.")
        timeframe = body.get("timeframe", "1h")
        if timeframe != "1h":
            raise ValueError("Model training is fixed to hourly candles.")
        try:
            candle_limit = int(body.get("candle_limit", 10_000))
        except (TypeError, ValueError) as exc:
            raise ValueError("Candle limit must be an integer.") from exc
        if candle_limit < 1_000 or candle_limit > 50_000:
            raise ValueError("Candle limit must be between 1000 and 50000.")
        job_id = f"train_{uuid.uuid4().hex[:12]}"
        now = _utc_now()
        with self._job_lock:
            with _connect(self.db_path) as conn:
                conn.execute(
                    """INSERT INTO training_jobs (
                           job_id, status, symbols_json, timeframe, candle_limit,
                           progress_json, created_at_utc, updated_at_utc
                       ) VALUES (?, 'queued', ?, ?, ?, '{}', ?, ?)""",
                    (job_id, json.dumps(symbols), timeframe, candle_limit, now, now),
                )
            worker = threading.Thread(
                target=self._run_training_job,
                args=(job_id, symbols, candle_limit),
                name=f"training-{job_id}",
                daemon=True,
            )
            worker.start()
        return {"job_id": job_id, "status": "queued", "symbols": symbols, "timeframe": timeframe}

    def _run_training_job(self, job_id: str, symbols: list[str], candle_limit: int) -> None:
        try:
            with _connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE training_jobs SET status='running', updated_at_utc=? WHERE job_id=?",
                    (_utc_now(), job_id),
                )
            for index, symbol in enumerate(symbols, start=1):
                with _connect(self.db_path) as conn:
                    conn.execute(
                        "UPDATE training_jobs SET progress_json=?, updated_at_utc=? WHERE job_id=?",
                        (json.dumps({"current_symbol": symbol, "completed_symbols": index - 1,
                                     "total_symbols": len(symbols)}), _utc_now(), job_id),
                    )
                fetch_and_train(
                    symbol,
                    candle_limit=candle_limit,
                    db_path=self.db_path,
                    artifact_root=self.artifact_root,
                )
            with _connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE training_jobs SET status='completed', progress_json=?, updated_at_utc=? WHERE job_id=?",
                    (json.dumps({"completed_symbols": len(symbols), "total_symbols": len(symbols)}),
                     _utc_now(), job_id),
                )
        except Exception as exc:
            logger.exception("Model training job %s failed", job_id)
            with _connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE training_jobs SET status='failed', error_message=?, updated_at_utc=? WHERE job_id=?",
                    (f"{type(exc).__name__}: {exc}"[:2_000], _utc_now(), job_id),
                )


def pd_read_candles(path: Path):
    import pandas as pd

    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


class DashboardHandler(SimpleHTTPRequestHandler):
    service: DashboardService

    def __init__(self, *args: Any, service: DashboardService, **kwargs: Any) -> None:
        self.service = service
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.info("%s - %s", self.address_string(), fmt % args)

    def _send_json(self, payload: Any, status: int = 200) -> None:
        data = json.dumps(_json_safe(payload), allow_nan=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_error_json(self, message: str, status: int) -> None:
        self._send_json({"error": message}, status=status)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length < 1 or length > 16_384:
            raise ValueError("Request body must be between 1 byte and 16 KB.")
        if "application/json" not in self.headers.get("Content-Type", ""):
            raise ValueError("Content-Type must be application/json.")
        try:
            payload = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("Request body must contain valid JSON.") from exc
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")
        return payload

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/"):
            if parsed.path in {"/", "/index.html"}:
                self.path = "/index.html"
            return super().do_GET()
        try:
            params = parse_qs(parsed.query)
            if parsed.path == "/api/overview":
                result = self.service.overview()
            elif parsed.path == "/api/activity" or parsed.path == "/api/decisions":
                result = self.service.activity(params)
            elif parsed.path.startswith("/api/decisions/") and parsed.path.endswith("/reviews"):
                decision_id = self._decision_id(parsed.path)
                result = {"items": self.service.review_history(decision_id)}
            elif parsed.path == "/api/reviews/export.csv":
                csv_data = self.service.export_reviews_csv().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", "attachment; filename=decision-annotations.csv")
                self.send_header("Content-Length", str(len(csv_data)))
                self.end_headers()
                self.wfile.write(csv_data)
                return
            elif parsed.path == "/api/market":
                result = self.service.market(params.get("symbol", ["BTC/USDT"])[0])
            elif parsed.path == "/api/journal":
                result = self.service.journal()
            elif parsed.path == "/api/models":
                result = {"items": self.service.list_models()}
            elif parsed.path == "/api/jobs":
                result = {"items": self.service.list_jobs()}
            else:
                self._send_error_json("Not found", 404)
                return
            self._send_json(result)
        except ValueError as exc:
            self._send_error_json(str(exc), 400)
        except (StorageError, sqlite3.Error) as exc:
            logger.exception("Dashboard data request failed")
            self._send_error_json(f"Unable to load dashboard data: {exc}", 500)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            body = self._body()
            if parsed.path == "/api/models/train":
                self._send_json(self.service.start_training(body), status=202)
                return
            match = re.fullmatch(r"/api/decisions/(\d+)/review", parsed.path)
            if match:
                result = self.service.save_review(int(match.group(1)), body)
                self._send_json(result, status=201)
                return
            self._send_error_json("Not found", 404)
        except ValueError as exc:
            self._send_error_json(str(exc), 400)
        except StorageError as exc:
            status = 404 if "unknown decision" in str(exc).lower() else 422
            self._send_error_json(str(exc), status)
        except sqlite3.IntegrityError as exc:
            self._send_error_json(f"A training job may already be active: {exc}", 409)
        except (sqlite3.Error, OSError) as exc:
            logger.exception("Dashboard update failed")
            self._send_error_json(f"Unable to save the requested change: {exc}", 500)


def serve_dashboard(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    db_path: str | Path | None = None,
    artifact_root: str | Path = "artifacts",
) -> None:
    settings = load_settings()
    resolved_db = db_path or settings.sqlite_db_path
    service = DashboardService(resolved_db, artifact_root, settings.paper_account_id)
    handler = lambda *args, **kwargs: DashboardHandler(*args, service=service, **kwargs)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Research dashboard: http://{host}:{port}")
    print(f"Paper account: {settings.paper_account_id} (live trading disabled)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Start the local crypto research dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address; loopback is the safe default.")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--artifact-root", default="artifacts")
    args = parser.parse_args(argv)
    if args.port < 1 or args.port > 65_535:
        parser.error("--port must be between 1 and 65535")
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("For safety, the dashboard may bind only to this computer's loopback interface.")
    serve_dashboard(
        host=args.host,
        port=args.port,
        db_path=args.db_path,
        artifact_root=args.artifact_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
