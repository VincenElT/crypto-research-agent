from __future__ import annotations

import sqlite3
import time

import pytest

from crypto_research.dashboard import DashboardService
from crypto_research.exceptions import StorageError
from crypto_research.storage import init_db


def add_decision(db_path) -> int:
    with init_db(db_path) as conn:
        cursor = conn.execute(
            """INSERT INTO agent_decisions (
                as_of_utc, symbol, scout_action, scout_confidence, scout_thesis,
                scout_evidence_json, scout_risks_json, scout_missing_data_json,
                skeptic_verdict, skeptic_agrees, skeptic_confidence, skeptic_rationale,
                skeptic_weaknesses_json, skeptic_missing_evidence_json, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("2025-01-01T00:00:00+00:00", "BTC/USDT", "BUY", .7, "thesis",
             "[]", "[]", "[]", "APPROVE", 1, .6, "review", "[]", "[]", "2025-01-01T00:00:01+00:00"),
        )
        return int(cursor.lastrowid)


def test_reviews_are_persistent_and_append_revision_history(tmp_path) -> None:
    db = tmp_path / "research.sqlite"
    decision_id = add_decision(db)
    service = DashboardService(db, tmp_path / "artifacts")
    first = service.save_review(decision_id, {"label": "good_reasoning_bad_outcome", "comment": "Sound logic."})
    second = service.save_review(decision_id, {"label": "uncertain", "comment": "Need more data."})
    history = service.review_history(decision_id)
    assert len(history) == 2
    assert first["review_id"] < second["review_id"]
    assert history[-1]["label"] == "uncertain"
    assert service.activity({"review_label": ["uncertain"]})["total"] == 1


def test_review_validation_and_unknown_decision_fail_closed(tmp_path) -> None:
    service = DashboardService(tmp_path / "research.sqlite")
    with pytest.raises(StorageError, match="Unsupported"):
        service.save_review(1, {"label": "bad_outcome"})
    with pytest.raises(StorageError, match="unknown decision"):
        service.save_review(999, {"label": "needs_review"})
    with pytest.raises(StorageError, match="2000"):
        service.save_review(1, {"label": "needs_review", "comment": "x" * 2001})


def test_activity_filters_and_pagination_are_bounded(tmp_path) -> None:
    db = tmp_path / "research.sqlite"
    add_decision(db)
    service = DashboardService(db)
    assert service.activity({"symbol": ["BTC/USDT"], "confidence_min": ["0.6"]})["total"] == 1
    with pytest.raises(ValueError, match="Unsupported symbol"):
        service.activity({"symbol": ["DOGE/USDT"]})
    with pytest.raises(ValueError, match="between"):
        service.activity({"limit": ["999"]})


def test_training_api_accepts_only_fixed_hourly_research_inputs(tmp_path) -> None:
    service = DashboardService(tmp_path / "research.sqlite")
    with pytest.raises(ValueError, match="hourly"):
        service.start_training({"symbols": ["BTC/USDT"], "timeframe": "1d"})
    with pytest.raises(ValueError, match="supported"):
        service.start_training({"symbols": ["DOGE/USDT"]})


def test_training_job_records_completion_and_failure(tmp_path, monkeypatch) -> None:
    import crypto_research.dashboard as dashboard

    service = DashboardService(tmp_path / "research.sqlite", tmp_path / "artifacts")
    monkeypatch.setattr(dashboard, "fetch_and_train", lambda *args, **kwargs: [{"model_version": "test"}])
    success = service.start_training({"symbols": ["BTC/USDT"], "candle_limit": 1_000})
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and service.list_jobs()[0]["status"] in {"queued", "running"}:
        time.sleep(.02)
    assert service.list_jobs()[0]["status"] == "completed"

    def fail(*args, **kwargs):
        raise RuntimeError("fixture fetch failed")

    monkeypatch.setattr(dashboard, "fetch_and_train", fail)
    failed = service.start_training({"symbols": ["ETH/USDT"], "candle_limit": 1_000})
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and service.list_jobs()[0]["status"] in {"queued", "running"}:
        time.sleep(.02)
    jobs = {job["job_id"]: job for job in service.list_jobs()}
    assert jobs[success["job_id"]]["status"] == "completed"
    assert jobs[failed["job_id"]]["status"] == "failed"
    assert "fixture fetch failed" in jobs[failed["job_id"]]["error_message"]


def test_legacy_agent_decision_schema_is_migrated_without_losing_rows(tmp_path) -> None:
    db = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE agent_decisions (
            id INTEGER PRIMARY KEY, as_of_utc TEXT NOT NULL, symbol TEXT NOT NULL,
            scout_action TEXT NOT NULL, scout_confidence REAL NOT NULL, scout_thesis TEXT NOT NULL,
            scout_evidence_json TEXT NOT NULL, scout_risks_json TEXT NOT NULL,
            scout_missing_data_json TEXT NOT NULL, skeptic_verdict TEXT NOT NULL,
            skeptic_agrees INTEGER NOT NULL, skeptic_confidence REAL NOT NULL,
            skeptic_rationale TEXT NOT NULL, skeptic_weaknesses_json TEXT NOT NULL,
            skeptic_missing_evidence_json TEXT NOT NULL, created_at_utc TEXT NOT NULL
        )"""
    )
    conn.execute(
        """INSERT INTO agent_decisions VALUES (1, '2025-01-01T00:00:00+00:00', 'BTC/USDT',
           'HOLD', 0.5, 'legacy', '[]', '[]', '[]', 'REJECT', 0, 0.5, 'review', '[]', '[]',
           '2025-01-01T00:00:01+00:00')"""
    )
    conn.commit()
    conn.close()
    with init_db(db) as migrated:
        columns = {row[1] for row in migrated.execute("PRAGMA table_info(agent_decisions)")}
        row = migrated.execute("SELECT scout_thesis, model_version, prediction_horizon FROM agent_decisions WHERE id=1").fetchone()
        assert {"model_version", "prediction_horizon"}.issubset(columns)
        assert tuple(row) == ("legacy", None, None)
