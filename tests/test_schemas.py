"""Unit tests for strict Scout/Skeptic JSON schema validation."""

from __future__ import annotations

import pytest

from crypto_research.exceptions import AgentResponseError
from crypto_research.schemas import ScoutProposal, SkepticReview, parse_agent_payload

VALID_SCOUT = {
    "action": "HOLD",
    "confidence": 0.4,
    "thesis": "SMA 50 is below SMA 200 and 30-day momentum is negative.",
    "evidence": ["SMA 50 below SMA 200", "negative 30-day momentum"],
    "risks": ["trend can reverse", "payload has no order-book data"],
    "missing_data": ["volume regime vs history"],
}

VALID_SKEPTIC = {
    "verdict": "APPROVE",
    "agrees_with_action": True,
    "rationale": "HOLD matches the downtrend label in the payload.",
    "missing_evidence": ["no volatility measure"],
    "weaknesses": ["does not quantify how far price is from SMA 200"],
    "confidence": 0.55,
}


def test_valid_scout_json_parses() -> None:
    proposal = parse_agent_payload(ScoutProposal, VALID_SCOUT)
    assert proposal.action == "HOLD"
    assert proposal.confidence == 0.4


def test_scout_markdown_fence_is_stripped() -> None:
    fenced = "```json\n" + __import__("json").dumps(VALID_SCOUT) + "\n```"
    proposal = parse_agent_payload(ScoutProposal, fenced)
    assert proposal.action == "HOLD"


def test_invalid_action_is_rejected() -> None:
    payload = {**VALID_SCOUT, "action": "HODL"}
    with pytest.raises(AgentResponseError, match="schema validation"):
        parse_agent_payload(ScoutProposal, payload)


def test_confidence_out_of_range_is_rejected() -> None:
    payload = {**VALID_SCOUT, "confidence": 1.2}
    with pytest.raises(AgentResponseError, match="schema validation"):
        parse_agent_payload(ScoutProposal, payload)


def test_missing_evidence_is_rejected() -> None:
    payload = {**VALID_SCOUT, "evidence": []}
    with pytest.raises(AgentResponseError, match="schema validation"):
        parse_agent_payload(ScoutProposal, payload)


def test_extra_fields_are_rejected() -> None:
    payload = {**VALID_SCOUT, "place_order": True}
    with pytest.raises(AgentResponseError, match="schema validation"):
        parse_agent_payload(ScoutProposal, payload)


def test_malformed_json_is_rejected() -> None:
    with pytest.raises(AgentResponseError, match="schema validation"):
        parse_agent_payload(ScoutProposal, "{not json")


def test_valid_skeptic_json_parses() -> None:
    review = parse_agent_payload(SkepticReview, VALID_SKEPTIC)
    assert review.verdict == "APPROVE"


def test_invalid_verdict_is_rejected() -> None:
    payload = {**VALID_SKEPTIC, "verdict": "MAYBE"}
    with pytest.raises(AgentResponseError, match="schema validation"):
        parse_agent_payload(SkepticReview, payload)
