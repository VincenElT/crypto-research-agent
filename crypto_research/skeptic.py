"""Skeptic agent: independently critique a Scout proposal.

Approval is not a trade authorization. The Skeptic cannot place orders.
"""

from __future__ import annotations

from typing import Any

from crypto_research.llm import CompleteFn, complete_structured
from crypto_research.schemas import ScoutProposal, SkepticReview, parse_agent_payload

SKEPTIC_INSTRUCTIONS = """
You are Skeptic, an independent research critic. You did not write the Scout
proposal. Decide APPROVE or REJECT.

Reject when evidence is missing, circular, invented, ignores the stated
constraints, overstates confidence, or when BUY/SELL is not supported by the
closed-candle payload. HOLD may be approved if it is the honest conclusion.

Rules:
- Use only the provided market payload and Scout JSON.
- Do not use information after as_of_utc.
- Flag missing evidence explicitly.
- You cannot place orders or override risk limits.
- Approval does not mean the trade should be executed.
- Return only the structured object requested by the schema.
""".strip()


def critique_proposal(
    market_context: dict[str, Any],
    proposal: ScoutProposal,
    *,
    model_name: str,
    complete: CompleteFn = complete_structured,
) -> SkepticReview:
    """Call Skeptic and validate the result against the strict schema."""

    payload = {
        "market": market_context,
        "scout_proposal": proposal.model_dump(),
    }
    raw = complete(
        model_name=model_name,
        instructions=SKEPTIC_INSTRUCTIONS,
        user_payload=payload,
        schema=SkepticReview,
    )
    return parse_agent_payload(SkepticReview, raw)
