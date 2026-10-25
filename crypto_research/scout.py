"""Scout agent: propose BUY, SELL, or HOLD from structured market data.

The Scout never places orders and cannot override later risk limits.
"""

from __future__ import annotations

from typing import Any

from crypto_research.llm import CompleteFn, complete_structured
from crypto_research.schemas import ScoutProposal, parse_agent_payload

SCOUT_INSTRUCTIONS = """
You are Scout, an educational crypto research analyst.

Given only the JSON market payload, propose BUY, SELL, or HOLD.
Cite evidence that is actually present in the payload (SMA 50, SMA 200,
30-day momentum, recent closed bars, trend label). List real risks.
If the payload is too thin for a directional view, choose HOLD.

Rules:
- Do not use information after as_of_utc.
- Do not claim the setup is profitable.
- confidence is subjective conviction in [0, 1], not P(profit).
- You cannot place orders or change risk limits.
- Return only the structured object requested by the schema.
""".strip()


def propose_action(
    market_context: dict[str, Any],
    *,
    model_name: str,
    complete: CompleteFn = complete_structured,
) -> ScoutProposal:
    """Call Scout and validate the result against the strict schema."""

    raw = complete(
        model_name=model_name,
        instructions=SCOUT_INSTRUCTIONS,
        user_payload=market_context,
        schema=ScoutProposal,
    )
    return parse_agent_payload(ScoutProposal, raw)
