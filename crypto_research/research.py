"""Run Scout then Skeptic on a closed-candle market snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from crypto_research.llm import CompleteFn, complete_structured
from crypto_research.schemas import ScoutProposal, SkepticReview
from crypto_research.scout import propose_action
from crypto_research.skeptic import critique_proposal


@dataclass(frozen=True)
class DualAgentResult:
    """Validated Scout proposal plus Skeptic critique. Not an order."""

    proposal: ScoutProposal
    critique: SkepticReview


def run_dual_agents(
    market_context: dict[str, Any],
    *,
    model_name: str,
    complete: CompleteFn = complete_structured,
) -> DualAgentResult:
    """Scout first, then Skeptic. Both outputs are schema-validated."""

    proposal = propose_action(
        market_context, model_name=model_name, complete=complete
    )
    critique = critique_proposal(
        market_context, proposal, model_name=model_name, complete=complete
    )
    return DualAgentResult(proposal=proposal, critique=critique)
