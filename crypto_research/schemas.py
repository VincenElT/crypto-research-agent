"""Strict JSON schemas for Scout and Skeptic responses.

Pydantic models are the source of truth. Every model output is validated here
even when the OpenAI SDK already parsed structured output, so extra fields,
missing evidence, and invalid enums cannot leak into the pipeline.
"""

from __future__ import annotations

import json
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from crypto_research.exceptions import AgentResponseError

TModel = TypeVar("TModel", bound=BaseModel)

Action = Literal["BUY", "SELL", "HOLD"]
Verdict = Literal["APPROVE", "REJECT"]


class ScoutProposal(BaseModel):
    """Scout output. Confidence is not a calibrated probability of profit."""

    model_config = ConfigDict(extra="forbid")

    action: Action
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Subjective conviction in [0, 1], not a probability of profit.",
    )
    thesis: str = Field(min_length=1, max_length=2000)
    evidence: list[str] = Field(min_length=1, max_length=12)
    risks: list[str] = Field(min_length=1, max_length=12)
    missing_data: list[str] = Field(max_length=12)


class SkepticReview(BaseModel):
    """Skeptic output. Approval is not permission to place an order."""

    model_config = ConfigDict(extra="forbid")

    verdict: Verdict
    agrees_with_action: bool
    rationale: str = Field(min_length=1, max_length=2000)
    missing_evidence: list[str] = Field(min_length=0, max_length=12)
    weaknesses: list[str] = Field(min_length=1, max_length=12)
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Subjective conviction in the critique, not a probability of profit.",
    )


def strip_json_fence(text: str) -> str:
    """Remove optional markdown fences around a JSON object."""

    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def parse_agent_payload(model: type[TModel], payload: Any) -> TModel:
    """Validate an LLM payload against a strict schema.

    Raises:
        AgentResponseError: If JSON is malformed or does not match the schema.
    """

    try:
        if isinstance(payload, model):
            return model.model_validate(payload.model_dump())
        if isinstance(payload, BaseModel):
            return model.model_validate(payload.model_dump())
        if isinstance(payload, str):
            return model.model_validate_json(strip_json_fence(payload))
        if isinstance(payload, dict):
            return model.model_validate(payload)
        if payload is None:
            raise AgentResponseError(f"{model.__name__} response was empty.")
        raise AgentResponseError(
            f"{model.__name__} response had unsupported type {type(payload).__name__}."
        )
    except (ValidationError, json.JSONDecodeError, ValueError, TypeError) as exc:
        raise AgentResponseError(
            f"{model.__name__} failed strict JSON schema validation: {exc}"
        ) from exc
