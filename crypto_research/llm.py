"""OpenAI Responses API helper for structured agent JSON.

Uses `client.responses.parse` with a Pydantic `text_format`, then the result is
re-validated locally. The API key is read from the environment at call time and
is never logged.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any, TypeVar

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError
from pydantic import BaseModel

from crypto_research.exceptions import AgentResponseError, ConfigurationError
from crypto_research.schemas import parse_agent_payload

TModel = TypeVar("TModel", bound=BaseModel)

CompleteFn = Callable[..., TModel]


def require_openai_api_key() -> str:
    """Return the OpenAI API key or fail. Callers must not print this value."""

    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise ConfigurationError(
            "OPENAI_API_KEY is missing. Copy .env.example to .env and set the key "
            "to run Scout and Skeptic. Market data still does not need exchange keys."
        )
    return key


def complete_structured(
    *,
    model_name: str,
    instructions: str,
    user_payload: dict[str, Any],
    schema: type[TModel],
    client: OpenAI | None = None,
) -> TModel:
    """Ask the model for JSON matching `schema` and validate it strictly."""

    api_client = client or OpenAI(api_key=require_openai_api_key())
    try:
        response = api_client.responses.parse(
            model=model_name,
            instructions=instructions,
            input=json.dumps(user_payload, separators=(",", ":")),
            text_format=schema,
            temperature=0,
        )
    except RateLimitError as exc:
        raise AgentResponseError(
            f"OpenAI rate-limited {schema.__name__}: {exc}"
        ) from exc
    except APIConnectionError as exc:
        raise AgentResponseError(
            f"Could not reach OpenAI for {schema.__name__}: {exc}"
        ) from exc
    except APIStatusError as exc:
        raise AgentResponseError(
            f"OpenAI rejected {schema.__name__} request (HTTP {exc.status_code}). "
            "Use OPENAI_MODEL with a model that supports structured JSON output, "
            f"such as gpt-4o-mini. Details omitted to avoid leaking request data."
        ) from exc

    parsed = getattr(response, "output_parsed", None)
    if parsed is not None:
        return parse_agent_payload(schema, parsed)

    output_text = getattr(response, "output_text", None)
    if output_text:
        return parse_agent_payload(schema, output_text)

    raise AgentResponseError(
        f"OpenAI returned no parseable {schema.__name__} JSON (possible refusal)."
    )
