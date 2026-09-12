"""Text-only OpenAI calls for search query parsing."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any


INVALID_JSON_RETRY_PROMPT = "Return the same result as valid JSON only."

MAX_TOKENS = 1200
RETRY_DELAYS = [10, 20, 40]
ERROR_EXCERPT_CHARS = 500


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)
    if stripped.startswith("{") and stripped.endswith("}"):
        return json.loads(stripped)
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(match.group(0))


def _text_format(json_schema: dict[str, Any] | None, json_schema_name: str) -> dict[str, Any] | None:
    if json_schema is None:
        return None
    return {
        "format": {
            "type": "json_schema",
            "name": json_schema_name,
            "strict": True,
            "schema": json_schema,
        }
    }


def _create_kwargs(model: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "max_output_tokens": MAX_TOKENS,
    }
    # Sampling options are not universally supported by reasoning models.
    # Keep deterministic sampling for the established non-reasoning families;
    # use provider defaults for other models, including GPT-5 and o-series.
    if any(model == family or model.startswith(family + "-")
           for family in ("gpt-4.1", "gpt-4o")):
        kwargs["temperature"] = 0
    return kwargs


def call_text_json_with_error(
    *,
    client: Any,
    model: str,
    user_prompt: str,
    system_prompt_text: str,
    retry_prompt_text: str | None = None,
    json_schema: dict[str, Any] | None = None,
    json_schema_name: str = "text_json",
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    retry_text = retry_prompt_text or INVALID_JSON_RETRY_PROMPT
    text_format = _text_format(json_schema, json_schema_name)
    request_input = [
        {
            "role": "system",
            "content": [{"type": "input_text", "text": system_prompt_text}],
        },
        {
            "role": "user",
            "content": [{"type": "input_text", "text": user_prompt}],
        },
    ]

    create_kwargs = _create_kwargs(model)
    if text_format is not None:
        create_kwargs["text"] = text_format

    last_text: str | None = None
    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            response = client.responses.create(
                input=request_input,
                **create_kwargs,
            )
            last_text = response.output_text
            break
        except Exception as exc:
            name = exc.__class__.__name__
            if (
                name == "RateLimitError"
                and _is_retryable_rate_limit(exc)
                and attempt < len(RETRY_DELAYS)
            ):
                time.sleep(RETRY_DELAYS[attempt])
                continue
            return None, _llm_error(
                stage="api_call",
                exc=exc,
                attempt=attempt + 1,
                model=model,
            )

    if not last_text:
        return None, _llm_error(stage="empty_response", message="response.output_text was empty", model=model)

    try:
        return _extract_json(last_text), None
    except Exception as exc:
        first_parse_error = _llm_error(
            stage="json_parse",
            exc=exc,
            model=model,
            response_excerpt=_excerpt(last_text),
        )
        try:
            retry_response = client.responses.create(
                input=[
                    *request_input,
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": retry_text,
                            }
                        ],
                    },
                ],
                **create_kwargs,
            )
        except Exception as retry_exc:
            return None, _llm_error(
                stage="json_retry_api_call",
                exc=retry_exc,
                model=model,
                first_parse_error=first_parse_error,
            )
        try:
            return _extract_json(retry_response.output_text), None
        except Exception as retry_parse_exc:
            return None, _llm_error(
                stage="json_retry_parse",
                exc=retry_parse_exc,
                model=model,
                first_parse_error=first_parse_error,
                response_excerpt=_excerpt(retry_response.output_text),
            )


def _llm_error(
    *,
    stage: str,
    model: str,
    exc: Exception | None = None,
    message: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    payload = {
        "stage": stage,
        "model": model,
        "message": message or (str(exc) if exc else ""),
    }
    if exc is not None:
        payload["type"] = exc.__class__.__name__
        for key in ("status_code", "code", "param"):
            value = getattr(exc, key, None)
            if isinstance(value, (str, int)):
                payload[key] = value
    for key, value in extra.items():
        if value is not None:
            payload[key] = value
    return payload


def _is_retryable_rate_limit(exc: Exception) -> bool:
    message = str(exc).lower()
    return "insufficient_quota" not in message and "quota" not in message


def _excerpt(text: Any) -> str:
    if text is None:
        return ""
    value = str(text)
    return value[:ERROR_EXCERPT_CHARS]


def default_model() -> str:
    return os.getenv("FASHION_HOW_OPENAI_MODEL") or os.getenv("OPENAI_MODEL", "gpt-5.4-mini")


