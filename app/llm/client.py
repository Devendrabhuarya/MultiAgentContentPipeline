"""
LLM client abstraction.

Wraps an OpenAI-compatible chat model (via langchain-openai) and exposes a
single important method: `invoke_structured`, which asks the model to
return JSON matching a given Pydantic schema, parses it, and retries with
an error-correction prompt if parsing/validation fails.

A `MockLLMClient` subclass is provided for tests and offline/demo runs so
that the rest of the system never needs to special-case "no API key".
"""
from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

from app.config import settings

logger = logging.getLogger("pipeline.LLMClient")

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Raised when the LLM fails to produce a usable structured response."""


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_blob(text: str) -> str:
    """
    Extract the first complete, balanced JSON object/array from `text`.

    Some models (notably several free-tier ones) append extra commentary,
    a second copy of the JSON, or trailing notes after a valid JSON block.
    A naive greedy regex (first '{' to last '}') breaks in that case with
    a "json.JSONDecodeError: Extra data" error. This instead walks the
    string tracking brace/bracket depth (while respecting quoted strings
    and escape characters) and returns exactly the first balanced block.
    """
    text = _strip_code_fences(text)

    start = None
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            break
    if start is None:
        return text  # no JSON-looking content at all; let json.loads raise a clear error

    opening = text[start]
    closing = "}" if opening == "{" else "]"
    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == opening:
            depth += 1
        elif ch == closing:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    # Unbalanced (truncated output) -- return what we have and let json.loads
    # raise a clear, retryable error rather than silently guessing.
    return text[start:]


class BaseLLMClient(ABC):
    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return raw text completion from the model."""

    def invoke_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Type[T],
        max_retries: int | None = None,
    ) -> T:
        """
        Call the LLM and parse its response into `schema`.

        On invalid JSON / failed validation, retries with an augmented
        prompt asking the model to correct its output. Raises LLMError
        if all attempts fail.
        """
        retries = settings.llm_max_retries if max_retries is None else max_retries
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        full_system_prompt = (
            f"{system_prompt}\n\n"
            "You must respond with ONLY valid JSON matching this JSON Schema. "
            "Do not include any prose, explanation, or markdown code fences.\n\n"
            f"JSON Schema:\n{schema_json}"
        )

        last_error: Exception | None = None
        current_user_prompt = user_prompt

        for attempt in range(retries + 1):
            raw: str | None = None
            try:
                raw = self.complete(full_system_prompt, current_user_prompt)
                blob = _extract_json_blob(raw)
                data = json.loads(blob)
                return schema.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                logger.warning(
                    "Structured output parse failure (attempt %d/%d): %s",
                    attempt + 1, retries + 1, exc,
                )
                logger.debug(
                    "Raw model output that failed to parse (attempt %d):\n%s",
                    attempt + 1, (raw or "<no raw text captured>")[:2000],
                )
                current_user_prompt = (
                    f"{user_prompt}\n\n"
                    "Your previous response could not be parsed as valid JSON "
                    f"matching the required schema. Error: {exc}\n"
                    "Return ONLY corrected valid JSON, nothing else."
                )
            except Exception as exc:  # noqa: BLE001 - network/API errors
                last_error = exc
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s", attempt + 1, retries + 1, exc
                )

        raise LLMError(
            f"Failed to get valid structured output after {retries + 1} attempts: {last_error}"
        )


class OpenAICompatibleLLMClient(BaseLLMClient):
    """Real LLM client using an OpenAI-compatible Chat Completions API via LangChain."""

    def __init__(self, model_name: str | None = None, temperature: float | None = None):
        from langchain_openai import ChatOpenAI  # imported lazily so tests don't need it

        if not settings.openai_api_key:
            raise LLMError(
                "OPENAI_API_KEY is not set. Configure it in your .env file, "
                "or use MockLLMClient for offline testing."
            )

        self._model_name = model_name or settings.model_name

        kwargs = dict(
            model=self._model_name,
            temperature=settings.temperature if temperature is None else temperature,
            api_key=settings.openai_api_key,
            timeout=settings.api_timeout,
        )
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url

        # Some free-tier "reasoning" models (e.g. Groq's openai/gpt-oss-*,
        # qwen3-family models) route their answer into a separate
        # `reasoning`/`reasoning_content` field and can leave the normal
        # `content` field empty on longer prompts. Where the provider
        # supports it, ask it to skip/minimize reasoning so the actual
        # answer lands in `content` as expected. This is passed through
        # as an extra body param and silently ignored by providers/models
        # that don't recognize it.
        if settings.reasoning_effort:
            kwargs["model_kwargs"] = {"reasoning_effort": settings.reasoning_effort}

        self._chat = ChatOpenAI(**kwargs)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        response = self._chat.invoke(messages)
        content = response.content if isinstance(response.content, str) else str(response.content)

        if content.strip():
            return content

        # Fallback: some reasoning models (notably Groq's gpt-oss family)
        # can return an empty `content` while the actual answer ends up in
        # a `reasoning`/`reasoning_content` field instead. Recover it here
        # rather than failing the whole call outright.
        additional = getattr(response, "additional_kwargs", {}) or {}
        fallback_text = (
            additional.get("reasoning_content")
            or additional.get("reasoning")
            or ""
        )
        if fallback_text.strip():
            logger.warning(
                "Model '%s' returned empty `content`; recovered answer from "
                "the reasoning field instead. Consider setting a non-reasoning "
                "model or REASONING_EFFORT=none in .env.",
                self._model_name,
            )
            return fallback_text

        return content


class MockLLMClient(BaseLLMClient):
    """
    Deterministic offline stand-in for the real LLM.

    Used by tests, CLI dry-runs, and demos when no API key is configured.
    `responses` is an optional queue of canned JSON strings to return in
    order; if exhausted (or not provided), a schema-driven generic filler
    is produced instead so the pipeline still runs end-to-end.
    """

    def __init__(self, responses: list[str] | None = None):
        self._responses = list(responses or [])
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if self._responses:
            return self._responses.pop(0)
        return "{}"
