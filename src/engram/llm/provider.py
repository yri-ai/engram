"""LLM provider wrapper using LiteLLM."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from litellm import acompletion

logger = logging.getLogger(__name__)


class LLMProvider:
    """Async LLM wrapper with JSON mode and error handling."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.api_key = api_key

    async def complete_json(
        self,
        prompt: str,
        system: str = "You are a helpful assistant that outputs JSON.",
    ) -> dict[str, Any]:
        """Call LLM and parse JSON response."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "api_key": self.api_key,
        }
        # Some newer models (e.g. claude-sonnet-5) deprecate the temperature param.
        _m = (self.model or "").lower()
        if not ("sonnet-5" in _m or "opus-4-8" in _m or "fable-5" in _m):
            kwargs["temperature"] = self.temperature
        response = await acompletion(**kwargs)

        content = response.choices[0].message.content
        cleaned = (content or "").strip()
        # Strip markdown code fences (Anthropic often wraps JSON in ```json ... ```)
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as err:
            # Fallback: extract the outermost JSON object
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Failed to parse LLM response as JSON: {e}\nContent: {content}"
                    ) from e
            raise ValueError(f"Failed to parse LLM response as JSON.\nContent: {content}") from err
