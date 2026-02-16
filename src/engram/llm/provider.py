"""LLM provider wrapper using LiteLLM."""

from __future__ import annotations

import json
import logging
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

        response = await acompletion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            api_key=self.api_key,
        )

        content = response.choices[0].message.content
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse LLM response as JSON: {e}\nContent: {content}"
            ) from e
