"""Tests for LLM provider wrapper."""

from unittest.mock import AsyncMock, patch

import pytest

from engram.llm.provider import LLMProvider


async def test_complete_returns_parsed_json():
    provider = LLMProvider(model="gpt-4o-mini", temperature=0.1)
    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock(message=AsyncMock(content='{"entities": []}'))]

    with patch("engram.llm.provider.acompletion", return_value=mock_response):
        result = await provider.complete_json("Extract entities", system="You are a helper")
        assert result == {"entities": []}


async def test_complete_handles_invalid_json():
    provider = LLMProvider(model="gpt-4o-mini", temperature=0.1)
    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock(message=AsyncMock(content="not json"))]

    with (
        patch("engram.llm.provider.acompletion", return_value=mock_response),
        pytest.raises(ValueError, match="Failed to parse"),
    ):
        await provider.complete_json("Extract entities")
