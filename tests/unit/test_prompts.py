"""Tests for LLM extraction prompt templates."""

from engram.llm.prompts import build_entity_extraction_prompt, build_relationship_inference_prompt


def test_entity_prompt_includes_message():
    prompt = build_entity_extraction_prompt(
        message_text="Kendra loves Nike shoes",
        speaker="Kendra",
        timestamp="2024-01-15T10:00:00Z",
        context_entities=[],
    )
    assert "Kendra loves Nike shoes" in prompt
    assert "Kendra" in prompt
    assert "PERSON" in prompt  # Should mention entity types


def test_relationship_prompt_includes_entities():
    prompt = build_relationship_inference_prompt(
        message_text="Kendra loves Nike shoes",
        speaker="Kendra",
        timestamp="2024-01-15T10:00:00Z",
        entities=[
            {"name": "Kendra", "type": "PERSON"},
            {"name": "Nike shoes", "type": "PREFERENCE"},
        ],
        existing_relationships=[],
    )
    assert "Kendra" in prompt
    assert "Nike shoes" in prompt
    assert "prefers" in prompt  # Should mention relationship types
