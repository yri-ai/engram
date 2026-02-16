"""Extraction prompt templates for LLM-based entity and relationship extraction.

Prompts adapted from ARCHITECTURE.md sections 2.2 and 2.3.
Kept in a dedicated module for easy iteration.
"""

from __future__ import annotations

import json
from typing import Any


def build_entity_extraction_prompt(
    message_text: str,
    speaker: str,
    timestamp: str,
    context_entities: list[dict[str, Any]],
) -> str:
    """Build the Stage 1 entity extraction prompt.

    See ARCHITECTURE.md section 2.2.

    Args:
        message_text: Raw conversation message text.
        speaker: Name of the message speaker.
        timestamp: ISO 8601 timestamp of the message.
        context_entities: Recently mentioned entities in this conversation,
            each a dict with at least "name" and "type" keys.

    Returns:
        Formatted prompt string for LLM entity extraction.
    """
    context_str = json.dumps(context_entities, indent=2) if context_entities else "(none)"

    return f"""You are extracting entities from a conversation message for a knowledge graph.

Message: "{message_text}"
Timestamp: {timestamp}
Speaker: {speaker}

Context (entities mentioned recently in this conversation):
{context_str}

Extract entities following these rules:

1. Entity Types:
   - PERSON: Named individuals (participants or mentioned people)
   - PREFERENCE: Things someone likes/dislikes (brands, foods, activities)
   - GOAL: Stated objectives or intentions
   - CONCEPT: Abstract ideas or topics being discussed
   - EVENT: Specific time-bound occurrences (meetings, deadlines, races)
   - TOPIC: Conversation subjects or themes

2. Canonicalization:
   - Normalize names: "Kendra M" -> "Kendra Martinez" (use context)
   - Resolve pronouns: "she" -> match to recent Person entity
   - Handle variations: "running shoes" = "running shoe" (singular form)
   - Merge synonyms: "marathon" = "marathon race"

3. Confidence:
   - Explicit mention: confidence = 1.0
   - Pronoun resolution: confidence = 0.8
   - Inferred from context: confidence = 0.6

Return JSON:
{{
  "entities": [
    {{
      "name": "exact text from message",
      "canonical": "normalized canonical name",
      "type": "PERSON|PREFERENCE|GOAL|CONCEPT|EVENT|TOPIC",
      "confidence": 0.0-1.0
    }}
  ]
}}

Only extract entities that are clearly present. Do not infer entities not mentioned."""


def build_relationship_inference_prompt(
    message_text: str,
    speaker: str,
    timestamp: str,
    entities: list[dict[str, Any]],
    existing_relationships: list[dict[str, Any]],
) -> str:
    """Build the Stage 2 relationship inference prompt.

    See ARCHITECTURE.md section 2.3. Includes explicit relationship type
    constraint from Open Question #2 to prevent type explosion.

    Args:
        message_text: Raw conversation message text.
        speaker: Name of the message speaker.
        timestamp: ISO 8601 timestamp of the message.
        entities: Entities extracted in Stage 1, each a dict with
            at least "name" and "type" keys.
        existing_relationships: Currently active relationships involving
            these entities, for context.

    Returns:
        Formatted prompt string for LLM relationship inference.
    """
    entities_str = json.dumps(entities, indent=2)

    if existing_relationships:
        existing_str = json.dumps(existing_relationships, indent=2)
    else:
        existing_str = "(none)"

    return f"""You are inferring relationships between entities from a conversation message.

Message: "{message_text}"
Timestamp: {timestamp}
Speaker: {speaker}

Extracted Entities:
{entities_str}

Context (existing relationships involving these entities):
{existing_str}

Infer relationships following these rules:

1. Relationship Types:
   - prefers: Person -> Preference (positive sentiment)
   - avoids: Person -> Preference (negative sentiment)
   - knows: Person -> Person (social connection)
   - discussed: Person -> Topic (conversation participation)
   - mentioned_with: Entity -> Entity (co-occurrence, no strong semantic link)
   - has_goal: Person -> Goal (ownership of objective)
   - relates_to: Entity -> Entity (generic semantic relationship)

Use ONLY these relationship types:
- prefers: likes, dislikes, choices, opinions
- avoids: negative sentiment, rejection
- knows: social relationships, connections
- discussed: conversation participation
- mentioned_with: co-occurrence (fallback)
- has_goal: ownership of objective
- relates_to: generic connection (fallback if no better match)

NEVER create compound type names (e.g., "slightly_prefers", "knows_well").
Capture nuance via properties, not type names.

If relationship doesn't match any type, use "relates_to" and explain in evidence field.

2. Evidence:
   - Only infer relationships explicitly stated or strongly implied
   - Provide direct quote or paraphrased evidence
   - Mark uncertain inferences with confidence < 1.0

3. Temporal Markers:
   - "now prefers" -> new relationship, invalidate old
   - "used to like" -> relationship with valid_to = now
   - "always loved" -> relationship with valid_from = distant past
   - "recently switched to" -> relationship transition

4. Confidence Scoring:
   - Direct statement: 1.0 ("I love Nike shoes")
   - Strong implication: 0.8 ("Nike is the best")
   - Weak inference: 0.6 ("mentioned Nike positively")
   - Co-occurrence only: 0.4 ("talked about Nike and running")

Return JSON (SEMANTIC JUDGMENTS ONLY - NO IDs):
{{
  "relationships": [
    {{
      "source_mention": "exact entity mention from message",
      "target_mention": "exact entity mention from message",
      "type": "relationship type",
      "confidence": 0.0-1.0,
      "evidence": "quote or summary",
      "temporal_marker": "new|update|past|ongoing"
    }}
  ]
}}

CRITICAL: Return entity MENTIONS (text from message), NOT IDs or UUIDs.
Example: {{"source_mention": "Kendra", "target_mention": "Nike shoes"}}"""
