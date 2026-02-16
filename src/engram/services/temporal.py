"""Temporal reasoning: decay, reinforcement, point-in-time queries."""

from __future__ import annotations

import math
from datetime import datetime  # noqa: TC003

from engram.config import Settings  # noqa: TC001

CONFIDENCE_FLOOR = 0.1


def calculate_decayed_confidence(
    base_confidence: float,
    rel_type: str,
    last_mentioned: datetime,
    current_time: datetime,
    settings: Settings | None = None,
) -> float:
    """Exponential decay: confidence(t) = base * exp(-rate * days).

    Args:
        base_confidence: Initial confidence (0.0-1.0)
        rel_type: Relationship type (e.g., "prefers", "knows")
        last_mentioned: When relationship was last mentioned
        current_time: Current time for decay calculation
        settings: Optional Settings instance for configurable decay rates.
                 If None, uses default rates.

    Returns:
        Decayed confidence, floored at CONFIDENCE_FLOOR (0.1)

    Note: Open Question #5 (Decay Rate Tuning) — Now configurable via Settings.
    """
    days_elapsed = (current_time - last_mentioned).total_seconds() / 86400
    if days_elapsed <= 0:
        return base_confidence

    # Get decay rates from settings (configurable) or use defaults
    if settings:
        decay_rates = settings.get_decay_rates()
    else:
        # Fallback to hardcoded defaults if no settings provided
        decay_rates = {
            "prefers": 0.05,
            "avoids": 0.04,
            "knows": 0.005,
            "discussed": 0.03,
            "mentioned_with": 0.1,
            "has_goal": 0.02,
            "relates_to": 0.01,
            "default": 0.01,
        }

    decay_rate = decay_rates.get(rel_type, decay_rates["default"])
    decayed = base_confidence * math.exp(-decay_rate * days_elapsed)
    return max(decayed, CONFIDENCE_FLOOR)


def calculate_reinforced_confidence(
    current_decayed: float,
    new_mention_confidence: float,
) -> float:
    """Boost confidence on re-mention. 70% new, 30% old."""
    return min((new_mention_confidence * 0.7) + (current_decayed * 0.3), 1.0)
