import math
from datetime import UTC, datetime

from engram.services.temporal import calculate_decayed_confidence, calculate_reinforced_confidence


def test_no_decay_at_time_zero():
    confidence = calculate_decayed_confidence(
        base_confidence=1.0,
        rel_type="prefers",
        last_mentioned=datetime(2024, 1, 1, tzinfo=UTC),
        current_time=datetime(2024, 1, 1, tzinfo=UTC),
    )
    assert confidence == 1.0


def test_preference_decays_fast():
    """Preferences decay at 0.05/day. After 30 days: ~0.22"""
    confidence = calculate_decayed_confidence(
        base_confidence=1.0,
        rel_type="prefers",
        last_mentioned=datetime(2024, 1, 1, tzinfo=UTC),
        current_time=datetime(2024, 1, 31, tzinfo=UTC),
    )
    expected = math.exp(-0.05 * 30)
    assert abs(confidence - expected) < 0.01


def test_social_relationship_decays_slow():
    """Knows relationships decay at 0.005/day. After 30 days: ~0.86"""
    confidence = calculate_decayed_confidence(
        base_confidence=1.0,
        rel_type="knows",
        last_mentioned=datetime(2024, 1, 1, tzinfo=UTC),
        current_time=datetime(2024, 1, 31, tzinfo=UTC),
    )
    expected = math.exp(-0.005 * 30)
    assert abs(confidence - expected) < 0.01


def test_confidence_floors_at_minimum():
    """Confidence never drops below 0.1."""
    confidence = calculate_decayed_confidence(
        base_confidence=1.0,
        rel_type="prefers",
        last_mentioned=datetime(2024, 1, 1, tzinfo=UTC),
        current_time=datetime(2024, 7, 1, tzinfo=UTC),  # 6 months
    )
    assert confidence == 0.1


def test_reinforcement_boosts_confidence():
    result = calculate_reinforced_confidence(
        current_decayed=0.5,
        new_mention_confidence=0.9,
    )
    # 70% new + 30% old = 0.63 + 0.15 = 0.78
    assert abs(result - 0.78) < 0.01
