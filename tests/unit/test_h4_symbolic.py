"""Tests for H4 symbolic pruning."""

from datetime import date

from engram.models.track_b import TrackBEvent, DelinquencyBucket
from engram.services.h4_symbolic import prune_predictions, run_h4_experiment


def test_prune_predictions_loose():
    """Loose: only allow observed outcomes."""
    from collections import Counter
    constraints = {("current", "current", "d30"): Counter({"d60": 10, "current": 5})}
    predictions = {"current": 0.3, "d30": 0.4, "d60": 0.2, "d90": 0.1}
    pruned = prune_predictions(predictions, ("current", "current", "d30"), constraints, "loose")
    assert pruned.get("d30", 0.0) == 0.0  # not in constraint outcomes
    assert pruned.get("d90", 0.0) == 0.0
    assert pruned["d60"] > 0.0
    assert pruned["current"] > 0.0
    assert abs(sum(pruned.values()) - 1.0) < 0.001


def test_prune_predictions_hard():
    """Hard: only top 2 outcomes."""
    from collections import Counter
    constraints = {("a", "b", "c"): Counter({"x": 100, "y": 50, "z": 5})}
    predictions = {"x": 0.3, "y": 0.3, "z": 0.4}
    pruned = prune_predictions(predictions, ("a", "b", "c"), constraints, "hard")
    assert pruned.get("z", 0.0) == 0.0  # z is rank 3
    assert pruned["x"] > 0.0
    assert pruned["y"] > 0.0


def test_prune_no_constraint_passes_through():
    predictions = {"a": 0.5, "b": 0.5}
    pruned = prune_predictions(predictions, ("x", "y", "z"), {}, "loose")
    assert pruned == predictions


def _make_events(n_loans: int = 30) -> list[TrackBEvent]:
    events = []
    for i in range(n_loans):
        loan_id = f"LN{i:04d}"
        if i % 5 == 0:
            buckets = [
                DelinquencyBucket.CURRENT, DelinquencyBucket.CURRENT,
                DelinquencyBucket.D30, DelinquencyBucket.D60,
                DelinquencyBucket.D30, DelinquencyBucket.CURRENT,
                DelinquencyBucket.CURRENT, DelinquencyBucket.CURRENT,
                DelinquencyBucket.CURRENT, DelinquencyBucket.CURRENT,
                DelinquencyBucket.CURRENT, DelinquencyBucket.CURRENT,
            ]
        else:
            buckets = [DelinquencyBucket.CURRENT] * 12
        for m, b in enumerate(buckets):
            events.append(TrackBEvent(
                loan_id=loan_id, as_of=date(2025, (m % 12) + 1, 1),
                bucket=b, current_upb=100000.0 - m * 500,
            ))
    return events


def test_h4_experiment_produces_artifact():
    events = _make_events(30)
    result = run_h4_experiment(events, train_end="2025-06-30", eval_end="2025-12-31")

    assert "without_symbolic" in result
    assert "tightness" in result
    assert "contradiction_reduction" in result
    assert "recall_loss" in result
    assert "error_examples" in result

    assert "loose" in result["tightness"]
    assert "medium" in result["tightness"]
    assert "hard" in result["tightness"]
    assert "selected_tightness" in result["tightness"]

    for level in ["loose", "medium", "hard"]:
        assert "contradiction_rate" in result["tightness"][level]
        assert "recall" in result["tightness"][level]
        assert "novelty_prune_rate" in result["tightness"][level]
