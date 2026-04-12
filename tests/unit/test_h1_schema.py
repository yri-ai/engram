"""Tests for H1 schema induction and transfer."""

from datetime import date

from engram.models.track_b import TrackBEvent, DelinquencyBucket
from engram.services.h1_schema import induce_motifs, evaluate_schema_guided_accuracy
from engram.services.h1_transfer import run_h1_experiment, split_loans


def _make_events(n_loans: int = 20) -> list[TrackBEvent]:
    """Generate events with recurring motifs."""
    events = []
    for i in range(n_loans):
        loan_id = f"LN{i:04d}"
        if i % 4 == 0:
            # Deterioration motif: current → current → d30 → d60
            buckets = [
                DelinquencyBucket.CURRENT, DelinquencyBucket.CURRENT,
                DelinquencyBucket.D30, DelinquencyBucket.D60,
                DelinquencyBucket.D30, DelinquencyBucket.CURRENT,
                DelinquencyBucket.CURRENT, DelinquencyBucket.CURRENT,
            ]
        elif i % 4 == 1:
            # Same deterioration motif (reusable!)
            buckets = [
                DelinquencyBucket.CURRENT, DelinquencyBucket.CURRENT,
                DelinquencyBucket.D30, DelinquencyBucket.D60,
                DelinquencyBucket.D60, DelinquencyBucket.D30,
                DelinquencyBucket.CURRENT, DelinquencyBucket.CURRENT,
            ]
        else:
            buckets = [DelinquencyBucket.CURRENT] * 8

        for m, b in enumerate(buckets):
            events.append(TrackBEvent(
                loan_id=loan_id,
                as_of=date(2025, (m % 12) + 1, 1),
                bucket=b,
                current_upb=100000.0 - m * 1000,
            ))
    return events


def test_induce_motifs_finds_patterns():
    events = _make_events(20)
    motifs = induce_motifs(events, window=3, min_support=2)
    assert len(motifs) > 0
    # The "current-current-d30" → d60 motif should appear
    patterns = [m.pattern for m in motifs]
    assert ("current", "current", "d30") in patterns


def test_motifs_are_compact():
    events = _make_events(20)
    motifs = induce_motifs(events, window=3, min_support=2)
    for m in motifs:
        assert m.nodes == 3  # window size
        assert m.edges == 2


def test_motifs_have_support():
    events = _make_events(20)
    motifs = induce_motifs(events, window=3, min_support=3)
    for m in motifs:
        assert m.support_cases >= 3


def test_evaluate_schema_accuracy():
    events = _make_events(20)
    motifs = induce_motifs(events, window=3, min_support=2)
    result = evaluate_schema_guided_accuracy(motifs, events, window=3)
    assert result["accuracy"] > 0.0
    assert result["coverage"] > 0.0
    assert result["total_windows"] > 0


def test_split_loans():
    events = _make_events(20)
    train, eval_e = split_loans(events, train_frac=0.7, seed=42)
    train_loans = set(e.loan_id for e in train)
    eval_loans = set(e.loan_id for e in eval_e)
    # No overlap
    assert train_loans & eval_loans == set()
    assert len(train_loans) > 0
    assert len(eval_loans) > 0


def test_run_h1_experiment_produces_artifact():
    events = _make_events(20)
    result = run_h1_experiment(events, window=3, min_support=2, seed=42)

    assert result["family"] == "ginnie"
    assert len(result["motifs"]) > 0
    assert result["average_schema_size"] > 0
    assert result["transfer_score"] >= 0.0
    assert "in_family_baseline_accuracy" in result["transfer_eval"]
    assert "schema_guided_accuracy" in result["transfer_eval"]

    # Granularity sweep
    gs = result["granularity_sweep"]
    assert "event_only" in gs
    assert "event_plus_state" in gs
    assert "event_state_gate" in gs
    assert "selected_granularity" in gs


def test_h1_transfer_score_is_reasonable():
    events = _make_events(40)  # more data for better transfer
    result = run_h1_experiment(events, window=3, min_support=2, seed=42)
    # Transfer score should be positive (schemas found on train work on eval)
    assert result["transfer_score"] > 0.0
