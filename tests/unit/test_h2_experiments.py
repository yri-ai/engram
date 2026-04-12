"""Tests for H2 experiment runner."""

from datetime import date

from engram.models.track_b import TrackBEvent, DelinquencyBucket
from engram.services.h2_experiments import run_h2_experiment


def _make_events(n_loans: int = 30) -> list[TrackBEvent]:
    """Generate events with transitions for context ablation testing."""
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
                loan_id=loan_id,
                as_of=date(2025, (m % 12) + 1, 1),
                bucket=b,
                current_upb=100000.0 - m * 500,
                interest_rate=7.0,
                credit_score=720,
                state="CA",
            ))
    return events


def test_h2_experiment_produces_artifact():
    events = _make_events()
    artifact = run_h2_experiment(events, train_end="2025-06-30", eval_end="2025-12-31")

    assert "full" in artifact.profiles
    assert "top_k" in artifact.profiles
    assert "schema_guided" in artifact.profiles
    assert "minimal_discriminative" in artifact.profiles


def test_h2_artifact_schema():
    events = _make_events()
    artifact = run_h2_experiment(events, train_end="2025-06-30", eval_end="2025-12-31")
    d = artifact.to_dict()

    assert d["selected_primitive"] == "next_transition"
    assert "profiles" in d
    assert "evidence_gap_coverage" in d
    assert "competing_cause_discrimination" in d
    assert "distractor_report" in d

    for profile in ["full", "top_k", "schema_guided", "minimal_discriminative"]:
        assert profile in d["profiles"]
        assert "top1_accuracy_mean" in d["profiles"][profile]
        assert "brier_score_mean" in d["profiles"][profile]
        assert profile in d["distractor_report"]


def test_h2_competing_cause_has_required_keys():
    events = _make_events()
    artifact = run_h2_experiment(events, train_end="2025-06-30", eval_end="2025-12-31")
    cc = artifact.competing_cause_discrimination
    assert "subset_count" in cc
    assert "accuracy" in cc
    assert "mean_margin_top1_top2" in cc
