"""Tests for H3 experiment runner."""

from datetime import date

from engram.models.track_b import TrackBEvent, DelinquencyBucket
from engram.services.h3_experiments import run_h3_experiment


def _make_events(n_loans: int = 20) -> list[TrackBEvent]:
    """Generate multi-loan event sequences with some transitions."""
    events = []
    for i in range(n_loans):
        loan_id = f"LN{i:04d}"
        # Most loans stay current, some transition
        if i % 5 == 0:
            # This loan deteriorates then recovers
            buckets = [
                DelinquencyBucket.CURRENT,
                DelinquencyBucket.CURRENT,
                DelinquencyBucket.D30,
                DelinquencyBucket.D60,
                DelinquencyBucket.D30,
                DelinquencyBucket.CURRENT,
                DelinquencyBucket.CURRENT,
                DelinquencyBucket.CURRENT,
                DelinquencyBucket.CURRENT,
                DelinquencyBucket.CURRENT,
                DelinquencyBucket.CURRENT,
                DelinquencyBucket.CURRENT,
            ]
        else:
            buckets = [DelinquencyBucket.CURRENT] * 12

        for m, b in enumerate(buckets):
            events.append(TrackBEvent(
                loan_id=loan_id,
                as_of=date(2025, (m % 12) + 1, 1),
                bucket=b,
                current_upb=100000.0 - m * 500,
            ))
    return events


def test_h3_experiment_produces_artifact():
    events = _make_events(20)
    artifact = run_h3_experiment(
        events,
        train_end="2025-06-30",
        eval_end="2025-12-31",
        seeds=[7, 17],
    )

    # All four primitives populated
    assert "endpoint" in artifact.primitives
    assert "next_transition" in artifact.primitives
    assert "short_chain" in artifact.primitives
    assert "branch_ranking" in artifact.primitives

    # Metrics are real numbers
    nt = artifact.primitives["next_transition"]
    assert nt.top1_accuracy_mean > 0.0
    assert nt.brier_score_mean >= 0.0

    # Winner selected
    assert artifact.winner in ("endpoint", "next_transition", "short_chain", "branch_ranking")


def test_h3_artifact_schema():
    events = _make_events(20)
    artifact = run_h3_experiment(events, train_end="2025-06-30", eval_end="2025-12-31", seeds=[7])
    d = artifact.to_dict()

    # Required top-level keys
    assert "seed_list" in d
    assert "primitives" in d
    assert "distractor_robustness" in d
    assert "calibration" in d
    assert "chain_length_sensitivity" in d
    assert "observed_vs_latent" in d
    assert "delayed_outcome_horizons" in d
    assert "winner" in d

    # Chain sensitivity has step_1 through step_4
    for step in ["step_1", "step_2", "step_3", "step_4"]:
        assert step in d["chain_length_sensitivity"]
    assert "selected_horizon" in d["chain_length_sensitivity"]

    # Delayed outcome has 4 horizons
    for h in ["horizon_1m", "horizon_2m", "horizon_3m", "horizon_4m"]:
        assert h in d["delayed_outcome_horizons"]

    # Observed vs latent
    assert "observed_only" in d["observed_vs_latent"]
    assert "latent_enabled" in d["observed_vs_latent"]
    assert "latent_lift_accuracy" in d["observed_vs_latent"]
    assert "latent_lift_brier" in d["observed_vs_latent"]


def test_h3_distractor_robustness_populated():
    events = _make_events(20)
    artifact = run_h3_experiment(events, train_end="2025-06-30", eval_end="2025-12-31", seeds=[7])
    for prim in ["endpoint", "next_transition", "short_chain", "branch_ranking"]:
        assert prim in artifact.distractor_robustness
        assert "distractor_drop" in artifact.distractor_robustness[prim]
