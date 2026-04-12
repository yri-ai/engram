"""Tests for H5 cross-structure transfer."""

from datetime import date

from engram.models.track_b import TrackBEvent, DelinquencyBucket
from engram.services.h5_transfer import split_into_families, run_h5_experiment


def _make_events(n_loans: int = 40) -> list[TrackBEvent]:
    events = []
    for i in range(n_loans):
        loan_id = f"LN{i:04d}"
        if i % 4 == 0:
            # Volatile: delinquent in first half
            buckets = [
                DelinquencyBucket.D30, DelinquencyBucket.D60,
                DelinquencyBucket.D30, DelinquencyBucket.CURRENT,
                DelinquencyBucket.CURRENT, DelinquencyBucket.CURRENT,
                DelinquencyBucket.CURRENT, DelinquencyBucket.CURRENT,
                DelinquencyBucket.CURRENT, DelinquencyBucket.CURRENT,
                DelinquencyBucket.CURRENT, DelinquencyBucket.CURRENT,
            ]
        elif i % 4 == 1:
            # Another volatile pattern
            buckets = [
                DelinquencyBucket.CURRENT, DelinquencyBucket.D30,
                DelinquencyBucket.CURRENT, DelinquencyBucket.D30,
                DelinquencyBucket.CURRENT, DelinquencyBucket.CURRENT,
                DelinquencyBucket.CURRENT, DelinquencyBucket.CURRENT,
                DelinquencyBucket.CURRENT, DelinquencyBucket.CURRENT,
                DelinquencyBucket.D30, DelinquencyBucket.CURRENT,
            ]
        else:
            # Stable: all current
            buckets = [DelinquencyBucket.CURRENT] * 12

        for m, b in enumerate(buckets):
            events.append(TrackBEvent(
                loan_id=loan_id, as_of=date(2025, (m % 12) + 1, 1),
                bucket=b, current_upb=100000.0 - m * 500,
                state="CA" if i % 2 == 0 else "TX",
            ))
    return events


def test_split_into_families():
    events = _make_events(40)
    family_a, family_b = split_into_families(events)
    a_loans = set(e.loan_id for e in family_a)
    b_loans = set(e.loan_id for e in family_b)
    assert len(a_loans) > 0
    assert len(b_loans) > 0
    assert a_loans & b_loans == set()


def test_h5_experiment_produces_artifact():
    events = _make_events(40)
    result = run_h5_experiment(events, window=3, min_support=2)

    assert "families" in result
    assert "shared_core" in result
    assert "family_specific" in result
    assert "family_drift" in result
    assert "transferable_motif_score" in result
    assert "lift_vs_shared_core" in result
    assert "structural_drivers" in result

    drivers = result["structural_drivers"]
    assert "threshold_gates" in drivers
    assert "waterfall_logic" in drivers
    assert "optionality_collapse" in drivers
    assert "counterparty_behavior" in drivers
    assert "covenant_or_contract_triggers" in drivers
    assert sum(drivers.values()) > 0.0


def test_h5_transferable_motif_score():
    events = _make_events(40)
    result = run_h5_experiment(events, window=3, min_support=2)
    assert 0.0 <= result["transferable_motif_score"] <= 1.0
    assert 0.0 <= result["family_drift"] <= 1.0
