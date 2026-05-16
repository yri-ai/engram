"""Tests for H3 dataset builders."""

from datetime import date

from engram.models.track_b import TrackBEvent, DelinquencyBucket
from engram.services.h3_dataset import (
    build_endpoint_labels,
    build_next_transition_labels,
    build_short_chain_labels,
    build_branch_ranking_labels,
    add_distractor_features,
)


def _sample_events() -> list[TrackBEvent]:
    """6 months for one loan: current → current → d30 → d60 → d30 → current."""
    buckets = [
        DelinquencyBucket.CURRENT,
        DelinquencyBucket.CURRENT,
        DelinquencyBucket.D30,
        DelinquencyBucket.D60,
        DelinquencyBucket.D30,
        DelinquencyBucket.CURRENT,
    ]
    return [
        TrackBEvent(loan_id="LN1", as_of=date(2025, m, 1), bucket=b, current_upb=100000.0)
        for m, b in zip(range(1, 7), buckets)
    ]


def test_endpoint_labels():
    events = _sample_events()
    rows = build_endpoint_labels(events, horizon=3)
    # 6 events, horizon=3, so first 3 get labels
    assert len(rows) == 3
    # First row: Jan predict Apr (d60)
    assert rows[0]["label"]["next_bucket"] == "d60"
    assert rows[0]["label"]["horizon_months"] == 3


def test_next_transition_labels():
    events = _sample_events()
    rows = build_next_transition_labels(events)
    assert len(rows) == 5  # 6 events - 1
    assert rows[0]["label"]["next_bucket"] == "current"  # Jan→Feb
    assert rows[1]["label"]["next_bucket"] == "d30"  # Feb→Mar


def test_short_chain_labels():
    events = _sample_events()
    rows = build_short_chain_labels(events, chain_length=2)
    assert len(rows) == 4  # 6 - 2
    # Jan: next 2 = Feb(current), Mar(d30)
    assert rows[0]["label"]["next_bucket"] == "current-d30"


def test_branch_ranking_labels():
    events = _sample_events()
    rows = build_branch_ranking_labels(events, window=2)
    assert len(rows) == 4  # 6 - 2
    # Jan: next 2 months = current, d30 → deteriorate
    assert rows[0]["label"]["next_bucket"] == "deteriorate"
    # Mar: next 2 = d60, d30 → recover
    assert rows[2]["label"]["next_bucket"] == "recover"


def test_distractor_features():
    events = _sample_events()
    rows = build_next_transition_labels(events)
    distractor = add_distractor_features(rows)
    assert len(distractor) == len(rows)
    assert "distractor_score" in distractor[0]["features"]
    assert "noise_flag" in distractor[0]["features"]
    # Original features preserved
    assert distractor[0]["features"]["bucket"] == rows[0]["features"]["bucket"]
