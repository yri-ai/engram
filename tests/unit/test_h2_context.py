"""Tests for H2 context profiles and metrics."""

from engram.services.h2_context import (
    profile_full,
    profile_top_k,
    profile_schema_guided,
    profile_minimal_discriminative,
    compute_evidence_gaps,
    compute_competing_cause_discrimination,
)


def _sample_row():
    return {
        "event_id": "LN1-202501",
        "message_id": "track-b-LN1-202501",
        "loan_id": "LN1",
        "as_of": "2025-01-01",
        "features": {
            "bucket": "current",
            "current_upb": 100000.0,
            "interest_rate": 7.5,
            "credit_score": 720,
            "state": "CA",
            "prev_bucket": "current",
            "upb_change_pct": -0.005,
        },
        "label": {"next_bucket": "current", "horizon_months": 1},
        "split": "eval",
    }


def test_profile_full_keeps_all():
    row = _sample_row()
    result = profile_full(row)
    assert result["features"] == row["features"]


def test_profile_top_k_limits_features():
    row = _sample_row()
    result = profile_top_k(row, k=3)
    assert len(result["features"]) == 3
    assert "bucket" in result["features"]


def test_profile_schema_guided_only_bucket_and_momentum():
    row = _sample_row()
    result = profile_schema_guided(row)
    assert set(result["features"].keys()) == {"bucket", "prev_bucket"}


def test_profile_minimal_for_current_loan():
    row = _sample_row()
    result = profile_minimal_discriminative(row)
    # Current loan with prev_bucket → bucket + prev_bucket
    assert "bucket" in result["features"]


def test_profile_minimal_for_delinquent_loan():
    row = _sample_row()
    row["features"]["bucket"] = "d30"
    result = profile_minimal_discriminative(row)
    assert result["features"] == {"bucket": "d30"}


def test_evidence_gaps():
    # Predictions with narrow margins → gaps identified
    preds = [
        {"probabilities": {"current": 0.6, "d30": 0.4}},  # margin 0.2 < 0.3 → gap
        {"probabilities": {"current": 0.9, "d30": 0.1}},  # margin 0.8 ≥ 0.3 → no gap
        {"probabilities": {"current": 0.55, "d30": 0.45}},  # margin 0.1 < 0.3 → gap
    ]
    coverage = compute_evidence_gaps([], preds)
    assert abs(coverage - 2 / 3) < 0.01


def test_competing_cause_discrimination():
    rows = [
        {"label": {"next_bucket": "current"}},
        {"label": {"next_bucket": "d30"}},
    ]
    preds = [
        {"top_bucket": "current", "probabilities": {"current": 0.7, "d30": 0.3}},
        {"top_bucket": "d30", "probabilities": {"current": 0.4, "d30": 0.6}},
    ]
    result = compute_competing_cause_discrimination(rows, preds)
    assert result["subset_count"] == 2  # both have top < 0.95
    assert result["accuracy"] == 1.0  # both correct
    assert result["mean_margin_top1_top2"] > 0.0
