"""Task 0.4 walk-forward split and record-time replay tests."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from engram.forecasting.splits import (
    origination_cohort_splits,
    record_time_filter,
    walk_forward_windows,
)


def _row(loan_id: str, as_of: str, bucket: str = "current") -> dict[str, Any]:
    return {
        "event_id": f"{loan_id}-{as_of}",
        "message_id": f"track-b-{loan_id}-{as_of}",
        "loan_id": loan_id,
        "as_of": as_of,
        "split": "",
        "features": {"bucket": bucket, "current_upb": 100.0},
        "feature_provenance": {
            "bucket": [{"source_id": "bucket-source", "recorded_from": as_of}],
            "current_upb": [{"source_id": "upb-source", "recorded_from": as_of}],
        },
        "label": {"next_bucket": bucket, "horizon_months": 1},
    }


def test_walk_forward_windows_have_exact_boundaries_and_disjoint_train_eval():
    rows = [
        _row("L1", "2025-01-01"),
        _row("L1", "2025-02-01"),
        _row("L1", "2025-03-01"),
        _row("L1", "2025-04-01"),
    ]

    windows = walk_forward_windows(rows, n_windows=2, step_months=1)

    assert [[row["as_of"] for row in train] for train, _ in windows] == [
        ["2025-01-01"],
        ["2025-01-01", "2025-02-01"],
    ]
    assert [[row["as_of"] for row in eval_rows] for _, eval_rows in windows] == [
        ["2025-02-01"],
        ["2025-03-01"],
    ]
    for train_rows, eval_rows in windows:
        assert {row["message_id"] for row in train_rows}.isdisjoint(
            {row["message_id"] for row in eval_rows}
        )
        assert {row["split"] for row in train_rows} == {"train"}
        assert {row["split"] for row in eval_rows} == {"eval"}


def test_walk_forward_windows_do_not_mutate_source_rows():
    rows = [_row("L1", "2025-01-01"), _row("L1", "2025-02-01")]
    before = deepcopy(rows)

    walk_forward_windows(rows, n_windows=1, step_months=1)

    assert rows == before


def test_walk_forward_windows_step_months_advances_eval_start_without_overlap():
    rows = [
        _row("L1", "2025-01-01"),
        _row("L1", "2025-02-01"),
        _row("L1", "2025-03-01"),
        _row("L1", "2025-04-01"),
    ]

    windows = walk_forward_windows(rows, n_windows=2, step_months=2)

    assert [[row["as_of"] for row in eval_rows] for _, eval_rows in windows] == [
        ["2025-02-01", "2025-03-01"],
        ["2025-04-01"],
    ]


def test_record_time_filter_removes_future_recorded_features_only():
    row = _row("L1", "2025-06-01")
    row["features"]["future_recorded_canary"] = "d90_plus"
    row["feature_provenance"]["future_recorded_canary"] = [
        {"source_id": "future", "recorded_from": "2025-12-01"}
    ]
    row["features"]["same_day_feature"] = 1.0
    row["feature_provenance"]["same_day_feature"] = [
        {"source_id": "same-day", "recorded_from": "2025-06-01"}
    ]

    filtered = record_time_filter(row, as_of="2025-06-01")

    assert "future_recorded_canary" not in filtered["features"]
    assert "future_recorded_canary" not in filtered["feature_provenance"]
    assert filtered["features"]["same_day_feature"] == 1.0
    assert filtered["features"]["bucket"] == "current"
    assert "future_recorded_canary" in row["features"], "source row must not be mutated"


def test_origination_cohort_splits_partition_by_first_seen_year():
    rows = [
        _row("OLD", "2024-12-01"),
        _row("OLD", "2025-01-01"),
        _row("NEW", "2025-03-01"),
        _row("LATER", "2026-01-01"),
    ]

    cohorts = origination_cohort_splits(rows)

    assert {year: [row["loan_id"] for row in cohort] for year, cohort in cohorts.items()} == {
        "2024": ["OLD", "OLD"],
        "2025": ["NEW"],
        "2026": ["LATER"],
    }
