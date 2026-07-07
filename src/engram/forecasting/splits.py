"""Temporal split and record-time filtering utilities for forecasting."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import UTC, date, datetime
from typing import Any

from engram.services.track_b_dataset import validate_no_leakage

Row = dict[str, Any]


def walk_forward_windows(
    rows: list[Row], n_windows: int, step_months: int
) -> list[tuple[list[Row], list[Row]]]:
    """Build leakage-checked walk-forward train/eval windows.

    Window i trains on rows strictly before the eval start month and evaluates
    on the next ``step_months`` months. Returned rows are deep copies with
    window-local split labels so source rows are never mutated.
    """
    if n_windows < 0:
        raise ValueError("n_windows must be non-negative")
    if step_months <= 0:
        raise ValueError("step_months must be positive")

    unique_dates = sorted({date.fromisoformat(row["as_of"]) for row in rows})
    windows: list[tuple[list[Row], list[Row]]] = []
    for window_index in range(n_windows):
        eval_start_index = 1 + (window_index * step_months)
        if eval_start_index >= len(unique_dates):
            break
        eval_start = unique_dates[eval_start_index]
        eval_end = _add_months(eval_start, step_months)

        train_rows = [
            deepcopy(row) for row in rows if date.fromisoformat(row["as_of"]) < eval_start
        ]
        eval_rows = [
            deepcopy(row)
            for row in rows
            if eval_start <= date.fromisoformat(row["as_of"]) < eval_end
        ]
        for row in train_rows:
            row["split"] = "train"
        for row in eval_rows:
            row["split"] = "eval"

        validate_no_leakage([*train_rows, *eval_rows])
        windows.append((train_rows, eval_rows))
    return windows


def record_time_filter(row: Row, as_of: str) -> Row:
    """Return a copy with features/context unavailable at ``as_of`` removed."""
    filtered = deepcopy(row)
    features = filtered.get("features", {})
    provenance = filtered.get("feature_provenance", {})
    if isinstance(features, dict) and isinstance(provenance, dict):
        for feature_name in list(features):
            records = provenance.get(feature_name)
            if _has_future_record(records, as_of):
                features.pop(feature_name, None)
                provenance.pop(feature_name, None)

    context_items = filtered.get("context")
    if isinstance(context_items, list):
        filtered["context"] = [
            item
            for item in context_items
            if not (isinstance(item, dict) and _has_future_record([item], as_of))
        ]
    return filtered


def origination_cohort_splits(rows: list[Row]) -> dict[str, list[Row]]:
    """Group rows by the first-seen year of each loan."""
    first_seen: dict[str, str] = {}
    for row in sorted(rows, key=lambda item: (item["loan_id"], item["as_of"])):
        first_seen.setdefault(row["loan_id"], row["as_of"][:4])

    cohorts: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        cohorts[first_seen[row["loan_id"]]].append(deepcopy(row))
    return dict(sorted(cohorts.items()))


def _has_future_record(records: Any, as_of: str) -> bool:
    if not isinstance(records, list):
        return False
    as_of_dt = _parse_record_time(as_of)
    return any(
        isinstance(record, dict)
        and isinstance(record.get("recorded_from"), str)
        and _parse_record_time(record["recorded_from"]) > as_of_dt
        for record in records
    )


def _parse_record_time(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    if "T" not in normalized:
        normalized = f"{normalized}T23:59:59+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _add_months(value: date, months: int) -> date:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    return date(year, month, value.day)
