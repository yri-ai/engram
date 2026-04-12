"""Track B dataset builder with leakage-safe splits."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from engram.models.track_b import TrackBEvent


class LeakageError(Exception):
    """Raised when data leakage is detected in train/eval/holdout splits."""


def build_labeled_rows(events: list[TrackBEvent]) -> list[dict[str, Any]]:
    """Build labeled rows from a chronological sequence of events.

    For each loan, pairs consecutive months: the current month's features
    become the input, and the next month's bucket becomes the label.
    The last observation per loan is dropped (no label available).

    Args:
        events: List of TrackBEvent objects (need not be sorted).

    Returns:
        List of canonical row dicts with features and labels.
    """
    # Group by loan_id, sort by as_of
    by_loan: dict[str, list[TrackBEvent]] = defaultdict(list)
    for e in events:
        by_loan[e.loan_id].append(e)

    rows: list[dict[str, Any]] = []
    for loan_id in sorted(by_loan):
        loan_events = sorted(by_loan[loan_id], key=lambda e: e.as_of)
        for i in range(len(loan_events) - 1):
            current = loan_events[i]
            next_event = loan_events[i + 1]
            row = current.to_canonical_row(
                next_bucket=next_event.bucket,
                split="",  # assigned later
            )
            rows.append(row)

    return rows


def assign_splits(
    rows: list[dict[str, Any]],
    train_end: str,
    eval_end: str,
) -> list[dict[str, Any]]:
    """Assign train/eval/holdout splits based on as_of date.

    Args:
        rows: Canonical row dicts from build_labeled_rows.
        train_end: ISO date string. Rows with as_of <= this are train.
        eval_end: ISO date string. Rows with as_of <= this (and > train_end) are eval.
            Rows after eval_end are holdout.
    """
    for row in rows:
        as_of = row["as_of"]
        if as_of <= train_end:
            row["split"] = "train"
        elif as_of <= eval_end:
            row["split"] = "eval"
        else:
            row["split"] = "holdout"
    return rows


def validate_no_leakage(rows: list[dict[str, Any]]) -> None:
    """Validate that no data leakage exists across splits.

    Checks:
    1. No duplicate message_id across different splits.
    2. No same (loan_id, year-month) appearing in multiple splits.

    Raises:
        LeakageError: If leakage is detected.
    """
    # Check 1: duplicate message_id across splits
    msg_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        msg_splits[row["message_id"]].add(row["split"])
    for msg_id, splits in msg_splits.items():
        if len(splits) > 1:
            raise LeakageError(
                f"message_id {msg_id} appears in multiple splits: {splits}"
            )

    # Check 2: same (loan_id, year-month) across splits
    loan_month_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        year_month = row["as_of"][:7]  # "2025-10"
        key = (row["loan_id"], year_month)
        loan_month_splits[key].add(row["split"])
    for key, splits in loan_month_splits.items():
        if len(splits) > 1:
            raise LeakageError(
                f"loan_id={key[0]} month={key[1]} appears in multiple splits: {splits}"
            )
