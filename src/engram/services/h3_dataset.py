"""Dataset preparation for H3 predictive primitive experiments.

Builds labeled datasets for each of the four predictive primitives
from the same underlying event stream.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from engram.models.track_b import DelinquencyBucket, TrackBEvent


def _group_by_loan(events: list[TrackBEvent]) -> dict[str, list[TrackBEvent]]:
    by_loan: dict[str, list[TrackBEvent]] = defaultdict(list)
    for e in events:
        by_loan[e.loan_id].append(e)
    for loan_id in by_loan:
        by_loan[loan_id].sort(key=lambda e: e.as_of)
    return by_loan


def build_endpoint_labels(
    events: list[TrackBEvent],
    horizon: int = 6,
) -> list[dict[str, Any]]:
    """Endpoint primitive: predict bucket at N months in the future.

    For each event, the label is the bucket `horizon` months later.
    Events without enough future data are dropped.
    """
    by_loan = _group_by_loan(events)
    rows = []
    for loan_events in by_loan.values():
        for i in range(len(loan_events) - horizon):
            current = loan_events[i]
            future = loan_events[i + horizon]
            rows.append({
                "event_id": current.event_id,
                "message_id": current.message_id,
                "loan_id": current.loan_id,
                "as_of": current.as_of.isoformat(),
                "features": current.features_dict(),
                "label": {"next_bucket": future.bucket.value, "horizon_months": horizon},
            })
    return rows


def build_next_transition_labels(
    events: list[TrackBEvent],
) -> list[dict[str, Any]]:
    """Next-transition primitive: predict the next month's bucket.

    This is the same as the Gate 1 baseline.
    """
    by_loan = _group_by_loan(events)
    rows = []
    for loan_events in by_loan.values():
        for i in range(len(loan_events) - 1):
            current = loan_events[i]
            next_e = loan_events[i + 1]
            rows.append({
                "event_id": current.event_id,
                "message_id": current.message_id,
                "loan_id": current.loan_id,
                "as_of": current.as_of.isoformat(),
                "features": current.features_dict(),
                "label": {"next_bucket": next_e.bucket.value, "horizon_months": 1},
            })
    return rows


def build_short_chain_labels(
    events: list[TrackBEvent],
    chain_length: int = 3,
) -> list[dict[str, Any]]:
    """Short-chain primitive: predict the next N-month sequence as a composite label.

    The label is a hyphenated string of buckets, e.g., "current-current-d30".
    """
    by_loan = _group_by_loan(events)
    rows = []
    for loan_events in by_loan.values():
        for i in range(len(loan_events) - chain_length):
            current = loan_events[i]
            chain = [loan_events[i + j + 1].bucket.value for j in range(chain_length)]
            chain_label = "-".join(chain)
            rows.append({
                "event_id": current.event_id,
                "message_id": current.message_id,
                "loan_id": current.loan_id,
                "as_of": current.as_of.isoformat(),
                "features": current.features_dict(),
                "label": {"next_bucket": chain_label, "horizon_months": chain_length},
            })
    return rows


# Branch definitions for ranking
BRANCHES = {
    "stable": lambda buckets: all(b == buckets[0] for b in buckets),
    "deteriorate": lambda buckets: any(
        DelinquencyBucket(buckets[i + 1]) > DelinquencyBucket(buckets[i])
        for i in range(len(buckets) - 1)
        if buckets[i] != "reo" and buckets[i + 1] != "reo"
    ),
    "recover": lambda buckets: any(
        DelinquencyBucket(buckets[i + 1]) < DelinquencyBucket(buckets[i])
        for i in range(len(buckets) - 1)
    ),
}


def build_branch_ranking_labels(
    events: list[TrackBEvent],
    window: int = 3,
) -> list[dict[str, Any]]:
    """Branch-ranking primitive: classify the next N months into a trajectory type.

    Labels: stable, deteriorate, recover. Checked in priority order:
    deteriorate > recover > stable.
    """
    by_loan = _group_by_loan(events)
    rows = []
    for loan_events in by_loan.values():
        for i in range(len(loan_events) - window):
            current = loan_events[i]
            future_buckets = [loan_events[i + j + 1].bucket.value for j in range(window)]

            # Classify trajectory
            if BRANCHES["deteriorate"](future_buckets):
                branch = "deteriorate"
            elif BRANCHES["recover"](future_buckets):
                branch = "recover"
            else:
                branch = "stable"

            rows.append({
                "event_id": current.event_id,
                "message_id": current.message_id,
                "loan_id": current.loan_id,
                "as_of": current.as_of.isoformat(),
                "features": current.features_dict(),
                "label": {"next_bucket": branch, "horizon_months": window},
            })
    return rows


def add_distractor_features(
    rows: list[dict[str, Any]],
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Add distractor noise to features for robustness testing."""
    import hashlib
    distractor_rows = []
    for row in rows:
        new_row = {**row, "features": {**row["features"]}}
        # Deterministic noise based on message_id + seed
        h = int(hashlib.sha256(f"{row['message_id']}:{seed}".encode()).hexdigest()[:8], 16)
        new_row["features"]["distractor_score"] = (h % 1000) / 1000.0
        new_row["features"]["noise_flag"] = h % 2 == 0
        distractor_rows.append(new_row)
    return distractor_rows
