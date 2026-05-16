"""Extract features from Track B event history for forecasting.

This module computes features from the event timeline for a given loan,
without requiring a live graph database. It operates on the event list
directly, providing the bridge between raw events and the forecasting model.

When the graph is populated (via LLM ingestion), this can be extended
to pull graph-derived features (entity counts, relationship types, etc.).
"""

from __future__ import annotations

from datetime import date
from typing import Any

from engram.models.track_b import TrackBEvent


def extract_features_from_event_history(
    events: list[TrackBEvent],
    as_of: date,
    loan_id: str,
    lookback_months: int = 3,
) -> dict[str, Any] | None:
    """Extract features for a loan at a point in time.

    Args:
        events: All events (may include multiple loans).
        as_of: The observation date.
        loan_id: Which loan to extract features for.
        lookback_months: How many months of history to consider.

    Returns:
        Feature dict keyed by message_id, or None if loan not found.
    """
    # Filter to this loan, up to as_of
    loan_events = sorted(
        [e for e in events if e.loan_id == loan_id and e.as_of <= as_of],
        key=lambda e: e.as_of,
    )

    if not loan_events:
        return None

    current = loan_events[-1]
    recent = loan_events[-lookback_months:] if len(loan_events) >= lookback_months else loan_events

    # Count bucket changes in lookback window
    bucket_changes = 0
    for i in range(1, len(recent)):
        if recent[i].bucket != recent[i - 1].bucket:
            bucket_changes += 1

    # UPB change
    upb_change_pct = None
    if len(recent) >= 2 and recent[0].current_upb > 0:
        upb_change_pct = round(
            (current.current_upb - recent[0].current_upb) / recent[0].current_upb, 4
        )

    # Previous bucket
    prev_bucket = recent[-2].bucket.value if len(recent) >= 2 else None

    features: dict[str, Any] = {
        "message_id": current.message_id,
        "bucket": current.bucket.value,
        "current_upb": current.current_upb,
        "months_observed": len(loan_events),
        f"bucket_changes_{lookback_months}m": bucket_changes,
        "prev_bucket": prev_bucket,
        "upb_change_pct": upb_change_pct,
    }

    if current.interest_rate is not None:
        features["interest_rate"] = current.interest_rate
    if current.credit_score is not None:
        features["credit_score"] = current.credit_score
    if current.state is not None:
        features["state"] = current.state

    return features
