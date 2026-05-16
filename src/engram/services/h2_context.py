"""Context profile builders for H2 ablation.

Each profile function takes a full-featured row and returns a row
with a restricted feature set, simulating different context budgets.
"""

from __future__ import annotations

from typing import Any


def profile_full(row: dict[str, Any]) -> dict[str, Any]:
    """Full context — all available features. No filtering."""
    return row


def profile_top_k(row: dict[str, Any], k: int = 3) -> dict[str, Any]:
    """Top-K features by pre-computed importance ranking.

    For loan delinquency, the ranking is:
    1. bucket (current delinquency status)
    2. prev_bucket (momentum signal from Gate 2)
    3. upb_change_pct (balance trajectory)
    """
    importance_order = ["bucket", "prev_bucket", "upb_change_pct",
                        "interest_rate", "credit_score", "current_upb", "state"]
    features = row["features"]
    restricted = {}
    kept = 0
    for key in importance_order:
        if key in features and kept < k:
            restricted[key] = features[key]
            kept += 1
    # Always include bucket (needed by model)
    if "bucket" not in restricted:
        restricted["bucket"] = features.get("bucket", "current")
    return {**row, "features": restricted}


def profile_schema_guided(row: dict[str, Any]) -> dict[str, Any]:
    """Schema-guided: only features that follow the delinquency schema.

    The schema says: delinquency transitions are driven by
    current status + recent momentum. Strip everything else.
    """
    features = row["features"]
    restricted = {
        "bucket": features.get("bucket", "current"),
    }
    if "prev_bucket" in features:
        restricted["prev_bucket"] = features["prev_bucket"]
    return {**row, "features": restricted}


def profile_minimal_discriminative(row: dict[str, Any]) -> dict[str, Any]:
    """Minimal discriminative: only the single feature that most
    distinguishes the competing outcomes for *this specific row*.

    For transitions: if the loan is already delinquent, bucket alone
    is sufficient. If current, prev_bucket (momentum) is the discriminator.
    """
    features = row["features"]
    bucket = features.get("bucket", "current")
    if bucket != "current" and bucket != "":
        # Already delinquent — bucket is the key signal
        restricted = {"bucket": bucket}
    elif "prev_bucket" in features and features["prev_bucket"] is not None:
        # Currently performing but has history — momentum matters
        restricted = {"bucket": bucket, "prev_bucket": features["prev_bucket"]}
    else:
        restricted = {"bucket": bucket}
    return {**row, "features": restricted}


PROFILES = {
    "full": profile_full,
    "top_k": profile_top_k,
    "schema_guided": profile_schema_guided,
    "minimal_discriminative": profile_minimal_discriminative,
}


def compute_evidence_gaps(
    eval_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> float:
    """Compute evidence-gap coverage.

    For each eval row, check if the model can identify what
    missing evidence would flip its prediction. A prediction
    has a gap if the margin between top-1 and top-2 is narrow
    (< 0.3), indicating the model is uncertain and additional
    evidence could change the outcome.

    Returns fraction of eval rows with identified evidence gaps.
    """
    if not predictions:
        return 0.0
    gaps = 0
    for pred in predictions:
        probs = pred.get("probabilities", {})
        if len(probs) < 2:
            continue
        sorted_probs = sorted(probs.values(), reverse=True)
        margin = sorted_probs[0] - sorted_probs[1]
        if margin < 0.3:
            gaps += 1
    return gaps / len(predictions)


def compute_competing_cause_discrimination(
    eval_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute competing-cause discrimination metrics.

    Focuses on rows where the model must distinguish between
    2+ plausible next states (not just "stay current").
    These are the decision-relevant cases.
    """
    # Find rows where the true label is NOT the overwhelmingly dominant class
    # i.e., cases with actual competing causes
    subset_preds = []
    for row, pred in zip(eval_rows, predictions, strict=True):
        probs = pred.get("probabilities", {})
        if len(probs) < 2:
            continue
        sorted_probs = sorted(probs.values(), reverse=True)
        # Only include rows where top prediction has < 0.95 probability
        # (meaning there's meaningful competition)
        if sorted_probs[0] < 0.95:
            subset_preds.append((row, pred))

    if not subset_preds:
        return {"subset_count": 0, "accuracy": 0.0, "mean_margin_top1_top2": 0.0}

    correct = 0
    margins = []
    for row, pred in subset_preds:
        truth = row["label"]["next_bucket"]
        if pred["top_bucket"] == truth:
            correct += 1
        probs = pred.get("probabilities", {})
        sorted_probs = sorted(probs.values(), reverse=True)
        margins.append(sorted_probs[0] - sorted_probs[1])

    return {
        "subset_count": len(subset_preds),
        "accuracy": correct / len(subset_preds),
        "mean_margin_top1_top2": sum(margins) / len(margins),
    }
