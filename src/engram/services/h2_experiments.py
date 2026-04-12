"""H2 context ablation experiment runner."""

from __future__ import annotations

from typing import Any

from engram.models.h2 import H2Artifact, ProfileResult
from engram.models.track_b import TrackBEvent
from engram.services.h2_context import (
    PROFILES,
    compute_competing_cause_discrimination,
    compute_evidence_gaps,
)
from engram.services.h3_dataset import (
    add_distractor_features,
    build_next_transition_labels,
)
from engram.services.h3_primitives import LatentTransitionPrimitive, TransitionMatrixPrimitive
from engram.services.track_b_dataset import assign_splits


def _split(rows: list[dict[str, Any]], train_end: str, eval_end: str):
    rows = assign_splits(rows, train_end=train_end, eval_end=eval_end)
    train = [r for r in rows if r["split"] == "train"]
    eval_rows = [r for r in rows if r["split"] == "eval"]
    return train, eval_rows


def _add_prev_bucket(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_loan: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_loan.setdefault(r["loan_id"], []).append(r)
    for loan_rows in by_loan.values():
        loan_rows.sort(key=lambda r: r["as_of"])
        for i in range(1, len(loan_rows)):
            loan_rows[i]["features"]["prev_bucket"] = loan_rows[i - 1]["features"]["bucket"]
    return rows


def _make_model(profile_name: str):
    """Use latent model for profiles that include prev_bucket, basic otherwise."""
    if profile_name in ("full", "top_k", "schema_guided"):
        return LatentTransitionPrimitive()
    return TransitionMatrixPrimitive(profile_name)


def run_h2_experiment(
    events: list[TrackBEvent],
    train_end: str = "2025-06-30",
    eval_end: str = "2025-12-31",
) -> H2Artifact:
    """Run the full H2 context ablation experiment."""
    artifact = H2Artifact(selected_primitive="next_transition")

    # Build base dataset with prev_bucket
    all_rows = build_next_transition_labels(events)
    all_rows = _add_prev_bucket(all_rows)

    for profile_name, profile_fn in PROFILES.items():
        # Apply context profile
        profiled_rows = [profile_fn(r) for r in all_rows]
        train, eval_rows = _split(profiled_rows, train_end, eval_end)

        if not train or not eval_rows:
            artifact.profiles[profile_name] = ProfileResult(name=profile_name)
            artifact.distractor_report[profile_name] = {"distractor_drop": 0.0}
            continue

        # Train and evaluate
        model = _make_model(profile_name)
        model.fit(train)
        metrics = model.backtest(eval_rows)

        # Collect predictions for evidence gap and competing cause analysis
        predictions = []
        for row in eval_rows:
            pred = model.predict(row["features"])
            predictions.append(pred)

        # Distractor robustness
        distractor_eval = add_distractor_features(eval_rows)
        distractor_profiled = [profile_fn(r) for r in distractor_eval]
        distractor_metrics = model.backtest(distractor_profiled)
        distractor_drop = metrics["top1_accuracy"] - distractor_metrics["top1_accuracy"]

        artifact.profiles[profile_name] = ProfileResult(
            name=profile_name,
            top1_accuracy_mean=metrics["top1_accuracy"],
            brier_score_mean=metrics["brier_score"],
            distractor_drop=distractor_drop,
        )
        artifact.distractor_report[profile_name] = {"distractor_drop": distractor_drop}

    # Evidence gap coverage (using the minimal_discriminative profile)
    min_rows = [PROFILES["minimal_discriminative"](r) for r in all_rows]
    _, min_eval = _split(min_rows, train_end, eval_end)
    if min_eval:
        min_model = _make_model("minimal_discriminative")
        min_train, _ = _split([PROFILES["minimal_discriminative"](r) for r in all_rows], train_end, eval_end)
        min_model.fit(min_train)
        min_preds = [min_model.predict(r["features"]) for r in min_eval]
        artifact.evidence_gap_coverage = compute_evidence_gaps(min_eval, min_preds)

    # Competing cause discrimination (using full profile)
    full_rows = [PROFILES["full"](r) for r in all_rows]
    full_train, full_eval = _split(full_rows, train_end, eval_end)
    if full_train and full_eval:
        full_model = _make_model("full")
        full_model.fit(full_train)
        full_preds = [full_model.predict(r["features"]) for r in full_eval]
        artifact.competing_cause_discrimination = compute_competing_cause_discrimination(
            full_eval, full_preds
        )

    return artifact
