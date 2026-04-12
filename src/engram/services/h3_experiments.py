"""H3 experiment runner: compare 4 primitives with sub-experiments."""

from __future__ import annotations

import random
import statistics
from typing import Any

from engram.models.h3 import H3Artifact, PrimitiveResult
from engram.models.track_b import TrackBEvent
from engram.services.h3_dataset import (
    build_endpoint_labels,
    build_next_transition_labels,
    build_short_chain_labels,
    build_branch_ranking_labels,
    add_distractor_features,
)
from engram.services.h3_primitives import (
    TransitionMatrixPrimitive,
    BranchRankingPrimitive,
    LatentTransitionPrimitive,
    compute_ece,
)
from engram.services.track_b_dataset import assign_splits


def _split_rows(
    rows: list[dict[str, Any]],
    train_end: str,
    eval_end: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = assign_splits(rows, train_end=train_end, eval_end=eval_end)
    train = [r for r in rows if r["split"] == "train"]
    eval_rows = [r for r in rows if r["split"] == "eval"]
    return train, eval_rows


def _add_prev_bucket(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add prev_bucket feature by looking at previous row for same loan."""
    by_loan: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_loan.setdefault(r["loan_id"], []).append(r)
    for loan_rows in by_loan.values():
        loan_rows.sort(key=lambda r: r["as_of"])
        for i in range(1, len(loan_rows)):
            loan_rows[i]["features"]["prev_bucket"] = loan_rows[i - 1]["features"]["bucket"]
    return rows


def _subsample(rows: list[dict[str, Any]], seed: int, frac: float = 0.8) -> list[dict[str, Any]]:
    """Deterministic subsample for seed-based variation."""
    rng = random.Random(seed)
    k = int(len(rows) * frac)
    return rng.sample(rows, k)


def run_h3_experiment(
    events: list[TrackBEvent],
    train_end: str = "2025-06-30",
    eval_end: str = "2025-12-31",
    seeds: list[int] | None = None,
) -> H3Artifact:
    """Run the full H3 comparison experiment."""
    if seeds is None:
        seeds = [7, 17, 27]

    artifact = H3Artifact(seed_list=seeds)

    # --- Main primitive comparison ---
    primitive_builders = {
        "endpoint": lambda: build_endpoint_labels(events, horizon=6),
        "next_transition": lambda: build_next_transition_labels(events),
        "short_chain": lambda: build_short_chain_labels(events, chain_length=3),
        "branch_ranking": lambda: build_branch_ranking_labels(events, window=3),
    }

    for prim_name, builder in primitive_builders.items():
        rows = builder()
        train, eval_rows = _split_rows(rows, train_end, eval_end)

        if not train or not eval_rows:
            artifact.primitives[prim_name] = PrimitiveResult(name=prim_name)
            artifact.distractor_robustness[prim_name] = {"distractor_drop": 0.0}
            artifact.calibration[prim_name] = {"ece_mean": 0.0}
            continue

        # Run across seeds
        accs, briers = [], []
        for seed in seeds:
            train_sub = _subsample(train, seed)
            if prim_name == "branch_ranking":
                model = BranchRankingPrimitive()
            else:
                model = TransitionMatrixPrimitive(prim_name)
            model.fit(train_sub)
            metrics = model.backtest(eval_rows)
            accs.append(metrics["top1_accuracy"])
            briers.append(metrics["brier_score"])

        artifact.primitives[prim_name] = PrimitiveResult(
            name=prim_name,
            top1_accuracy_mean=statistics.mean(accs),
            top1_accuracy_std=statistics.stdev(accs) if len(accs) > 1 else 0.0,
            brier_score_mean=statistics.mean(briers),
            brier_score_std=statistics.stdev(briers) if len(briers) > 1 else 0.0,
        )

        # Calibration (ECE) on last seed's model
        artifact.calibration[prim_name] = {"ece_mean": compute_ece(eval_rows, model)}

        # Distractor robustness
        distractor_eval = add_distractor_features(eval_rows)
        distractor_metrics = model.backtest(distractor_eval)
        clean_acc = statistics.mean(accs)
        artifact.distractor_robustness[prim_name] = {
            "distractor_drop": clean_acc - distractor_metrics["top1_accuracy"],
        }

    # --- Chain-length sensitivity sweep ---
    chain_results = {}
    for steps in [1, 2, 3, 4]:
        if steps == 1:
            rows = build_next_transition_labels(events)
        else:
            rows = build_short_chain_labels(events, chain_length=steps)
        train, eval_rows = _split_rows(rows, train_end, eval_end)
        if not train or not eval_rows:
            chain_results[f"step_{steps}"] = {"top1_accuracy_mean": 0.0, "brier_score_mean": 0.0}
            continue
        model = TransitionMatrixPrimitive(f"chain_{steps}")
        model.fit(train)
        m = model.backtest(eval_rows)
        chain_results[f"step_{steps}"] = {
            "top1_accuracy_mean": m["top1_accuracy"],
            "brier_score_mean": m["brier_score"],
        }

    # Select best horizon by Brier
    best_step = min(chain_results, key=lambda k: chain_results[k]["brier_score_mean"])
    chain_results["selected_horizon"] = best_step
    artifact.chain_length_sensitivity = chain_results

    # --- Observed vs latent ---
    next_rows = build_next_transition_labels(events)
    next_rows = _add_prev_bucket(next_rows)
    train, eval_rows = _split_rows(next_rows, train_end, eval_end)

    if train and eval_rows:
        # Observed only
        obs_model = TransitionMatrixPrimitive("observed")
        obs_model.fit(train)
        obs_m = obs_model.backtest(eval_rows)

        # Latent (uses prev_bucket momentum)
        lat_model = LatentTransitionPrimitive()
        lat_model.fit(train)
        lat_m = lat_model.backtest(eval_rows)

        artifact.observed_vs_latent = {
            "observed_only": {"top1_accuracy_mean": obs_m["top1_accuracy"], "brier_score_mean": obs_m["brier_score"]},
            "latent_enabled": {"top1_accuracy_mean": lat_m["top1_accuracy"], "brier_score_mean": lat_m["brier_score"]},
            "latent_lift_accuracy": lat_m["top1_accuracy"] - obs_m["top1_accuracy"],
            "latent_lift_brier": obs_m["brier_score"] - lat_m["brier_score"],  # positive = latent is better
        }

    # --- Delayed-outcome horizons ---
    horizon_results = {}
    for h in [1, 2, 3, 4]:
        rows = build_endpoint_labels(events, horizon=h)
        train, eval_rows = _split_rows(rows, train_end, eval_end)
        if not train or not eval_rows:
            horizon_results[f"horizon_{h}m"] = {"top1_accuracy_mean": 0.0, "brier_score_mean": 0.0}
            continue
        model = TransitionMatrixPrimitive(f"horizon_{h}")
        model.fit(train)
        m = model.backtest(eval_rows)
        horizon_results[f"horizon_{h}m"] = {
            "top1_accuracy_mean": m["top1_accuracy"],
            "brier_score_mean": m["brier_score"],
        }
    artifact.delayed_outcome_horizons = horizon_results

    # Select winner
    artifact.select_winner()

    return artifact
