"""H1 transfer evaluation and experiment runner."""

from __future__ import annotations

import random
from typing import Any

from engram.models.track_b import TrackBEvent
from engram.services.h1_schema import evaluate_schema_guided_accuracy, induce_motifs


def split_loans(
    events: list[TrackBEvent],
    train_frac: float = 0.7,
    seed: int = 42,
) -> tuple[list[TrackBEvent], list[TrackBEvent]]:
    """Split events by loan_id (not by time) for transfer testing."""
    loan_ids = sorted(set(e.loan_id for e in events))
    rng = random.Random(seed)
    rng.shuffle(loan_ids)
    split_idx = int(len(loan_ids) * train_frac)
    train_loans = set(loan_ids[:split_idx])
    eval_loans = set(loan_ids[split_idx:])
    train_events = [e for e in events if e.loan_id in train_loans]
    eval_events = [e for e in events if e.loan_id in eval_loans]
    return train_events, eval_events


def run_h1_experiment(
    events: list[TrackBEvent],
    window: int = 3,
    min_support: int = 3,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the full H1 schema induction experiment."""

    # Split by loan for transfer testing
    train_events, eval_events = split_loans(events, train_frac=0.7, seed=seed)

    # Induce motifs from train set
    motifs = induce_motifs(train_events, window=window, min_support=min_support, granularity="event_only")

    # Evaluate on eval set (transfer)
    transfer_eval = evaluate_schema_guided_accuracy(motifs, eval_events, granularity="event_only", window=window)

    # In-family baseline: induce on eval set itself (no transfer — upper bound)
    self_motifs = induce_motifs(eval_events, window=window, min_support=max(1, min_support // 2), granularity="event_only")
    baseline_eval = evaluate_schema_guided_accuracy(self_motifs, eval_events, granularity="event_only", window=window)

    transfer_score = (
        transfer_eval["accuracy"] / baseline_eval["accuracy"]
        if baseline_eval["accuracy"] > 0 else 0.0
    )

    # Schema compactness
    avg_size = sum(m.nodes for m in motifs) / len(motifs) if motifs else 0.0

    # Granularity sweep
    granularity_results = {}
    for gran in ["event_only", "event_plus_state", "event_state_gate"]:
        gran_motifs = induce_motifs(train_events, window=window, min_support=min_support, granularity=gran)
        gran_eval = evaluate_schema_guided_accuracy(gran_motifs, eval_events, granularity=gran, window=window)
        gran_avg_nodes = sum(m.nodes for m in gran_motifs) / len(gran_motifs) if gran_motifs else 0.0
        granularity_results[gran] = {
            "accuracy": gran_eval["accuracy"],
            "avg_nodes": gran_avg_nodes,
            "motif_count": len(gran_motifs),
        }

    # Select minimal granularity within 0.01 of best
    best_acc = max(g["accuracy"] for g in granularity_results.values())
    selected = "event_only"  # default to simplest
    for gran in ["event_only", "event_plus_state", "event_state_gate"]:
        if granularity_results[gran]["accuracy"] >= best_acc - 0.01:
            selected = gran
            break
    granularity_results["selected_granularity"] = selected

    return {
        "family": "ginnie",
        "motifs": [m.to_dict() for m in motifs[:50]],  # top 50 by support
        "average_schema_size": avg_size,
        "transfer_score": transfer_score,
        "transfer_eval": {
            "in_family_baseline_accuracy": baseline_eval["accuracy"],
            "schema_guided_accuracy": transfer_eval["accuracy"],
            "schema_coverage": transfer_eval["coverage"],
        },
        "granularity_sweep": granularity_results,
        "total_motifs_induced": len(motifs),
    }
