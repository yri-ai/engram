"""H4 symbolic pruning: use motif library to constrain predictions.

Three tightness levels:
- loose: prune outcomes never observed for this pattern
- medium: prune outcomes with <5% frequency for this pattern
- hard: only allow top 2 outcomes for this pattern
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from engram.models.track_b import TrackBEvent, DelinquencyBucket
from engram.services.h1_schema import induce_motifs, ENCODERS
from engram.services.h3_primitives import TransitionMatrixPrimitive
from engram.services.h3_dataset import build_next_transition_labels
from engram.services.track_b_dataset import assign_splits


def _build_motif_constraints(
    motifs: list[Any],
) -> dict[tuple[str, ...], Counter[str]]:
    """Build a lookup: pattern → outcome frequencies."""
    constraints: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    for m in motifs:
        pattern = tuple(m.pattern) if hasattr(m, "pattern") else tuple(m["pattern"])
        outcome = m.outcome if hasattr(m, "outcome") else m["outcome"]
        support = m.support_cases if hasattr(m, "support_cases") else m["support_cases"]
        constraints[pattern][outcome] += support
    return constraints


def prune_predictions(
    predictions: dict[str, float],
    pattern: tuple[str, ...],
    constraints: dict[tuple[str, ...], Counter[str]],
    tightness: str = "loose",
) -> dict[str, float]:
    """Apply symbolic pruning to prediction probabilities.

    Returns a new probability dict with pruned outcomes zeroed and renormalized.
    """
    if pattern not in constraints:
        return predictions  # No constraint info — pass through

    outcome_counts = constraints[pattern]
    total = sum(outcome_counts.values())

    if tightness == "loose":
        # Only allow outcomes that have been observed
        allowed = set(outcome_counts.keys())
    elif tightness == "medium":
        # Allow outcomes with >= 5% frequency
        allowed = {o for o, c in outcome_counts.items() if c / total >= 0.05}
    elif tightness == "hard":
        # Only top 2 outcomes
        top2 = [o for o, _ in outcome_counts.most_common(2)]
        allowed = set(top2)
    else:
        return predictions

    if not allowed:
        return predictions

    pruned = {k: (v if k in allowed else 0.0) for k, v in predictions.items()}
    total_prob = sum(pruned.values())
    if total_prob > 0:
        pruned = {k: v / total_prob for k, v in pruned.items()}
    return pruned


def run_h4_experiment(
    events: list[TrackBEvent],
    train_end: str = "2025-06-30",
    eval_end: str = "2025-12-31",
    window: int = 3,
    min_support: int = 3,
) -> dict[str, Any]:
    """Run the H4 symbolic pruning ablation."""

    # Build next-transition dataset
    rows = build_next_transition_labels(events)
    rows = assign_splits(rows, train_end=train_end, eval_end=eval_end)
    train = [r for r in rows if r["split"] == "train"]
    eval_rows = [r for r in rows if r["split"] == "eval"]

    # Train base model (without symbolic)
    model = TransitionMatrixPrimitive("base")
    model.fit(train)

    # Induce motifs from train events only
    train_loans = set(r["loan_id"] for r in train)
    train_events = [e for e in events if e.loan_id in train_loans]
    motifs = induce_motifs(train_events, window=window, min_support=min_support)
    constraints = _build_motif_constraints(motifs)

    # Build pattern lookup for eval rows
    encoder = ENCODERS["event_only"]
    by_loan: dict[str, list[TrackBEvent]] = defaultdict(list)
    for e in events:
        by_loan[e.loan_id].append(e)
    for lid in by_loan:
        by_loan[lid].sort(key=lambda e: e.as_of)

    # Map each eval row to its preceding pattern
    eval_patterns: dict[str, tuple[str, ...]] = {}
    for lid, loan_events in by_loan.items():
        encoded = [encoder(loan_events, i) for i in range(len(loan_events))]
        for i in range(window, len(encoded)):
            event = loan_events[i]
            msg_id = event.message_id
            pattern = tuple(encoded[i - window : i])
            eval_patterns[msg_id] = pattern

    # Evaluate without symbolic pruning
    without_correct = 0
    without_contradictions = 0
    total_eval = len(eval_rows)

    for row in eval_rows:
        truth = row["label"]["next_bucket"]
        pred = model.predict(row["features"])
        if pred["top_bucket"] == truth:
            without_correct += 1
        # A "contradiction" is when the prediction is impossible per motifs
        pattern = eval_patterns.get(row["message_id"])
        if pattern and pattern in constraints:
            if pred["top_bucket"] not in constraints[pattern]:
                without_contradictions += 1

    without_recall = without_correct / total_eval if total_eval else 0.0
    without_contradiction_rate = without_contradictions / total_eval if total_eval else 0.0

    # Evaluate with symbolic pruning at each tightness
    tightness_results = {}
    for tightness in ["loose", "medium", "hard"]:
        correct = 0
        contradictions = 0
        novelty_pruned = 0

        for row in eval_rows:
            truth = row["label"]["next_bucket"]
            pred = model.predict(row["features"])
            pattern = eval_patterns.get(row["message_id"])

            if pattern:
                pruned_probs = prune_predictions(pred["probabilities"], pattern, constraints, tightness)
                top = max(pruned_probs, key=lambda k: pruned_probs[k])

                # Check if pruning removed the truth
                if truth in pred["probabilities"] and pruned_probs.get(truth, 0.0) == 0.0:
                    novelty_pruned += 1

                if pattern in constraints and top not in constraints[pattern]:
                    contradictions += 1
            else:
                top = pred["top_bucket"]

            if top == truth:
                correct += 1

        recall = correct / total_eval if total_eval else 0.0
        contradiction_rate = contradictions / total_eval if total_eval else 0.0
        novelty_rate = novelty_pruned / total_eval if total_eval else 0.0

        tightness_results[tightness] = {
            "contradiction_rate": contradiction_rate,
            "recall": recall,
            "novelty_prune_rate": novelty_rate,
        }

    # Select best tightness: minimize contradictions, keep novelty_prune_rate <= 0.10
    valid = {k: v for k, v in tightness_results.items() if v["novelty_prune_rate"] <= 0.10}
    if valid:
        selected = min(valid, key=lambda k: valid[k]["contradiction_rate"])
    else:
        selected = "loose"
    tightness_results["selected_tightness"] = selected

    best_recall = tightness_results[selected]["recall"]
    best_contradiction = tightness_results[selected]["contradiction_rate"]

    # Error examples: cases where pruning changed the prediction incorrectly
    error_examples = []
    for row in eval_rows[:1000]:
        truth = row["label"]["next_bucket"]
        pred = model.predict(row["features"])
        pattern = eval_patterns.get(row["message_id"])
        if pattern:
            pruned = prune_predictions(pred["probabilities"], pattern, constraints, selected)
            pruned_top = max(pruned, key=lambda k: pruned[k])
            if pruned_top != pred["top_bucket"] and pruned_top != truth:
                error_examples.append({
                    "message_id": row["message_id"],
                    "pattern": list(pattern),
                    "truth": truth,
                    "unpruned_top": pred["top_bucket"],
                    "pruned_top": pruned_top,
                })
                if len(error_examples) >= 10:
                    break

    return {
        "without_symbolic": {
            "contradiction_rate": without_contradiction_rate,
            "recall": without_recall,
        },
        "tightness": tightness_results,
        "contradiction_reduction": without_contradiction_rate - best_contradiction,
        "recall_loss": without_recall - best_recall,
        "error_examples": error_examples,
    }
