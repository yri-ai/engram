"""H5 cross-structure transfer: test motif reusability across families.

Since we only have Ginnie Mae data, we create sub-families by splitting
loans by characteristics (e.g., delinquency history profile). This tests
whether motifs induced from one sub-population transfer to another.

Family A: "stable" loans (never delinquent in first half of history)
Family B: "volatile" loans (at least one delinquency in first half)

This is a proxy for cross-structure transfer. With real multi-source
data (Ginnie + Fannie, or CMBS + RMBS), families would be genuinely
different product types.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from engram.models.track_b import TrackBEvent, DelinquencyBucket
from engram.services.h1_schema import induce_motifs, evaluate_schema_guided_accuracy


def split_into_families(
    events: list[TrackBEvent],
) -> tuple[list[TrackBEvent], list[TrackBEvent]]:
    """Split events into two families based on loan behavior profile.

    Family A (stable): loans that were current for entire first 24 months
    Family B (volatile): loans with any delinquency in first 24 months
    """
    by_loan: dict[str, list[TrackBEvent]] = defaultdict(list)
    for e in events:
        by_loan[e.loan_id].append(e)
    for lid in by_loan:
        by_loan[lid].sort(key=lambda e: e.as_of)

    family_a_loans: set[str] = set()
    family_b_loans: set[str] = set()

    for lid, loan_events in by_loan.items():
        first_half = loan_events[: len(loan_events) // 2]
        if any(e.bucket != DelinquencyBucket.CURRENT for e in first_half):
            family_b_loans.add(lid)
        else:
            family_a_loans.add(lid)

    family_a = [e for e in events if e.loan_id in family_a_loans]
    family_b = [e for e in events if e.loan_id in family_b_loans]
    return family_a, family_b


def compute_motif_overlap(
    motifs_a: list[Any],
    motifs_b: list[Any],
    top_n: int = 20,
) -> float:
    """Compute Jaccard overlap of top-N motif patterns between two families."""
    patterns_a = set()
    for m in sorted(motifs_a, key=lambda x: -x.support_cases)[:top_n]:
        patterns_a.add(m.pattern)

    patterns_b = set()
    for m in sorted(motifs_b, key=lambda x: -x.support_cases)[:top_n]:
        patterns_b.add(m.pattern)

    if not patterns_a and not patterns_b:
        return 0.0
    intersection = patterns_a & patterns_b
    union = patterns_a | patterns_b
    return len(intersection) / len(union) if union else 0.0


def characterize_structural_drivers(
    events: list[TrackBEvent],
) -> dict[str, float]:
    """Characterize what drives outcomes in a family.

    Simplified structural driver analysis based on observable patterns:
    - threshold_gates: fraction of transitions at bucket boundaries
    - waterfall_logic: fraction of loans that deteriorate step-by-step
    - optionality_collapse: fraction going directly to d90+
    - counterparty_behavior: variance in transition rates across states
    - covenant_or_contract_triggers: fraction with sudden bucket jumps
    """
    by_loan: dict[str, list[TrackBEvent]] = defaultdict(list)
    for e in events:
        by_loan[e.loan_id].append(e)

    total_transitions = 0
    threshold_count = 0
    waterfall_count = 0
    collapse_count = 0
    jump_count = 0

    bucket_order = list(DelinquencyBucket)

    for loan_events in by_loan.values():
        sorted_events = sorted(loan_events, key=lambda e: e.as_of)
        for i in range(1, len(sorted_events)):
            prev = sorted_events[i - 1].bucket
            curr = sorted_events[i].bucket
            if prev == curr:
                continue
            total_transitions += 1

            prev_idx = bucket_order.index(prev)
            curr_idx = bucket_order.index(curr)
            step = curr_idx - prev_idx

            # Threshold: transitions at bucket boundary (±1 step)
            if abs(step) == 1:
                threshold_count += 1
            # Waterfall: deterioration by exactly 1 step
            if step == 1:
                waterfall_count += 1
            # Collapse: jump of 3+ steps toward delinquent
            if step >= 3:
                collapse_count += 1
            # Jump: any non-adjacent transition
            if abs(step) >= 2:
                jump_count += 1

    # State-based variance (proxy for counterparty behavior)
    state_transition_rates: dict[str | None, list[float]] = defaultdict(list)
    for loan_events in by_loan.values():
        sorted_events = sorted(loan_events, key=lambda e: e.as_of)
        transitions = sum(1 for i in range(1, len(sorted_events))
                          if sorted_events[i].bucket != sorted_events[i - 1].bucket)
        rate = transitions / max(len(sorted_events) - 1, 1)
        state = sorted_events[0].state
        state_transition_rates[state].append(rate)

    state_means = [sum(rates) / len(rates) for rates in state_transition_rates.values() if rates]
    counterparty = (max(state_means) - min(state_means)) if len(state_means) >= 2 else 0.0

    t = max(total_transitions, 1)
    return {
        "threshold_gates": threshold_count / t,
        "waterfall_logic": waterfall_count / t,
        "optionality_collapse": collapse_count / t,
        "counterparty_behavior": counterparty,
        "covenant_or_contract_triggers": jump_count / t,
    }


def run_h5_experiment(
    events: list[TrackBEvent],
    window: int = 3,
    min_support: int = 3,
) -> dict[str, Any]:
    """Run H5 cross-structure transfer experiment."""

    family_a, family_b = split_into_families(events)

    if not family_a or not family_b:
        return {
            "families": ["stable", "volatile"],
            "shared_core": {"accuracy": 0.0},
            "family_specific": {"accuracy": 0.0},
            "family_drift": 0.0,
            "transferable_motif_score": 0.0,
            "lift_vs_shared_core": 0.0,
            "structural_drivers": characterize_structural_drivers(events),
        }

    # Induce motifs per family
    motifs_a = induce_motifs(family_a, window=window, min_support=min_support)
    motifs_b = induce_motifs(family_b, window=window, min_support=min_support)

    # Shared-core: induce on all data
    motifs_shared = induce_motifs(events, window=window, min_support=min_support)

    # Evaluate shared motifs on family B
    shared_eval = evaluate_schema_guided_accuracy(motifs_shared, family_b, window=window)

    # Evaluate family-B-specific motifs on family B
    specific_eval = evaluate_schema_guided_accuracy(motifs_b, family_b, window=window)

    # Evaluate family A motifs on family B (cross-family transfer)
    cross_eval = evaluate_schema_guided_accuracy(motifs_a, family_b, window=window)

    # Motif overlap
    jaccard = compute_motif_overlap(motifs_a, motifs_b, top_n=20)
    family_drift = 1.0 - jaccard

    # Transferable motif score: motifs that appear in both families with support >= 3
    patterns_a = {m.pattern for m in motifs_a}
    patterns_b = {m.pattern for m in motifs_b}
    transferable = patterns_a & patterns_b
    total_patterns = patterns_a | patterns_b
    transferable_score = len(transferable) / len(total_patterns) if total_patterns else 0.0

    # Structural drivers for family B
    drivers = characterize_structural_drivers(family_b)

    return {
        "families": ["stable", "volatile"],
        "family_sizes": {"stable": len(set(e.loan_id for e in family_a)),
                         "volatile": len(set(e.loan_id for e in family_b))},
        "shared_core": {"accuracy": shared_eval["accuracy"]},
        "family_specific": {"accuracy": specific_eval["accuracy"]},
        "cross_family": {"accuracy": cross_eval["accuracy"]},
        "family_drift": family_drift,
        "transferable_motif_score": transferable_score,
        "lift_vs_shared_core": specific_eval["accuracy"] - shared_eval["accuracy"],
        "structural_drivers": drivers,
        "motif_counts": {"shared": len(motifs_shared), "family_a": len(motifs_a), "family_b": len(motifs_b)},
    }
