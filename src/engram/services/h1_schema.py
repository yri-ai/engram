"""Schema induction: discover reusable precursor motifs from loan histories.

A motif is a short sequence pattern (2-4 steps) that recurs across
multiple loans and precedes a specific outcome. Motifs are the
"precursor schemas" from the research thesis.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from engram.models.track_b import TrackBEvent


@dataclass
class Motif:
    """A discovered precursor schema."""

    motif_id: str
    pattern: tuple[str, ...]  # sequence of states
    outcome: str  # what follows this pattern
    support_cases: int = 0  # how many loans exhibit this
    loan_ids: list[str] = field(default_factory=list)

    @property
    def nodes(self) -> int:
        return len(self.pattern)

    @property
    def edges(self) -> int:
        return max(0, len(self.pattern) - 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "motif_id": self.motif_id,
            "pattern": list(self.pattern),
            "outcome": self.outcome,
            "nodes": self.nodes,
            "edges": self.edges,
            "support_cases": self.support_cases,
        }


def _encode_event_only(event: TrackBEvent) -> str:
    """Event-only encoding: just the bucket."""
    return event.bucket.value


def _encode_event_plus_state(event: TrackBEvent, prev: TrackBEvent | None) -> str:
    """Event + state: bucket + direction of change."""
    bucket = event.bucket.value
    if prev is None:
        return f"{bucket}:flat"
    if event.bucket > prev.bucket:
        return f"{bucket}:up"
    if event.bucket < prev.bucket:
        return f"{bucket}:down"
    return f"{bucket}:flat"


def _encode_event_state_gate(event: TrackBEvent, prev: TrackBEvent | None) -> str:
    """Event + state + gate: bucket + direction + UPB declining flag."""
    base = _encode_event_plus_state(event, prev)
    if prev is not None and prev.current_upb > 0:
        declining = event.current_upb < prev.current_upb
        return f"{base}|{'dec' if declining else 'inc'}"
    return f"{base}|unk"


ENCODERS = {
    "event_only": lambda events, i: _encode_event_only(events[i]),
    "event_plus_state": lambda events, i: _encode_event_plus_state(
        events[i], events[i - 1] if i > 0 else None
    ),
    "event_state_gate": lambda events, i: _encode_event_state_gate(
        events[i], events[i - 1] if i > 0 else None
    ),
}


def induce_motifs(
    events: list[TrackBEvent],
    window: int = 3,
    min_support: int = 3,
    granularity: str = "event_only",
) -> list[Motif]:
    """Induce precursor motifs from event histories.

    Groups events by loan, extracts all windows of length `window`,
    records the outcome (next bucket after the window), and keeps
    patterns that appear in >= min_support distinct loans.
    """
    encoder = ENCODERS[granularity]

    # Group by loan
    by_loan: dict[str, list[TrackBEvent]] = defaultdict(list)
    for e in events:
        by_loan[e.loan_id].append(e)
    for loan_id in by_loan:
        by_loan[loan_id].sort(key=lambda e: e.as_of)

    # Extract (pattern, outcome) from each loan
    pattern_loans: dict[tuple[tuple[str, ...], str], set[str]] = defaultdict(set)

    for loan_id, loan_events in by_loan.items():
        encoded = [encoder(loan_events, i) for i in range(len(loan_events))]
        for i in range(len(encoded) - window):
            pattern = tuple(encoded[i : i + window])
            outcome = encoded[i + window]
            pattern_loans[(pattern, outcome)].add(loan_id)

    # Filter by support and build motifs
    motifs = []
    for idx, ((pattern, outcome), loans) in enumerate(
        sorted(pattern_loans.items(), key=lambda x: -len(x[1]))
    ):
        if len(loans) < min_support:
            continue
        motifs.append(Motif(
            motif_id=f"M{idx + 1}",
            pattern=pattern,
            outcome=outcome,
            support_cases=len(loans),
            loan_ids=sorted(loans)[:10],  # keep sample, not all
        ))

    return motifs


def evaluate_schema_guided_accuracy(
    motifs: list[Motif],
    events: list[TrackBEvent],
    granularity: str = "event_only",
    window: int = 3,
) -> dict[str, float]:
    """Evaluate how well the motif library predicts outcomes.

    For each window in the eval set, check if a matching motif exists.
    If so, use its outcome as the prediction. Measure accuracy.
    """
    encoder = ENCODERS[granularity]

    # Build motif lookup: pattern → Counter of outcomes
    motif_lookup: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    for m in motifs:
        motif_lookup[m.pattern][m.outcome] += m.support_cases

    by_loan: dict[str, list[TrackBEvent]] = defaultdict(list)
    for e in events:
        by_loan[e.loan_id].append(e)
    for loan_id in by_loan:
        by_loan[loan_id].sort(key=lambda e: e.as_of)

    correct = 0
    total = 0
    matched = 0

    for loan_events in by_loan.values():
        encoded = [encoder(loan_events, i) for i in range(len(loan_events))]
        for i in range(len(encoded) - window):
            pattern = tuple(encoded[i : i + window])
            truth = encoded[i + window]
            total += 1

            if pattern in motif_lookup:
                matched += 1
                # Predict most common outcome for this pattern
                predicted = motif_lookup[pattern].most_common(1)[0][0]
                if predicted == truth:
                    correct += 1

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "coverage": matched / total if total > 0 else 0.0,
        "total_windows": total,
        "matched_windows": matched,
    }
