"""Deterministic in-context row sampling for tabular foundation heads."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Literal

Strategy = Literal["random", "class_balanced", "transition_balanced", "recency_weighted"]


def sample_context(
    rows: list[dict[str, Any]], budget: int, strategy: Strategy = "random", *, seed: int = 42
) -> list[dict[str, Any]]:
    """Sample at most ``budget`` rows without crossing caller-supplied boundaries."""
    if budget <= 0:
        return []
    rng = random.Random(seed)
    pool = list(rows)
    if strategy == "random":
        rng.shuffle(pool)
        return pool[:budget]
    if strategy == "recency_weighted":
        return sorted(pool, key=lambda row: str(row.get("as_of", "")), reverse=True)[:budget]
    key_name = "next_bucket" if strategy == "class_balanced" else "transition"
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        if key_name == "transition":
            key = f"{row.get('features', {}).get('bucket')}->{row.get('label', {}).get('next_bucket')}"
        else:
            key = str(row.get("label", {}).get("next_bucket"))
        groups[key].append(row)
    for group in groups.values():
        rng.shuffle(group)
    selected: list[dict[str, Any]] = []
    while len(selected) < budget and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(selected) < budget:
                selected.append(groups[key].pop())
    return selected
