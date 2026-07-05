"""Collateral scenario generation."""

from __future__ import annotations

import random


def sample_collateral_paths(marginals: list[float], n_paths: int, *, seed: int = 42) -> list[list[float]]:
    rng = random.Random(seed)
    paths: list[list[float]] = []
    for _ in range(n_paths):
        macro = rng.uniform(0.9, 1.1)
        paths.append([max(value * macro, 0.0) for value in marginals])
    return paths
