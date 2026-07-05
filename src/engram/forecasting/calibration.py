"""Post-hoc calibration utilities."""

from __future__ import annotations

import math


def temperature_scale(probabilities: dict[str, float], temperature: float) -> dict[str, float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    raw = {
        key: math.exp(math.log(max(value, 1e-12)) / temperature)
        for key, value in probabilities.items()
    }
    total = sum(raw.values())
    return {key: value / total for key, value in raw.items()}


def fit_temperature(prior_windows: list[tuple[dict[str, float], str]]) -> float:
    """Choose from a tiny deterministic grid using prior windows only."""
    if not prior_windows:
        return 1.0
    candidates = [0.75, 1.0, 1.5, 2.0]

    def loss(temp: float) -> float:
        return -sum(
            math.log(max(temperature_scale(probs, temp).get(label, 0.0), 1e-12))
            for probs, label in prior_windows
        )

    return min(candidates, key=loss)
