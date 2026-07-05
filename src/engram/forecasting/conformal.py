"""Hand-rolled split-conformal prediction sets."""

from __future__ import annotations

import math


def conformal_threshold(calibration: list[tuple[dict[str, float], str]], alpha: float = 0.1) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    scores = sorted(1.0 - probs[label] for probs, label in calibration)
    if not scores:
        return 1.0
    index = math.ceil((len(scores) + 1) * (1 - alpha)) - 1
    index = min(len(scores) - 1, max(index, 0))
    return scores[index]


def prediction_set(probabilities: dict[str, float], threshold: float) -> list[str]:
    return sorted(bucket for bucket, prob in probabilities.items() if 1.0 - prob <= threshold)


def weighted_conformal_threshold(
    calibration: list[tuple[dict[str, float], str, float]], alpha: float = 0.1
) -> float:
    expanded: list[tuple[dict[str, float], str]] = []
    for probs, label, weight in calibration:
        if weight < 0:
            raise ValueError("weights must be non-negative")
        if weight == 0:
            continue
        expanded.extend([(probs, label)] * max(1, round(weight)))
    return conformal_threshold(expanded, alpha)


def timing_band(event_times: list[float], alpha: float = 0.1) -> tuple[float, float]:
    ordered = sorted(event_times)
    if not ordered:
        return (0.0, 0.0)
    lo = int(len(ordered) * alpha / 2)
    hi = min(len(ordered) - 1, int(len(ordered) * (1 - alpha / 2)))
    return (ordered[lo], ordered[hi])
