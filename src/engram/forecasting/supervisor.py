"""Supervisor reconciliation over forecast head distributions."""

from __future__ import annotations


def reconcile_distributions(
    distributions: list[dict[str, float]], weights: list[float] | None = None
) -> dict[str, float]:
    if not distributions:
        raise ValueError("at least one distribution is required")
    active_weights = weights or [1.0 / len(distributions)] * len(distributions)
    if len(active_weights) != len(distributions):
        raise ValueError("weights length must match distributions")
    if any(weight < 0 for weight in active_weights):
        raise ValueError("weights must be non-negative")
    buckets = sorted({bucket for distribution in distributions for bucket in distribution})
    raw = {
        bucket: sum(weight * distribution.get(bucket, 0.0) for weight, distribution in zip(active_weights, distributions, strict=True))
        for bucket in buckets
    }
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("weighted distributions must have positive mass")
    return {bucket: value / total for bucket, value in raw.items()}
