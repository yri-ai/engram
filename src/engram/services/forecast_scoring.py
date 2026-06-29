"""Scoring and provisional calibration reports for resolved forecasts."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import UTC, datetime
from typing import Protocol

from engram.models.forecasting import (
    CalibrationSummary,
    ForecastResolution,
    ForecastRun,
    ForecastScore,
)


class ForecastScoringRepository(Protocol):
    """Repository surface needed to score stored forecast runs."""

    def list_runs(self) -> list[ForecastRun]: ...

    def list_resolutions(self) -> list[ForecastResolution]: ...

    def save_score(self, score: ForecastScore) -> None: ...


def binary_brier_score(probability: float, resolved: bool) -> float:
    """Return binary Brier score for a single event probability."""

    _validate_probability(probability)
    outcome = 1.0 if resolved else 0.0
    return (probability - outcome) ** 2


def multiclass_brier_score(probabilities: dict[str, float], resolved_branch: str) -> float:
    """Return multiclass Brier score across the full closed outcome space."""

    if resolved_branch not in probabilities:
        raise ValueError("resolved branch is not present in probabilities")
    return sum(
        (probability - (1.0 if branch_id == resolved_branch else 0.0)) ** 2
        for branch_id, probability in probabilities.items()
    )


def log_score(probability: float, epsilon: float = 1e-15) -> float:
    """Return negative log probability, clipping away from 0 and 1."""

    _validate_probability(probability)
    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must be between 0 and 0.5")
    clipped = min(max(probability, epsilon), 1.0 - epsilon)
    return -math.log(clipped)


def top_1_accuracy(run: ForecastRun, resolved_branch: str) -> bool:
    """Return whether the deterministic top branch matched the resolution."""

    return _ranked_branch_ids(run)[0] == resolved_branch


def top_k_accuracy(run: ForecastRun, resolved_branch: str, k: int) -> bool:
    """Return whether the resolved branch is among the top-k probabilities."""

    if k < 1:
        raise ValueError("k must be at least 1")
    return resolved_branch in _ranked_branch_ids(run)[:k]


def probability_assigned_to_resolved_branch(run: ForecastRun, resolved_branch: str) -> float:
    """Return the run probability assigned to the ultimately resolved branch."""

    try:
        return run.probabilities[resolved_branch]
    except KeyError as exc:
        raise ValueError("resolved branch is not present in run probabilities") from exc


def assign_calibration_bucket(probability: float, bucket_count: int = 10) -> int:
    """Assign a probability to a zero-indexed calibration bucket."""

    _validate_probability(probability)
    if bucket_count < 1:
        raise ValueError("bucket_count must be at least 1")
    if probability == 1.0:
        return bucket_count - 1
    return int(probability * bucket_count)


def score_forecast_run(
    run: ForecastRun,
    resolution: ForecastResolution,
    *,
    top_k: int | None = None,
    bucket_count: int = 10,
    epsilon: float = 1e-15,
) -> ForecastScore:
    """Build a score record for one run and its matching resolution."""

    if run.question_id != resolution.question_id:
        raise ValueError("run and resolution question_id must match")
    if set(run.branch_ids) != set(resolution.branch_ids):
        raise ValueError("run and resolution branch ids must match")

    probability_assigned = probability_assigned_to_resolved_branch(run, resolution.resolved_branch)
    top_k_correct = (
        top_k_accuracy(run, resolution.resolved_branch, top_k) if top_k is not None else None
    )
    return ForecastScore(
        id=f"score-{run.id}",
        run_id=run.id,
        question_id=run.question_id,
        resolved_branch=resolution.resolved_branch,
        probability_assigned=probability_assigned,
        brier_score=multiclass_brier_score(run.probabilities, resolution.resolved_branch),
        log_score=log_score(probability_assigned, epsilon=epsilon),
        top_1_correct=top_1_accuracy(run, resolution.resolved_branch),
        top_k_correct=top_k_correct,
        metadata={
            "resolution_id": resolution.id,
            "calibration_bucket": assign_calibration_bucket(probability_assigned, bucket_count),
            "top_branch_probability": run.probabilities[run.top_branch],
        },
    )


def build_calibration_report(
    repository: ForecastScoringRepository,
    *,
    bucket_count: int = 10,
    low_sample_threshold: int = 30,
    skip_missing_resolutions: bool = False,
    top_k: int | None = None,
    persist_scores: bool = True,
) -> CalibrationSummary:
    """Score repository runs with resolutions and aggregate a calibration summary."""

    if bucket_count < 1:
        raise ValueError("bucket_count must be at least 1")
    if low_sample_threshold < 0:
        raise ValueError("low_sample_threshold must be non-negative")

    resolutions_by_question = _resolutions_by_question_id(repository.list_resolutions())
    scores: list[ForecastScore] = []
    skipped_run_ids: list[str] = []

    for run in repository.list_runs():
        resolution = resolutions_by_question.get(run.question_id)
        if resolution is None:
            if not skip_missing_resolutions:
                raise ValueError(f"missing resolution for run {run.id}")
            skipped_run_ids.append(run.id)
            continue

        score = score_forecast_run(run, resolution, top_k=top_k, bucket_count=bucket_count)
        scores.append(score)
        if persist_scores:
            repository.save_score(score)

    buckets = _build_buckets(scores, bucket_count)
    run_count = len(scores)
    return CalibrationSummary(
        id=f"calibration-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
        run_count=run_count,
        mean_brier_score=_mean([score.brier_score for score in scores]),
        mean_log_score=_mean([score.log_score for score in scores]),
        bucket_count=bucket_count,
        buckets=buckets,
        low_sample_warning=run_count < low_sample_threshold,
        metadata={
            "scored_run_ids": [score.run_id for score in scores],
            "skipped_run_count": len(skipped_run_ids),
            "skipped_run_ids": skipped_run_ids,
        },
    )


def _ranked_branch_ids(run: ForecastRun) -> list[str]:
    return sorted(
        run.probabilities, key=lambda branch_id: (-run.probabilities[branch_id], branch_id)
    )


def _validate_probability(probability: float) -> None:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")


def _resolutions_by_question_id(
    resolutions: list[ForecastResolution],
) -> dict[str, ForecastResolution]:
    by_question: dict[str, ForecastResolution] = {}
    for resolution in resolutions:
        if resolution.question_id in by_question:
            raise ValueError(f"multiple resolutions for question {resolution.question_id}")
        by_question[resolution.question_id] = resolution
    return by_question


def _build_buckets(
    scores: list[ForecastScore], bucket_count: int
) -> list[dict[str, float | int | None]]:
    scores_by_bucket: dict[int, list[ForecastScore]] = defaultdict(list)
    for score in scores:
        bucket_probability = score.metadata.get(
            "top_branch_probability", score.probability_assigned
        )
        bucket = assign_calibration_bucket(float(bucket_probability), bucket_count)
        scores_by_bucket[bucket].append(score)

    buckets: list[dict[str, float | int | None]] = []
    for bucket in range(bucket_count):
        bucket_scores = scores_by_bucket[bucket]
        buckets.append(
            {
                "bucket": bucket,
                "lower_bound": bucket / bucket_count,
                "upper_bound": (bucket + 1) / bucket_count,
                "count": len(bucket_scores),
                "mean_probability": _mean(
                    [
                        float(
                            score.metadata.get("top_branch_probability", score.probability_assigned)
                        )
                        for score in bucket_scores
                    ]
                ),
                "observed_frequency": _mean(
                    [1.0 if score.top_1_correct else 0.0 for score in bucket_scores]
                ),
            }
        )
    return buckets


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
