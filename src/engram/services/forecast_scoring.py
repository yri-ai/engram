"""Scoring utilities for resolved forecast runs."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import UTC, datetime
from typing import Protocol, TypedDict

from engram.models.forecasting import (
    BeliefUpdate,
    CalibrationSummary,
    ForecastResolution,
    ForecastRun,
    ForecastScore,
)


class ScoredQuestionRow(TypedDict):
    question_id: str
    run_id: str
    target_entity_id: str | None
    forecast_as_of: str
    top_branch: str
    outcome_branch: str
    resolution_source: str
    selected_evidence_ids: list[str]
    extraction_variant: str
    brier_score: float
    top_1_correct: bool
    calibration_bucket: str
    expected_calibration_error: float
    sample_count: int


class ForecastScoringRepository(Protocol):
    """Repository surface needed to score stored forecast runs."""

    def list_runs(self) -> list[ForecastRun]: ...

    def list_resolutions(self) -> list[ForecastResolution]: ...

    def save_score(self, score: ForecastScore) -> None: ...

    def list_updates(self) -> list[BeliefUpdate]: ...


class ForecastScorer:
    """Compute per-question and aggregate forecast metrics."""

    def score_runs(
        self,
        runs: list[ForecastRun],
        resolutions: list[ForecastResolution],
        *,
        bins: int = 10,
    ) -> dict[str, object]:
        resolution_by_run_id = {
            resolution.run_id: resolution
            for resolution in resolutions
            if resolution.run_id is not None
        }
        scores: list[ForecastScore] = []
        per_question: list[ScoredQuestionRow] = []

        for run in runs:
            resolution = resolution_by_run_id.get(run.id)
            if resolution is None:
                continue

            brier_score = self._multiclass_brier(run, resolution)
            probabilities = _run_probabilities(run)
            top_confidence = probabilities.get(run.top_branch, 0.0)
            outcome_branch = _resolution_branch(resolution)
            top_1_correct = run.top_branch == outcome_branch
            scores.append(
                ForecastScore(
                    question_id=run.question_id,
                    run_id=run.id,
                    brier_score=brier_score,
                    top_1_correct=top_1_correct,
                    calibration_bucket=self._bucket_label(top_confidence, bins),
                    expected_calibration_error=abs(top_confidence - float(top_1_correct)),
                    sample_count=1,
                )
            )
            per_question.append(
                ScoredQuestionRow(
                    question_id=run.question_id,
                    run_id=run.id,
                    target_entity_id=run.metadata.get("target_entity_id"),
                    forecast_as_of=run.forecast_as_of.isoformat().replace("+00:00", "Z"),
                    top_branch=run.top_branch,
                    outcome_branch=outcome_branch,
                    resolution_source=resolution.source or "",
                    selected_evidence_ids=run.selected_evidence_ids,
                    extraction_variant=run.metadata.get("extraction_variant", "default"),
                    brier_score=brier_score,
                    top_1_correct=top_1_correct,
                    calibration_bucket=self._bucket_label(top_confidence, bins),
                    expected_calibration_error=abs(top_confidence - float(top_1_correct)),
                    sample_count=1,
                )
            )

        aggregate = self._aggregate(scores, bins)
        by_extraction_variant = self._aggregate_by_variant(per_question)
        return {
            "aggregate": aggregate,
            "per_question": per_question,
            "by_extraction_variant": by_extraction_variant,
        }

    @staticmethod
    def _multiclass_brier(run: ForecastRun, resolution: ForecastResolution) -> float:
        probabilities = _run_probabilities(run)
        outcome_branch = _resolution_branch(resolution)
        total = 0.0
        for branch, probability in probabilities.items():
            outcome = 1.0 if branch == outcome_branch else 0.0
            total += (probability - outcome) ** 2
        return total

    def _aggregate(self, scores: list[ForecastScore], bins: int) -> dict[str, float | int]:
        sample_count = len(scores)
        if sample_count == 0:
            return {
                "sample_count": 0,
                "top_1_accuracy": 0.0,
                "brier_score": 0.0,
                "expected_calibration_error": 0.0,
            }

        top_1_accuracy = sum(score.top_1_correct for score in scores) / sample_count
        brier_score = sum(score.brier_score for score in scores) / sample_count

        buckets: dict[str, list[ForecastScore]] = defaultdict(list)
        for score in scores:
            buckets[score.calibration_bucket or self._bucket_label(0.0, bins)].append(score)

        ece = 0.0
        for bucket_scores in buckets.values():
            avg_error = sum(
                score.expected_calibration_error or 0.0 for score in bucket_scores
            ) / len(bucket_scores)
            ece += (len(bucket_scores) / sample_count) * avg_error

        return {
            "sample_count": sample_count,
            "top_1_accuracy": top_1_accuracy,
            "brier_score": brier_score,
            "expected_calibration_error": ece,
        }

    @staticmethod
    def _bucket_label(confidence: float, bins: int) -> str:
        index = min(int(confidence * bins), bins - 1)
        start = index / bins
        end = (index + 1) / bins
        return f"{start:.1f}-{end:.1f}"

    @staticmethod
    def _aggregate_by_variant(
        per_question: list[ScoredQuestionRow],
    ) -> dict[str, dict[str, float | int]]:
        grouped: dict[str, list[ScoredQuestionRow]] = defaultdict(list)
        for item in per_question:
            variant = item.get("extraction_variant", "default")
            grouped[str(variant)].append(item)

        summary: dict[str, dict[str, float | int]] = {}
        for variant, rows in grouped.items():
            sample_count = len(rows)
            summary[variant] = {
                "sample_count": sample_count,
                "top_1_accuracy": sum(1 for row in rows if row["top_1_correct"]) / sample_count,
                "brier_score": sum(float(row["brier_score"]) for row in rows) / sample_count,
                "expected_calibration_error": sum(
                    float(row["expected_calibration_error"]) for row in rows
                )
                / sample_count,
            }
        return summary


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

    probabilities = _run_probabilities(run)
    try:
        return probabilities[resolved_branch]
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
    probabilities = _run_probabilities(run)
    resolved_branch = _resolution_branch(resolution)
    if (
        run.branch_ids
        and resolution.branch_ids
        and set(run.branch_ids) != set(resolution.branch_ids)
    ):
        raise ValueError("run and resolution branch ids must match")

    probability_assigned = probability_assigned_to_resolved_branch(run, resolved_branch)
    top_k_correct = top_k_accuracy(run, resolved_branch, top_k) if top_k is not None else None
    return ForecastScore(
        id=f"score-{run.id}",
        run_id=run.id,
        question_id=run.question_id,
        resolved_branch=resolved_branch,
        probability_assigned=probability_assigned,
        brier_score=multiclass_brier_score(probabilities, resolved_branch),
        log_score=log_score(probability_assigned, epsilon=epsilon),
        top_1_correct=top_1_accuracy(run, resolved_branch),
        top_k_correct=top_k_correct,
        metadata={
            "resolution_id": resolution.id,
            "calibration_bucket": assign_calibration_bucket(probability_assigned, bucket_count),
            "top_branch_probability": probabilities[run.top_branch],
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
        mean_log_score=_mean([score.log_score or 0.0 for score in scores]),
        bucket_count=bucket_count,
        buckets=buckets,
        low_sample_warning=run_count < low_sample_threshold,
        metadata={
            "scored_run_ids": [score.run_id for score in scores],
            "skipped_run_count": len(skipped_run_ids),
            "skipped_run_ids": skipped_run_ids,
            "update_quality": build_update_quality_report(repository),
        },
    )


def build_update_quality_report(repository: ForecastScoringRepository) -> dict[str, object]:
    """Report prequential update quality via log-score delta toward resolution."""

    runs_by_id = {run.id: run for run in repository.list_runs()}
    resolutions_by_question = _resolutions_by_question_id(repository.list_resolutions())
    rows: list[dict[str, object]] = []
    for update in repository.list_updates():
        prior = runs_by_id[update.prior_run_id]
        posterior = runs_by_id[update.posterior_run_id]
        resolution = resolutions_by_question.get(prior.question_id)
        if resolution is None:
            continue
        branch = _resolution_branch(resolution)
        prior_log = log_score(probability_assigned_to_resolved_branch(prior, branch))
        posterior_log = log_score(probability_assigned_to_resolved_branch(posterior, branch))
        rows.append(
            {
                "update_id": update.update_id,
                "prior_run_id": prior.id,
                "posterior_run_id": posterior.id,
                "resolved_branch": branch,
                "log_score_delta": prior_log - posterior_log,
                "improved": posterior_log < prior_log,
            }
        )
    return {
        "update_count": len(rows),
        "improved_count": sum(1 for row in rows if row["improved"]),
        "mean_log_score_delta": _mean(
            [
                float(row["log_score_delta"])
                for row in rows
                if isinstance(row["log_score_delta"], int | float)
            ]
        ),
        "updates": rows,
    }


def _run_probabilities(run: ForecastRun) -> dict[str, float]:
    return run.probabilities or run.branch_probabilities


def _resolution_branch(resolution: ForecastResolution) -> str:
    branch = resolution.resolved_branch or resolution.outcome_branch
    if branch is None:
        raise ValueError("resolution branch is required")
    return branch


def _ranked_branch_ids(run: ForecastRun) -> list[str]:
    probabilities = _run_probabilities(run)
    return sorted(probabilities, key=lambda branch_id: (-probabilities[branch_id], branch_id))


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
        bucket_probability = _score_bucket_probability(score)
        bucket = assign_calibration_bucket(bucket_probability, bucket_count)
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
                    [_score_bucket_probability(score) for score in bucket_scores]
                ),
                "observed_frequency": _mean(
                    [1.0 if score.top_1_correct else 0.0 for score in bucket_scores]
                ),
            }
        )
    return buckets


def _score_bucket_probability(score: ForecastScore) -> float:
    raw = score.metadata.get("top_branch_probability", score.probability_assigned)
    return float(raw if raw is not None else 0.0)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
