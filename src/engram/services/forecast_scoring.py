"""Scoring utilities for resolved forecast runs."""

from __future__ import annotations

from collections import defaultdict
from typing import TypedDict

from engram.models.forecasting import ForecastResolution, ForecastRun, ForecastScore


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


class ForecastScorer:
    """Compute per-question and aggregate forecast metrics."""

    def score_runs(
        self,
        runs: list[ForecastRun],
        resolutions: list[ForecastResolution],
        *,
        bins: int = 10,
    ) -> dict[str, object]:
        resolution_by_run_id = {resolution.run_id: resolution for resolution in resolutions}
        scores: list[ForecastScore] = []
        per_question: list[ScoredQuestionRow] = []

        for run in runs:
            resolution = resolution_by_run_id.get(run.id)
            if resolution is None:
                continue

            brier_score = self._multiclass_brier(run, resolution)
            top_confidence = run.branch_probabilities.get(run.top_branch, 0.0)
            scores.append(
                ForecastScore(
                    question_id=run.question_id,
                    run_id=run.id,
                    brier_score=brier_score,
                    top_1_correct=run.top_branch == resolution.outcome_branch,
                    calibration_bucket=self._bucket_label(top_confidence, bins),
                    expected_calibration_error=abs(
                        top_confidence - float(run.top_branch == resolution.outcome_branch)
                    ),
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
                    outcome_branch=resolution.outcome_branch,
                    resolution_source=resolution.source,
                    selected_evidence_ids=run.selected_evidence_ids,
                    extraction_variant=run.metadata.get("extraction_variant", "default"),
                    brier_score=brier_score,
                    top_1_correct=run.top_branch == resolution.outcome_branch,
                    calibration_bucket=self._bucket_label(top_confidence, bins),
                    expected_calibration_error=abs(
                        top_confidence - float(run.top_branch == resolution.outcome_branch)
                    ),
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
        total = 0.0
        for branch, probability in run.branch_probabilities.items():
            outcome = 1.0 if branch == resolution.outcome_branch else 0.0
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
            buckets[score.calibration_bucket].append(score)

        ece = 0.0
        for bucket_scores in buckets.values():
            avg_error = sum(score.expected_calibration_error for score in bucket_scores) / len(
                bucket_scores
            )
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
