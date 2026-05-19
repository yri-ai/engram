"""Scoring utilities for resolved forecast runs."""

from __future__ import annotations

from collections import defaultdict

from engram.models.forecasting import ForecastResolution, ForecastRun, ForecastScore


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
                    expected_calibration_error=abs(top_confidence - float(run.top_branch == resolution.outcome_branch)),
                    sample_count=1,
                )
            )

        aggregate = self._aggregate(scores, bins)
        return {
            "aggregate": aggregate,
            "per_question": [score.model_dump(mode="json") for score in scores],
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
            avg_error = sum(score.expected_calibration_error for score in bucket_scores) / len(bucket_scores)
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
