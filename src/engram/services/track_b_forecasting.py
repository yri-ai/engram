"""Baseline forecaster for Track B loan delinquency prediction.

Uses transition probability matrices estimated from training data.
This replaces the hash-based placeholder scoring with a real
(simple) statistical model.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from engram.models.track_b import DelinquencyBucket


ALL_BUCKETS = [b.value for b in DelinquencyBucket]


class BaselineForecaster:
    """Transition matrix forecaster.

    Estimates P(next_bucket | current_bucket) from training data
    and uses it to predict the most likely next state.
    """

    def __init__(self) -> None:
        # transition_counts[current_bucket][next_bucket] = count
        self._transition_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self._global_counts: Counter[str] = Counter()

    def fit(self, train_rows: list[dict[str, Any]]) -> None:
        """Fit transition probabilities from labeled training rows."""
        self._transition_counts.clear()
        self._global_counts.clear()

        for row in train_rows:
            current = row["features"]["bucket"]
            next_b = row["label"]["next_bucket"]
            self._transition_counts[current][next_b] += 1
            self._global_counts[next_b] += 1

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """Predict next bucket probabilities.

        Returns dict with 'top_bucket' and 'probabilities' over all buckets.
        Falls back to global distribution for unseen buckets.
        """
        current = features["bucket"]
        counts = self._transition_counts.get(current, self._global_counts)

        if not counts:
            # No data at all — uniform
            n = len(ALL_BUCKETS)
            probs = {b: 1.0 / n for b in ALL_BUCKETS}
        else:
            total = sum(counts.values())
            probs = {b: counts.get(b, 0) / total for b in ALL_BUCKETS}

        top_bucket = max(probs, key=lambda b: probs[b])

        return {
            "top_bucket": top_bucket,
            "probabilities": probs,
        }

    def backtest(self, eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Run backtest on eval rows, return accuracy and Brier score."""
        correct = 0
        brier_sum = 0.0

        for row in eval_rows:
            truth = row["label"]["next_bucket"]
            pred = self.predict(row["features"])

            if pred["top_bucket"] == truth:
                correct += 1

            # Brier score: sum of squared errors over all classes
            for bucket in ALL_BUCKETS:
                p = pred["probabilities"].get(bucket, 0.0)
                y = 1.0 if bucket == truth else 0.0
                brier_sum += (p - y) ** 2

        n = len(eval_rows)
        return {
            "sample_count": n,
            "top1_accuracy": correct / n if n > 0 else 0.0,
            "brier_score": brier_sum / n if n > 0 else 0.0,
        }
