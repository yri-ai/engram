"""Predictive primitives for H3 comparison.

Each primitive is a forecaster that takes features and produces
predictions in a common format: {top_bucket, probabilities}.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


class TransitionMatrixPrimitive:
    """Base primitive using transition probability matrix.

    Used by endpoint, next_transition, and short_chain — they differ
    in how labels are constructed, but the model is the same.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._counts: dict[str, Counter[str]] = defaultdict(Counter)
        self._global: Counter[str] = Counter()
        self._classes: list[str] = []

    def fit(self, rows: list[dict[str, Any]]) -> None:
        self._counts.clear()
        self._global.clear()
        for row in rows:
            current = row["features"]["bucket"]
            target = row["label"]["next_bucket"]
            self._counts[current][target] += 1
            self._global[target] += 1
        self._classes = sorted(set(self._global.keys()))

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        current = features["bucket"]
        counts = self._counts.get(current, self._global)
        if not counts:
            n = max(len(self._classes), 1)
            probs = {c: 1.0 / n for c in self._classes}
        else:
            total = sum(counts.values())
            probs = {c: counts.get(c, 0) / total for c in self._classes}
        top = max(probs, key=lambda c: probs[c]) if probs else ""
        return {"top_bucket": top, "probabilities": probs}

    def backtest(self, rows: list[dict[str, Any]]) -> dict[str, float]:
        correct = 0
        brier_sum = 0.0
        for row in rows:
            truth = row["label"]["next_bucket"]
            pred = self.predict(row["features"])
            if pred["top_bucket"] == truth:
                correct += 1
            for c in self._classes:
                p = pred["probabilities"].get(c, 0.0)
                y = 1.0 if c == truth else 0.0
                brier_sum += (p - y) ** 2
        n = len(rows)
        return {
            "top1_accuracy": correct / n if n else 0.0,
            "brier_score": brier_sum / n if n else 0.0,
        }


class BranchRankingPrimitive:
    """Branch ranking: predicts trajectory type (stable/deteriorate/recover).

    Uses the same transition matrix approach but over branch labels.
    """

    def __init__(self) -> None:
        self.name = "branch_ranking"
        self._inner = TransitionMatrixPrimitive("branch_ranking")

    def fit(self, rows: list[dict[str, Any]]) -> None:
        self._inner.fit(rows)

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        return self._inner.predict(features)

    def backtest(self, rows: list[dict[str, Any]]) -> dict[str, float]:
        return self._inner.backtest(rows)


class LatentTransitionPrimitive:
    """Transition primitive with latent state inference.

    Extends the basic transition matrix by adding a "momentum" feature:
    if the loan transitioned in the previous month, it's more likely to
    continue transitioning. This is the simplest latent state.
    """

    def __init__(self) -> None:
        self.name = "latent_transition"
        # (current_bucket, had_recent_change) -> next_bucket
        self._counts: dict[tuple[str, bool], Counter[str]] = defaultdict(Counter)
        self._global: Counter[str] = Counter()
        self._classes: list[str] = []

    def fit(self, rows: list[dict[str, Any]]) -> None:
        self._counts.clear()
        self._global.clear()
        for row in rows:
            current = row["features"]["bucket"]
            changed = row["features"].get("prev_bucket") is not None and \
                      row["features"]["prev_bucket"] != current
            target = row["label"]["next_bucket"]
            self._counts[(current, changed)][target] += 1
            self._global[target] += 1
        self._classes = sorted(set(self._global.keys()))

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        current = features["bucket"]
        changed = features.get("prev_bucket") is not None and \
                  features["prev_bucket"] != current
        counts = self._counts.get((current, changed), self._global)
        if not counts:
            n = max(len(self._classes), 1)
            probs = {c: 1.0 / n for c in self._classes}
        else:
            total = sum(counts.values())
            probs = {c: counts.get(c, 0) / total for c in self._classes}
        top = max(probs, key=lambda c: probs[c]) if probs else ""
        return {"top_bucket": top, "probabilities": probs}

    def backtest(self, rows: list[dict[str, Any]]) -> dict[str, float]:
        correct = 0
        brier_sum = 0.0
        for row in rows:
            truth = row["label"]["next_bucket"]
            pred = self.predict(row["features"])
            if pred["top_bucket"] == truth:
                correct += 1
            for c in self._classes:
                p = pred["probabilities"].get(c, 0.0)
                y = 1.0 if c == truth else 0.0
                brier_sum += (p - y) ** 2
        n = len(rows)
        return {
            "top1_accuracy": correct / n if n else 0.0,
            "brier_score": brier_sum / n if n else 0.0,
        }


def compute_ece(
    rows: list[dict[str, Any]],
    primitive: TransitionMatrixPrimitive | BranchRankingPrimitive | LatentTransitionPrimitive,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error."""
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for row in rows:
        pred = primitive.predict(row["features"])
        truth = row["label"]["next_bucket"]
        top_prob = max(pred["probabilities"].values()) if pred["probabilities"] else 0.0
        correct = pred["top_bucket"] == truth
        bin_idx = min(int(top_prob * n_bins), n_bins - 1)
        bins[bin_idx].append((top_prob, correct))

    ece = 0.0
    total = len(rows)
    for bin_items in bins:
        if not bin_items:
            continue
        avg_conf = sum(p for p, _ in bin_items) / len(bin_items)
        avg_acc = sum(1 for _, c in bin_items if c) / len(bin_items)
        ece += len(bin_items) / total * abs(avg_conf - avg_acc)
    return ece
