"""Models for H3 predictive primitive comparison."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PrimitiveResult:
    """Metrics for a single predictive primitive across seeds."""

    name: str
    top1_accuracy_mean: float = 0.0
    top1_accuracy_std: float = 0.0
    brier_score_mean: float = 0.0
    brier_score_std: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "top1_accuracy_mean": self.top1_accuracy_mean,
            "top1_accuracy_std": self.top1_accuracy_std,
            "brier_score_mean": self.brier_score_mean,
            "brier_score_std": self.brier_score_std,
        }


@dataclass
class H3Artifact:
    """Full Gate 2 artifact."""

    seed_list: list[int] = field(default_factory=lambda: [7, 17, 27])
    primitives: dict[str, PrimitiveResult] = field(default_factory=dict)
    distractor_robustness: dict[str, dict[str, float]] = field(default_factory=dict)
    calibration: dict[str, dict[str, float]] = field(default_factory=dict)
    chain_length_sensitivity: dict[str, Any] = field(default_factory=dict)
    observed_vs_latent: dict[str, Any] = field(default_factory=dict)
    delayed_outcome_horizons: dict[str, Any] = field(default_factory=dict)
    winner: str = ""

    def select_winner(self) -> str:
        """Select the primitive with best Brier score, tie-break by accuracy."""
        if not self.primitives:
            return ""
        best = min(
            self.primitives.values(),
            key=lambda p: (p.brier_score_mean, -p.top1_accuracy_mean),
        )
        self.winner = best.name
        return self.winner

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "seed_list": self.seed_list,
            "primitives": {k: v.to_dict() for k, v in self.primitives.items()},
            "distractor_robustness": self.distractor_robustness,
            "calibration": self.calibration,
            "chain_length_sensitivity": self.chain_length_sensitivity,
            "observed_vs_latent": self.observed_vs_latent,
            "delayed_outcome_horizons": self.delayed_outcome_horizons,
            "winner": self.winner,
        }
        return d
