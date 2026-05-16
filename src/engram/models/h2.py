"""Models for H2 context ablation experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProfileResult:
    """Metrics for a single context profile."""

    name: str
    top1_accuracy_mean: float = 0.0
    brier_score_mean: float = 0.0
    distractor_drop: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "top1_accuracy_mean": self.top1_accuracy_mean,
            "brier_score_mean": self.brier_score_mean,
            "distractor_drop": self.distractor_drop,
        }


@dataclass
class H2Artifact:
    """Full Gate 3 artifact."""

    selected_primitive: str = "next_transition"
    profiles: dict[str, ProfileResult] = field(default_factory=dict)
    evidence_gap_coverage: float = 0.0
    competing_cause_discrimination: dict[str, Any] = field(default_factory=dict)
    distractor_report: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_primitive": self.selected_primitive,
            "profiles": {k: v.to_dict() for k, v in self.profiles.items()},
            "evidence_gap_coverage": self.evidence_gap_coverage,
            "competing_cause_discrimination": self.competing_cause_discrimination,
            "distractor_report": self.distractor_report,
        }
