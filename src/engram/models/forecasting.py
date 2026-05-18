"""Forecast lifecycle models for persisted questions, runs, and outcomes."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Self

from pydantic import BaseModel, Field, model_validator


class ForecastQuestion(BaseModel):
    """Canonical definition of a forecastable question."""

    id: str
    tenant_id: str
    target_entity_id: str
    objective: str
    structural_family: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    forecast_as_of: datetime
    horizon: str
    resolution_due_at: datetime
    resolution_criteria: str
    allowed_branch_names: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)


class ForecastRun(BaseModel):
    """Immutable record of a single forecast execution."""

    id: str
    question_id: str
    model_or_engine: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    forecast_as_of: datetime
    branch_probabilities: dict[str, float]
    top_branch: str
    selected_evidence_ids: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    rationale: str
    config: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_probabilities(self) -> Self:
        if self.top_branch not in self.branch_probabilities:
            raise ValueError("top_branch must be present in branch_probabilities")

        probability_sum = sum(self.branch_probabilities.values())
        if abs(probability_sum - 1.0) > 1e-6:
            raise ValueError("branch_probabilities must sum to 1.0")

        return self


class ForecastResolution(BaseModel):
    """Observed outcome for a forecast question."""

    question_id: str
    run_id: str
    resolved_at: datetime
    outcome_branch: str
    outcome_probability_target: float | None = None
    resolution_notes: str | None = None
    resolved_by: str
    source: str


class ForecastScore(BaseModel):
    """Stored scoring result for one resolved forecast run."""

    question_id: str
    run_id: str
    brier_score: float = Field(ge=0.0, le=1.0)
    top_1_correct: bool
    calibration_bucket: str
    expected_calibration_error: float = Field(ge=0.0, le=1.0)
    sample_count: int = Field(ge=1)
