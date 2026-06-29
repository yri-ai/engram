"""Contracts for temporal forecasting questions, evidence, runs, and scores."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ForecastQuestionType(StrEnum):
    """Supported closed outcome spaces for forecast questions."""

    BINARY = "binary"
    CLOSED_BRANCH = "closed_branch"


class QuestionStatus(StrEnum):
    """Lifecycle state for a forecast question."""

    DRAFT = "draft"
    ACTIVE = "active"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"


class OutcomeBranch(BaseModel):
    """A mutually exclusive outcome option for a forecast question."""

    id: str
    label: str
    description: str | None = None
    prior: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResolutionCriteria(BaseModel):
    """Human-auditable rule for deciding a question's resolved branch."""

    description: str = Field(min_length=1)
    resolved_by: datetime | None = None
    source_requirements: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ForecastQuestion(BaseModel):
    """A closed forecast question bound to an as-of date and resolution rule."""

    id: str
    tenant_id: str = "default"
    title: str
    question_type: ForecastQuestionType
    forecast_as_of: datetime
    horizon: str = Field(min_length=1)
    resolution_criteria: ResolutionCriteria
    branches: list[OutcomeBranch] = Field(min_length=2)
    target_id: str | None = None
    status: QuestionStatus = QuestionStatus.DRAFT
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_branches(self) -> ForecastQuestion:
        branch_ids = [branch.id for branch in self.branches]
        if len(branch_ids) != len(set(branch_ids)):
            raise ValueError("branch ids must be unique")
        if self.question_type == ForecastQuestionType.BINARY and len(self.branches) != 2:
            raise ValueError("binary forecast questions must have exactly two branches")
        return self


class EvidenceItem(BaseModel):
    """A leakage-safe evidence packet with explicit valid and recorded time."""

    id: str
    text: str
    valid_from: datetime
    valid_to: datetime | None = None
    recorded_from: datetime
    recorded_to: datetime | None = None
    source_id: str
    source_span: str | None = None
    supports_branch: list[str] = Field(default_factory=list)
    opposes_branch: list[str] = Field(default_factory=list)
    supersession_status: str
    supersedes_id: str | None = None
    superseded_by_id: str | None = None
    contradicts_ids: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceDossier(BaseModel):
    """Evidence compiled for one question using only information known as-of."""

    id: str
    question_id: str
    forecast_as_of: datetime
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    excluded_counts: dict[str, int] = Field(default_factory=dict)
    missing_evidence: list[str] = Field(default_factory=list)
    compiler: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class ForecastRun(BaseModel):
    """An immutable forecast probability distribution for a question."""

    model_config = ConfigDict(frozen=True)

    id: str
    question_id: str
    dossier_id: str
    forecast_as_of: datetime
    branch_ids: list[str] = Field(min_length=2)
    probabilities: dict[str, float]
    top_branch: str
    protocol: str
    model_name: str | None = None
    protocol_config: dict[str, Any] = Field(default_factory=dict)
    model_config_snapshot: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    rationale: str | None = None
    is_append_only: bool = True
    replaces_run_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_distribution(self) -> ForecastRun:
        if len(self.branch_ids) != len(set(self.branch_ids)):
            raise ValueError("branch ids must be unique")
        if set(self.probabilities) != set(self.branch_ids):
            raise ValueError("probability keys must match branch ids")
        if any(not math.isfinite(probability) for probability in self.probabilities.values()):
            raise ValueError("probabilities must be finite")
        if any(
            probability < 0.0 or probability > 1.0 for probability in self.probabilities.values()
        ):
            raise ValueError("probabilities must be between 0 and 1")
        if abs(sum(self.probabilities.values()) - 1.0) > 1e-6:
            raise ValueError("probabilities must sum to 1.0")
        if self.top_branch not in self.branch_ids:
            raise ValueError("top branch must be one of the branch ids")
        max_probability = max(self.probabilities.values())
        expected_top = sorted(
            branch_id
            for branch_id, probability in self.probabilities.items()
            if probability == max_probability
        )[0]
        if self.top_branch != expected_top:
            raise ValueError("top branch must have the highest probability")
        return self


class ForecastResolution(BaseModel):
    """Resolved outcome for a forecast question."""

    id: str
    question_id: str
    branch_ids: list[str] = Field(min_length=2)
    resolved_branch: str
    resolved_at: datetime
    evidence_ids: list[str] = Field(default_factory=list)
    notes: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_resolved_branch(self) -> ForecastResolution:
        if self.resolved_branch not in self.branch_ids:
            raise ValueError("resolved branch must be one of the branch ids")
        return self


class ForecastScore(BaseModel):
    """Proper scoring output for one resolved forecast run."""

    id: str
    run_id: str
    question_id: str
    resolved_branch: str
    probability_assigned: float = Field(ge=0.0, le=1.0)
    brier_score: float = Field(ge=0.0)
    log_score: float = Field(ge=0.0)
    top_1_correct: bool
    top_k_correct: bool | None = None
    scored_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class CalibrationSummary(BaseModel):
    """Aggregate calibration metrics over a set of scored forecast runs."""

    id: str
    run_count: int = Field(ge=0)
    mean_brier_score: float | None = Field(default=None, ge=0.0)
    mean_log_score: float | None = Field(default=None, ge=0.0)
    bucket_count: int = Field(default=10, ge=1)
    buckets: list[dict[str, Any]] = Field(default_factory=list)
    low_sample_warning: bool = False
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)
