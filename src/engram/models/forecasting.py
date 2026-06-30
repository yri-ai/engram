"""Forecasting contracts, ledger models, and lifecycle persistence records."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Self

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
    """Canonical definition of a forecastable closed question.

    The model accepts both the JSON-ledger temporal kernel shape and the
    graph-backed forecast lifecycle shape used by earlier CLI commands.
    """

    id: str
    tenant_id: str = "default"

    # Temporal kernel / JSON ledger shape.
    title: str | None = None
    question_type: ForecastQuestionType | None = None
    resolution_criteria: ResolutionCriteria | str
    branches: list[OutcomeBranch] = Field(default_factory=list)
    target_id: str | None = None
    status: QuestionStatus = QuestionStatus.DRAFT

    # Graph-backed lifecycle shape.
    target_entity_id: str | None = None
    objective: str | None = None
    structural_family: str | None = None
    resolution_due_at: datetime | None = None
    allowed_branch_names: list[str] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    forecast_as_of: datetime
    horizon: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def build_id(
        *, tenant_id: str, target_entity_id: str, objective: str, forecast_as_of: datetime
    ) -> str:
        digest = _forecast_digest(
            tenant_id,
            target_entity_id,
            objective.strip().casefold(),
            forecast_as_of.isoformat(),
        )
        return f"{tenant_id}:forecast-question:{digest}"

    @model_validator(mode="after")
    def validate_branches(self) -> Self:
        if self.branches:
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
    """Immutable record of a single forecast probability distribution."""

    model_config = ConfigDict(frozen=True)

    id: str
    question_id: str
    forecast_as_of: datetime
    top_branch: str

    # Temporal kernel / JSON ledger shape.
    dossier_id: str | None = None
    branch_ids: list[str] = Field(default_factory=list)
    probabilities: dict[str, float] = Field(default_factory=dict)
    protocol: str | None = None
    model_name: str | None = None
    protocol_config: dict[str, Any] = Field(default_factory=dict)
    model_config_snapshot: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    is_append_only: bool = True
    replaces_run_id: str | None = None

    # Graph-backed lifecycle shape.
    model_or_engine: str | None = None
    branch_probabilities: dict[str, float] = Field(default_factory=dict)
    selected_evidence_ids: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    rationale: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def build_id(
        *,
        question_id: str,
        model_or_engine: str,
        forecast_as_of: datetime,
        config: dict[str, Any],
    ) -> str:
        digest = _forecast_digest(
            question_id,
            model_or_engine.strip().casefold(),
            forecast_as_of.isoformat(),
            json.dumps(config, sort_keys=True, separators=(",", ":")),
        )
        return f"{question_id}:forecast-run:{digest}"

    @model_validator(mode="after")
    def validate_distribution(self) -> Self:
        distributions: list[tuple[str, dict[str, float], list[str]]] = []
        if self.probabilities:
            distributions.append(("probabilities", self.probabilities, self.branch_ids))
        if self.branch_probabilities:
            distributions.append(("branch_probabilities", self.branch_probabilities, []))

        for field_name, probabilities, branch_ids in distributions:
            if branch_ids:
                if len(branch_ids) != len(set(branch_ids)):
                    raise ValueError("branch ids must be unique")
                if set(probabilities) != set(branch_ids):
                    raise ValueError("probability keys must match branch ids")
            if any(not math.isfinite(probability) for probability in probabilities.values()):
                raise ValueError("probabilities must be finite")
            if any(
                probability < 0.0 or probability > 1.0 for probability in probabilities.values()
            ):
                raise ValueError("probabilities must be between 0 and 1")
            if abs(sum(probabilities.values()) - 1.0) > 1e-6:
                if field_name == "branch_probabilities":
                    raise ValueError("branch_probabilities must sum to 1.0")
                raise ValueError("probabilities must sum to 1.0")
            if self.top_branch not in probabilities:
                if field_name == "branch_probabilities":
                    raise ValueError("top_branch must be present in branch_probabilities")
                raise ValueError("top branch must be one of the branch ids")

            if field_name == "probabilities":
                max_probability = max(probabilities.values())
                expected_top = sorted(
                    branch_id
                    for branch_id, probability in probabilities.items()
                    if probability == max_probability
                )[0]
                if self.top_branch != expected_top:
                    raise ValueError("top branch must have the highest probability")
        return self


class ForecastResolution(BaseModel):
    """Observed outcome for a forecast question/run."""

    # Temporal kernel / JSON ledger shape.
    id: str | None = None
    question_id: str
    branch_ids: list[str] = Field(default_factory=list)
    resolved_branch: str | None = None
    resolved_at: datetime
    evidence_ids: list[str] = Field(default_factory=list)
    notes: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Graph-backed lifecycle shape.
    run_id: str | None = None
    outcome_branch: str | None = None
    outcome_probability_target: float | None = None
    resolution_notes: str | None = None
    resolved_by: str | None = None
    source: str | None = None

    @staticmethod
    def build_id(*, question_id: str, run_id: str) -> str:
        digest = _forecast_digest(question_id, run_id)
        return f"{question_id}:forecast-resolution:{digest}"

    @model_validator(mode="after")
    def validate_resolved_branch(self) -> Self:
        if (
            self.resolved_branch is not None
            and self.branch_ids
            and self.resolved_branch not in self.branch_ids
        ):
            raise ValueError("resolved branch must be one of the branch ids")
        return self


class ForecastScore(BaseModel):
    """Stored scoring output for a resolved forecast run."""

    id: str | None = None
    run_id: str
    question_id: str

    # Temporal kernel / JSON ledger shape.
    resolved_branch: str | None = None
    probability_assigned: float | None = Field(default=None, ge=0.0, le=1.0)
    brier_score: float = Field(ge=0.0)
    log_score: float | None = Field(default=None, ge=0.0)
    top_1_correct: bool
    top_k_correct: bool | None = None
    scored_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Graph-backed lifecycle shape.
    calibration_bucket: str | None = None
    expected_calibration_error: float | None = Field(default=None, ge=0.0, le=1.0)
    sample_count: int | None = Field(default=None, ge=1)


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


def _forecast_digest(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
