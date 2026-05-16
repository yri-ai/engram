"""Contracts for schema-guided branch forecasting."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    """A candidate historical claim or event used for branch selection."""

    id: str
    text: str
    event_type: str
    source: str | None = None
    timestamp: str | None = None
    salience: float = Field(default=1.0, ge=0.0)
    tokens: int = Field(default=1, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BranchDefinition(BaseModel):
    """A plausible future branch with compact precursor requirements."""

    name: str
    description: str
    precursor_events: list[str] = Field(default_factory=list)
    blocked_by_events: list[str] = Field(default_factory=list)
    prior: float = Field(default=0.5, ge=0.0)


class ContextBudget(BaseModel):
    """Limits for evidence passed into a branch forecast."""

    max_items: int = Field(default=6, ge=1)
    max_tokens: int = Field(default=1200, ge=1)
    min_score: float = Field(default=0.0, ge=0.0)


class BranchScore(BaseModel):
    """Scored branch with evidence and a human-auditable rationale."""

    branch: str
    score: float = Field(ge=0.0, le=1.0)
    matched_evidence_ids: list[str] = Field(default_factory=list)
    missing_precursors: list[str] = Field(default_factory=list)
    blocked_by_evidence_ids: list[str] = Field(default_factory=list)
    rationale: str


class BranchForecast(BaseModel):
    """Result of a branch-sensitive transition forecast."""

    objective: str
    structural_family: str
    top_branch: str
    scores: list[BranchScore]
    selected_context: list[EvidenceItem]
    evidence_gaps: list[str] = Field(default_factory=list)
    mode: Literal["schema_guided", "minimal_discriminative"] = "schema_guided"


class BranchFeedback(BaseModel):
    """Observed usefulness signal for online Bayesian updates."""

    branch: str
    objective: str
    useful: bool
    weight: float = Field(default=1.0, gt=0.0)
