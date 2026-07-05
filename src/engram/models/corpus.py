"""Public forecast testing corpus models."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 - pydantic needs runtime type resolution.
from typing import Literal

from pydantic import BaseModel, Field, model_validator

DEFAULT_TAXONOMY_ID = "public_real_estate_milestone_v1"
DEFAULT_BRANCHES = {
    "advance_or_close": "Deal advances materially or closes.",
    "reprice_or_restructure": "Deal reprices, restructures, or terms materially change.",
    "terminated_or_failed": "Deal terminates, fails, or is abandoned.",
}


class CorpusBranchTaxonomy(BaseModel):
    """Closed branch taxonomy for public corpus annotations."""

    taxonomy_id: str = DEFAULT_TAXONOMY_ID
    branches: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_BRANCHES))

    def validate_branch_id(self, branch_id: str) -> None:
        if branch_id not in self.branches:
            raise ValueError(f"branch id is not in taxonomy {self.taxonomy_id}: {branch_id}")


DEFAULT_CORPUS_TAXONOMY = CorpusBranchTaxonomy()


class PublicEvidenceDoc(BaseModel):
    """A public source document or excerpt for one deal."""

    doc_id: str
    url: str
    published_at: datetime
    retrieved_at: datetime
    text_ref: str | None = None
    summary: str
    role: Literal["forecast_evidence", "resolution_evidence"] = "forecast_evidence"


class PublicMilestone(BaseModel):
    """A timestamped public deal milestone."""

    at: datetime
    kind: str
    description: str


class PublicDeal(BaseModel):
    """A licensed-clean public deal timeline for forecast lifecycle testing."""

    deal_id: str
    source_kind: Literal["edgar_reit", "courtlistener", "recorder", "other"]
    evidence_docs: list[PublicEvidenceDoc] = Field(min_length=1)
    milestones: list[PublicMilestone] = Field(min_length=1)
    resolved_branch: str
    resolved_at: datetime
    branch_taxonomy_id: str = DEFAULT_TAXONOMY_ID

    @model_validator(mode="after")
    def validate_chronology_and_taxonomy(self) -> PublicDeal:
        if self.branch_taxonomy_id != DEFAULT_CORPUS_TAXONOMY.taxonomy_id:
            raise ValueError(f"unsupported branch_taxonomy_id: {self.branch_taxonomy_id}")
        DEFAULT_CORPUS_TAXONOMY.validate_branch_id(self.resolved_branch)
        for doc in self.evidence_docs:
            if doc.retrieved_at < doc.published_at:
                raise ValueError("retrieved_at must be on or after published_at")
        latest_pre_resolution = max(
            [doc.published_at for doc in self.evidence_docs]
            + [milestone.at for milestone in self.milestones]
        )
        if self.resolved_at < latest_pre_resolution:
            raise ValueError("resolved_at must be on or after all evidence docs and milestones")
        return self
