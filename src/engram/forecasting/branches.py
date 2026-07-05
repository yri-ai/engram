"""Branch forecast contracts for the agentic LLM head."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from engram.models.track_b import DelinquencyBucket


class Branch(BaseModel):
    target_bucket: DelinquencyBucket
    horizon_months: int = Field(default=1, ge=1, le=24)
    chain: list[DelinquencyBucket] = Field(default_factory=list)


class BranchForecast(BaseModel):
    branch: Branch
    probability: float = Field(ge=0.0, le=1.0)
    evidence_refs: list[str] = Field(default_factory=list)
    flip_conditions: list[str] = Field(default_factory=list)


class ScenarioNode(BaseModel):
    node_id: str
    branch: Branch | None = None
    probability: float = Field(default=1.0, ge=0.0, le=1.0)
    children: list[ScenarioNode] = Field(default_factory=list)

    @field_validator("children")
    @classmethod
    def _children_sum_to_at_most_one(cls, children: list[ScenarioNode]) -> list[ScenarioNode]:
        if sum(child.probability for child in children) > 1.000001:
            raise ValueError("child probabilities must sum to at most 1")
        return children


def enumerate_bucket_branches(horizon_months: int = 1) -> list[Branch]:
    return [Branch(target_bucket=bucket, horizon_months=horizon_months) for bucket in DelinquencyBucket]
