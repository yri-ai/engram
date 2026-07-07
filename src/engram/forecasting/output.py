"""Public prediction output contract."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionReport(BaseModel):
    prediction_id: str
    as_of: str
    calibrated_probabilities: dict[str, float]
    conformal_set: list[str]
    evidence_chain: list[str] = Field(default_factory=list)
    flip_conditions: list[str] = Field(default_factory=list)
    model_attribution: dict[str, float] = Field(default_factory=dict)
    cost: dict[str, float] = Field(default_factory=dict)
