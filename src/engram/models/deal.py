"""Deal specification contracts for deterministic waterfall simulation."""

from __future__ import annotations

import ast
from datetime import datetime  # noqa: TC003
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Tranche(BaseModel):
    tranche_id: str
    balance: float = Field(ge=0.0)
    coupon: float = Field(ge=0.0)
    seniority: int = Field(ge=1)


class Trigger(BaseModel):
    trigger_id: str
    trigger_type: Literal["OC", "IC"]
    formula: str
    threshold: float

    @field_validator("formula")
    @classmethod
    def _safe_formula(cls, value: str) -> str:
        tree = ast.parse(value, mode="eval")
        allowed = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Name,
            ast.Load,
            ast.Constant,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.USub,
        )
        if any(not isinstance(node, allowed) for node in ast.walk(tree)):
            raise ValueError("formula must use the restricted expression language")
        return value


class Covenant(BaseModel):
    covenant_id: str
    description: str


class WaterfallStep(BaseModel):
    step_id: str
    tranche_id: str
    rule: Literal["interest", "principal", "residual"]
    priority: int = Field(ge=1)


class DealSpec(BaseModel):
    schema_version: int = 1
    spec_id: str
    deal_id: str
    valid_from: datetime
    recorded_from: datetime
    supersedes_spec_id: str | None = None
    verified: bool = False
    verified_by: str | None = None
    verified_at: datetime | None = None
    source_ids: list[str] = Field(default_factory=list)
    tranches: list[Tranche] = Field(default_factory=list)
    triggers: list[Trigger] = Field(default_factory=list)
    covenants: list[Covenant] = Field(default_factory=list)
    waterfall: list[WaterfallStep] = Field(default_factory=list)

    def require_verified(self) -> None:
        if not self.verified:
            raise ValueError("deal spec must be human verified before simulation")
