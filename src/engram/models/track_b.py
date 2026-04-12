"""Canonical Track B models for loan-level forecasting."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Any


class DelinquencyBucket(StrEnum):
    CURRENT = "current"
    D30 = "d30"
    D60 = "d60"
    D90 = "d90"
    D90_PLUS = "d90_plus"
    REO = "reo"

    @classmethod
    def from_raw(cls, status: str, months: str) -> DelinquencyBucket:
        """Convert Ginnie Mae raw delinquency fields to bucket.

        Args:
            status: F (foreclosure), V (voluntary), R (REO)
            months: months delinquent (1-based: 1=current, 2=30-day, etc.)
        """
        if status.upper().startswith("R"):
            return cls.REO
        months_int = int(months) if months.strip() else 1
        # Ginnie Mae uses 1-based: 1=current, 2=30-day, 3=60-day, etc.
        return cls.from_months_delinquent(months_int - 1)

    @classmethod
    def from_months_delinquent(cls, months: int) -> DelinquencyBucket:
        """Convert 0-based months delinquent to bucket."""
        if months <= 0:
            return cls.CURRENT
        if months == 1:
            return cls.D30
        if months == 2:
            return cls.D60
        if months == 3:
            return cls.D90
        return cls.D90_PLUS


@dataclass
class TrackBEvent:
    """A single loan-month observation."""

    loan_id: str
    as_of: date
    bucket: DelinquencyBucket
    current_upb: float
    interest_rate: float | None = None
    credit_score: int | None = None
    state: str | None = None
    original_upb: float | None = None

    @property
    def event_id(self) -> str:
        return f"{self.loan_id}-{self.as_of.strftime('%Y%m')}"

    @property
    def message_id(self) -> str:
        return f"track-b-{self.loan_id}-{self.as_of.strftime('%Y%m')}"

    def features_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "bucket": self.bucket.value,
            "current_upb": self.current_upb,
        }
        if self.interest_rate is not None:
            d["interest_rate"] = self.interest_rate
        if self.credit_score is not None:
            d["credit_score"] = self.credit_score
        if self.state is not None:
            d["state"] = self.state
        if self.original_upb is not None:
            d["original_upb"] = self.original_upb
        return d

    def to_canonical_row(
        self,
        next_bucket: DelinquencyBucket,
        split: str,
        horizon_months: int = 1,
    ) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "message_id": self.message_id,
            "loan_id": self.loan_id,
            "as_of": self.as_of.isoformat(),
            "split": split,
            "features": self.features_dict(),
            "label": {
                "next_bucket": next_bucket.value,
                "horizon_months": horizon_months,
            },
        }
