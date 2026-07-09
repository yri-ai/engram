"""In-memory deal spec repository seam with immutable as-of semantics."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engram.models.deal import DealSpec


@dataclass(slots=True)
class DealRepository:
    specs: dict[str, DealSpec] = field(default_factory=dict)
    manifest: list[str] = field(default_factory=list)

    def write(self, spec: DealSpec) -> None:
        if spec.spec_id in self.specs:
            raise ValueError("deal specs are immutable by spec_id")
        self.specs[spec.spec_id] = spec
        self.manifest.append(spec.spec_id)

    def latest_as_of(self, deal_id: str, as_of: str) -> DealSpec | None:
        as_of_dt = _parse_as_of(as_of)
        candidates = [
            spec
            for spec in self.specs.values()
            if spec.deal_id == deal_id
            and _as_aware(spec.recorded_from) <= as_of_dt
            and _as_aware(spec.valid_from) <= as_of_dt
        ]
        superseded = {spec.supersedes_spec_id for spec in candidates if spec.supersedes_spec_id}
        current = [spec for spec in candidates if spec.spec_id not in superseded]
        return max(current, key=lambda spec: _as_aware(spec.recorded_from), default=None)

    def rollback_manifest(self) -> list[str]:
        return list(self.manifest)


def _parse_as_of(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    if "T" not in normalized:
        normalized = f"{normalized}T23:59:59+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _as_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value
