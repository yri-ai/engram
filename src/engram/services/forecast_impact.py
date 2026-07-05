"""Decision-impact reporting for forecast lifecycle ledgers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from engram.models.forecasting import BaselineDecisionRecord, DecisionRecord, ForecastResolution
    from engram.services.forecast_repository import JsonForecastRepository


@dataclass(frozen=True)
class TimeWindow:
    """Closed-open UTC-ish datetime window parsed from CLI text."""

    start: datetime
    end: datetime

    @classmethod
    def parse(cls, value: str) -> TimeWindow:
        separator = ".." if ".." in value else ","
        parts = [part.strip() for part in value.split(separator)]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError("window must use START..END or START,END")
        start = _coerce_aware_utc(datetime.fromisoformat(parts[0].replace("Z", "+00:00")))
        end = _coerce_aware_utc(datetime.fromisoformat(parts[1].replace("Z", "+00:00")))
        if start >= end:
            raise ValueError("window start must be before end")
        return cls(start=start, end=end)

    def contains(self, value: datetime) -> bool:
        return self.start <= _coerce_aware_utc(value) < self.end


def build_impact_report(
    repository: JsonForecastRepository,
    *,
    baseline_window: TimeWindow,
    measure_window: TimeWindow,
    min_resolved_records: int = 10,
) -> dict[str, Any]:
    """Build Layer 3 baseline-vs-forecast decision impact metrics."""

    baseline_records = [
        decision
        for decision in repository.list_baseline_decisions()
        if baseline_window.contains(decision.decided_at)
    ]
    forecast_records = [
        decision for decision in repository.list_decisions() if measure_window.contains(decision.decided_at)
    ]
    baseline_metrics = _baseline_metrics(baseline_records)
    forecast_metrics = _forecast_linked_metrics(repository, forecast_records)
    if baseline_metrics["resolved_count"] < min_resolved_records:
        raise ValueError(
            "baseline comparison side has fewer than "
            f"{min_resolved_records} resolved records: {baseline_metrics['resolved_count']}"
        )
    if forecast_metrics["resolved_count"] < min_resolved_records:
        raise ValueError(
            "measurement comparison side has fewer than "
            f"{min_resolved_records} resolved records: {forecast_metrics['resolved_count']}"
        )
    baseline_hit_rate = baseline_metrics["hit_rate"]
    forecast_hit_rate = forecast_metrics["hit_rate"]
    return {
        "schema_version": 1,
        "baseline_window": _window_payload(baseline_window),
        "measure_window": _window_payload(measure_window),
        "min_resolved_records": min_resolved_records,
        "baseline": baseline_metrics,
        "forecast_linked": forecast_metrics,
        "comparison": {
            "hit_rate_delta": None
            if baseline_hit_rate is None or forecast_hit_rate is None
            else forecast_hit_rate - baseline_hit_rate,
            "avoided_loss_delta": forecast_metrics["avoided_loss"] - baseline_metrics["avoided_loss"],
        },
    }


def _baseline_metrics(records: list[BaselineDecisionRecord]) -> dict[str, Any]:
    resolved = [record for record in records if record.realized_outcome_branch is not None]
    hits = sum(1 for record in resolved if record.expected_outcome_branch == record.realized_outcome_branch)
    return _metrics_payload(records, resolved_count=len(resolved), hits=hits)


def _forecast_linked_metrics(
    repository: JsonForecastRepository, records: list[DecisionRecord]
) -> dict[str, Any]:
    resolved_records: list[DecisionRecord] = []
    hits = 0
    pending_forecast_linked_count = 0
    for record in records:
        run = repository.load_run(record.primary_forecast_run_id)
        resolution = _resolution_for_question(repository, run.question_id)
        resolution_branch = _resolution_branch(resolution)
        if record.realized_outcome_branch is None or resolution_branch is None:
            pending_forecast_linked_count += 1
            continue
        resolved_records.append(record)
        if record.expected_outcome_branch == record.realized_outcome_branch:
            hits += 1
    payload = _metrics_payload(records, resolved_count=len(resolved_records), hits=hits)
    payload["pending_forecast_linked_count"] = pending_forecast_linked_count
    return payload


def _metrics_payload(
    records: list[BaselineDecisionRecord] | list[DecisionRecord], *, resolved_count: int, hits: int
) -> dict[str, Any]:
    avoided_loss = sum(
        record.impact_value or 0.0
        for record in records
        if record.impact_kind == "avoided_loss" and record.realized_outcome_branch is not None
    )
    pending_count = sum(1 for record in records if record.realized_outcome_branch is None)
    return {
        "total_count": len(records),
        "resolved_count": resolved_count,
        "pending_baseline_count": pending_count,
        "hit_count": hits,
        "hit_rate": None if resolved_count == 0 else hits / resolved_count,
        "avoided_loss": avoided_loss,
        "decision_ids": [record.decision_id for record in records],
    }


def _coerce_aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _resolution_branch(resolution: ForecastResolution | None) -> str | None:
    if resolution is None:
        return None
    return resolution.resolved_branch or resolution.outcome_branch


def _resolution_for_question(
    repository: JsonForecastRepository, question_id: str
) -> ForecastResolution | None:
    for resolution in repository.list_resolutions():
        if resolution.question_id == question_id:
            return resolution
    return None


def _window_payload(window: TimeWindow) -> dict[str, str]:
    return {"start": window.start.isoformat(), "end": window.end.isoformat()}
