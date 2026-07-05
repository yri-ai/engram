"""Lifecycle audit reports for forecast ledgers."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from engram.services.forecast_repository import JsonForecastRepository

_CLOCK_SKEW_TOLERANCE = timedelta(minutes=5)


def build_audit_report(repository: JsonForecastRepository, *, spot_check: int = 0) -> dict[str, Any]:
    """Audit Layer 1 lifecycle integrity for a JSON forecast ledger."""

    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    resolutions_by_question = {resolution.question_id: resolution for resolution in repository.list_resolutions()}
    dossiers_by_id = {dossier.id: dossier for dossier in repository.list_dossiers()}

    for question in repository.list_questions():
        resolution = resolutions_by_question.get(question.id)
        if resolution is None:
            failures.append({"kind": "missing_resolution", "question_id": question.id})
        else:
            if not question.created_at < resolution.resolved_at:
                failures.append({"kind": "invalid_question_resolution_order", "question_id": question.id})
        if question.forecast_as_of > question.created_at + _CLOCK_SKEW_TOLERANCE:
            failures.append({"kind": "forecast_as_of_after_created_at", "question_id": question.id})

    for index, run in enumerate(repository.list_runs()):
        if run.forecast_as_of is None:
            failures.append({"kind": "run_missing_forecast_as_of", "run_id": run.id})
        if not run.dossier_id:
            failures.append({"kind": "unauditable_evidence", "run_id": run.id})
            continue
        dossier = dossiers_by_id.get(run.dossier_id)
        if dossier is None:
            failures.append({"kind": "missing_dossier", "run_id": run.id, "dossier_id": run.dossier_id})
            continue
        dossier_evidence_ids = {item.id for item in dossier.evidence_items}
        cited_ids = set(run.evidence_ids)
        if not cited_ids and run.metadata.get("cited_evidence_ids"):
            cited_ids = set(run.metadata["cited_evidence_ids"])
        missing_citations = sorted(cited_ids - dossier_evidence_ids)
        if missing_citations:
            failures.append({"kind": "citation_not_in_dossier", "run_id": run.id, "evidence_ids": missing_citations})
        if index < spot_check:
            for item in dossier.evidence_items:
                if item.recorded_from > run.forecast_as_of:
                    failures.append({"kind": "post_cutoff_evidence", "run_id": run.id, "evidence_id": item.id})
            warnings.append({"kind": "json_evidence_self_consistency_check", "run_id": run.id})

    auditable_runs = [run for run in repository.list_runs() if run.dossier_id in dossiers_by_id]
    passed = not failures
    return {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "question_count": len(repository.list_questions()),
        "run_count": len(repository.list_runs()),
        "auditable_run_count": len(auditable_runs),
        "unauditable_run_count": len(repository.list_runs()) - len(auditable_runs),
        "failure_count": len(failures),
        "failures": failures,
        "warnings": warnings,
    }
