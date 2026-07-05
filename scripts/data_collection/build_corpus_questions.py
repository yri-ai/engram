#!/usr/bin/env python3
"""Build forecast lifecycle ledger questions/dossiers from PublicDeal fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from engram.models.corpus import DEFAULT_CORPUS_TAXONOMY, PublicDeal
from engram.models.forecasting import (
    EvidenceDossier,
    EvidenceItem,
    ForecastQuestion,
    ForecastQuestionType,
    ForecastResolution,
    OutcomeBranch,
    QuestionStatus,
    ResolutionCriteria,
)
from engram.services.forecast_protocol import DeterministicForecastProtocol
from engram.services.forecast_repository import JsonForecastRepository


def deal_to_question(deal: PublicDeal) -> ForecastQuestion:
    forecast_docs = [doc for doc in deal.evidence_docs if doc.role == "forecast_evidence"]
    forecast_as_of = max((doc.published_at for doc in forecast_docs), default=min(milestone.at for milestone in deal.milestones))
    return ForecastQuestion(
        id=f"corpus-question:{deal.deal_id}",
        title=f"Public deal outcome for {deal.deal_id}",
        question_type=ForecastQuestionType.CLOSED_BRANCH,
        resolution_criteria=ResolutionCriteria(
            description="Resolve to the public real-estate milestone taxonomy branch evidenced by public records.",
            resolved_by=deal.resolved_at,
        ),
        branches=[
            OutcomeBranch(id=branch_id, label=branch_id, description=description)
            for branch_id, description in DEFAULT_CORPUS_TAXONOMY.branches.items()
        ],
        status=QuestionStatus.ACTIVE,
        created_at=forecast_as_of,
        forecast_as_of=forecast_as_of,
        horizon="public_corpus_resolution",
        metadata={"deal_id": deal.deal_id, "branch_taxonomy_id": deal.branch_taxonomy_id},
    )


def deal_to_dossier(deal: PublicDeal, question: ForecastQuestion) -> EvidenceDossier:
    items: list[EvidenceItem] = []
    for doc in deal.evidence_docs:
        if doc.role != "forecast_evidence":
            continue
        if doc.published_at > question.forecast_as_of:
            raise ValueError(f"evidence doc published after forecast_as_of: {doc.doc_id}")
        first_milestone = min(deal.milestones, key=lambda milestone: milestone.at)
        items.append(
            EvidenceItem(
                id=f"corpusdoc:{deal.deal_id}:{doc.doc_id}",
                text=doc.summary,
                valid_from=first_milestone.at,
                recorded_from=doc.published_at,
                source_id=doc.doc_id,
                source_span=doc.text_ref,
                supersession_status="current_as_of",
                metadata={
                    "deal_id": deal.deal_id,
                    "source_kind": deal.source_kind,
                    "url": doc.url,
                    "retrieved_at": doc.retrieved_at.isoformat(),
                    "text_ref": doc.text_ref,
                },
            )
        )
    return EvidenceDossier(
        id=f"corpus-dossier:{deal.deal_id}",
        question_id=question.id,
        forecast_as_of=question.forecast_as_of,
        evidence_items=items,
        compiler="public_corpus.v1",
        metadata={"deal_id": deal.deal_id, "unannotated": True},
    )


def build_corpus_ledger(input_dir: Path, repo_dir: Path) -> dict[str, int]:
    repository = JsonForecastRepository(repo_dir)
    protocol = DeterministicForecastProtocol()
    deal_count = 0
    for path in sorted(input_dir.glob("*.json")):
        deal = PublicDeal.model_validate_json(path.read_text(encoding="utf-8"))
        question = deal_to_question(deal)
        dossier = deal_to_dossier(deal, question)
        repository.save_question(question)
        repository.save_dossier(dossier)
        run = protocol.create_run(question, dossier, run_id=f"corpus-run:{deal.deal_id}")
        repository.save_run(run)
        repository.save_resolution(
            ForecastResolution(
                id=f"corpus-resolution:{deal.deal_id}",
                question_id=question.id,
                branch_ids=[branch.id for branch in question.branches],
                resolved_branch=deal.resolved_branch,
                resolved_at=deal.resolved_at,
                evidence_ids=[
                    f"corpusdoc:{deal.deal_id}:{doc.doc_id}"
                    for doc in deal.evidence_docs
                    if doc.role == "resolution_evidence"
                ],
            )
        )
        deal_count += 1
    return {
        "deal_count": deal_count,
        "question_count": len(repository.list_questions()),
        "dossier_count": len(repository.list_dossiers()),
        "run_count": len(repository.list_runs()),
        "resolution_count": len(repository.list_resolutions()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = build_corpus_ledger(args.input_dir, args.repo)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
