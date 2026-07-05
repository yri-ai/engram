"""Repositories for persisted forecast lifecycle artifacts."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from engram.models.fact import Fact
from engram.models.forecasting import (
    BaselineDecisionRecord,
    BeliefUpdate,
    DecisionRecord,
    EvidenceDossier,
    ForecastQuestion,
    ForecastResolution,
    ForecastRun,
    ForecastScore,
    QuestionStatus,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from engram.storage.base import GraphStore

ModelT = TypeVar("ModelT", bound=BaseModel)
SUPPORTED_SCHEMA_VERSION = 1


class SchemaVersionError(ValueError):
    """Raised when a persisted forecast artifact uses an unsupported schema."""


class JsonForecastRepository:
    """Store forecast questions, runs, resolutions, and scores as JSON files.

    Concurrency contract (MVP): the ledger is single-writer per directory for
    questions, resolutions, and scores. Forecast run writes are race-safe: run
    IDs are claimed with exclusive-create semantics (``os.link``), so concurrent
    writers cannot silently overwrite an existing run — the loser gets
    ``FileExistsError``. Multi-writer support for the other record types is a
    post-MVP concern (see master plan v2 persistence rules).
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.questions_dir = self.root / "questions"
        self.runs_dir = self.root / "runs"
        self.resolutions_dir = self.root / "resolutions"
        self.scores_dir = self.root / "scores"
        self.dossiers_dir = self.root / "dossiers"
        self.decisions_dir = self.root / "decisions"
        self.baseline_decisions_dir = self.root / "baseline_decisions"
        self.updates_dir = self.root / "updates"
        for directory in (
            self.questions_dir,
            self.dossiers_dir,
            self.runs_dir,
            self.resolutions_dir,
            self.scores_dir,
            self.decisions_dir,
            self.baseline_decisions_dir,
            self.updates_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def save_question(self, question: ForecastQuestion) -> None:
        if question.status != QuestionStatus.DRAFT and not question.branches:
            raise ValueError(
                "non-draft questions in the JSON ledger require a non-empty "
                "'branches' set (kernel shape); resolution matching depends on it"
            )
        path = self._path(self.questions_dir, question.id)
        if path.exists():
            existing = self.load_question(question.id)
            if existing.status != QuestionStatus.DRAFT:
                raise ValueError("active question changes require a new id")
        self._atomic_write(path, question)

    def load_question(self, question_id: str) -> ForecastQuestion:
        return self._load(self._path(self.questions_dir, question_id), ForecastQuestion)

    def list_questions(self) -> list[ForecastQuestion]:
        return self._list(self.questions_dir, ForecastQuestion)

    def save_dossier(self, dossier: EvidenceDossier) -> None:
        self._atomic_create(
            self._path(self.dossiers_dir, dossier.id),
            dossier,
            exists_message="evidence dossier already exists",
        )

    def load_dossier(self, dossier_id: str) -> EvidenceDossier:
        return self._load(self._path(self.dossiers_dir, dossier_id), EvidenceDossier)

    def list_dossiers(self) -> list[EvidenceDossier]:
        return self._list(self.dossiers_dir, EvidenceDossier)

    def save_run(self, run: ForecastRun) -> None:
        if run.dossier_id is not None:
            dossier_path = self._path(self.dossiers_dir, run.dossier_id)
            if not dossier_path.exists():
                raise ValueError(f"dossier_id does not exist in ledger: {run.dossier_id}")
            dossier = self.load_dossier(run.dossier_id)
            if dossier.question_id != run.question_id:
                raise ValueError("run dossier_id must reference a dossier for the same question_id")
            if dossier.forecast_as_of != run.forecast_as_of:
                raise ValueError("run dossier_id must reference a dossier with matching forecast_as_of")
        path = self._path(self.runs_dir, run.id)
        self._atomic_create(path, run, exists_message="forecast run already exists")

    def load_run(self, run_id: str) -> ForecastRun:
        return self._load(self._path(self.runs_dir, run_id), ForecastRun)

    def list_runs(self) -> list[ForecastRun]:
        return self._list(self.runs_dir, ForecastRun)

    def save_resolution(self, resolution: ForecastResolution) -> None:
        question = self.load_question(resolution.question_id)
        question_branch_ids = {branch.id for branch in question.branches}
        if question_branch_ids:
            if resolution.resolved_branch not in question_branch_ids:
                raise ValueError("resolved branch must be one of the question branches")
            if set(resolution.branch_ids) != question_branch_ids:
                raise ValueError("resolution branch ids must match question branches")
        for existing in self.list_resolutions():
            if existing.question_id == resolution.question_id:
                raise ValueError(f"question already has a resolution: {resolution.question_id}")
        resolution_id = resolution.id
        if resolution_id is None:
            raise ValueError("resolution id is required for JSON forecast repository")
        self._atomic_write(self._path(self.resolutions_dir, resolution_id), resolution)

    def load_resolution(self, resolution_id: str) -> ForecastResolution:
        return self._load(self._path(self.resolutions_dir, resolution_id), ForecastResolution)

    def list_resolutions(self) -> list[ForecastResolution]:
        return self._list(self.resolutions_dir, ForecastResolution)

    def save_score(self, score: ForecastScore) -> None:
        if score.id is None:
            raise ValueError("score id is required for JSON forecast repository")
        self._atomic_write(self._path(self.scores_dir, score.id), score)

    def load_score(self, score_id: str) -> ForecastScore:
        return self._load(self._path(self.scores_dir, score_id), ForecastScore)

    def list_scores(self) -> list[ForecastScore]:
        return self._list(self.scores_dir, ForecastScore)

    def save_decision(self, decision: DecisionRecord) -> None:
        if decision.primary_forecast_run_id in decision.supporting_forecast_run_ids:
            raise ValueError("primary forecast run must not be supporting")
        if not self._path(self.runs_dir, decision.primary_forecast_run_id).exists():
            raise ValueError(
                f"primary_forecast_run_id does not exist: {decision.primary_forecast_run_id}"
            )
        for run_id in decision.supporting_forecast_run_ids:
            if not self._path(self.runs_dir, run_id).exists():
                raise ValueError(f"supporting_forecast_run_id does not exist: {run_id}")
        primary_run = self.load_run(decision.primary_forecast_run_id)
        primary_question = self.load_question(primary_run.question_id)
        branch_ids = {branch.id for branch in primary_question.branches} or set(
            primary_question.allowed_branch_names
        )
        if decision.expected_outcome_branch not in branch_ids:
            raise ValueError("expected outcome branch must be one of the primary question branches")
        self._atomic_create(
            self._path(self.decisions_dir, decision.decision_id),
            decision,
            exists_message="decision already exists",
        )

    def load_decision(self, decision_id: str) -> DecisionRecord:
        return self._load(self._path(self.decisions_dir, decision_id), DecisionRecord)

    def list_decisions(self) -> list[DecisionRecord]:
        return self._list(self.decisions_dir, DecisionRecord)

    def save_baseline_decision(self, decision: BaselineDecisionRecord) -> None:
        self._atomic_create(
            self._path(self.baseline_decisions_dir, decision.decision_id),
            decision,
            exists_message="baseline decision already exists",
        )

    def load_baseline_decision(self, decision_id: str) -> BaselineDecisionRecord:
        return self._load(
            self._path(self.baseline_decisions_dir, decision_id), BaselineDecisionRecord
        )

    def list_baseline_decisions(self) -> list[BaselineDecisionRecord]:
        return self._list(self.baseline_decisions_dir, BaselineDecisionRecord)

    def resolve_decision(
        self,
        decision_id: str,
        *,
        realized_outcome_branch: str,
        impact_value: float | None,
        impact_kind: str | None,
    ) -> DecisionRecord:
        decision = self.load_decision(decision_id)
        if (
            decision.realized_outcome_branch is not None
            or decision.impact_value is not None
            or decision.impact_kind is not None
        ):
            raise ValueError(f"decision is already resolved: {decision_id}")
        primary_run = self.load_run(decision.primary_forecast_run_id)
        question = self.load_question(primary_run.question_id)
        branch_ids = {branch.id for branch in question.branches} or set(question.allowed_branch_names)
        if branch_ids and realized_outcome_branch not in branch_ids:
            raise ValueError("realized_outcome_branch must be one of the primary question branches")
        resolved = DecisionRecord.model_validate(
            decision.model_dump()
            | {
                "realized_outcome_branch": realized_outcome_branch,
                "impact_value": impact_value,
                "impact_kind": impact_kind,
            }
        )
        self._atomic_write(self._path(self.decisions_dir, decision_id), resolved)
        return resolved

    def resolve_baseline_decision(
        self,
        decision_id: str,
        *,
        realized_outcome_branch: str,
        impact_value: float | None,
        impact_kind: str | None,
    ) -> BaselineDecisionRecord:
        decision = self.load_baseline_decision(decision_id)
        if (
            decision.realized_outcome_branch is not None
            or decision.impact_value is not None
            or decision.impact_kind is not None
        ):
            raise ValueError(f"baseline decision is already resolved: {decision_id}")
        resolved = BaselineDecisionRecord.model_validate(
            decision.model_dump()
            | {
                "realized_outcome_branch": realized_outcome_branch,
                "impact_value": impact_value,
                "impact_kind": impact_kind,
            }
        )
        self._atomic_write(self._path(self.baseline_decisions_dir, decision_id), resolved)
        return resolved


    def save_update(self, update: BeliefUpdate) -> None:
        prior = self.load_run(update.prior_run_id)
        posterior = self.load_run(update.posterior_run_id)
        if prior.question_id != posterior.question_id:
            raise ValueError("belief update runs must belong to the same question")
        if posterior.forecast_as_of < prior.forecast_as_of:
            raise ValueError("posterior run must not predate prior run")
        if update.update_at < prior.forecast_as_of or update.update_at > posterior.forecast_as_of:
            raise ValueError("update_at must fall between prior and posterior forecast_as_of")
        self._atomic_create(
            self._path(self.updates_dir, update.update_id),
            update,
            exists_message="belief update already exists",
        )

    def load_update(self, update_id: str) -> BeliefUpdate:
        return self._load(self._path(self.updates_dir, update_id), BeliefUpdate)

    def list_updates(self) -> list[BeliefUpdate]:
        return self._list(self.updates_dir, BeliefUpdate)

    @staticmethod
    def _path(directory: Path, record_id: str) -> Path:
        if "/" in record_id or "\\" in record_id or record_id in {"", ".", ".."}:
            raise ValueError("record id must be a file-safe name")
        return directory / f"{record_id}.json"

    @staticmethod
    def _load(path: Path, model: type[ModelT]) -> ModelT:
        raw = json.loads(path.read_text(encoding="utf-8"))
        version = raw.get("schema_version", SUPPORTED_SCHEMA_VERSION)
        if version != SUPPORTED_SCHEMA_VERSION:
            raise SchemaVersionError(
                f"Unsupported schema_version {version} for {model.__name__}: {path}"
            )
        return model.model_validate(raw)

    def _list(self, directory: Path, model: type[ModelT]) -> list[ModelT]:
        return [self._load(path, model) for path in sorted(directory.glob("*.json"))]

    @staticmethod
    def _atomic_write(path: Path, record: BaseModel) -> None:
        tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp_path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
        os.replace(tmp_path, path)

    @staticmethod
    def _atomic_create(path: Path, record: BaseModel, *, exists_message: str) -> None:
        """Exclusive-create write: fails if path exists, race-safe across writers.

        Writes to a temp file, then ``os.link`` claims the final name atomically.
        Two concurrent writers cannot both succeed for the same ID.
        """
        tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp_path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
        try:
            os.link(tmp_path, path)
        except FileExistsError:
            raise FileExistsError(f"{exists_message}: {path.stem}") from None
        finally:
            tmp_path.unlink(missing_ok=True)


def migrate_json_forecast_ledger(from_dir: str | Path, to_dir: str | Path) -> dict[str, list[str]]:
    """Forward-copy a JSON forecast ledger and verify stable semantic equality."""

    source_path = Path(from_dir).resolve()
    target_path = Path(to_dir).resolve()
    if source_path == target_path:
        raise ValueError("in-place migration is not allowed")
    if not source_path.exists():
        raise ValueError(f"migration source does not exist: {source_path}")
    if not source_path.is_dir():
        raise ValueError(f"migration source must be a directory: {source_path}")
    if target_path.exists() and any(target_path.iterdir()):
        raise ValueError("migration target must be empty or absent")

    source = JsonForecastRepository(source_path)
    if target_path.exists():
        shutil.rmtree(target_path)
    target = JsonForecastRepository(target_path)

    for question in source.list_questions():
        target.save_question(question)
    for dossier in source.list_dossiers():
        target.save_dossier(dossier)
    for run in source.list_runs():
        # Migration must preserve pre-M7 runs whose dossier files were not yet
        # ledger-persisted. New writes are guarded by save_run(); migration is a
        # forward rewrite of historical artifacts and audit will classify those
        # old runs as unauditable until rerun or retired.
        target._atomic_create(  # noqa: SLF001
            target._path(target.runs_dir, run.id),  # noqa: SLF001
            run,
            exists_message="forecast run already exists",
        )
    for resolution in source.list_resolutions():
        target.save_resolution(resolution)
    for decision in source.list_decisions():
        target.save_decision(decision)
    for baseline_decision in source.list_baseline_decisions():
        target.save_baseline_decision(baseline_decision)
    for update in source.list_updates():
        target.save_update(update)
    for score in source.list_scores():
        target.save_score(score)

    summary = {
        "questions": _verify_records(source.list_questions(), target.list_questions()),
        "dossiers": _verify_records(source.list_dossiers(), target.list_dossiers()),
        "runs": _verify_records(source.list_runs(), target.list_runs()),
        "resolutions": _verify_records(source.list_resolutions(), target.list_resolutions()),
        "decisions": _verify_records(source.list_decisions(), target.list_decisions()),
        "baseline_decisions": _verify_records(
            source.list_baseline_decisions(), target.list_baseline_decisions()
        ),
        "updates": _verify_records(source.list_updates(), target.list_updates()),
        "scores": _verify_records(
            source.list_scores(), target.list_scores(), normalize_fields={"scored_at"}
        ),
    }
    return summary


def _verify_records(
    source_records: Sequence[BaseModel],
    target_records: Sequence[BaseModel],
    *,
    normalize_fields: set[str] | None = None,
) -> list[str]:
    source_payloads = [_normalized_record(record, normalize_fields or set()) for record in source_records]
    target_payloads = [_normalized_record(record, normalize_fields or set()) for record in target_records]
    if source_payloads != target_payloads:
        raise ValueError("migration verification failed")
    return [_record_identifier(payload) for payload in source_payloads]


def _record_identifier(payload: dict[str, Any]) -> str:
    return str(payload.get("id") or payload.get("decision_id") or payload.get("update_id"))


def _normalized_record(record: BaseModel, normalize_fields: set[str]) -> dict[str, Any]:
    payload = record.model_dump(mode="json")
    for field in normalize_fields:
        payload.pop(field, None)
    return dict(sorted(payload.items()))


class ForecastRepository:
    """Persist forecast lifecycle models using graph facts."""

    QUESTION_FACT_KEY = "forecast.question"
    RUN_FACT_KEY = "forecast.run"
    RESOLUTION_FACT_KEY = "forecast.resolution"

    def __init__(
        self,
        store: GraphStore,
        *,
        tenant_id: str,
        conversation_id: str,
        message_id: str,
    ) -> None:
        self._store = store
        self._tenant_id = tenant_id
        self._conversation_id = conversation_id
        self._message_id = message_id

    async def save_question(self, question: ForecastQuestion) -> ForecastQuestion:
        if question.target_entity_id is None:
            raise ValueError("target_entity_id is required for graph forecast repository")
        await self._store.save_fact(
            self._build_fact(
                question.target_entity_id, self.QUESTION_FACT_KEY, question.id, question
            )
        )
        return question

    async def list_questions(self, *, target_entity_id: str) -> list[ForecastQuestion]:
        facts = await self._store.get_facts(
            self._tenant_id, target_entity_id, fact_key=self.QUESTION_FACT_KEY
        )
        return [ForecastQuestion.model_validate(self._payload_from_fact(fact)) for fact in facts]

    async def save_run(self, *, target_entity_id: str, run: ForecastRun) -> ForecastRun:
        await self._store.save_fact(
            self._build_fact(target_entity_id, self.RUN_FACT_KEY, run.id, run)
        )
        return run

    async def list_runs(self, *, target_entity_id: str, question_id: str) -> list[ForecastRun]:
        facts = await self._store.get_facts(
            self._tenant_id, target_entity_id, fact_key=self.RUN_FACT_KEY
        )
        runs = [ForecastRun.model_validate(self._payload_from_fact(fact)) for fact in facts]
        return [run for run in runs if run.question_id == question_id]

    async def save_resolution(
        self, *, target_entity_id: str, resolution: ForecastResolution
    ) -> ForecastResolution:
        if resolution.run_id is None:
            raise ValueError("run_id is required for graph forecast repository")
        await self._store.save_fact(
            self._build_fact(
                target_entity_id,
                self.RESOLUTION_FACT_KEY,
                ForecastResolution.build_id(
                    question_id=resolution.question_id,
                    run_id=resolution.run_id,
                ),
                resolution,
            )
        )
        return resolution

    async def get_resolution(
        self, *, target_entity_id: str, question_id: str
    ) -> ForecastResolution | None:
        facts = await self._store.get_facts(
            self._tenant_id,
            target_entity_id,
            fact_key=self.RESOLUTION_FACT_KEY,
        )
        for fact in facts:
            payload = self._payload_from_fact(fact)
            if payload.get("question_id") == question_id:
                return ForecastResolution.model_validate(payload)
        return None

    def _build_fact(self, entity_id: str, fact_key: str, record_id: str, model: Any) -> Fact:
        return Fact(
            id=f"{self._tenant_id}:{fact_key}:{record_id}",
            tenant_id=self._tenant_id,
            conversation_id=self._conversation_id,
            message_id=self._message_id,
            entity_id=entity_id,
            fact_key=fact_key,
            fact_text=record_id,
            confidence=1.0,
            metadata={"payload": model.model_dump(mode="json")},
        )

    @staticmethod
    def _payload_from_fact(fact: Fact) -> dict[str, Any]:
        payload = fact.metadata.get("payload", {})
        return payload if isinstance(payload, dict) else {}
