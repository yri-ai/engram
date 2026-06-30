"""Repositories for persisted forecast lifecycle artifacts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from engram.models.fact import Fact
from engram.models.forecasting import (
    ForecastQuestion,
    ForecastResolution,
    ForecastRun,
    ForecastScore,
    QuestionStatus,
)

if TYPE_CHECKING:
    from engram.storage.base import GraphStore

ModelT = TypeVar("ModelT", bound=BaseModel)


class JsonForecastRepository:
    """Store forecast questions, runs, resolutions, and scores as JSON files."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.questions_dir = self.root / "questions"
        self.runs_dir = self.root / "runs"
        self.resolutions_dir = self.root / "resolutions"
        self.scores_dir = self.root / "scores"
        for directory in (
            self.questions_dir,
            self.runs_dir,
            self.resolutions_dir,
            self.scores_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def save_question(self, question: ForecastQuestion) -> None:
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

    def save_run(self, run: ForecastRun) -> None:
        path = self._path(self.runs_dir, run.id)
        if path.exists():
            raise FileExistsError(f"forecast run already exists: {run.id}")
        self._atomic_write(path, run)

    def load_run(self, run_id: str) -> ForecastRun:
        return self._load(self._path(self.runs_dir, run_id), ForecastRun)

    def list_runs(self) -> list[ForecastRun]:
        return self._list(self.runs_dir, ForecastRun)

    def save_resolution(self, resolution: ForecastResolution) -> None:
        question = self.load_question(resolution.question_id)
        question_branch_ids = {branch.id for branch in question.branches}
        if resolution.resolved_branch not in question_branch_ids:
            raise ValueError("resolved branch must be one of the question branches")
        if set(resolution.branch_ids) != question_branch_ids:
            raise ValueError("resolution branch ids must match question branches")
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

    @staticmethod
    def _path(directory: Path, record_id: str) -> Path:
        if "/" in record_id or "\\" in record_id or record_id in {"", ".", ".."}:
            raise ValueError("record id must be a file-safe name")
        return directory / f"{record_id}.json"

    @staticmethod
    def _load(path: Path, model: type[ModelT]) -> ModelT:
        return model.model_validate_json(path.read_text(encoding="utf-8"))

    def _list(self, directory: Path, model: type[ModelT]) -> list[ModelT]:
        return [self._load(path, model) for path in sorted(directory.glob("*.json"))]

    @staticmethod
    def _atomic_write(path: Path, record: BaseModel) -> None:
        tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp_path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
        os.replace(tmp_path, path)


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
