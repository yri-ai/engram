"""Filesystem-backed repository for temporal forecast ledger records."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from engram.models.forecasting import (
    ForecastQuestion,
    ForecastResolution,
    ForecastRun,
    ForecastScore,
    QuestionStatus,
)

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
        self._atomic_write(self._path(self.resolutions_dir, resolution.id), resolution)

    def load_resolution(self, resolution_id: str) -> ForecastResolution:
        return self._load(self._path(self.resolutions_dir, resolution_id), ForecastResolution)

    def list_resolutions(self) -> list[ForecastResolution]:
        return self._list(self.resolutions_dir, ForecastResolution)

    def save_score(self, score: ForecastScore) -> None:
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
