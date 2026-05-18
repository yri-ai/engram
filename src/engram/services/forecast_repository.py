"""Repository for persisted forecast lifecycle artifacts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from engram.models.fact import Fact
from engram.models.forecasting import ForecastQuestion, ForecastResolution, ForecastRun

if TYPE_CHECKING:
    from engram.storage.base import GraphStore


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
        await self._store.save_fact(self._build_fact(question.target_entity_id, self.QUESTION_FACT_KEY, question.id, question))
        return question

    async def list_questions(self, *, target_entity_id: str) -> list[ForecastQuestion]:
        facts = await self._store.get_facts(self._tenant_id, target_entity_id, fact_key=self.QUESTION_FACT_KEY)
        return [ForecastQuestion.model_validate(self._payload_from_fact(fact)) for fact in facts]

    async def save_run(self, *, target_entity_id: str, run: ForecastRun) -> ForecastRun:
        await self._store.save_fact(self._build_fact(target_entity_id, self.RUN_FACT_KEY, run.id, run))
        return run

    async def list_runs(self, *, target_entity_id: str, question_id: str) -> list[ForecastRun]:
        facts = await self._store.get_facts(self._tenant_id, target_entity_id, fact_key=self.RUN_FACT_KEY)
        runs = [ForecastRun.model_validate(self._payload_from_fact(fact)) for fact in facts]
        return [run for run in runs if run.question_id == question_id]

    async def save_resolution(
        self, *, target_entity_id: str, resolution: ForecastResolution
    ) -> ForecastResolution:
        await self._store.save_fact(
            self._build_fact(
                target_entity_id,
                self.RESOLUTION_FACT_KEY,
                resolution.question_id,
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
