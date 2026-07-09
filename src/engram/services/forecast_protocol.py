"""Deterministic baseline protocol for temporal forecast runs."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from engram.models.forecasting import EvidenceDossier, ForecastQuestion, ForecastRun


class ForecastProtocol(Protocol):
    """Synchronous forecast protocol seam."""

    def create_run(
        self,
        question: ForecastQuestion,
        dossier: EvidenceDossier,
        *,
        run_id: str | None = None,
        cited_evidence_ids: list[str] | None = None,
    ) -> ForecastRun: ...


class AsyncForecastProtocol(Protocol):
    """Asynchronous forecast protocol seam for LLM-backed forecasters."""

    async def create_run(
        self,
        question: ForecastQuestion,
        dossier: EvidenceDossier,
        *,
        run_id: str | None = None,
        cited_evidence_ids: list[str] | None = None,
    ) -> ForecastRun: ...


@dataclass(frozen=True)
class DeterministicForecastProtocolConfig:
    """Tunable weights for the deterministic baseline protocol."""

    support_weight: float = 0.7
    oppose_weight: float = 0.9
    missing_penalty: float = 0.25
    temperature: float = 1.0
    probability_sum_tolerance: float = 1e-6


class DeterministicForecastProtocol:
    """Map explicit branch evidence into a reproducible probability distribution."""

    protocol = "deterministic-baseline"
    model_name = "deterministic-forecast-protocol"

    def __init__(self, config: DeterministicForecastProtocolConfig | None = None) -> None:
        self.config = config or DeterministicForecastProtocolConfig()
        if self.config.temperature <= 0:
            raise ValueError("temperature must be greater than zero")

    def create_run(
        self,
        question: ForecastQuestion,
        dossier: EvidenceDossier,
        *,
        run_id: str | None = None,
        cited_evidence_ids: list[str] | None = None,
    ) -> ForecastRun:
        """Create a scoreable forecast run from a question and as-of dossier."""

        if dossier.question_id != question.id:
            raise ValueError("dossier question_id must match question id")
        if dossier.forecast_as_of != question.forecast_as_of:
            raise ValueError("dossier forecast_as_of must match question forecast_as_of")
        if not question.branches:
            raise ValueError(
                "DeterministicForecastProtocol requires a temporal-kernel question "
                "with a non-empty 'branches' set (graph-lifecycle questions using "
                "allowed_branch_names are not scoreable by this protocol)"
            )
        if dossier.metadata.get("audit_mode"):
            raise ValueError(
                "dossier was compiled in audit mode (contains post-as_of supersession "
                "metadata) and must not be used for forecasting"
            )

        branch_ids = [branch.id for branch in question.branches]
        branch_id_set = set(branch_ids)
        evidence_ids = [item.id for item in dossier.evidence_items]
        citation_ids = cited_evidence_ids if cited_evidence_ids is not None else evidence_ids
        self._validate_evidence_citations(citation_ids, evidence_ids)

        priors = self._branch_priors(question)
        support_counts = dict.fromkeys(branch_ids, 0)
        opposition_counts = dict.fromkeys(branch_ids, 0)

        for item in dossier.evidence_items:
            invalid_supports = set(item.supports_branch) - branch_id_set
            invalid_oppositions = set(item.opposes_branch) - branch_id_set
            if invalid_supports or invalid_oppositions:
                raise ValueError("evidence branch references must match question branches")
            for branch_id in item.supports_branch:
                support_counts[branch_id] += 1
            for branch_id in item.opposes_branch:
                opposition_counts[branch_id] += 1

        missing_count = len(dossier.missing_evidence)
        raw_scores = {
            branch_id: self._log_prior(priors[branch_id])
            + self.config.support_weight * support_counts[branch_id]
            - self.config.oppose_weight * opposition_counts[branch_id]
            - self.config.missing_penalty * missing_count
            for branch_id in branch_ids
        }
        probabilities = self._stable_softmax(raw_scores)
        probability_sum = sum(probabilities.values())
        if abs(probability_sum - 1.0) > self.config.probability_sum_tolerance:
            raise ValueError("probabilities must sum to 1.0")

        top_branch = self._top_branch(probabilities)
        run_evidence_ids = list(dict.fromkeys(citation_ids))
        rationale = self._rationale(run_evidence_ids, missing_count)

        return ForecastRun(
            id=run_id or f"run-{question.id}-{dossier.id}-{self.protocol}",
            question_id=question.id,
            dossier_id=dossier.id,
            forecast_as_of=question.forecast_as_of,
            branch_ids=branch_ids,
            probabilities=probabilities,
            top_branch=top_branch,
            protocol=self.protocol,
            model_name=self.model_name,
            protocol_config=asdict(self.config),
            model_config_snapshot={"type": "deterministic_baseline"},
            evidence_ids=run_evidence_ids,
            rationale=rationale,
            metadata={
                "raw_scores": raw_scores,
                "support_counts": support_counts,
                "opposition_counts": opposition_counts,
                "missing_evidence_count": missing_count,
            },
        )

    def _branch_priors(self, question: ForecastQuestion) -> dict[str, float]:
        explicit_priors = [branch.prior for branch in question.branches]
        has_any_prior = any(prior is not None for prior in explicit_priors)
        has_missing_prior = any(prior is None for prior in explicit_priors)
        if has_any_prior and has_missing_prior:
            raise ValueError("branch priors must be provided for all branches or none")
        if not has_any_prior:
            prior = 1.0 / len(question.branches)
            return {branch.id: prior for branch in question.branches}

        priors = {
            branch.id: branch.prior for branch in question.branches if branch.prior is not None
        }
        prior_sum = sum(priors.values())
        if prior_sum <= 0:
            raise ValueError("branch priors must sum to a positive value")
        return {branch_id: prior / prior_sum for branch_id, prior in priors.items()}

    @staticmethod
    def _log_prior(prior: float) -> float:
        if prior == 0:
            return -math.inf
        return math.log(prior)

    def _stable_softmax(self, raw_scores: dict[str, float]) -> dict[str, float]:
        scaled_scores = {
            branch_id: score / self.config.temperature for branch_id, score in raw_scores.items()
        }
        finite_scores = [score for score in scaled_scores.values() if math.isfinite(score)]
        if not finite_scores:
            raise ValueError("at least one branch must have a finite score")
        max_score = max(finite_scores)
        exp_scores = {
            branch_id: 0.0 if not math.isfinite(score) else math.exp(score - max_score)
            for branch_id, score in scaled_scores.items()
        }
        total = sum(exp_scores.values())
        if total <= 0:
            raise ValueError("softmax total must be positive")
        probabilities = {branch_id: value / total for branch_id, value in exp_scores.items()}

        drift = 1.0 - sum(probabilities.values())
        if probabilities and abs(drift) <= self.config.probability_sum_tolerance:
            top_branch = self._top_branch(probabilities)
            probabilities[top_branch] += drift
        return probabilities

    @staticmethod
    def _top_branch(probabilities: dict[str, float]) -> str:
        max_probability = max(probabilities.values())
        return sorted(
            branch_id
            for branch_id, probability in probabilities.items()
            if probability == max_probability
        )[0]

    @staticmethod
    def _validate_evidence_citations(
        cited_evidence_ids: list[str], dossier_evidence_ids: list[str]
    ) -> None:
        unknown_ids = set(cited_evidence_ids) - set(dossier_evidence_ids)
        if unknown_ids:
            raise ValueError("cited evidence ids must exist in dossier evidence")

    @staticmethod
    def _rationale(evidence_ids: list[str], missing_count: int) -> str:
        if evidence_ids:
            cited = ", ".join(f"[{evidence_id}]" for evidence_id in evidence_ids)
            return f"Deterministic baseline used cited evidence {cited}; missing evidence count: {missing_count}."
        return f"Deterministic baseline used no cited evidence; missing evidence count: {missing_count}."


class LLMForecastProtocol:
    """LLM-assisted forecast protocol using an async JSON-completion provider."""

    protocol = "llm.v1"

    def __init__(
        self, provider: Any, *, model_name: str = "configured-llm", temperature: float = 0.0
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature

    async def create_run(
        self,
        question: ForecastQuestion,
        dossier: EvidenceDossier,
        *,
        run_id: str | None = None,
        cited_evidence_ids: list[str] | None = None,
    ) -> ForecastRun:
        if dossier.question_id != question.id:
            raise ValueError("dossier question_id must match question id")
        if dossier.forecast_as_of != question.forecast_as_of:
            raise ValueError("dossier forecast_as_of must match question forecast_as_of")
        if dossier.metadata.get("audit_mode"):
            raise ValueError(
                "dossier was compiled in audit mode and must not be used for forecasting"
            )
        branch_ids = [branch.id for branch in question.branches]
        evidence_ids = [item.id for item in dossier.evidence_items]
        citations = cited_evidence_ids if cited_evidence_ids is not None else evidence_ids
        DeterministicForecastProtocol._validate_evidence_citations(citations, evidence_ids)
        prompt = _build_llm_prompt(question, dossier)
        response = await self.provider.complete_json(prompt)
        probabilities = response.get("probabilities")
        rationale = response.get("rationale", "")
        if not isinstance(probabilities, dict):
            raise ValueError("LLM forecast response must include probabilities object")
        probability_payload = {str(key): float(value) for key, value in probabilities.items()}
        if set(probability_payload) != set(branch_ids):
            raise ValueError("LLM probabilities must cover exactly the question branches")
        if any(
            value < 0.0 or value > 1.0 or not math.isfinite(value)
            for value in probability_payload.values()
        ):
            raise ValueError("LLM probabilities must be finite values between 0 and 1")
        if abs(sum(probability_payload.values()) - 1.0) > 1e-6:
            raise ValueError("LLM probabilities must sum to 1.0")
        top_branch = DeterministicForecastProtocol._top_branch(probability_payload)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        return ForecastRun(
            id=run_id or f"run-{question.id}-{dossier.id}-{self.protocol}",
            question_id=question.id,
            dossier_id=dossier.id,
            forecast_as_of=question.forecast_as_of,
            branch_ids=branch_ids,
            probabilities=probability_payload,
            top_branch=top_branch,
            protocol=self.protocol,
            model_name=self.model_name,
            protocol_config={
                "model": self.model_name,
                "temperature": self.temperature,
                "prompt_hash": prompt_hash,
            },
            model_config_snapshot={"type": "llm_forecast", "prompt_hash": prompt_hash},
            evidence_ids=list(dict.fromkeys(citations)),
            rationale=str(rationale),
            metadata={"provider_response": response},
        )


def _build_llm_prompt(question: ForecastQuestion, dossier: EvidenceDossier) -> str:
    return json.dumps(
        {
            "task": "Assign calibrated probabilities to each closed forecast branch using only the provided as-of evidence.",
            "question": question.model_dump(mode="json"),
            "dossier": dossier.model_dump(mode="json"),
            "response_schema": {
                "probabilities": {branch.id: "float" for branch in question.branches},
                "rationale": "string",
            },
        },
        sort_keys=True,
    )
