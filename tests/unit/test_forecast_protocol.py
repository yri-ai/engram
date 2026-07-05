from datetime import UTC, datetime

import pytest

from engram.models.forecasting import (
    EvidenceDossier,
    EvidenceItem,
    ForecastQuestion,
    ForecastQuestionType,
    OutcomeBranch,
    ResolutionCriteria,
)
from engram.services.forecast_protocol import DeterministicForecastProtocol

NOW = datetime(2026, 1, 15, tzinfo=UTC)
LATER = datetime(2026, 2, 15, tzinfo=UTC)


def test_uniform_priors_produce_equal_probabilities_and_lexicographic_top_branch():
    question = _question(
        [OutcomeBranch(id="beta", label="Beta"), OutcomeBranch(id="alpha", label="Alpha")]
    )
    dossier = _dossier(question, [])

    run = DeterministicForecastProtocol().create_run(question, dossier, run_id="run-uniform")

    assert run.probabilities == {"beta": 0.5, "alpha": 0.5}
    assert run.top_branch == "alpha"
    assert run.protocol == "deterministic-baseline"
    assert run.model_name == "deterministic-forecast-protocol"


def test_branch_priors_are_used_as_base_probabilities():
    question = _question(
        [
            OutcomeBranch(id="yes", label="Yes", prior=0.8),
            OutcomeBranch(id="no", label="No", prior=0.2),
        ]
    )
    dossier = _dossier(question, [])

    run = DeterministicForecastProtocol().create_run(question, dossier, run_id="run-priors")

    assert run.probabilities["yes"] == pytest.approx(0.8)
    assert run.probabilities["no"] == pytest.approx(0.2)
    assert run.top_branch == "yes"


def test_mixed_branch_priors_are_rejected():
    question = _question(
        [
            OutcomeBranch(id="yes", label="Yes", prior=0.8),
            OutcomeBranch(id="no", label="No"),
        ]
    )
    dossier = _dossier(question, [])

    with pytest.raises(ValueError, match="branch priors must be provided for all branches or none"):
        DeterministicForecastProtocol().create_run(question, dossier, run_id="run-mixed-priors")


def test_support_and_opposition_evidence_adjust_probabilities():
    question = _question([OutcomeBranch(id="yes", label="Yes"), OutcomeBranch(id="no", label="No")])
    dossier = _dossier(
        question,
        [
            _evidence("e-support", supports_branch=["yes"]),
            _evidence("e-oppose", opposes_branch=["no"]),
        ],
    )

    run = DeterministicForecastProtocol().create_run(question, dossier, run_id="run-evidence")

    assert run.probabilities["yes"] > run.probabilities["no"]
    assert run.metadata["support_counts"] == {"yes": 1, "no": 0}
    assert run.metadata["opposition_counts"] == {"yes": 0, "no": 1}
    assert run.evidence_ids == ["e-support", "e-oppose"]


def test_missing_evidence_penalty_is_recorded_in_raw_scores():
    question = _question([OutcomeBranch(id="yes", label="Yes"), OutcomeBranch(id="no", label="No")])
    without_missing = _dossier(question, [])
    with_missing = _dossier(question, [], missing_evidence=["signed contract", "recent financials"])

    protocol = DeterministicForecastProtocol()
    run_without_missing = protocol.create_run(question, without_missing, run_id="run-no-missing")
    run_with_missing = protocol.create_run(question, with_missing, run_id="run-missing")

    assert run_with_missing.metadata["missing_evidence_count"] == 2
    assert run_with_missing.metadata["raw_scores"]["yes"] == pytest.approx(
        run_without_missing.metadata["raw_scores"]["yes"] - 0.5
    )


def test_probabilities_are_softmax_normalized_and_sum_within_tolerance():
    question = _question([OutcomeBranch(id="yes", label="Yes"), OutcomeBranch(id="no", label="No")])
    dossier = _dossier(
        question,
        [_evidence(f"e-{index}", supports_branch=["yes"]) for index in range(100)],
    )

    run = DeterministicForecastProtocol().create_run(question, dossier, run_id="run-stable")

    assert abs(sum(run.probabilities.values()) - 1.0) <= 1e-6
    assert run.probabilities["yes"] > 0.999999
    assert run.top_branch == "yes"


def test_cited_evidence_ids_must_exist_in_dossier():
    question = _question([OutcomeBranch(id="yes", label="Yes"), OutcomeBranch(id="no", label="No")])
    dossier = _dossier(question, [_evidence("e-known", supports_branch=["yes"])])

    with pytest.raises(ValueError, match="cited evidence ids must exist"):
        DeterministicForecastProtocol().create_run(
            question,
            dossier,
            run_id="run-bad-citation",
            cited_evidence_ids=["e-known", "e-missing"],
        )


def test_question_and_dossier_must_match():
    question = _question([OutcomeBranch(id="yes", label="Yes"), OutcomeBranch(id="no", label="No")])
    dossier = EvidenceDossier(
        id="dossier-other",
        question_id="other-question",
        forecast_as_of=question.forecast_as_of,
    )

    with pytest.raises(ValueError, match="dossier question_id must match"):
        DeterministicForecastProtocol().create_run(question, dossier, run_id="run-mismatch")


def test_protocol_rejects_graph_lifecycle_questions_without_branches():
    """Graph-shape questions (allowed_branch_names only) cannot be scored: no div-by-zero."""
    question = ForecastQuestion(
        id="q-graph-shape",
        forecast_as_of=NOW,
        horizon="30d",
        resolution_criteria="Milestone recorded.",
        allowed_branch_names=["advance", "reprice"],
    )
    dossier = EvidenceDossier(
        id="dossier-q-graph-shape",
        question_id="q-graph-shape",
        forecast_as_of=NOW,
    )

    with pytest.raises(ValueError, match="non-empty 'branches'"):
        DeterministicForecastProtocol().create_run(question, dossier, run_id="run-graph-shape")


def test_protocol_rejects_audit_mode_dossiers():
    """Dossiers compiled with future supersession audit metadata are not forecast input."""
    question = _question([OutcomeBranch(id="yes", label="Yes"), OutcomeBranch(id="no", label="No")])
    dossier = EvidenceDossier(
        id=f"dossier-{question.id}",
        question_id=question.id,
        forecast_as_of=question.forecast_as_of,
        metadata={"audit_mode": True},
    )

    with pytest.raises(ValueError, match="audit mode"):
        DeterministicForecastProtocol().create_run(question, dossier, run_id="run-audit")


@pytest.mark.asyncio
async def test_llm_forecast_protocol_rejects_forecast_as_of_mismatch():
    from engram.services.forecast_protocol import LLMForecastProtocol

    question = _question([OutcomeBranch(id="yes", label="Yes"), OutcomeBranch(id="no", label="No")])
    dossier = _dossier(question, [_evidence("e-known", supports_branch=["yes"])]).model_copy(
        update={"forecast_as_of": LATER}
    )

    with pytest.raises(ValueError, match="dossier forecast_as_of must match"):
        await LLMForecastProtocol(_StubLLMProvider(), model_name="mock-model").create_run(
            question, dossier, run_id="run-llm-mismatch"
        )


def _question(branches: list[OutcomeBranch]) -> ForecastQuestion:
    return ForecastQuestion(
        id="q-protocol",
        title="Will the contract renew?",
        question_type=ForecastQuestionType.BINARY
        if len(branches) == 2
        else ForecastQuestionType.CLOSED_BRANCH,
        forecast_as_of=NOW,
        horizon="30d",
        resolution_criteria=ResolutionCriteria(
            description="Renewal is recorded in contract data.",
            resolved_by=LATER,
        ),
        branches=branches,
    )


def _dossier(
    question: ForecastQuestion,
    evidence_items: list[EvidenceItem],
    missing_evidence: list[str] | None = None,
) -> EvidenceDossier:
    return EvidenceDossier(
        id=f"dossier-{question.id}",
        question_id=question.id,
        forecast_as_of=question.forecast_as_of,
        evidence_items=evidence_items,
        missing_evidence=missing_evidence or [],
    )


def _evidence(
    evidence_id: str,
    supports_branch: list[str] | None = None,
    opposes_branch: list[str] | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        id=evidence_id,
        text="Relevant renewal evidence.",
        valid_from=NOW,
        recorded_from=NOW,
        source_id="source-1",
        supports_branch=supports_branch or [],
        opposes_branch=opposes_branch or [],
        supersession_status="current_as_of",
    )


class _StubLLMProvider:
    async def complete_json(self, prompt: str):  # type: ignore[no-untyped-def]
        assert "probabilities" in prompt
        return {"probabilities": {"yes": 0.8, "no": 0.2}, "rationale": "Mocked forecast."}


@pytest.mark.asyncio
async def test_llm_forecast_protocol_builds_scoreable_run():
    from engram.services.forecast_protocol import LLMForecastProtocol

    question = _question([OutcomeBranch(id="yes", label="Yes"), OutcomeBranch(id="no", label="No")])
    dossier = _dossier(question, [_evidence("e-known", supports_branch=["yes"])])

    run = await LLMForecastProtocol(_StubLLMProvider(), model_name="mock-model").create_run(
        question, dossier, run_id="run-llm"
    )

    assert run.id == "run-llm"
    assert run.protocol == "llm.v1"
    assert run.model_name == "mock-model"
    assert run.probabilities == {"yes": 0.8, "no": 0.2}
    assert run.top_branch == "yes"
    assert run.protocol_config["prompt_hash"]


@pytest.mark.asyncio
async def test_llm_forecast_protocol_rejects_malformed_output():
    from engram.services.forecast_protocol import LLMForecastProtocol

    question = _question([OutcomeBranch(id="yes", label="Yes"), OutcomeBranch(id="no", label="No")])
    dossier = _dossier(question, [_evidence("e-known", supports_branch=["yes"])])

    class BadProvider:
        async def complete_json(self, prompt: str):  # type: ignore[no-untyped-def]
            return {"probabilities": {"yes": 1.0}}

    with pytest.raises(ValueError, match="cover exactly"):
        await LLMForecastProtocol(BadProvider()).create_run(question, dossier)
