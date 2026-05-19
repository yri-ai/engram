"""Tests for the Engram CLI commands."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import httpx
import pytest
from typer.testing import CliRunner

from engram.cli import main as cli
from engram.models.forecasting import ForecastQuestion, ForecastRun

if TYPE_CHECKING:
    from engram.models.forecasting import ForecastResolution


def test_http_client_ingest_messages_sets_defaults() -> None:
    """EngramHTTPClient.ingest_messages should fill defaults before POSTing."""

    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        captured.append(payload)
        return httpx.Response(
            200,
            json={
                "message_id": payload["message_id"],
                "entities_extracted": 2,
                "relationships_inferred": 1,
                "conflicts_resolved": 0,
                "processing_time_ms": 123.4,
            },
        )

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport, base_url="http://testserver")
    client = cli.EngramHTTPClient(api_url="http://testserver", client=http_client)

    messages = [
        {
            "text": "Kendra switched to Adidas",
            "speaker": "Coach",
        }
    ]

    results = client.ingest_messages(
        messages=messages,
        conversation_id="conv-1",
        tenant_id="tenant-1",
        group_id=None,
    )

    assert len(results) == 1
    assert captured[0]["conversation_id"] == "conv-1"
    # group_id should fall back to conversation when not provided
    assert captured[0]["group_id"] == "conv-1"

    client.close()
    http_client.close()


class _StubClient:
    def __init__(self) -> None:
        self.ingest_calls: list[tuple] = []
        self.search_calls: list[tuple] = []
        self.closed = False

    def ingest_messages(self, **kwargs):  # type: ignore[no-untyped-def]
        self.ingest_calls.append(kwargs)
        return [
            {
                "message_id": "m-1",
                "entities_extracted": 2,
                "relationships_inferred": 1,
                "conflicts_resolved": 0,
                "processing_time_ms": 123,
            }
        ]

    def search_entities(self, query: str, tenant_id: str):  # type: ignore[override]
        self.search_calls.append((query, tenant_id))
        return [
            {"id": "entity-1", "conversation_id": "default"},
        ]

    def get_entity(self, entity_id: str):  # type: ignore[override]
        assert entity_id == "entity-1"
        return {"id": entity_id, "canonical_name": "Kendra"}

    def get_active_relationships(self, **_kwargs):  # type: ignore[no-untyped-def]
        return [
            cli.RelationshipRow(
                target_id="Adidas",
                rel_type="prefers",
                confidence=0.9,
                valid_from="2024-01-01T00:00:00Z",
                valid_to=None,
                evidence="Switched to Adidas for arch support",
            )
        ]

    def point_in_time(self, **_kwargs):  # type: ignore[no-untyped-def]
        return []

    def close(self) -> None:
        self.closed = True


class _StubForecastRepository:
    def __init__(self) -> None:
        self.listed_questions: list[ForecastQuestion] = []
        self.saved_questions: list[ForecastQuestion] = []
        self.saved_runs: list[tuple[str, ForecastRun]] = []
        self.saved_resolutions: list[tuple[str, ForecastResolution]] = []
        self.listed_runs: list[ForecastRun] = []
        self.listed_resolutions: list[ForecastResolution] = []

    async def save_question(self, question: ForecastQuestion) -> ForecastQuestion:
        self.saved_questions.append(question)
        return question

    async def list_questions(self, *, target_entity_id: str) -> list[ForecastQuestion]:
        return [question for question in self.listed_questions if question.target_entity_id == target_entity_id]

    async def save_run(self, *, target_entity_id: str, run: ForecastRun) -> ForecastRun:
        self.saved_runs.append((target_entity_id, run))
        return run

    async def save_resolution(
        self, *, target_entity_id: str, resolution: ForecastResolution
    ) -> ForecastResolution:
        self.saved_resolutions.append((target_entity_id, resolution))
        return resolution

    async def list_runs(self, *, target_entity_id: str, question_id: str) -> list[ForecastRun]:
        return [run for run in self.listed_runs if run.question_id == question_id]

    async def get_resolution(
        self, *, target_entity_id: str, question_id: str
    ) -> ForecastResolution | None:
        for resolution in self.listed_resolutions:
            if resolution.question_id == question_id:
                return resolution
        return None


class _StubForecastContext:
    def __init__(self, repository: _StubForecastRepository, scorer=None) -> None:  # type: ignore[no-untyped-def]
        self.repository = repository
        self.scorer = scorer
        self.closed = False

    async def close(self) -> None:
        self.closed = True


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def test_cli_ingest_command(monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path) -> None:
    stub = _StubClient()
    monkeypatch.setattr(cli, "_build_client", lambda *_, **__: stub)

    payload = {"messages": [{"text": "Hello", "speaker": "Coach"}]}
    file_path = tmp_path / "messages.json"
    file_path.write_text(json.dumps(payload))

    result = runner.invoke(cli.app, ["ingest", str(file_path)])

    assert result.exit_code == 0
    assert stub.ingest_calls, "ingest_messages should be called"
    assert stub.closed is True


def test_cli_query_command(monkeypatch: pytest.MonkeyPatch, runner: CliRunner) -> None:
    stub = _StubClient()
    monkeypatch.setattr(cli, "_build_client", lambda *_, **__: stub)

    result = runner.invoke(cli.app, ["query", "Kendra"])

    assert result.exit_code == 0
    assert "Active relationships" in result.stdout
    assert stub.search_calls == [("Kendra", "default")]
    assert stub.closed is True


def test_cli_forecast_command(runner: CliRunner, tmp_path) -> None:
    payload = {
        "evidence": [
            {
                "id": "costs",
                "text": "Freight and commodity cost pressure intensified.",
                "event_type": "input_cost_pressure",
                "salience": 0.9,
            },
            {
                "id": "demand",
                "text": "Demand weakness appeared in the discretionary segment.",
                "event_type": "demand_weakness",
                "salience": 0.8,
            },
        ]
    }
    file_path = tmp_path / "forecast.json"
    output_path = tmp_path / "forecast-output.json"
    file_path.write_text(json.dumps(payload))

    result = runner.invoke(
        cli.app,
        [
            "forecast",
            str(file_path),
            "--objective",
            "Q4 gross margin risk",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert "Top branch: margin_compression" in result.stdout
    written = json.loads(output_path.read_text())
    assert written["top_branch"] == "margin_compression"
    assert written["selected_context"][0]["id"] == "costs"


def test_cli_forecast_command_accepts_directory(runner: CliRunner, tmp_path) -> None:
    (tmp_path / "Rent Roll").mkdir()
    (tmp_path / "Financials").mkdir()
    (tmp_path / "Rent Roll" / "Sterling Town Center RR - 4.24.2023.xls").write_text("")
    (tmp_path / "Financials" / "Sterling Town Center T12 - 3.2023.xlsx").write_text("")
    (tmp_path / "Sterling TC Underwriting.xlsx").write_text("")

    result = runner.invoke(
        cli.app,
        [
            "forecast",
            str(tmp_path),
            "--objective",
            "acquisition diligence risk",
            "--structural-family",
            "real_estate_acquisition",
        ],
    )

    assert result.exit_code == 0
    assert "family=real_estate_acquisition" in result.stdout
    assert "Top branch:" in result.stdout


def test_cli_create_forecast_question_command(monkeypatch: pytest.MonkeyPatch, runner: CliRunner) -> None:
    repository = _StubForecastRepository()
    context = _StubForecastContext(repository)
    monkeypatch.setattr(cli, "_open_forecast_context", lambda **_kwargs: context)

    result = runner.invoke(
        cli.app,
        [
            "create-forecast-question",
            "--target-entity-id",
            "deal-123",
            "--objective",
            "Predict the next deal branch",
            "--structural-family",
            "real_estate_acquisition",
            "--forecast-as-of",
            "2026-05-01T00:00:00Z",
            "--horizon",
            "30d",
            "--resolution-due-at",
            "2026-06-01T00:00:00Z",
            "--resolution-criteria",
            "Resolve when the deal closes, reprices, or terminates",
            "--allowed-branch",
            "advance_diligence",
            "--allowed-branch",
            "reprice_or_restructure",
        ],
    )

    assert result.exit_code == 0
    assert repository.saved_questions
    saved = repository.saved_questions[0]
    assert saved.target_entity_id == "deal-123"
    assert saved.allowed_branch_names == ["advance_diligence", "reprice_or_restructure"]
    assert saved.id == ForecastQuestion.build_id(
        tenant_id="default",
        target_entity_id="deal-123",
        objective="Predict the next deal branch",
        forecast_as_of=datetime(2026, 5, 1, tzinfo=UTC),
    )
    assert context.closed is True


def test_cli_run_forecast_command_persists_forecast_run(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path
) -> None:
    repository = _StubForecastRepository()
    context = _StubForecastContext(repository)
    monkeypatch.setattr(cli, "_open_forecast_context", lambda **_kwargs: context)

    payload = {
        "evidence": [
            {
                "id": "costs",
                "text": "Freight and commodity cost pressure intensified.",
                "event_type": "input_cost_pressure",
                "salience": 0.9,
            },
            {
                "id": "demand",
                "text": "Demand weakness appeared in the discretionary segment.",
                "event_type": "demand_weakness",
                "salience": 0.8,
            },
        ]
    }
    file_path = tmp_path / "forecast.json"
    file_path.write_text(json.dumps(payload))

    result = runner.invoke(
        cli.app,
        [
            "run-forecast",
            str(file_path),
            "--question-id",
            "fq-1",
            "--target-entity-id",
            "deal-123",
            "--objective",
            "Q4 gross margin risk",
            "--structural-family",
            "margin_analysis",
            "--forecast-as-of",
            "2026-05-01T00:00:00Z",
        ],
    )

    assert result.exit_code == 0
    assert repository.saved_runs
    target_entity_id, saved = repository.saved_runs[0]
    assert target_entity_id == "deal-123"
    assert saved.question_id == "fq-1"
    assert saved.id == ForecastRun.build_id(
        question_id="fq-1",
        model_or_engine="branch_forecaster",
        forecast_as_of=datetime(2026, 5, 1, tzinfo=UTC),
        config={"max_items": 6, "max_tokens": 1200, "min_score": 0.0},
    )
    assert saved.top_branch == "margin_compression"
    assert abs(sum(saved.branch_probabilities.values()) - 1.0) < 1e-6
    assert context.closed is True


def test_cli_resolve_forecast_command(monkeypatch: pytest.MonkeyPatch, runner: CliRunner) -> None:
    repository = _StubForecastRepository()
    context = _StubForecastContext(repository)
    monkeypatch.setattr(cli, "_open_forecast_context", lambda **_kwargs: context)

    result = runner.invoke(
        cli.app,
        [
            "resolve-forecast",
            "--question-id",
            "fq-1",
            "--run-id",
            "fr-1",
            "--target-entity-id",
            "deal-123",
            "--outcome-branch",
            "reprice_or_restructure",
            "--resolved-at",
            "2026-06-15T00:00:00Z",
            "--resolved-by",
            "analyst@example.com",
            "--source",
            "ic_memo_2026_06_15",
            "--resolution-notes",
            "Seller accepted revised price after diligence.",
        ],
    )

    assert result.exit_code == 0
    assert repository.saved_resolutions
    target_entity_id, saved = repository.saved_resolutions[0]
    assert target_entity_id == "deal-123"
    assert saved.question_id == "fq-1"
    assert saved.run_id == "fr-1"
    assert saved.resolved_at == datetime(2026, 6, 15, tzinfo=UTC)
    assert context.closed is True


def test_cli_score_forecasts_command(monkeypatch: pytest.MonkeyPatch, runner: CliRunner) -> None:
    repository = _StubForecastRepository()
    repository.listed_runs = [
        ForecastRun(
            id="fr-1",
            question_id="fq-1",
            model_or_engine="branch_forecaster",
            forecast_as_of=datetime(2026, 5, 1, tzinfo=UTC),
            branch_probabilities={"advance": 0.8, "reprice": 0.2},
            top_branch="advance",
            selected_evidence_ids=["fact-1"],
            rationale="test rationale",
        )
    ]
    repository.listed_resolutions = [
        cli.ForecastResolution(
            question_id="fq-1",
            run_id="fr-1",
            resolved_at=datetime(2026, 6, 1, tzinfo=UTC),
            outcome_branch="advance",
            resolved_by="analyst@example.com",
            source="memo",
        )
    ]
    scorer = cli.ForecastScorer()
    context = _StubForecastContext(repository, scorer=scorer)
    monkeypatch.setattr(cli, "_open_forecast_context", lambda **_kwargs: context)

    result = runner.invoke(
        cli.app,
        [
            "score-forecasts",
            "--target-entity-id",
            "deal-123",
            "--question-id",
            "fq-1",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["aggregate"]["sample_count"] == 1
    assert payload["aggregate"]["top_1_accuracy"] == 1.0
    assert payload["per_question"][0]["question_id"] == "fq-1"
    assert context.closed is True


def test_cli_score_forecasts_command_scores_all_questions_for_target(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner
) -> None:
    repository = _StubForecastRepository()
    repository.listed_questions = [
        ForecastQuestion(
            id="fq-1",
            tenant_id="default",
            target_entity_id="deal-123",
            objective="Predict branch one",
            structural_family="real_estate_acquisition",
            forecast_as_of=datetime(2026, 5, 1, tzinfo=UTC),
            horizon="30d",
            resolution_due_at=datetime(2026, 6, 1, tzinfo=UTC),
            resolution_criteria="criteria",
            allowed_branch_names=["advance", "reprice"],
        ),
        ForecastQuestion(
            id="fq-2",
            tenant_id="default",
            target_entity_id="deal-123",
            objective="Predict branch two",
            structural_family="real_estate_acquisition",
            forecast_as_of=datetime(2026, 5, 2, tzinfo=UTC),
            horizon="30d",
            resolution_due_at=datetime(2026, 6, 2, tzinfo=UTC),
            resolution_criteria="criteria",
            allowed_branch_names=["advance", "reprice"],
        ),
    ]
    repository.listed_runs = [
        ForecastRun(
            id="fr-1",
            question_id="fq-1",
            model_or_engine="branch_forecaster",
            forecast_as_of=datetime(2026, 5, 1, tzinfo=UTC),
            branch_probabilities={"advance": 0.8, "reprice": 0.2},
            top_branch="advance",
            selected_evidence_ids=["fact-1"],
            rationale="test rationale",
            metadata={"target_entity_id": "deal-123", "extraction_variant": "baseline"},
        ),
        ForecastRun(
            id="fr-2",
            question_id="fq-2",
            model_or_engine="branch_forecaster",
            forecast_as_of=datetime(2026, 5, 2, tzinfo=UTC),
            branch_probabilities={"advance": 0.4, "reprice": 0.6},
            top_branch="reprice",
            selected_evidence_ids=["fact-2"],
            rationale="test rationale",
            metadata={"target_entity_id": "deal-123", "extraction_variant": "structured_v1"},
        ),
    ]
    repository.listed_resolutions = [
        cli.ForecastResolution(
            question_id="fq-1",
            run_id="fr-1",
            resolved_at=datetime(2026, 6, 1, tzinfo=UTC),
            outcome_branch="advance",
            resolved_by="analyst@example.com",
            source="memo-1",
        ),
        cli.ForecastResolution(
            question_id="fq-2",
            run_id="fr-2",
            resolved_at=datetime(2026, 6, 2, tzinfo=UTC),
            outcome_branch="advance",
            resolved_by="analyst@example.com",
            source="memo-2",
        ),
    ]
    scorer = cli.ForecastScorer()
    context = _StubForecastContext(repository, scorer=scorer)
    monkeypatch.setattr(cli, "_open_forecast_context", lambda **_kwargs: context)

    result = runner.invoke(
        cli.app,
        [
            "score-forecasts",
            "--target-entity-id",
            "deal-123",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["aggregate"]["sample_count"] == 2
    assert len(payload["per_question"]) == 2
    assert set(payload["by_extraction_variant"]) == {"baseline", "structured_v1"}
    assert context.closed is True
