"""Tests for the Engram CLI commands."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

import httpx
import pytest
from typer.testing import CliRunner

from engram.cli import main as cli
from engram.models.forecasting import ForecastQuestion, ForecastResolution, ForecastRun


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


def test_forecast_kernel_cli_flow_uses_json_dossier_without_neo4j(
    runner: CliRunner, tmp_path
) -> None:
    repo_path = tmp_path / "forecast-ledger"
    evidence_path = tmp_path / "evidence.json"
    dossier_path = tmp_path / "dossier.json"
    run_path = tmp_path / "run.json"
    report_path = tmp_path / "report.json"
    as_of = "2026-01-15T00:00:00+00:00"
    resolved_at = "2026-02-15T00:00:00+00:00"

    create_result = runner.invoke(
        cli.app,
        [
            "forecast-question-create",
            "--repo",
            str(repo_path),
            "--question-id",
            "q-cli",
            "--title",
            "Will Alice renew?",
            "--forecast-as-of",
            as_of,
            "--horizon",
            "30d",
            "--resolution-criteria",
            "Renewal is recorded by the horizon.",
            "--resolved-by",
            resolved_at,
            "--branch",
            "yes:Yes",
            "--branch",
            "no:No",
            "--status",
            "active",
        ],
    )

    assert create_result.exit_code == 0
    question = json.loads((repo_path / "questions" / "q-cli.json").read_text())
    assert question["title"] == "Will Alice renew?"
    assert question["question_type"] == "binary"

    evidence_path.write_text(
        json.dumps(
            [
                {
                    "id": "e-renewal-signal",
                    "text": "Alice requested renewal paperwork.",
                    "valid_from": as_of,
                    "recorded_from": as_of,
                    "source_id": "source-1",
                    "supports_branch": ["yes"],
                    "supersession_status": "current_as_of",
                }
            ]
        )
    )

    dossier_result = runner.invoke(
        cli.app,
        [
            "forecast-dossier-compile",
            "--repo",
            str(repo_path),
            "--question-id",
            "q-cli",
            "--evidence-json",
            str(evidence_path),
            "--output",
            str(dossier_path),
        ],
    )

    assert dossier_result.exit_code == 0
    dossier = json.loads(dossier_path.read_text())
    assert dossier["question_id"] == "q-cli"
    assert dossier["evidence_items"][0]["id"] == "e-renewal-signal"
    assert dossier["compiler"] == "json_evidence.v1"

    run_result = runner.invoke(
        cli.app,
        [
            "forecast-run-create",
            "--repo",
            str(repo_path),
            "--question-id",
            "q-cli",
            "--dossier",
            str(dossier_path),
            "--run-id",
            "run-q-cli",
            "--output",
            str(run_path),
        ],
    )

    assert run_result.exit_code == 0
    run = json.loads((repo_path / "runs" / "run-q-cli.json").read_text())
    assert run["top_branch"] == "yes"
    assert json.loads(run_path.read_text())["id"] == "run-q-cli"

    resolve_result = runner.invoke(
        cli.app,
        [
            "forecast-resolve-create",
            "--repo",
            str(repo_path),
            "--question-id",
            "q-cli",
            "--resolved-branch",
            "yes",
            "--resolved-at",
            resolved_at,
            "--evidence-id",
            "resolution-source",
        ],
    )

    assert resolve_result.exit_code == 0
    resolution = json.loads((repo_path / "resolutions" / "resolution-q-cli.json").read_text())
    assert resolution["resolved_branch"] == "yes"

    report_result = runner.invoke(
        cli.app,
        [
            "forecast-score-report",
            "--repo",
            str(repo_path),
            "--bucket-count",
            "5",
            "--low-sample-threshold",
            "2",
            "--output",
            str(report_path),
        ],
    )

    assert report_result.exit_code == 0
    report = json.loads(report_path.read_text())
    assert report["run_count"] == 1
    assert report["low_sample_warning"] is True
    assert (repo_path / "scores" / "score-run-q-cli.json").exists()


def test_forecast_dossier_compile_rejects_future_recorded_json_evidence(
    runner: CliRunner, tmp_path
) -> None:
    repo_path = tmp_path / "forecast-ledger"
    evidence_path = tmp_path / "future-evidence.json"
    as_of = "2026-01-15T00:00:00+00:00"

    create_result = runner.invoke(
        cli.app,
        [
            "forecast-question-create",
            "--repo",
            str(repo_path),
            "--question-id",
            "q-leakage",
            "--title",
            "Will Alice renew?",
            "--forecast-as-of",
            as_of,
            "--horizon",
            "30d",
            "--resolution-criteria",
            "Renewal is recorded by the horizon.",
            "--branch",
            "yes:Yes",
            "--branch",
            "no:No",
        ],
    )
    assert create_result.exit_code == 0

    evidence_path.write_text(
        json.dumps(
            [
                {
                    "id": "e-future",
                    "text": "Future evidence.",
                    "valid_from": as_of,
                    "recorded_from": "2026-02-01T00:00:00+00:00",
                    "source_id": "source-1",
                    "supports_branch": ["yes"],
                    "supersession_status": "current_as_of",
                }
            ]
        )
    )

    result = runner.invoke(
        cli.app,
        [
            "forecast-dossier-compile",
            "--repo",
            str(repo_path),
            "--question-id",
            "q-leakage",
            "--evidence-json",
            str(evidence_path),
            "--output",
            str(tmp_path / "dossier.json"),
        ],
    )

    assert result.exit_code == 1
    assert "recorded after forecast_as_of" in result.stdout


def test_forecast_factory_seams_are_used(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path
) -> None:
    class StubRepository:
        def __init__(self) -> None:
            self.saved_questions = []

        def save_question(self, question):  # type: ignore[no-untyped-def]
            self.saved_questions.append(question)

    stub_repository = StubRepository()
    monkeypatch.setattr(cli, "_build_forecast_repository", lambda _path: stub_repository)

    result = runner.invoke(
        cli.app,
        [
            "forecast-question-create",
            "--repo",
            str(tmp_path),
            "--question-id",
            "q-seam",
            "--title",
            "Seam check",
            "--forecast-as-of",
            datetime(2026, 1, 15, tzinfo=UTC).isoformat(),
            "--horizon",
            "30d",
            "--resolution-criteria",
            "Resolved by records.",
            "--branch",
            "yes:Yes",
            "--branch",
            "no:No",
        ],
    )

    assert result.exit_code == 0
    assert stub_repository.saved_questions[0].id == "q-seam"


# Forecast lifecycle CLI tests merged from master.
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
        return [
            question
            for question in self.listed_questions
            if question.target_entity_id == target_entity_id
        ]

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


class _LoopBoundForecastRepository(_StubForecastRepository):
    def __init__(self) -> None:
        super().__init__()
        self.bound_loop = asyncio.get_running_loop()

    def _assert_same_loop(self) -> None:
        assert asyncio.get_running_loop() is self.bound_loop

    async def save_question(self, question: ForecastQuestion) -> ForecastQuestion:
        self._assert_same_loop()
        return await super().save_question(question)


class _LoopBoundForecastContext(_StubForecastContext):
    def __init__(self) -> None:
        super().__init__(_LoopBoundForecastRepository())
        self.bound_loop = asyncio.get_running_loop()

    async def close(self) -> None:
        assert asyncio.get_running_loop() is self.bound_loop
        await super().close()


def test_cli_create_forecast_question_command(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner
) -> None:
    repository = _StubForecastRepository()
    context = _StubForecastContext(repository)

    async def open_context(**_kwargs):  # type: ignore[no-untyped-def]
        return context

    monkeypatch.setattr(cli, "_open_forecast_context", open_context)

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

    async def open_context(**_kwargs):  # type: ignore[no-untyped-def]
        return context

    monkeypatch.setattr(cli, "_open_forecast_context", open_context)

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
    assert saved.metadata["target_entity_id"] == "deal-123"
    assert saved.metadata["extraction_variant"] == "default"
    assert abs(sum(saved.branch_probabilities.values()) - 1.0) < 1e-6
    assert context.closed is True


def test_cli_resolve_forecast_command(monkeypatch: pytest.MonkeyPatch, runner: CliRunner) -> None:
    repository = _StubForecastRepository()
    context = _StubForecastContext(repository)

    async def open_context(**_kwargs):  # type: ignore[no-untyped-def]
        return context

    monkeypatch.setattr(cli, "_open_forecast_context", open_context)

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

    async def open_context(**_kwargs):  # type: ignore[no-untyped-def]
        return context

    monkeypatch.setattr(cli, "_open_forecast_context", open_context)

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

    async def open_context(**_kwargs):  # type: ignore[no-untyped-def]
        return context

    monkeypatch.setattr(cli, "_open_forecast_context", open_context)

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


def test_cli_create_forecast_question_uses_single_event_loop(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner
) -> None:
    async def open_context(**_kwargs):  # type: ignore[no-untyped-def]
        return _LoopBoundForecastContext()

    monkeypatch.setattr(cli, "_open_forecast_context", open_context)

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
        ],
    )

    assert result.exit_code == 0
