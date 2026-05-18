"""Tests for the Engram CLI commands."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import httpx
import pytest
from typer.testing import CliRunner

from engram.cli import main as cli

if TYPE_CHECKING:
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


class _StubForecastRepository:
    def __init__(self) -> None:
        self.saved_questions: list[ForecastQuestion] = []
        self.saved_runs: list[tuple[str, ForecastRun]] = []
        self.saved_resolutions: list[tuple[str, ForecastResolution]] = []

    async def save_question(self, question: ForecastQuestion) -> ForecastQuestion:
        self.saved_questions.append(question)
        return question

    async def save_run(self, *, target_entity_id: str, run: ForecastRun) -> ForecastRun:
        self.saved_runs.append((target_entity_id, run))
        return run

    async def save_resolution(
        self, *, target_entity_id: str, resolution: ForecastResolution
    ) -> ForecastResolution:
        self.saved_resolutions.append((target_entity_id, resolution))
        return resolution


class _StubForecastContext:
    def __init__(self, repository: _StubForecastRepository) -> None:
        self.repository = repository
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
