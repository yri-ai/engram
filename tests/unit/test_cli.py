"""Tests for the Engram CLI commands."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest
from typer.testing import CliRunner

from engram.cli import main as cli


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
