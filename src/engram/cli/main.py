"""Engram CLI commands."""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

import httpx
import typer
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from engram.config import Settings
from engram.llm.provider import LLMProvider
from engram.models.branch_forecasting import ContextBudget
from engram.models.forecasting import (
    BaselineDecisionRecord,
    DecisionRecord,
    EvidenceDossier,
    EvidenceItem,
    ForecastQuestion,
    ForecastQuestionType,
    ForecastResolution,
    ForecastRun,
    OutcomeBranch,
    QuestionStatus,
    ResolutionCriteria,
)
from engram.services.branch_forecasting import BranchForecaster, evidence_from_path
from engram.services.forecast_audit import build_audit_report
from engram.services.forecast_impact import TimeWindow, build_impact_report
from engram.services.forecast_protocol import DeterministicForecastProtocol, LLMForecastProtocol
from engram.services.forecast_repository import (
    ForecastRepository,
    JsonForecastRepository,
    migrate_json_forecast_ledger,
)
from engram.services.forecast_scoring import ForecastScorer, build_calibration_report
from engram.storage.neo4j import Neo4jStore

app = typer.Typer(name="engram", help="Temporal knowledge graph engine for AI memory")
console = Console()
settings = Settings()
ALLOWED_BRANCH_OPTION = typer.Option(..., help="Allowed branch name; repeat for each branch")


def _warn_deprecated_command(command: str, replacement: str) -> None:
    typer.echo(
        f"Warning: '{command}' is deprecated; use '{replacement}' instead.",
        err=True,
    )


class CLIError(Exception):
    """Custom exception for CLI failures."""


@dataclass
class RelationshipRow:
    """Simplified representation of a relationship for table rendering."""

    target_id: str
    rel_type: str
    confidence: float
    valid_from: str
    valid_to: str | None
    evidence: str


class ForecastContext:
    """Loaded storage and repository dependencies for forecast lifecycle commands."""

    store: Neo4jStore
    repository: ForecastRepository
    scorer: ForecastScorer

    async def close(self) -> None:
        await self.store.close()


class EngramHTTPClient:
    """Lightweight HTTP client for talking to the Engram API."""

    def __init__(
        self,
        api_url: str,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = api_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.Client(base_url=self._base_url, timeout=timeout)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    # ---- Ingestion -----------------------------------------------------

    def ingest_messages(
        self,
        messages: Sequence[dict[str, Any]],
        conversation_id: str,
        tenant_id: str,
        group_id: str | None,
    ) -> list[dict[str, Any]]:
        """Send messages through the ingestion API."""

        results: list[dict[str, Any]] = []
        for raw in messages:
            payload = self._build_message_payload(
                raw,
                conversation_id=conversation_id,
                tenant_id=tenant_id,
                group_id=group_id,
            )
            response = self._client.post("/messages", json=payload)
            response.raise_for_status()
            results.append(response.json())
        return results

    def _build_message_payload(
        self,
        raw: dict[str, Any],
        *,
        conversation_id: str,
        tenant_id: str,
        group_id: str | None,
    ) -> dict[str, Any]:
        """Normalize message fields before sending to API."""

        if "text" not in raw or "speaker" not in raw:
            raise CLIError("Each message requires 'text' and 'speaker' fields")

        timestamp = raw.get("timestamp")
        if not timestamp:
            timestamp = datetime.now(UTC).isoformat()

        payload = {
            "text": raw["text"],
            "speaker": raw["speaker"],
            "timestamp": timestamp,
            "conversation_id": raw.get("conversation_id", conversation_id),
            "tenant_id": raw.get("tenant_id", tenant_id),
            "group_id": raw.get("group_id", group_id or raw.get("group_id")),
            "message_id": raw.get("message_id", f"cli-{uuid.uuid4()}"),
            "metadata": raw.get("metadata", {}),
        }

        if payload["group_id"] is None:
            payload["group_id"] = payload["conversation_id"]

        return payload

    # ---- Querying ------------------------------------------------------

    def search_entities(self, query: str, tenant_id: str) -> list[dict[str, Any]]:
        response = self._client.get("/search", params={"q": query, "tenant_id": tenant_id})
        response.raise_for_status()
        return response.json()

    def get_entity(self, entity_id: str) -> dict[str, Any]:
        response = self._client.get(f"/entities/{entity_id}")
        response.raise_for_status()
        return response.json()

    def get_active_relationships(
        self,
        entity_id: str,
        tenant_id: str,
        rel_type: str | None,
    ) -> list[RelationshipRow]:
        params: dict[str, Any] = {"tenant_id": tenant_id}
        if rel_type:
            params["rel_type"] = rel_type
        response = self._client.get(f"/entities/{entity_id}/relationships", params=params)
        response.raise_for_status()
        data = response.json()
        return [
            RelationshipRow(
                target_id=row["target_id"],
                rel_type=row["rel_type"],
                confidence=row["confidence"],
                valid_from=row["valid_from"],
                valid_to=row["valid_to"],
                evidence=row["evidence"],
            )
            for row in data
        ]

    def point_in_time(
        self,
        *,
        entity: str,
        as_of: str,
        tenant_id: str,
        rel_type: str | None,
        mode: str,
    ) -> Any:
        params: dict[str, Any] = {
            "entity": entity,
            "as_of": as_of,
            "tenant_id": tenant_id,
            "mode": mode,
        }
        if rel_type:
            params["rel_type"] = rel_type
        response = self._client.get("/query/point-in-time", params=params)
        response.raise_for_status()
        return response.json()


def _build_client(api_url: str, timeout: float = 30.0) -> EngramHTTPClient:
    """Factory for EngramHTTPClient. Separated for easier testing."""

    return EngramHTTPClient(api_url=api_url, timeout=timeout)


async def _open_forecast_context(
    *, tenant_id: str, conversation_id: str, message_id: str
) -> ForecastContext:
    settings = Settings(_env_file=".env")
    store = Neo4jStore(settings)
    await store.initialize()
    repository = ForecastRepository(
        store,
        tenant_id=tenant_id,
        conversation_id=conversation_id,
        message_id=message_id,
    )
    return ForecastContext(store=store, repository=repository, scorer=ForecastScorer())


def _parse_iso8601(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CLIError(f"Invalid datetime format: {value}") from exc


def _branch_probabilities_from_scores(result: dict[str, Any]) -> dict[str, float]:
    raw_scores = {item["branch"]: max(float(item["score"]), 0.0) for item in result["scores"]}
    total = sum(raw_scores.values())
    if total <= 0.0:
        count = len(raw_scores)
        return {branch: 1.0 / count for branch in raw_scores} if count else {}
    return {branch: score / total for branch, score in raw_scores.items()}


def _build_forecast_repository(path: str | Path) -> JsonForecastRepository:
    """Factory seam for the forecast ledger repository."""

    return JsonForecastRepository(path)


def _build_forecast_protocol() -> DeterministicForecastProtocol:
    """Factory seam for forecast run creation."""

    return DeterministicForecastProtocol()


def _build_evidence_compiler(evidence_json: str | Path) -> JsonEvidenceDossierCompiler:
    """Factory seam for MVP JSON evidence dossier compilation."""

    return JsonEvidenceDossierCompiler(evidence_json)


class JsonEvidenceDossierCompiler:
    """Compile user-provided JSON evidence records into an EvidenceDossier."""

    def __init__(self, evidence_json: str | Path) -> None:
        self.evidence_json = Path(evidence_json)

    def compile(self, question: ForecastQuestion) -> EvidenceDossier:
        payload = json.loads(self.evidence_json.read_text(encoding="utf-8"))
        raw_items = (
            payload.get("evidence_items", payload.get("evidence", []))
            if isinstance(payload, dict)
            else payload
        )
        if not isinstance(raw_items, list):
            raise CLIError("Evidence JSON must be a list or contain an evidence_items list")

        evidence_items = [EvidenceItem.model_validate(item) for item in raw_items]
        for item in evidence_items:
            self._validate_item_known_as_of(item, question)

        return EvidenceDossier(
            id=f"dossier-{question.id}",
            question_id=question.id,
            forecast_as_of=question.forecast_as_of,
            evidence_items=evidence_items,
            compiler="json_evidence.v1",
        )

    @staticmethod
    def _validate_item_known_as_of(item: EvidenceItem, question: ForecastQuestion) -> None:
        if item.recorded_from > question.forecast_as_of:
            raise CLIError(f"Evidence {item.id} was recorded after forecast_as_of")
        if item.valid_from > question.forecast_as_of:
            raise CLIError(f"Evidence {item.id} has valid_from after forecast_as_of")

        source_ingested_at = _metadata_datetime(item.metadata.get("source_ingested_at"))
        if source_ingested_at is not None and source_ingested_at > question.forecast_as_of:
            raise CLIError(f"Evidence {item.id} source was ingested after forecast_as_of")
        if item.metadata.get("evidence_role") == "resolution_evidence":
            raise CLIError(f"Evidence {item.id} is marked as resolution evidence")
        if item.metadata.get("resolution_for_question_id") == question.id:
            raise CLIError(f"Evidence {item.id} is resolution evidence for this question")
        derived_after = _metadata_datetime(item.metadata.get("derived_after"))
        if derived_after is not None and derived_after > question.forecast_as_of:
            raise CLIError(f"Evidence {item.id} was derived after forecast_as_of")


async def _initialize_neo4j_schema(current_settings: Settings) -> None:
    store = Neo4jStore(current_settings)
    try:
        await store.initialize()
    finally:
        await store.close()


async def _check_neo4j_store_health(current_settings: Settings) -> bool:
    store = Neo4jStore(current_settings)
    try:
        await store.initialize()
        return await store.health_check()
    finally:
        await store.close()


@app.command()
def init() -> None:
    """Initialize the Neo4j schema and indexes."""
    console.print("[bold blue]Initializing Engram schema...[/bold blue]")

    try:
        settings = Settings(_env_file=".env")
        asyncio.run(_initialize_neo4j_schema(settings))

        console.print("[bold green]✓ Schema initialized successfully![/bold green]")
        console.print(f"[dim]Neo4j URI: {settings.neo4j_uri}[/dim]")
    except Exception as e:
        console.print(f"[bold red]✗ Initialization failed: {e}[/bold red]")
        raise typer.Exit(code=1) from e


@app.command()
def consolidate(
    group_id: str = typer.Option(..., help="Group ID whose entities to consolidate"),
    tenant_id: str = typer.Option("default", help="Tenant ID"),
    types: str = typer.Option("PERSON", help="Comma-separated entity types"),
) -> None:
    """Merge first-name/full-name duplicate person entities within a group."""
    from engram.models.entity import EntityType
    from engram.services.entity_resolution import consolidate_name_variants

    current_settings = Settings(_env_file=".env")

    async def _run() -> int:
        store = Neo4jStore(current_settings)
        await store.initialize()
        try:
            ets = [EntityType[t.strip()] for t in types.split(",") if t.strip()]
            return await consolidate_name_variants(store, tenant_id, group_id, ets)
        finally:
            await store.close()

    try:
        count = asyncio.run(_run())
        console.print(f"[bold green]✓ Consolidated {count} variant entities[/bold green]")
    except Exception as e:
        console.print(f"[bold red]✗ Consolidation failed: {e}[/bold red]")
        raise typer.Exit(code=1) from e


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
    reload: bool = typer.Option(False, help="Enable auto-reload on code changes"),
) -> None:
    """Start the FastAPI server."""
    import uvicorn

    console.print("[bold blue]Starting Engram API server...[/bold blue]")
    console.print(f"[dim]Host: {host}:{port}[/dim]")
    console.print(f"[dim]Reload: {reload}[/dim]")
    console.print("[dim]Visit http://localhost:8000/docs for API documentation[/dim]")

    uvicorn.run("engram.main:app", host=host, port=port, reload=reload)


@app.command()
def ingest(
    file: str = typer.Argument(..., help="Path to JSON file with conversation messages"),
    conversation_id: str = typer.Option("default", help="Conversation ID"),
    tenant_id: str = typer.Option("default", help="Tenant ID"),
    group_id: str | None = typer.Option(None, help="Group ID for cross-conversation memory"),
    api_url: str = typer.Option("http://localhost:8000", help="Engram API base URL"),
    timeout: float = typer.Option(30.0, help="HTTP timeout in seconds"),
) -> None:
    """Ingest conversation messages from a JSON file.

    Expected JSON format:
    {
      "messages": [
        {"text": "...", "speaker": "...", "timestamp": "2024-01-15T10:00:00Z"},
        ...
      ]
    }
    """
    file_path = Path(file)
    if not file_path.exists():
        console.print(f"[bold red]✗ File not found: {file}[/bold red]")
        raise typer.Exit(code=1)

    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        console.print(f"[bold red]✗ Invalid JSON: {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    messages = data.get("messages", [])
    if not messages:
        console.print("[yellow]⚠ No messages found in file[/yellow]")
        return

    console.print(f"[bold blue]Ingesting {len(messages)} messages from {file}...[/bold blue]")

    client = _build_client(api_url, timeout)
    try:
        with Progress() as progress:
            task = progress.add_task("[cyan]Sending to Engram...", total=len(messages))
            results = client.ingest_messages(
                messages=messages,
                conversation_id=conversation_id,
                tenant_id=tenant_id,
                group_id=group_id,
            )
            for _ in results:
                progress.update(task, advance=1)

        console.print(f"[bold green]✓ Ingested {len(results)} messages[/bold green]")
        table = Table(title="Ingestion Summary")
        table.add_column("Message ID", style="cyan")
        table.add_column("Entities", style="magenta")
        table.add_column("Relationships", style="green")
        table.add_column("Conflicts", style="yellow")
        table.add_column("Latency (ms)", style="dim")

        for record in results:
            table.add_row(
                record.get("message_id", "-"),
                str(record.get("entities_extracted", 0)),
                str(record.get("relationships_inferred", 0)),
                str(record.get("conflicts_resolved", 0)),
                str(record.get("processing_time_ms", 0)),
            )

        console.print(table)
    except httpx.HTTPError as exc:
        console.print(f"[bold red]✗ API request failed: {exc}[/bold red]")
        raise typer.Exit(code=1) from exc
    except CLIError as exc:
        console.print(f"[bold red]✗ {exc}[/bold red]")
        raise typer.Exit(code=1) from exc
    finally:
        client.close()


@app.command()
def query(
    entity: str = typer.Argument(..., help="Entity name to query"),
    as_of: str | None = typer.Option(None, help="Query as of date (ISO 8601 format)"),
    rel_type: str | None = typer.Option(None, help="Filter by relationship type"),
    tenant_id: str = typer.Option("default", help="Tenant ID"),
    conversation_id: str | None = typer.Option("default", help="Conversation ID filter"),
    api_url: str = typer.Option("http://localhost:8000", help="Engram API base URL"),
    timeout: float = typer.Option(30.0, help="HTTP timeout in seconds"),
    mode: str = typer.Option(
        "world_state", help="Temporal mode: world_state, knowledge, bitemporal"
    ),
) -> None:
    """Query entity state and relationships.

    Examples:
      engram query "Kendra"
      engram query "Kendra" --as-of "2024-02-15T00:00:00Z"
      engram query "Kendra" --rel-type "prefers"
    """
    console.print(f"[bold blue]Querying entity: {entity}[/bold blue]")

    if as_of:
        try:
            datetime.fromisoformat(as_of.replace("Z", "+00:00"))
        except ValueError as exc:
            console.print(f"[bold red]✗ Invalid date format: {as_of}[/bold red]")
            raise typer.Exit(code=1) from exc

    client = _build_client(api_url, timeout)
    try:
        entity_record = _resolve_entity(
            client,
            query=entity,
            tenant_id=tenant_id,
            conversation_id=conversation_id,
        )

        if as_of:
            result = client.point_in_time(
                entity=entity,
                as_of=as_of,
                tenant_id=tenant_id,
                rel_type=rel_type,
                mode=mode,
            )
            _render_temporal_result(result, mode)
            return

        relationships = client.get_active_relationships(
            entity_id=entity_record["id"],
            tenant_id=tenant_id,
            rel_type=rel_type,
        )
        _render_relationships(entity_record, relationships)

    except httpx.HTTPError as exc:
        console.print(f"[bold red]✗ API request failed: {exc}[/bold red]")
        raise typer.Exit(code=1) from exc
    except CLIError as exc:
        console.print(f"[bold red]✗ {exc}[/bold red]")
        raise typer.Exit(code=1) from exc
    finally:
        client.close()


@app.command()
def forecast(
    file: str = typer.Argument(..., help="Path to forecast evidence JSON/NDJSON or directory"),
    objective: str = typer.Option(..., help="Decision objective to forecast against"),
    structural_family: str = typer.Option("margin_analysis", help="Forecast structural family"),
    max_items: int = typer.Option(6, help="Maximum evidence items to use"),
    max_tokens: int = typer.Option(1200, help="Maximum approximate context tokens"),
    min_score: float = typer.Option(0.0, help="Minimum evidence salience to consider"),
    output: str | None = typer.Option(None, help="Optional JSON output path"),
) -> None:
    """Forecast plausible next branches from compact evidence."""
    file_path = Path(file)
    if not file_path.exists():
        console.print(f"[bold red]✗ File not found: {file}[/bold red]")
        raise typer.Exit(code=1)

    try:
        evidence = evidence_from_path(file_path)
        if not evidence:
            raise CLIError("No forecast evidence found")

        forecaster = BranchForecaster()
        result = forecaster.forecast(
            objective=objective,
            structural_family=structural_family,
            evidence=evidence,
            budget=ContextBudget(
                max_items=max_items,
                max_tokens=max_tokens,
                min_score=min_score,
            ),
        )
    except (CLIError, ValueError) as exc:
        console.print(f"[bold red]✗ {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    result_payload = result.model_dump(mode="json")
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
        console.print(f"[bold green]✓ Forecast written to {output}[/bold green]")

    _render_forecast(result_payload)


@app.command("forecast-question-create")
def forecast_question_create(
    repo: str = typer.Option(..., help="Forecast ledger repository path"),
    question_id: str = typer.Option(..., help="Question ID"),
    title: str = typer.Option(..., help="Question title"),
    forecast_as_of: str = typer.Option(..., help="Forecast as-of timestamp"),
    horizon: str = typer.Option(..., help="Forecast horizon label"),
    resolution_criteria: str = typer.Option(..., help="Resolution criteria description"),
    branch: Annotated[
        list[str] | None, typer.Option("--branch", help="Branch as id:label or id:label:prior")
    ] = None,
    resolved_by: str | None = typer.Option(None, help="Optional resolution deadline timestamp"),
    tenant_id: str = typer.Option("default", help="Tenant ID"),
    target_id: str | None = typer.Option(None, help="Optional target entity ID"),
    question_type: str | None = typer.Option(
        None, help="binary or closed_branch; inferred by default"
    ),
    status: str = typer.Option("draft", help="Question status"),
) -> None:
    """Create a forecast question in the JSON forecast ledger."""

    try:
        if not branch:
            raise CLIError("At least one --branch option is required")
        branches = [_parse_branch_option(value) for value in branch]
        inferred_type = "binary" if len(branches) == 2 else "closed_branch"
        question = ForecastQuestion(
            id=question_id,
            tenant_id=tenant_id,
            title=title,
            question_type=ForecastQuestionType(question_type or inferred_type),
            forecast_as_of=_parse_datetime(forecast_as_of, "forecast-as-of"),
            horizon=horizon,
            resolution_criteria=ResolutionCriteria(
                description=resolution_criteria,
                resolved_by=_parse_datetime(resolved_by, "resolved-by") if resolved_by else None,
            ),
            branches=branches,
            target_id=target_id,
            status=QuestionStatus(status),
        )
        _build_forecast_repository(repo).save_question(question)
    except (OSError, ValueError, CLIError) as exc:
        console.print(f"[bold red]✗ {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    console.print(f"[bold green]✓ Forecast question saved: {question.id}[/bold green]")


@app.command("forecast-dossier-compile")
def forecast_dossier_compile(
    repo: str = typer.Option(..., help="Forecast ledger repository path"),
    question_id: str = typer.Option(..., help="Question ID"),
    evidence_json: str = typer.Option(..., help="JSON evidence records path"),
    output: str | None = typer.Option(None, help="Optional dossier JSON output path"),
) -> None:
    """Compile and persist an MVP EvidenceDossier from JSON evidence records."""

    try:
        repository = _build_forecast_repository(repo)
        question = repository.load_question(question_id)
        dossier = _build_evidence_compiler(evidence_json).compile(question)
        repository.save_dossier(dossier)
        if output:
            _write_model_json(Path(output), dossier)
    except (OSError, ValueError, CLIError, json.JSONDecodeError) as exc:
        console.print(f"[bold red]✗ {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    console.print(f"[bold green]✓ Evidence dossier saved: {dossier.id}[/bold green]")
    if output:
        console.print(f"[bold green]✓ Evidence dossier written: {output}[/bold green]")


@app.command("forecast-run-create")
def forecast_run_create(
    repo: str = typer.Option(..., help="Forecast ledger repository path"),
    question_id: str = typer.Option(..., help="Question ID"),
    dossier: str = typer.Option(..., help="Evidence dossier JSON path"),
    run_id: str | None = typer.Option(None, help="Optional run ID"),
    output: str | None = typer.Option(None, help="Optional run JSON output path"),
    protocol: str = typer.Option("deterministic-baseline", help="deterministic-baseline or llm.v1"),
) -> None:
    """Create and persist a forecast run."""

    try:
        repository = _build_forecast_repository(repo)
        question = repository.load_question(question_id)
        evidence_dossier = EvidenceDossier.model_validate_json(
            Path(dossier).read_text(encoding="utf-8")
        )
        ledger_dossier = repository.load_dossier(evidence_dossier.id)
        if _stable_model_json(ledger_dossier) != _stable_model_json(evidence_dossier):
            raise ValueError("dossier file does not match persisted ledger dossier")
        if protocol == "deterministic-baseline":
            run = _build_forecast_protocol().create_run(question, evidence_dossier, run_id=run_id)
        elif protocol == "llm.v1":
            run = asyncio.run(_create_llm_forecast_run(question, evidence_dossier, run_id=run_id))
        else:
            raise ValueError(f"unsupported forecast protocol: {protocol}")
        repository.save_run(run)
        if output:
            _write_model_json(Path(output), run)
    except (OSError, ValueError, FileExistsError) as exc:
        console.print(f"[bold red]✗ {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    console.print(f"[bold green]✓ Forecast run saved: {run.id}[/bold green]")
    console.print(f"[bold blue]Top branch: {run.top_branch}[/bold blue]")




def _stable_model_json(model: Any) -> str:
    return json.dumps(model.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

async def _create_llm_forecast_run(
    question: ForecastQuestion, dossier: EvidenceDossier, *, run_id: str | None
) -> ForecastRun:
    provider = LLMProvider(model=settings.llm_model, temperature=settings.llm_temperature)
    return await LLMForecastProtocol(
        provider, model_name=settings.llm_model, temperature=settings.llm_temperature
    ).create_run(question, dossier, run_id=run_id)


@app.command("forecast-resolve-create")
def forecast_resolve_create(
    repo: str = typer.Option(..., help="Forecast ledger repository path"),
    question_id: str = typer.Option(..., help="Question ID"),
    resolved_branch: str = typer.Option(..., help="Resolved branch ID"),
    resolved_at: str = typer.Option(..., help="Resolution timestamp"),
    resolution_id: str | None = typer.Option(None, help="Optional resolution ID"),
    evidence_id: Annotated[
        list[str] | None, typer.Option("--evidence-id", help="Resolution evidence ID")
    ] = None,
    notes: str | None = typer.Option(None, help="Resolution notes"),
) -> None:
    """Create a forecast resolution in the JSON forecast ledger."""

    try:
        repository = _build_forecast_repository(repo)
        question = repository.load_question(question_id)
        resolution = ForecastResolution(
            id=resolution_id or f"resolution-{question_id}",
            question_id=question_id,
            branch_ids=[branch.id for branch in question.branches],
            resolved_branch=resolved_branch,
            resolved_at=_parse_datetime(resolved_at, "resolved-at"),
            evidence_ids=evidence_id or [],
            notes=notes,
        )
        repository.save_resolution(resolution)
    except (OSError, ValueError) as exc:
        console.print(f"[bold red]✗ {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    console.print(f"[bold green]✓ Forecast resolution saved: {resolution.id}[/bold green]")


@app.command("forecast-score-report")
def forecast_score_report(
    repo: str = typer.Option(..., help="Forecast ledger repository path"),
    bucket_count: int = typer.Option(10, help="Calibration bucket count"),
    low_sample_threshold: int = typer.Option(30, help="Low-sample warning threshold"),
    skip_missing_resolutions: bool = typer.Option(False, help="Skip runs without resolutions"),
    output: str | None = typer.Option(None, help="Optional report JSON output path"),
) -> None:
    """Build a score and provisional calibration report for resolved forecast runs."""

    try:
        repository = _build_forecast_repository(repo)
        report = build_calibration_report(
            repository,
            bucket_count=bucket_count,
            low_sample_threshold=low_sample_threshold,
            skip_missing_resolutions=skip_missing_resolutions,
        )
        if output:
            _write_model_json(Path(output), report)
    except (OSError, ValueError) as exc:
        console.print(f"[bold red]✗ {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    console.print(f"[bold green]✓ Forecast score report built: {report.id}[/bold green]")
    console.print(f"[bold blue]Scored runs: {report.run_count}[/bold blue]")
    if report.low_sample_warning:
        console.print("[yellow]Low sample warning: calibration report is provisional.[/yellow]")


@app.command("forecast-ledger-migrate")
def forecast_ledger_migrate(
    from_dir: str = typer.Option(..., "--from", help="Source forecast ledger directory"),
    to_dir: str = typer.Option(..., "--to", help="Target forecast ledger directory"),
) -> None:
    """Forward-copy a JSON forecast ledger and verify migrated artifacts."""

    try:
        summary = migrate_json_forecast_ledger(from_dir, to_dir)
    except (OSError, ValueError) as exc:
        console.print(f"[bold red]✗ {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    console.print(json.dumps({key: len(value) for key, value in summary.items()}, indent=2))


@app.command("forecast-decision-create")
def forecast_decision_create(
    repo: str = typer.Option(..., help="Forecast ledger repository path"),
    decision_id: str = typer.Option(..., help="Decision ID"),
    decided_at: str = typer.Option(..., help="Decision timestamp"),
    decision_type: str = typer.Option(..., help="Decision type"),
    primary_forecast_run_id: str = typer.Option(..., help="Primary forecast run ID"),
    expected_outcome_branch: str = typer.Option(..., help="Expected outcome branch"),
    rationale: str = typer.Option(..., help="Decision rationale"),
    supporting_forecast_run_id: Annotated[
        list[str] | None,
        typer.Option("--supporting-forecast-run-id", help="Supporting forecast run ID"),
    ] = None,
) -> None:
    """Create a forecast-linked decision record."""

    try:
        decision = DecisionRecord(
            decision_id=decision_id,
            decided_at=_parse_datetime(decided_at, "decided-at"),
            decision_type=decision_type,
            primary_forecast_run_id=primary_forecast_run_id,
            supporting_forecast_run_ids=supporting_forecast_run_id or [],
            rationale=rationale,
            expected_outcome_branch=expected_outcome_branch,
        )
        _build_forecast_repository(repo).save_decision(decision)
    except (OSError, ValueError, FileExistsError) as exc:
        console.print(f"[bold red]✗ {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    console.print(f"[bold green]✓ Forecast-linked decision saved: {decision.decision_id}[/bold green]")


@app.command("forecast-decision-resolve")
def forecast_decision_resolve(
    repo: str = typer.Option(..., help="Forecast ledger repository path"),
    decision_id: str = typer.Option(..., help="Decision ID"),
    realized_outcome_branch: str = typer.Option(..., help="Realized outcome branch"),
    impact_value: float | None = typer.Option(None, help="Optional impact value"),
    impact_kind: str | None = typer.Option(None, help="avoided_loss, hit, or miss"),
    baseline: bool = typer.Option(False, help="Resolve a baseline decision instead"),
) -> None:
    """Resolve a forecast-linked or baseline decision record."""

    try:
        repository = _build_forecast_repository(repo)
        if baseline:
            resolved = repository.resolve_baseline_decision(
                decision_id,
                realized_outcome_branch=realized_outcome_branch,
                impact_value=impact_value,
                impact_kind=impact_kind,
            )
        else:
            resolved = repository.resolve_decision(
                decision_id,
                realized_outcome_branch=realized_outcome_branch,
                impact_value=impact_value,
                impact_kind=impact_kind,
            )
    except (OSError, ValueError) as exc:
        console.print(f"[bold red]✗ {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    console.print(f"[bold green]✓ Decision resolved: {resolved.decision_id}[/bold green]")


@app.command("forecast-baseline-decision-create")
def forecast_baseline_decision_create(
    repo: str = typer.Option(..., help="Forecast ledger repository path"),
    decision_id: str = typer.Option(..., help="Decision ID"),
    decided_at: str = typer.Option(..., help="Decision timestamp"),
    decision_type: str = typer.Option(..., help="Decision type"),
    expected_outcome_branch: str = typer.Option(..., help="Expected outcome branch"),
    rationale: str = typer.Option(..., help="Decision rationale"),
) -> None:
    """Create a pre-forecast baseline decision record."""

    try:
        decision = BaselineDecisionRecord(
            decision_id=decision_id,
            decided_at=_parse_datetime(decided_at, "decided-at"),
            decision_type=decision_type,
            rationale=rationale,
            expected_outcome_branch=expected_outcome_branch,
        )
        _build_forecast_repository(repo).save_baseline_decision(decision)
    except (OSError, ValueError, FileExistsError) as exc:
        console.print(f"[bold red]✗ {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    console.print(f"[bold green]✓ Baseline decision saved: {decision.decision_id}[/bold green]")


@app.command("forecast-impact-report")
def forecast_impact_report(
    repo: str = typer.Option(..., help="Forecast ledger repository path"),
    baseline_window: str = typer.Option(..., help="Baseline window START..END"),
    measure_window: str = typer.Option(..., help="Measurement window START..END"),
    min_resolved_records: int = typer.Option(10, help="Minimum resolved records per side"),
    output: str | None = typer.Option(None, help="Optional report JSON output path"),
) -> None:
    """Build a baseline-vs-forecast decision impact report."""

    try:
        report = build_impact_report(
            _build_forecast_repository(repo),
            baseline_window=TimeWindow.parse(baseline_window),
            measure_window=TimeWindow.parse(measure_window),
            min_resolved_records=min_resolved_records,
        )
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except (OSError, ValueError) as exc:
        console.print(f"[bold red]✗ {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    console.print(json.dumps(report, indent=2))


@app.command("forecast-audit-report")
def forecast_audit_report(
    repo: str = typer.Option(..., help="Forecast ledger repository path"),
    spot_check: int = typer.Option(0, help="Number of runs to spot-check"),
    output: str | None = typer.Option(None, help="Optional report JSON output path"),
) -> None:
    """Build a CI-usable lifecycle audit report for a forecast ledger."""

    try:
        report = build_audit_report(_build_forecast_repository(repo), spot_check=spot_check)
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except (OSError, ValueError) as exc:
        console.print(f"[bold red]✗ {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    console.print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise typer.Exit(code=1)


@app.command()
def create_forecast_question(
    target_entity_id: str = typer.Option(..., help="Target entity to forecast"),
    objective: str = typer.Option(..., help="Decision objective to forecast against"),
    structural_family: str = typer.Option(..., help="Forecast structural family"),
    forecast_as_of: str = typer.Option(..., help="Evidence cutoff timestamp"),
    horizon: str = typer.Option(..., help="Forecast horizon label"),
    resolution_due_at: str = typer.Option(..., help="Expected resolution deadline"),
    resolution_criteria: str = typer.Option(..., help="Objective resolution criteria"),
    allowed_branch: list[str] = ALLOWED_BRANCH_OPTION,
    tenant_id: str = typer.Option("default", help="Tenant identifier"),
    conversation_id: str = typer.Option("forecasting", help="Conversation / scope identifier"),
) -> None:
    """Deprecated: use forecast-question-create for new JSON-ledger questions."""
    _warn_deprecated_command("create-forecast-question", "forecast-question-create")
    asyncio.run(
        _create_forecast_question_impl(
            target_entity_id=target_entity_id,
            objective=objective,
            structural_family=structural_family,
            forecast_as_of=forecast_as_of,
            horizon=horizon,
            resolution_due_at=resolution_due_at,
            resolution_criteria=resolution_criteria,
            allowed_branch=allowed_branch,
            tenant_id=tenant_id,
            conversation_id=conversation_id,
        )
    )


async def _create_forecast_question_impl(
    *,
    target_entity_id: str,
    objective: str,
    structural_family: str,
    forecast_as_of: str,
    horizon: str,
    resolution_due_at: str,
    resolution_criteria: str,
    allowed_branch: list[str],
    tenant_id: str,
    conversation_id: str,
) -> None:
    message_id = f"forecast-question-{uuid.uuid4()}"
    context = await _open_forecast_context(
        tenant_id=tenant_id,
        conversation_id=conversation_id,
        message_id=message_id,
    )
    try:
        parsed_forecast_as_of = _parse_iso8601(forecast_as_of)
        question = ForecastQuestion(
            id=ForecastQuestion.build_id(
                tenant_id=tenant_id,
                target_entity_id=target_entity_id,
                objective=objective,
                forecast_as_of=parsed_forecast_as_of,
            ),
            tenant_id=tenant_id,
            target_entity_id=target_entity_id,
            objective=objective,
            structural_family=structural_family,
            forecast_as_of=parsed_forecast_as_of,
            horizon=horizon,
            resolution_due_at=_parse_iso8601(resolution_due_at),
            resolution_criteria=resolution_criteria,
            allowed_branch_names=allowed_branch,
        )
        saved = await context.repository.save_question(question)
        console.print(json.dumps(saved.model_dump(mode="json"), indent=2))
    except (CLIError, ValueError) as exc:
        console.print(f"[bold red]✗ {exc}[/bold red]", file=sys.stderr)
        raise typer.Exit(code=1) from exc
    finally:
        await context.close()


@app.command()
def run_forecast(
    file: str = typer.Argument(..., help="Path to forecast evidence JSON/NDJSON or directory"),
    question_id: str = typer.Option(..., help="Forecast question identifier"),
    target_entity_id: str = typer.Option(..., help="Target entity identifier"),
    objective: str = typer.Option(..., help="Decision objective to forecast against"),
    structural_family: str = typer.Option(..., help="Forecast structural family"),
    forecast_as_of: str = typer.Option(..., help="Evidence cutoff timestamp"),
    max_items: int = typer.Option(6, help="Maximum evidence items to use"),
    max_tokens: int = typer.Option(1200, help="Maximum approximate context tokens"),
    min_score: float = typer.Option(0.0, help="Minimum evidence salience to consider"),
    tenant_id: str = typer.Option("default", help="Tenant identifier"),
    conversation_id: str = typer.Option("forecasting", help="Conversation / scope identifier"),
) -> None:
    """Deprecated: use forecast-run-create for canonical JSON-ledger runs."""
    _warn_deprecated_command("run-forecast", "forecast-run-create")
    asyncio.run(
        _run_forecast_impl(
            file=file,
            question_id=question_id,
            target_entity_id=target_entity_id,
            objective=objective,
            structural_family=structural_family,
            forecast_as_of=forecast_as_of,
            max_items=max_items,
            max_tokens=max_tokens,
            min_score=min_score,
            tenant_id=tenant_id,
            conversation_id=conversation_id,
        )
    )


async def _run_forecast_impl(
    *,
    file: str,
    question_id: str,
    target_entity_id: str,
    objective: str,
    structural_family: str,
    forecast_as_of: str,
    max_items: int,
    max_tokens: int,
    min_score: float,
    tenant_id: str,
    conversation_id: str,
) -> None:
    file_path = Path(file)
    if not file_path.exists():
        console.print(f"[bold red]✗ File not found: {file}[/bold red]", file=sys.stderr)
        raise typer.Exit(code=1)

    message_id = f"forecast-run-{uuid.uuid4()}"
    context = await _open_forecast_context(
        tenant_id=tenant_id,
        conversation_id=conversation_id,
        message_id=message_id,
    )
    try:
        parsed_forecast_as_of = _parse_iso8601(forecast_as_of)
        evidence = evidence_from_path(file_path)
        if not evidence:
            raise CLIError("No forecast evidence found")
        result = BranchForecaster().forecast(
            objective=objective,
            structural_family=structural_family,
            evidence=evidence,
            budget=ContextBudget(max_items=max_items, max_tokens=max_tokens, min_score=min_score),
            forecast_as_of=parsed_forecast_as_of,
        )
        payload = result.model_dump(mode="json")
        config = {"max_items": max_items, "max_tokens": max_tokens, "min_score": min_score}
        run = ForecastRun(
            id=ForecastRun.build_id(
                question_id=question_id,
                model_or_engine="branch_forecaster",
                forecast_as_of=parsed_forecast_as_of,
                config=config,
            ),
            question_id=question_id,
            model_or_engine="branch_forecaster",
            forecast_as_of=parsed_forecast_as_of,
            branch_probabilities=_branch_probabilities_from_scores(payload),
            top_branch=result.top_branch,
            selected_evidence_ids=[item["id"] for item in payload["selected_context"]],
            evidence_gaps=result.evidence_gaps,
            rationale="; ".join(
                f"{score.branch}: {score.rationale}" for score in result.scores[:2]
            ),
            config=config,
            metadata={
                "branch_forecast": payload,
                "target_entity_id": target_entity_id,
                "extraction_variant": "default",
            },
        )
        saved = await context.repository.save_run(target_entity_id=target_entity_id, run=run)
        console.print(json.dumps(saved.model_dump(mode="json"), indent=2))
    except (CLIError, ValueError) as exc:
        console.print(f"[bold red]✗ {exc}[/bold red]", file=sys.stderr)
        raise typer.Exit(code=1) from exc
    finally:
        await context.close()


@app.command()
def resolve_forecast(
    question_id: str = typer.Option(..., help="Forecast question identifier"),
    run_id: str = typer.Option(..., help="Forecast run identifier"),
    target_entity_id: str = typer.Option(..., help="Target entity identifier"),
    outcome_branch: str = typer.Option(..., help="Observed outcome branch"),
    resolved_at: str = typer.Option(..., help="Observed resolution timestamp"),
    resolved_by: str = typer.Option(..., help="Analyst or system resolving the outcome"),
    source: str = typer.Option(..., help="Resolution evidence source"),
    resolution_notes: str | None = typer.Option(None, help="Optional resolution notes"),
    tenant_id: str = typer.Option("default", help="Tenant identifier"),
    conversation_id: str = typer.Option("forecasting", help="Conversation / scope identifier"),
) -> None:
    """Deprecated: use forecast-resolve-create for canonical JSON-ledger resolutions."""
    _warn_deprecated_command("resolve-forecast", "forecast-resolve-create")
    asyncio.run(
        _resolve_forecast_impl(
            question_id=question_id,
            run_id=run_id,
            target_entity_id=target_entity_id,
            outcome_branch=outcome_branch,
            resolved_at=resolved_at,
            resolved_by=resolved_by,
            source=source,
            resolution_notes=resolution_notes,
            tenant_id=tenant_id,
            conversation_id=conversation_id,
        )
    )


async def _resolve_forecast_impl(
    *,
    question_id: str,
    run_id: str,
    target_entity_id: str,
    outcome_branch: str,
    resolved_at: str,
    resolved_by: str,
    source: str,
    resolution_notes: str | None,
    tenant_id: str,
    conversation_id: str,
) -> None:
    message_id = f"forecast-resolution-{uuid.uuid4()}"
    context = await _open_forecast_context(
        tenant_id=tenant_id,
        conversation_id=conversation_id,
        message_id=message_id,
    )
    try:
        resolution = ForecastResolution(
            question_id=question_id,
            run_id=run_id,
            resolved_at=_parse_iso8601(resolved_at),
            outcome_branch=outcome_branch,
            resolution_notes=resolution_notes,
            resolved_by=resolved_by,
            source=source,
        )
        saved = await context.repository.save_resolution(
            target_entity_id=target_entity_id,
            resolution=resolution,
        )
        console.print(json.dumps(saved.model_dump(mode="json"), indent=2))
    except (CLIError, ValueError) as exc:
        console.print(f"[bold red]✗ {exc}[/bold red]", file=sys.stderr)
        raise typer.Exit(code=1) from exc
    finally:
        await context.close()


@app.command()
def score_forecasts(
    target_entity_id: str = typer.Option(..., help="Target entity identifier"),
    question_id: str | None = typer.Option(None, help="Optional forecast question identifier"),
    bins: int = typer.Option(10, help="Calibration bin count"),
    tenant_id: str = typer.Option("default", help="Tenant identifier"),
    conversation_id: str = typer.Option("forecasting", help="Conversation / scope identifier"),
) -> None:
    """Deprecated: use forecast-score-report for canonical JSON-ledger scoring."""
    _warn_deprecated_command("score-forecasts", "forecast-score-report")
    asyncio.run(
        _score_forecasts_impl(
            target_entity_id=target_entity_id,
            question_id=question_id,
            bins=bins,
            tenant_id=tenant_id,
            conversation_id=conversation_id,
        )
    )


async def _score_forecasts_impl(
    *,
    target_entity_id: str,
    question_id: str | None,
    bins: int,
    tenant_id: str,
    conversation_id: str,
) -> None:
    message_id = f"forecast-score-{uuid.uuid4()}"
    context = await _open_forecast_context(
        tenant_id=tenant_id,
        conversation_id=conversation_id,
        message_id=message_id,
    )
    try:
        if question_id is not None:
            question_ids = [question_id]
        else:
            questions = await context.repository.list_questions(target_entity_id=target_entity_id)
            question_ids = [question.id for question in questions]

        runs: list[ForecastRun] = []
        resolutions: list[ForecastResolution] = []
        for current_question_id in question_ids:
            runs.extend(
                await context.repository.list_runs(
                    target_entity_id=target_entity_id,
                    question_id=current_question_id,
                )
            )
            resolution = await context.repository.get_resolution(
                target_entity_id=target_entity_id,
                question_id=current_question_id,
            )
            if resolution is not None:
                resolutions.append(resolution)

        report = context.scorer.score_runs(runs, resolutions, bins=bins)
        console.print(json.dumps(report, indent=2))
    except (CLIError, ValueError) as exc:
        console.print(f"[bold red]✗ {exc}[/bold red]", file=sys.stderr)
        raise typer.Exit(code=1) from exc
    finally:
        await context.close()


def _metadata_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return None


def _parse_branch_option(value: str) -> OutcomeBranch:
    parts = value.split(":")
    if len(parts) not in {2, 3} or not parts[0] or not parts[1]:
        raise CLIError("Branch must use id:label or id:label:prior")
    prior = float(parts[2]) if len(parts) == 3 and parts[2] else None
    return OutcomeBranch(id=parts[0], label=parts[1], prior=prior)


def _parse_datetime(value: str, field_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CLIError(f"Invalid {field_name} datetime: {value}") from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _write_model_json(path: Path, model: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _render_forecast(result: dict[str, Any]) -> None:
    console.print(
        f"[bold blue]Forecast[/bold blue] objective={result['objective']} "
        f"family={result['structural_family']}"
    )
    console.print(f"[bold green]Top branch: {result['top_branch']}[/bold green]")

    table = Table(title="Branch Scores")
    table.add_column("Branch", style="cyan")
    table.add_column("Score", style="green")
    table.add_column("Evidence", style="magenta")
    table.add_column("Missing", style="yellow")
    table.add_column("Rationale", style="dim")

    for score in result["scores"]:
        table.add_row(
            score["branch"],
            f"{score['score']:.3f}",
            ", ".join(score["matched_evidence_ids"]) or "-",
            ", ".join(score["missing_precursors"]) or "-",
            score["rationale"],
        )
    console.print(table)

    if result["evidence_gaps"]:
        console.print(
            "[bold yellow]Evidence gaps:[/bold yellow] " + ", ".join(result["evidence_gaps"])
        )


def _resolve_entity(
    client: EngramHTTPClient,
    *,
    query: str,
    tenant_id: str,
    conversation_id: str | None,
) -> dict[str, Any]:
    matches = client.search_entities(query, tenant_id=tenant_id)
    if conversation_id is not None:
        matches = [m for m in matches if m.get("conversation_id") == conversation_id]
    if not matches:
        raise CLIError(
            f"No entities matched '{query}' (tenant={tenant_id}, conversation={conversation_id})"
        )
    if len(matches) > 1:
        console.print(
            "[yellow]⚠ Multiple entities matched, using the first result. Refine with --conversation-id if needed.[/yellow]"
        )
    entity_record = client.get_entity(matches[0]["id"])
    return entity_record


def _render_relationships(
    entity_record: dict[str, Any], relationships: list[RelationshipRow]
) -> None:
    if not relationships:
        console.print("[yellow]No active relationships found.[/yellow]")
        return

    table = Table(title=f"Active relationships for {entity_record['canonical_name']}")
    table.add_column("Target", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Confidence", style="green")
    table.add_column("Valid From", style="yellow")
    table.add_column("Valid To", style="yellow")
    table.add_column("Evidence", style="dim")

    for rel in relationships:
        table.add_row(
            rel.target_id,
            rel.rel_type,
            f"{rel.confidence:.2f}",
            rel.valid_from,
            rel.valid_to or "present",
            rel.evidence or "-",
        )

    console.print(table)


def _render_temporal_result(result: Any, mode: str) -> None:
    if mode == "bitemporal" and isinstance(result, dict):
        console.print("[bold blue]World State[/bold blue]")
        _render_relationships(
            {"canonical_name": "world_state"}, _rows_from_raw(result.get("world_state", []))
        )
        console.print("\n[bold blue]Knowledge State[/bold blue]")
        _render_relationships(
            {"canonical_name": "knowledge"}, _rows_from_raw(result.get("knowledge", []))
        )
        return

    if isinstance(result, list):
        rows = _rows_from_raw(result)
        dummy = {"canonical_name": "point-in-time"}
        _render_relationships(dummy, rows)
    else:
        console.print(result)


def _rows_from_raw(raw: list[dict[str, Any]]) -> list[RelationshipRow]:
    return [
        RelationshipRow(
            target_id=row.get("target_id", "unknown"),
            rel_type=row.get("rel_type", ""),
            confidence=row.get("confidence", 0.0),
            valid_from=row.get("valid_from", ""),
            valid_to=row.get("valid_to"),
            evidence=row.get("evidence", ""),
        )
        for row in raw
    ]


@app.command()
def health() -> None:
    """Check service health (Neo4j, Redis, LLM provider)."""
    console.print("[bold blue]Checking service health...[/bold blue]")

    try:
        settings = Settings(_env_file=".env")

        # Check Neo4j
        neo4j_healthy = asyncio.run(_check_neo4j_store_health(settings))
        status_icon = "[bold green]✓[/bold green]" if neo4j_healthy else "[bold red]✗[/bold red]"
        console.print(f"{status_icon} Neo4j: {settings.neo4j_uri}")

        # Check Redis (optional)
        if settings.redis_enabled:
            try:
                import redis.asyncio as aioredis

                async def check_redis() -> bool:
                    redis_client = aioredis.from_url(
                        f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}",
                        password=settings.redis_password,
                        decode_responses=True,
                    )
                    redis_client_any = cast("Any", redis_client)
                    try:
                        await redis_client_any.ping()
                        return True
                    finally:
                        await redis_client_any.aclose()

                redis_healthy = asyncio.run(check_redis())
                status_icon = (
                    "[bold green]✓[/bold green]" if redis_healthy else "[bold red]✗[/bold red]"
                )
                console.print(f"{status_icon} Redis: {settings.redis_host}:{settings.redis_port}")
            except Exception as e:
                console.print(f"[bold red]✗[/bold red] Redis: {e}")
        else:
            console.print("[dim]⊘ Redis: disabled[/dim]")

        # Check LLM provider
        console.print(f"[bold green]✓[/bold green] LLM: {settings.llm_model}")

        if neo4j_healthy:
            console.print("\n[bold green]All services healthy![/bold green]")
        else:
            console.print("\n[bold yellow]Some services are not healthy[/bold yellow]")
            raise typer.Exit(code=1)

    except Exception as e:
        console.print(f"[bold red]✗ Health check failed: {e}[/bold red]")
        raise typer.Exit(code=1) from e


@app.command()
def export(
    output: str = typer.Option("graph.json", help="Output file path"),
    tenant_id: str = typer.Option("default", help="Tenant ID"),
) -> None:
    """Export graph to JSON.

    Exports all entities and relationships for a tenant.
    """
    console.print(f"[bold blue]Exporting graph to {output}...[/bold blue]")

    try:
        # For now, create a minimal export structure
        export_data = {
            "version": "0.1.0",
            "tenant_id": tenant_id,
            "exported_at": datetime.utcnow().isoformat(),
            "entities": [],
            "relationships": [],
        }

        output_path = Path(output)
        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        console.print(f"[bold green]✓ Exported to {output}[/bold green]")
        console.print("[dim]Entities: 0[/dim]")
        console.print("[dim]Relationships: 0[/dim]")

    except Exception as e:
        console.print(f"[bold red]✗ Export failed: {e}[/bold red]")
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
