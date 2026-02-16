# Engram v0.1.0 MVP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a working end-to-end temporal knowledge graph engine that extracts entities/relationships from conversations, versions them bitemporally, and exposes query APIs -- running locally via Docker Compose in < 5 minutes.

**Architecture:** Python FastAPI monolith connecting to Neo4j Community Edition (graph + vector) and Redis (idempotency). LLM extraction via LiteLLM (multi-provider). 3-stage pipeline: entity extraction -> relationship inference -> conflict resolution with bitemporal versioning. TDD throughout.

**Tech Stack:** Python 3.11+ | uv (package manager) | FastAPI + Pydantic v2 | Neo4j 5.16+ Community | Redis 7.x | LiteLLM | ruff (lint/format) | pytest + pytest-asyncio | Docker Compose

**Reference Docs:**
- `ARCHITECTURE.md` -- full schema, Cypher queries, pipeline design, API spec
- `RESEARCH.md` -- competitive analysis, temporal patterns, database comparison

---

## Research Findings (Pre-Implementation Validation)

### Neo4j Community Edition Vector Index: CONFIRMED
- `CREATE VECTOR INDEX` is a core Cypher feature (documented in general Cypher manual, not Enterprise-only)
- Available since Neo4j 5.11+, stable in 5.16+
- Supports: 1536 dimensions, cosine similarity, `db.index.vector.queryNodes()` procedure
- **Decision: Proceed with Neo4j-only architecture (no pgvector needed)**

### Neo4j Python Driver: Full Async Support
- `AsyncGraphDatabase.driver()` for async connections
- `session.execute_read()` / `session.execute_write()` for managed transactions with auto-retry
- `session.begin_transaction()` for explicit transactions (needed for atomic conflict resolution)
- Connection pooling built-in (configurable max pool size)

### LiteLLM: JSON Mode + Fallback/Retry
- `response_format={"type": "json_object"}` for JSON mode across providers
- Router with fallbacks, retries, cooldowns between providers
- Works with OpenAI, Anthropic, Ollama

### Python Tooling: uv + ruff + pytest-asyncio
- `uv` is the modern standard (fast, lockfile, replaces poetry/pip)
- `ruff` for linting + formatting (replaces black + flake8 + isort)
- `pytest-asyncio` for async test support

---

## Project Structure

```
engram/
+-- pyproject.toml
+-- uv.lock
+-- .env.example
+-- .python-version              # "3.11"
+-- Dockerfile
+-- docker-compose.yml
+-- src/
|   +-- engram/
|       +-- __init__.py
|       +-- main.py              # FastAPI app + lifespan
|       +-- config.py            # Settings (pydantic-settings)
|       +-- models/
|       |   +-- __init__.py
|       |   +-- entity.py        # Entity, EntityType
|       |   +-- relationship.py  # Relationship, RelationshipType, ExclusivityPolicy
|       |   +-- message.py       # Message, IngestRequest, IngestResponse
|       |   +-- temporal.py      # TemporalQuery, DecayConfig
|       +-- storage/
|       |   +-- __init__.py
|       |   +-- base.py          # Abstract GraphStore interface
|       |   +-- neo4j.py         # Neo4j implementation
|       |   +-- memory.py        # In-memory implementation (testing)
|       +-- services/
|       |   +-- __init__.py
|       |   +-- extraction.py    # LLM extraction pipeline (Stage 1-3)
|       |   +-- temporal.py      # Decay, point-in-time, evolution queries
|       |   +-- resolution.py    # Conflict resolution + exclusivity
|       |   +-- dedup.py         # Redis idempotency service
|       +-- llm/
|       |   +-- __init__.py
|       |   +-- provider.py      # LiteLLM wrapper
|       |   +-- prompts.py       # Extraction prompt templates
|       +-- api/
|       |   +-- __init__.py
|       |   +-- routes.py        # API route definitions
|       |   +-- deps.py          # FastAPI dependencies (DI)
|       +-- cli/
|           +-- __init__.py
|           +-- main.py          # CLI commands (typer)
+-- tests/
|   +-- __init__.py
|   +-- conftest.py              # Shared fixtures
|   +-- unit/
|   |   +-- __init__.py
|   |   +-- test_models.py
|   |   +-- test_temporal.py
|   |   +-- test_resolution.py
|   |   +-- test_dedup.py
|   |   +-- test_extraction.py
|   +-- integration/
|   |   +-- __init__.py
|   |   +-- conftest.py          # Neo4j + Redis fixtures
|   |   +-- test_neo4j_store.py
|   |   +-- test_api.py
|   +-- e2e/
|       +-- __init__.py
|       +-- test_coaching_demo.py
+-- examples/
    +-- coaching-demo.json       # Kendra's shoe preference scenario
```

---

## Phase 1: Foundation (Tasks 1-6)

### Task 1: Project Initialization

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `src/engram/__init__.py`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "engram"
version = "0.1.0"
description = "Temporal knowledge graph engine for AI memory"
readme = "README.md"
license = "MIT"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.34.0",
    "pydantic>=2.10.0",
    "pydantic-settings>=2.7.0",
    "neo4j>=5.27.0",
    "redis>=5.2.0",
    "litellm>=1.55.0",
    "openai>=1.60.0",
    "numpy>=2.0.0",
    "typer>=0.15.0",
    "rich>=13.9.0",
    "httpx>=0.28.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.25.0",
    "pytest-cov>=6.0.0",
    "ruff>=0.9.0",
    "mypy>=1.14.0",
    "pre-commit>=4.0.0",
]

[project.scripts]
engram = "engram.cli.main:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/engram"]

[tool.ruff]
target-version = "py311"
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP", "B", "A", "SIM", "TCH"]
ignore = ["E501"]

[tool.ruff.lint.isort]
known-first-party = ["engram"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = "-v --tb=short"

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true
plugins = ["pydantic.mypy"]

[[tool.mypy.overrides]]
module = ["litellm.*", "neo4j.*"]
ignore_missing_imports = true
```

**Step 2: Create .python-version**

```
3.11
```

**Step 3: Create .env.example**

```bash
# Required
OPENAI_API_KEY=sk-your-key-here

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_ENABLED=true

# LLM (defaults to gpt-4o-mini)
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.1

# Optional: Anthropic for fallback
# ANTHROPIC_API_KEY=sk-ant-your-key

# Optional: Ollama for local
# OLLAMA_BASE_URL=http://localhost:11434

# Decay Rates (Open Question #5: Configurable decay rates)
# Preset: balanced (default) | fast (2x decay) | slow (0.5x decay)
# DECAY_PRESET=balanced
# Fine-grained per-type overrides (optional):
# DECAY_RATE_PREFERS=0.05
# DECAY_RATE_KNOWS=0.005
# DECAY_RATE_DISCUSSED=0.03
```

**Step 4: Create .gitignore**

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
.venv/
venv/
ENV/
env/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/
.hypothesis/

# Environment
.env
.env.local

# Neo4j
neo4j_data/

# Redis
redis_data/

# Logs
*.log

# OS
.DS_Store
Thumbs.db

# uv
uv.lock
```

**Step 5: Create package init**

```python
# src/engram/__init__.py
"""Engram: Temporal knowledge graph engine for AI memory."""

__version__ = "0.1.0"
```

**Step 6: Initialize project with uv**

Run: `uv init --no-readme && uv sync`
Expected: `.venv` created, dependencies installed, `uv.lock` generated

**Step 7: Verify linting works**

Run: `uv run ruff check src/`
Expected: No errors (empty project)

**Step 8: Verify tests run**

Create empty test: `tests/__init__.py` and `tests/unit/__init__.py`
Run: `uv run pytest`
Expected: "no tests ran" (0 collected)

**Step 9: Commit**

```bash
git init && git add -A && git commit -m "feat: initialize engram project with uv, FastAPI, Neo4j, Redis"
```

---

### Task 2: Configuration Module

**Files:**
- Create: `src/engram/config.py`
- Test: `tests/unit/test_config.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_config.py
from engram.config import Settings


def test_default_settings():
    """Settings should have sane defaults for local development."""
    settings = Settings(
        openai_api_key="sk-test",
        _env_file=None,  # Don't read .env in tests
    )
    assert settings.neo4j_uri == "bolt://localhost:7687"
    assert settings.neo4j_user == "neo4j"
    assert settings.redis_host == "localhost"
    assert settings.redis_port == 6379
    assert settings.redis_enabled is True
    assert settings.llm_model == "gpt-4o-mini"
    assert settings.llm_temperature == 0.1


def test_settings_from_env(monkeypatch):
    """Settings should be overridable from environment variables."""
    monkeypatch.setenv("NEO4J_URI", "bolt://custom:7687")
    monkeypatch.setenv("REDIS_ENABLED", "false")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    settings = Settings(_env_file=None)
    assert settings.neo4j_uri == "bolt://custom:7687"
    assert settings.redis_enabled is False
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'engram'`

**Step 3: Write minimal implementation**

```python
# src/engram/config.py
"""Application configuration via environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Engram configuration. All values can be overridden via env vars."""

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str | None = None
    redis_db: int = 0
    redis_enabled: bool = True

    # LLM
    openai_api_key: str = ""
    anthropic_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1

    # Embedding
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Application
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Decay Rates (Open Question #5: Configurable decay rates)
    decay_preset: str = "balanced"  # balanced | fast | slow
    decay_rate_prefers: float = 0.05     # 1.5-day half-life
    decay_rate_avoids: float = 0.04
    decay_rate_knows: float = 0.005      # 7-day half-life
    decay_rate_discussed: float = 0.03
    decay_rate_mentioned_with: float = 0.1
    decay_rate_has_goal: float = 0.02
    decay_rate_relates_to: float = 0.01
    decay_rate_default: float = 0.01
    
    def get_decay_rates(self) -> dict[str, float]:
        """Get decay rates, applying preset multipliers if configured."""
        base_rates = {
            "prefers": self.decay_rate_prefers,
            "avoids": self.decay_rate_avoids,
            "knows": self.decay_rate_knows,
            "discussed": self.decay_rate_discussed,
            "mentioned_with": self.decay_rate_mentioned_with,
            "has_goal": self.decay_rate_has_goal,
            "relates_to": self.decay_rate_relates_to,
            "default": self.decay_rate_default,
        }
        
        # Apply preset multipliers (inspired by CortexGraph)
        if self.decay_preset == "fast":
            return {k: v * 2.0 for k, v in base_rates.items()}
        elif self.decay_preset == "slow":
            return {k: v * 0.5 for k, v in base_rates.items()}
        else:  # balanced
            return base_rates

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_config.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add configuration module with pydantic-settings"
```

---

### Task 3: Core Pydantic Models

**Files:**
- Create: `src/engram/models/entity.py`
- Create: `src/engram/models/relationship.py`
- Create: `src/engram/models/message.py`
- Create: `src/engram/models/temporal.py`
- Create: `src/engram/models/__init__.py`
- Test: `tests/unit/test_models.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_models.py
from datetime import datetime, timezone

from engram.models.entity import Entity, EntityType
from engram.models.relationship import Relationship, RelationshipType
from engram.models.message import IngestRequest


def test_entity_creation():
    entity = Entity(
        id="t1:c1:PERSON:kendra",
        tenant_id="t1",
        conversation_id="c1",
        entity_type=EntityType.PERSON,
        canonical_name="kendra",
        aliases=["Kendra", "Kendra M"],
    )
    assert entity.id == "t1:c1:PERSON:kendra"
    assert entity.entity_type == EntityType.PERSON


def test_entity_id_generation():
    entity_id = Entity.build_id("t1", "c1", EntityType.PERSON, "kendra")
    assert entity_id == "t1:c1:PERSON:kendra"


def test_relationship_creation():
    now = datetime.now(timezone.utc)
    rel = Relationship(
        tenant_id="t1",
        conversation_id="c1",
        message_id="msg1",
        source_id="t1:c1:PERSON:kendra",
        target_id="t1:c1:PREFERENCE:nike",
        rel_type=RelationshipType.PREFERS,
        confidence=0.95,
        evidence="I love Nike shoes",
        valid_from=now,
        valid_to=None,
        recorded_from=now,
        recorded_to=None,
        version=1,
    )
    assert rel.is_active is True
    assert rel.is_currently_believed is True


def test_relationship_terminated():
    now = datetime.now(timezone.utc)
    rel = Relationship(
        tenant_id="t1",
        conversation_id="c1",
        message_id="msg1",
        source_id="t1:c1:PERSON:kendra",
        target_id="t1:c1:PREFERENCE:nike",
        rel_type=RelationshipType.PREFERS,
        confidence=0.95,
        evidence="I love Nike shoes",
        valid_from=now,
        valid_to=now,  # Terminated
        recorded_from=now,
        recorded_to=now,  # No longer believed
        version=1,
    )
    assert rel.is_active is False
    assert rel.is_currently_believed is False


def test_ingest_request_validation():
    req = IngestRequest(
        text="Kendra loves Nike running shoes",
        speaker="Kendra",
        timestamp=datetime.now(timezone.utc),
        conversation_id="c1",
    )
    assert req.text == "Kendra loves Nike running shoes"
    assert req.tenant_id == "default"  # Default tenant for MVP
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/engram/models/entity.py
"""Entity model for knowledge graph nodes."""

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field


class EntityType(StrEnum):
    PERSON = "PERSON"
    PREFERENCE = "PREFERENCE"
    GOAL = "GOAL"
    CONCEPT = "CONCEPT"
    EVENT = "EVENT"
    TOPIC = "TOPIC"


class Entity(BaseModel):
    """A node in the knowledge graph."""

    id: str
    tenant_id: str
    conversation_id: str
    group_id: str | None = None  # NEW: Enables cross-conversation entity linking (defaults to conversation_id)
    entity_type: EntityType
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    embedding: list[float] | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_mentioned: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_messages: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    @staticmethod
    def build_id(
        tenant_id: str, entity_type: EntityType, canonical_name: str, group_id: str | None = None
    ) -> str:
        """Build deterministic entity ID.
        
        Args:
            tenant_id: Tenant identifier
            entity_type: Type of entity
            canonical_name: Normalized canonical name
            group_id: Optional group identifier for cross-conversation linking.
                     If None, entities are conversation-scoped (isolated).
                     If provided, entities are shared across all conversations in that group.
        
        Returns:
            Deterministic entity ID: {tenant}:{group_id}:{type}:{canonical}
        """
        # Default to conversation-scoped if no group_id provided (backward compatible)
        scope = group_id if group_id else "conversation"
        return f"{tenant_id}:{scope}:{entity_type}:{canonical_name}"

    @staticmethod
    def normalize_name(text: str) -> str:
        """Deterministic name normalization. Same input -> same output."""
        import re

        normalized = re.sub(r"[^\w\s]", "", text.lower())
        return re.sub(r"\s+", "-", normalized.strip())
```

```python
# src/engram/models/relationship.py
"""Relationship model with bitemporal versioning."""

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field


class RelationshipType(StrEnum):
    PREFERS = "prefers"
    AVOIDS = "avoids"
    KNOWS = "knows"
    DISCUSSED = "discussed"
    MENTIONED_WITH = "mentioned_with"
    HAS_GOAL = "has_goal"
    RELATES_TO = "relates_to"


class Relationship(BaseModel):
    """A bitemporal edge in the knowledge graph."""

    # Scoping
    tenant_id: str
    conversation_id: str
    group_id: str | None = None  # NEW: Must match entity group_id for cross-conversation relationships
    message_id: str

    # Endpoints
    source_id: str
    target_id: str

    # Semantics
    rel_type: RelationshipType
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str = ""

    # Bitemporal (4 time columns)
    valid_from: datetime
    valid_to: datetime | None = None  # NULL = still true
    recorded_from: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    recorded_to: datetime | None = None  # NULL = still believed

    # Evolution
    version: int = 1
    supersedes: str | None = None

    metadata: dict = Field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """True if relationship is currently valid (truth timeline)."""
        return self.valid_to is None

    @property
    def is_currently_believed(self) -> bool:
        """True if we still believe this relationship (knowledge timeline)."""
        return self.recorded_to is None


class ExclusivityPolicy(BaseModel):
    """Defines which relationships are mutually exclusive."""

    exclusivity_scope: tuple[str, ...] | None = None
    max_active: int | None = None
    close_on_new: bool = False
    exclusive_with: list[str] = Field(default_factory=list)
```

```python
# src/engram/models/message.py
"""Message models for ingestion."""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """Request to ingest a conversation message."""

    text: str
    speaker: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    conversation_id: str = "default"
    tenant_id: str = "default"
    group_id: str | None = None  # NEW (Open Question #3): Optional group for cross-conversation entity linking
    message_id: str | None = None  # Auto-generated if not provided
    metadata: dict = Field(default_factory=dict)


class IngestResponse(BaseModel):
    """Response from message ingestion."""

    message_id: str
    entities_extracted: int
    relationships_inferred: int
    conflicts_resolved: int
    processing_time_ms: float
```

```python
# src/engram/models/temporal.py
"""Temporal query models."""

from datetime import datetime

from pydantic import BaseModel


class PointInTimeQuery(BaseModel):
    """Query for a specific point in time."""

    tenant_id: str = "default"
    entity_name: str
    as_of: datetime
    relationship_type: str | None = None
    mode: str = "world_state"  # "world_state" | "knowledge" | "bitemporal"
    knowledge_date: datetime | None = None  # Only for bitemporal mode


class EvolutionQuery(BaseModel):
    """Query for relationship evolution over time."""

    tenant_id: str = "default"
    entity_name: str
    target_name: str | None = None
    relationship_type: str | None = None
```

```python
# src/engram/models/__init__.py
"""Engram data models."""

from engram.models.entity import Entity, EntityType
from engram.models.message import IngestRequest, IngestResponse
from engram.models.relationship import ExclusivityPolicy, Relationship, RelationshipType
from engram.models.temporal import EvolutionQuery, PointInTimeQuery

__all__ = [
    "Entity",
    "EntityType",
    "ExclusivityPolicy",
    "EvolutionQuery",
    "IngestRequest",
    "IngestResponse",
    "PointInTimeQuery",
    "Relationship",
    "RelationshipType",
]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_models.py -v`
Expected: 5 passed

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add core Pydantic models (Entity, Relationship, Message)"
```

---

### Task 4: Docker Compose (Neo4j + Redis)

**Files:**
- Create: `docker-compose.yml`
- Create: `Dockerfile`

**Step 1: Create docker-compose.yml**

```yaml
services:
  neo4j:
    image: neo4j:5.26-community
    environment:
      NEO4J_AUTH: neo4j/password
      NEO4J_PLUGINS: '["apoc"]'
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data
    healthcheck:
      test: ["CMD", "cypher-shell", "-u", "neo4j", "-p", "password", "RETURN 1"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  api:
    build: .
    environment:
      NEO4J_URI: bolt://neo4j:7687
      NEO4J_USER: neo4j
      NEO4J_PASSWORD: password
      REDIS_HOST: redis
      REDIS_PORT: 6379
      REDIS_ENABLED: "true"
      OPENAI_API_KEY: ${OPENAI_API_KEY}
    ports:
      - "8000:8000"
    depends_on:
      neo4j:
        condition: service_healthy
      redis:
        condition: service_healthy

volumes:
  neo4j_data:
  redis_data:
```

**Step 2: Create Dockerfile**

```dockerfile
# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev --no-install-project

# Copy source
COPY src/ src/

# Install project
RUN uv sync --frozen --no-dev

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Set PATH to use venv
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "engram.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Step 3: Verify Docker Compose starts Neo4j + Redis (no API yet)**

Run: `docker-compose up neo4j redis -d && sleep 15 && docker-compose ps`
Expected: Both services healthy

**Step 4: Verify Neo4j is accessible**

Run: `docker-compose exec neo4j cypher-shell -u neo4j -p password "RETURN 1 AS test"`
Expected: Returns `1`

**Step 5: Verify Redis is accessible**

Run: `docker-compose exec redis redis-cli ping`
Expected: `PONG`

**Step 6: Tear down**

Run: `docker-compose down`

**Step 7: Commit**

```bash
git add -A && git commit -m "feat: add Docker Compose with Neo4j and Redis"
```

---

### Task 5: Deduplication Service (Redis Idempotency)

**Files:**
- Create: `src/engram/services/dedup.py`
- Test: `tests/unit/test_dedup.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_dedup.py
import pytest

from engram.services.dedup import InMemoryDedup


class TestInMemoryDedup:
    def test_first_message_is_new(self):
        dedup = InMemoryDedup()
        assert dedup.check_and_mark("msg-1") is True

    def test_duplicate_message_is_not_new(self):
        dedup = InMemoryDedup()
        dedup.check_and_mark("msg-1")
        assert dedup.check_and_mark("msg-1") is False

    def test_different_messages_are_independent(self):
        dedup = InMemoryDedup()
        assert dedup.check_and_mark("msg-1") is True
        assert dedup.check_and_mark("msg-2") is True
        assert dedup.check_and_mark("msg-1") is False

    def test_eviction_on_max_size(self):
        dedup = InMemoryDedup(max_size=2)
        dedup.check_and_mark("msg-1")
        dedup.check_and_mark("msg-2")
        dedup.check_and_mark("msg-3")  # Triggers eviction
        # After eviction, old messages may be treated as new
        assert dedup.check_and_mark("msg-3") is False  # Recent one still tracked
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_dedup.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# src/engram/services/dedup.py
"""Message deduplication service.

Production: Redis SET NX (atomic, with TTL)
Development: In-memory set (not production-safe)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class DedupService(ABC):
    """Abstract deduplication interface."""

    @abstractmethod
    async def check_and_mark(self, message_id: str) -> bool:
        """Check if message is new and mark as processed.

        Returns True if message is NEW (first time seen).
        Returns False if message is a DUPLICATE.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        ...


class InMemoryDedup(DedupService):
    """In-memory deduplication (for testing and dev without Redis).

    WARNING: Not production-safe. No persistence, no TTL, naive eviction.
    """

    def __init__(self, max_size: int = 10_000) -> None:
        self._processed: set[str] = set()
        self._max_size = max_size

    async def check_and_mark(self, message_id: str) -> bool:
        if message_id in self._processed:
            return False
        if len(self._processed) >= self._max_size:
            # Naive eviction: clear half
            to_keep = list(self._processed)[self._max_size // 2 :]
            self._processed = set(to_keep)
            logger.warning("InMemoryDedup evicted old entries (not production-safe)")
        self._processed.add(message_id)
        return True

    async def close(self) -> None:
        self._processed.clear()


class RedisDedup(DedupService):
    """Redis-based deduplication (production).

    Uses SET NX with TTL for atomic idempotency checks.
    """

    def __init__(self, redis_client: object, ttl_seconds: int = 86400) -> None:
        self._redis = redis_client
        self._ttl = ttl_seconds

    async def check_and_mark(self, message_id: str) -> bool:
        key = f"engram:processed:{message_id}"
        is_new = await self._redis.set(key, "1", nx=True, ex=self._ttl)
        return is_new is not None  # Redis returns None if key already exists

    async def close(self) -> None:
        pass  # Redis client lifecycle managed externally
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_dedup.py -v`
Expected: 4 passed

NOTE: Tests use `InMemoryDedup` directly (sync-compatible via `asyncio.run`). For the unit tests, make the sync wrapper:

Actually, update the tests to use `pytest-asyncio`:

```python
# tests/unit/test_dedup.py (updated)
import pytest

from engram.services.dedup import InMemoryDedup


class TestInMemoryDedup:
    @pytest.fixture
    def dedup(self):
        return InMemoryDedup()

    async def test_first_message_is_new(self, dedup):
        assert await dedup.check_and_mark("msg-1") is True

    async def test_duplicate_message_is_not_new(self, dedup):
        await dedup.check_and_mark("msg-1")
        assert await dedup.check_and_mark("msg-1") is False

    async def test_different_messages_are_independent(self, dedup):
        assert await dedup.check_and_mark("msg-1") is True
        assert await dedup.check_and_mark("msg-2") is True
        assert await dedup.check_and_mark("msg-1") is False

    async def test_eviction_on_max_size(self):
        dedup = InMemoryDedup(max_size=2)
        await dedup.check_and_mark("msg-1")
        await dedup.check_and_mark("msg-2")
        await dedup.check_and_mark("msg-3")
        assert await dedup.check_and_mark("msg-3") is False
```

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add deduplication service (Redis + in-memory)"
```

---

### Task 6: Storage Adapter Interface

**Files:**
- Create: `src/engram/storage/base.py`
- Create: `src/engram/storage/__init__.py`

**Step 1: Write the abstract interface**

```python
# src/engram/storage/base.py
"""Abstract storage interface for graph operations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from engram.models.entity import Entity, EntityType
from engram.models.relationship import Relationship, RelationshipType


class GraphStore(ABC):
    """Abstract graph storage interface.

    Implementations: Neo4jStore (production), MemoryStore (testing).
    """

    # --- Lifecycle ---

    @abstractmethod
    async def initialize(self) -> None:
        """Create schema, indexes, constraints."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close connections and clean up."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if storage is healthy."""
        ...

    # --- Entity Operations ---

    @abstractmethod
    async def upsert_entity(self, entity: Entity) -> Entity:
        """Create or update entity. Idempotent via MERGE on entity.id."""
        ...

    @abstractmethod
    async def get_entity(self, entity_id: str) -> Entity | None:
        """Get entity by ID."""
        ...

    @abstractmethod
    async def get_entity_by_name(
        self, tenant_id: str, conversation_id: str, canonical_name: str
    ) -> Entity | None:
        """Find entity by canonical name within a conversation."""
        ...

    @abstractmethod
    async def list_entities(
        self,
        tenant_id: str,
        conversation_id: str | None = None,
        entity_type: EntityType | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Entity]:
        """List entities with filters."""
        ...

    # --- Relationship Operations ---

    @abstractmethod
    async def create_relationship(self, rel: Relationship) -> Relationship:
        """Create a new relationship."""
        ...

    @abstractmethod
    async def get_active_relationships(
        self,
        entity_id: str,
        rel_type: RelationshipType | None = None,
        tenant_id: str | None = None,
    ) -> list[Relationship]:
        """Get currently active relationships (valid_to IS NULL AND recorded_to IS NULL)."""
        ...

    @abstractmethod
    async def terminate_relationship(
        self,
        source_id: str,
        rel_type: RelationshipType,
        tenant_id: str,
        conversation_id: str,
        termination_time: datetime,
        exclude_target_id: str | None = None,
    ) -> int:
        """Terminate active relationships. Returns count of terminated."""
        ...

    # --- Temporal Queries ---

    @abstractmethod
    async def query_world_state_as_of(
        self,
        tenant_id: str,
        entity_name: str,
        as_of: datetime,
        rel_type: RelationshipType | None = None,
    ) -> list[Relationship]:
        """What was ACTUALLY TRUE at a point in time (valid time)."""
        ...

    @abstractmethod
    async def query_knowledge_as_of(
        self,
        tenant_id: str,
        entity_name: str,
        as_of: datetime,
        rel_type: RelationshipType | None = None,
    ) -> list[Relationship]:
        """What did we BELIEVE at a point in time (record time)."""
        ...

    @abstractmethod
    async def query_evolution(
        self,
        tenant_id: str,
        entity_name: str,
        target_name: str | None = None,
        rel_type: RelationshipType | None = None,
    ) -> list[Relationship]:
        """Get all versions of relationships for timeline view."""
        ...

    @abstractmethod
    async def get_recent_entities(
        self,
        tenant_id: str,
        conversation_id: str,
        since: datetime,
        limit: int = 20,
    ) -> list[Entity]:
        """Get recently mentioned entities (for LLM context building)."""
        ...
```

```python
# src/engram/storage/__init__.py
"""Storage adapters."""

from engram.storage.base import GraphStore

__all__ = ["GraphStore"]
```

**Step 2: Verify no syntax errors**

Run: `uv run ruff check src/engram/storage/ && uv run mypy src/engram/storage/`
Expected: No errors

**Step 3: Commit**

```bash
git add -A && git commit -m "feat: add abstract GraphStore interface"
```

---

## Phase 2: Storage Layer (Tasks 7-10)

### Task 7: In-Memory Storage (for Testing)

**Files:**
- Create: `src/engram/storage/memory.py`
- Test: `tests/unit/test_memory_store.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_memory_store.py
import pytest
from datetime import datetime, timezone, timedelta

from engram.models.entity import Entity, EntityType
from engram.models.relationship import Relationship, RelationshipType
from engram.storage.memory import MemoryStore


@pytest.fixture
async def store():
    s = MemoryStore()
    await s.initialize()
    yield s
    await s.close()


def _make_entity(name: str = "kendra", entity_type: EntityType = EntityType.PERSON) -> Entity:
    canonical = Entity.normalize_name(name)
    return Entity(
        id=Entity.build_id("t1", "c1", entity_type, canonical),
        tenant_id="t1",
        conversation_id="c1",
        entity_type=entity_type,
        canonical_name=canonical,
    )


def _make_rel(
    source_name: str = "kendra",
    target_name: str = "nike",
    rel_type: RelationshipType = RelationshipType.PREFERS,
    valid_from: datetime | None = None,
) -> Relationship:
    now = valid_from or datetime.now(timezone.utc)
    return Relationship(
        tenant_id="t1",
        conversation_id="c1",
        message_id="msg-1",
        source_id=Entity.build_id("t1", "c1", EntityType.PERSON, source_name),
        target_id=Entity.build_id("t1", "c1", EntityType.PREFERENCE, target_name),
        rel_type=rel_type,
        confidence=0.95,
        evidence="test evidence",
        valid_from=now,
        recorded_from=now,
    )


class TestMemoryStoreEntities:
    async def test_upsert_creates_entity(self, store):
        entity = _make_entity("kendra")
        result = await store.upsert_entity(entity)
        assert result.id == entity.id

    async def test_upsert_is_idempotent(self, store):
        entity = _make_entity("kendra")
        await store.upsert_entity(entity)
        await store.upsert_entity(entity)
        entities = await store.list_entities("t1", "c1")
        assert len(entities) == 1

    async def test_get_entity_by_id(self, store):
        entity = _make_entity("kendra")
        await store.upsert_entity(entity)
        result = await store.get_entity(entity.id)
        assert result is not None
        assert result.canonical_name == "kendra"

    async def test_get_entity_not_found(self, store):
        result = await store.get_entity("nonexistent")
        assert result is None

    async def test_list_entities_filtered(self, store):
        await store.upsert_entity(_make_entity("kendra", EntityType.PERSON))
        await store.upsert_entity(_make_entity("nike", EntityType.PREFERENCE))
        persons = await store.list_entities("t1", "c1", entity_type=EntityType.PERSON)
        assert len(persons) == 1
        assert persons[0].canonical_name == "kendra"


class TestMemoryStoreRelationships:
    async def test_create_and_get_active(self, store):
        await store.upsert_entity(_make_entity("kendra"))
        await store.upsert_entity(_make_entity("nike", EntityType.PREFERENCE))
        rel = _make_rel()
        await store.create_relationship(rel)
        active = await store.get_active_relationships(rel.source_id)
        assert len(active) == 1
        assert active[0].rel_type == RelationshipType.PREFERS

    async def test_terminate_relationship(self, store):
        await store.upsert_entity(_make_entity("kendra"))
        await store.upsert_entity(_make_entity("nike", EntityType.PREFERENCE))
        rel = _make_rel()
        await store.create_relationship(rel)
        now = datetime.now(timezone.utc)
        count = await store.terminate_relationship(
            source_id=rel.source_id,
            rel_type=RelationshipType.PREFERS,
            tenant_id="t1",
            conversation_id="c1",
            termination_time=now,
        )
        assert count == 1
        active = await store.get_active_relationships(rel.source_id)
        assert len(active) == 0


class TestMemoryStoreTemporalQueries:
    async def test_world_state_as_of(self, store):
        """Nike was valid W1-W3, Adidas from W3 onward."""
        await store.upsert_entity(_make_entity("kendra"))
        await store.upsert_entity(_make_entity("nike", EntityType.PREFERENCE))
        await store.upsert_entity(_make_entity("adidas", EntityType.PREFERENCE))

        w1 = datetime(2024, 1, 15, tzinfo=timezone.utc)
        w3 = datetime(2024, 3, 20, tzinfo=timezone.utc)

        # Nike: valid W1-W3
        nike_rel = _make_rel("kendra", "nike", valid_from=w1)
        nike_rel.valid_to = w3
        nike_rel.recorded_from = w1
        nike_rel.recorded_to = w3
        await store.create_relationship(nike_rel)

        # Adidas: valid W3+
        adidas_rel = _make_rel("kendra", "adidas", valid_from=w3)
        adidas_rel.message_id = "msg-2"
        adidas_rel.recorded_from = w3
        adidas_rel.version = 2
        await store.create_relationship(adidas_rel)

        # Query Week 2: Nike was true
        w2 = datetime(2024, 2, 15, tzinfo=timezone.utc)
        results = await store.query_world_state_as_of("t1", "kendra", w2)
        assert len(results) == 1
        assert results[0].target_id.endswith("nike")

        # Query Week 4: Adidas is true
        w4 = datetime(2024, 4, 15, tzinfo=timezone.utc)
        results = await store.query_world_state_as_of("t1", "kendra", w4)
        assert len(results) == 1
        assert results[0].target_id.endswith("adidas")

    async def test_knowledge_as_of(self, store):
        """Test 'what did we KNOW at time X' (record time queries)."""
        await store.upsert_entity(_make_entity("kendra"))
        await store.upsert_entity(_make_entity("nike", EntityType.PREFERENCE))
        await store.upsert_entity(_make_entity("adidas", EntityType.PREFERENCE))

        w1 = datetime(2024, 1, 15, tzinfo=timezone.utc)
        w3 = datetime(2024, 3, 20, tzinfo=timezone.utc)

        # Nike: recorded W1-W3, but was actually valid from W0 (backdated)
        w0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        nike_rel = _make_rel("kendra", "nike", valid_from=w0)
        nike_rel.valid_to = w3
        nike_rel.recorded_from = w1  # We learned about it in W1
        nike_rel.recorded_to = w3    # Retracted in W3
        await store.create_relationship(nike_rel)

        # Adidas: recorded from W3 onward
        adidas_rel = _make_rel("kendra", "adidas", valid_from=w3)
        adidas_rel.message_id = "msg-2"
        adidas_rel.recorded_from = w3
        adidas_rel.version = 2
        await store.create_relationship(adidas_rel)

        # Query: What did we KNOW in Week 2?
        w2 = datetime(2024, 2, 15, tzinfo=timezone.utc)
        results = await store.query_knowledge_as_of("t1", "kendra", w2, rel_type="prefers")
        assert len(results) == 1
        assert results[0].target_id.endswith("nike")  # We believed Nike

        # Query: What did we KNOW in Week 4?
        w4 = datetime(2024, 4, 15, tzinfo=timezone.utc)
        results = await store.query_knowledge_as_of("t1", "kendra", w4, rel_type="prefers")
        assert len(results) == 1
        assert results[0].target_id.endswith("adidas")  # We believe Adidas
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_memory_store.py -v`
Expected: FAIL

**Step 3: Implement MemoryStore**

Build `src/engram/storage/memory.py` that implements the `GraphStore` interface using in-memory dicts. This is the TDD reference implementation -- every test here validates the contract that `Neo4jStore` must also satisfy.

> **Note to implementer:** The full MemoryStore implementation is ~150 lines. Implement each method to pass the corresponding test. Use `dict[str, Entity]` for entities, `list[Relationship]` for relationships. Filter methods use list comprehensions.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_memory_store.py -v`
Expected: All passed

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add in-memory GraphStore implementation with temporal queries"
```

---

### Task 8: Neo4j Storage Implementation

**Files:**
- Create: `src/engram/storage/neo4j.py`
- Test: `tests/integration/test_neo4j_store.py`
- Test: `tests/integration/conftest.py`

**Prerequisites:** Docker Compose with Neo4j running (`docker-compose up neo4j -d`)

**Step 1: Write integration test fixtures**

```python
# tests/integration/conftest.py
import pytest
from engram.config import Settings
from engram.storage.neo4j import Neo4jStore


@pytest.fixture(scope="session")
def neo4j_settings():
    """Integration tests use local Docker Neo4j."""
    return Settings(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        openai_api_key="sk-test",
        _env_file=None,
    )


@pytest.fixture
async def neo4j_store(neo4j_settings):
    store = Neo4jStore(neo4j_settings)
    await store.initialize()
    yield store
    # Clean up test data
    await store._execute_write("MATCH (n) DETACH DELETE n")
    await store.close()
```

**Step 2: Write integration tests**

```python
# tests/integration/test_neo4j_store.py
"""Integration tests for Neo4j storage.

These tests mirror the MemoryStore unit tests but run against real Neo4j.
Requires: docker-compose up neo4j -d
"""
import pytest
from datetime import datetime, timezone

from engram.models.entity import Entity, EntityType
from engram.models.relationship import Relationship, RelationshipType
from engram.storage.neo4j import Neo4jStore


# Re-use helpers from unit tests (or import from shared module)
def _make_entity(name="kendra", entity_type=EntityType.PERSON):
    canonical = Entity.normalize_name(name)
    return Entity(
        id=Entity.build_id("t1", "c1", entity_type, canonical),
        tenant_id="t1",
        conversation_id="c1",
        entity_type=entity_type,
        canonical_name=canonical,
    )


@pytest.mark.integration
class TestNeo4jStoreEntities:
    async def test_health_check(self, neo4j_store):
        assert await neo4j_store.health_check() is True

    async def test_upsert_and_get_entity(self, neo4j_store):
        entity = _make_entity("kendra")
        await neo4j_store.upsert_entity(entity)
        result = await neo4j_store.get_entity(entity.id)
        assert result is not None
        assert result.canonical_name == "kendra"

    async def test_upsert_is_idempotent(self, neo4j_store):
        entity = _make_entity("kendra")
        await neo4j_store.upsert_entity(entity)
        await neo4j_store.upsert_entity(entity)
        entities = await neo4j_store.list_entities("t1", "c1")
        assert len(entities) == 1


@pytest.mark.integration
class TestNeo4jStoreSchema:
    async def test_indexes_created(self, neo4j_store):
        """Verify that initialize() created the required indexes."""
        result = await neo4j_store._execute_read("SHOW INDEXES YIELD name RETURN name")
        index_names = [r["name"] for r in result]
        assert "entity_id" in index_names
```

**Step 3: Implement Neo4jStore**

```python
# src/engram/storage/neo4j.py (outline -- implementer fills in methods)
"""Neo4j storage implementation."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession

from engram.config import Settings
from engram.models.entity import Entity, EntityType
from engram.models.relationship import Relationship, RelationshipType
from engram.storage.base import GraphStore

logger = logging.getLogger(__name__)


class Neo4jStore(GraphStore):
    def __init__(self, settings: Settings) -> None:
        self._driver: AsyncDriver | None = None
        self._settings = settings

    async def initialize(self) -> None:
        self._driver = AsyncGraphDatabase.driver(
            self._settings.neo4j_uri,
            auth=(self._settings.neo4j_user, self._settings.neo4j_password),
        )
        await self._create_indexes()

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()

    async def health_check(self) -> bool:
        try:
            await self._execute_read("RETURN 1 AS ok")
            return True
        except Exception:
            return False

    async def _create_indexes(self) -> None:
        """Create all indexes from ARCHITECTURE.md section 1.3."""
        indexes = [
            "CREATE INDEX entity_id IF NOT EXISTS FOR (e:Entity) ON (e.id)",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX entity_tenant IF NOT EXISTS FOR (e:Entity) ON (e.tenant_id)",
            "CREATE INDEX entity_conversation IF NOT EXISTS FOR (e:Entity) ON (e.conversation_id)",
            "CREATE INDEX rel_tenant IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() ON (r.tenant_id)",
            "CREATE INDEX rel_valid_from IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() ON (r.valid_from)",
            "CREATE INDEX rel_valid_to IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() ON (r.valid_to)",
            "CREATE INDEX rel_recorded_from IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() ON (r.recorded_from)",
            "CREATE INDEX rel_recorded_to IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() ON (r.recorded_to)",
        ]
        for idx in indexes:
            await self._execute_write(idx)

    async def _execute_read(self, query: str, **params: Any) -> list[dict]:
        assert self._driver is not None
        async with self._driver.session() as session:
            result = await session.run(query, params)
            return [dict(record) async for record in result]

    async def _execute_write(self, query: str, **params: Any) -> list[dict]:
        assert self._driver is not None
        async with self._driver.session() as session:
            result = await session.run(query, params)
            return [dict(record) async for record in result]

    # --- Entity Operations ---

    async def upsert_entity(self, entity: Entity) -> Entity:
        """Create or update entity. Idempotent via MERGE on entity.id."""
        await self._execute_write("""
            MERGE (e:Entity {id: $id})
            ON CREATE SET
                e.tenant_id = $tenant_id,
                e.conversation_id = $conversation_id,
                e.type = $type,
                e.canonical_name = $canonical_name,
                e.aliases = $aliases,
                e.embedding = $embedding,
                e.created_at = $created_at,
                e.last_mentioned = $last_mentioned,
                e.source_messages = $source_messages,
                e.metadata = $metadata
            ON MATCH SET
                e.last_mentioned = $last_mentioned,
                e.aliases = e.aliases + [a IN $aliases WHERE NOT a IN e.aliases],
                e.source_messages = e.source_messages + $source_messages
            RETURN e
        """, 
            id=entity.id,
            tenant_id=entity.tenant_id,
            conversation_id=entity.conversation_id,
            type=entity.entity_type.value,
            canonical_name=entity.canonical_name,
            aliases=entity.aliases,
            embedding=entity.embedding,
            created_at=entity.created_at.isoformat(),
            last_mentioned=entity.last_mentioned.isoformat(),
            source_messages=entity.source_messages,
            metadata=entity.metadata,
        )
        return entity

    async def get_entity(self, entity_id: str) -> Entity | None:
        results = await self._execute_read("""
            MATCH (e:Entity {id: $id})
            RETURN e
        """, id=entity_id)
        if not results:
            return None
        node = results[0]["e"]
        return self._node_to_entity(node)

    async def list_entities(
        self,
        tenant_id: str,
        conversation_id: str | None = None,
        entity_type: EntityType | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Entity]:
        query = "MATCH (e:Entity) WHERE e.tenant_id = $tenant_id"
        params = {"tenant_id": tenant_id, "limit": limit, "offset": offset}
        
        if conversation_id:
            query += " AND e.conversation_id = $conversation_id"
            params["conversation_id"] = conversation_id
        if entity_type:
            query += " AND e.type = $type"
            params["type"] = entity_type.value
        
        query += " RETURN e ORDER BY e.created_at DESC SKIP $offset LIMIT $limit"
        results = await self._execute_read(query, **params)
        return [self._node_to_entity(r["e"]) for r in results]

    def _node_to_entity(self, node: dict) -> Entity:
        """Convert Neo4j node to Entity model."""
        from datetime import datetime
        return Entity(
            id=node["id"],
            tenant_id=node["tenant_id"],
            conversation_id=node.get("conversation_id", ""),
            entity_type=EntityType(node["type"]),
            canonical_name=node["canonical_name"],
            aliases=node.get("aliases", []),
            embedding=node.get("embedding"),
            created_at=datetime.fromisoformat(node["created_at"]),
            last_mentioned=datetime.fromisoformat(node["last_mentioned"]),
            source_messages=node.get("source_messages", []),
            metadata=node.get("metadata", {}),
        )

    # --- Relationship Operations ---

    async def create_relationship(self, rel: Relationship) -> Relationship:
        """Create relationship. See ARCHITECTURE.md section 2.4 for full Cypher."""
        await self._execute_write("""
            MERGE (a:Entity {id: $source_id})
            MERGE (b:Entity {id: $target_id})
            CREATE (a)-[r:RELATIONSHIP {
                tenant_id: $tenant_id,
                conversation_id: $conversation_id,
                message_id: $message_id,
                type: $type,
                valid_from: $valid_from,
                valid_to: $valid_to,
                recorded_from: $recorded_from,
                recorded_to: $recorded_to,
                confidence: $confidence,
                evidence: $evidence,
                version: $version,
                supersedes: $supersedes,
                metadata: $metadata
            }]->(b)
        """,
            source_id=rel.source_id,
            target_id=rel.target_id,
            tenant_id=rel.tenant_id,
            conversation_id=rel.conversation_id,
            message_id=rel.message_id,
            type=rel.rel_type.value,
            valid_from=rel.valid_from.isoformat(),
            valid_to=rel.valid_to.isoformat() if rel.valid_to else None,
            recorded_from=rel.recorded_from.isoformat(),
            recorded_to=rel.recorded_to.isoformat() if rel.recorded_to else None,
            confidence=rel.confidence,
            evidence=rel.evidence,
            version=rel.version,
            supersedes=rel.supersedes,
            metadata=rel.metadata,
        )
        return rel

    async def get_active_relationships(
        self,
        entity_id: str,
        rel_type: RelationshipType | None = None,
        tenant_id: str | None = None,
    ) -> list[Relationship]:
        """Get currently active relationships (both timelines NULL)."""
        query = """
            MATCH (e:Entity {id: $entity_id})-[r:RELATIONSHIP]->(target)
            WHERE r.valid_to IS NULL
              AND r.recorded_to IS NULL
        """
        params = {"entity_id": entity_id}
        
        if rel_type:
            query += " AND r.type = $rel_type"
            params["rel_type"] = rel_type.value
        if tenant_id:
            query += " AND r.tenant_id = $tenant_id"
            params["tenant_id"] = tenant_id
        
        query += " RETURN r, target ORDER BY r.recorded_from DESC"
        results = await self._execute_read(query, **params)
        return [self._edge_to_relationship(r["r"], entity_id, r["target"]["id"]) for r in results]

    async def terminate_relationship(
        self,
        source_id: str,
        rel_type: RelationshipType,
        tenant_id: str,
        conversation_id: str,
        termination_time: datetime,
        exclude_target_id: str | None = None,
    ) -> int:
        """Terminate active relationships. Returns count terminated."""
        query = """
            MATCH (source:Entity {id: $source_id})-[r:RELATIONSHIP]->(target)
            WHERE r.type = $type
              AND r.tenant_id = $tenant_id
              AND r.conversation_id = $conversation_id
              AND r.valid_to IS NULL
              AND r.recorded_to IS NULL
        """
        params = {
            "source_id": source_id,
            "type": rel_type.value,
            "tenant_id": tenant_id,
            "conversation_id": conversation_id,
            "termination_time": termination_time.isoformat(),
        }
        
        if exclude_target_id:
            query += " AND target.id <> $exclude_target_id"
            params["exclude_target_id"] = exclude_target_id
        
        query += """
            SET r.valid_to = $termination_time,
                r.recorded_to = $termination_time
            RETURN count(r) AS count
        """
        result = await self._execute_write(query, **params)
        return result[0]["count"] if result else 0

    # --- Temporal Queries ---
    # (query_world_state_as_of, query_knowledge_as_of, query_evolution)
    # See ARCHITECTURE.md section 3.2 for WHERE clauses with valid_from/to, recorded_from/to

    async def query_world_state_as_of(
        self,
        tenant_id: str,
        entity_name: str,
        as_of: datetime,
        rel_type: RelationshipType | None = None,
    ) -> list[Relationship]:
        """What was ACTUALLY TRUE at a point in time (valid time)."""
        query = """
            MATCH (e:Entity)-[r:RELATIONSHIP]->(target)
            WHERE e.tenant_id = $tenant_id
              AND e.canonical_name = $entity_name
              AND $as_of >= r.valid_from
              AND ($as_of < r.valid_to OR r.valid_to IS NULL)
        """
        params = {"tenant_id": tenant_id, "entity_name": entity_name, "as_of": as_of.isoformat()}
        
        if rel_type:
            query += " AND r.type = $rel_type"
            params["rel_type"] = rel_type.value
        
        query += " RETURN e, r, target ORDER BY r.confidence DESC"
        results = await self._execute_read(query, **params)
        return [self._edge_to_relationship(r["r"], r["e"]["id"], r["target"]["id"]) for r in results]

    async def query_knowledge_as_of(
        self,
        tenant_id: str,
        entity_name: str,
        as_of: datetime,
        rel_type: RelationshipType | None = None,
    ) -> list[Relationship]:
        """What did we BELIEVE at a point in time (record time)."""
        query = """
            MATCH (e:Entity)-[r:RELATIONSHIP]->(target)
            WHERE e.tenant_id = $tenant_id
              AND e.canonical_name = $entity_name
              AND $as_of >= r.recorded_from
              AND ($as_of < r.recorded_to OR r.recorded_to IS NULL)
        """
        params = {"tenant_id": tenant_id, "entity_name": entity_name, "as_of": as_of.isoformat()}
        
        if rel_type:
            query += " AND r.type = $rel_type"
            params["rel_type"] = rel_type.value
        
        query += " RETURN e, r, target ORDER BY r.confidence DESC"
        results = await self._execute_read(query, **params)
        return [self._edge_to_relationship(r["r"], r["e"]["id"], r["target"]["id"]) for r in results]

    def _edge_to_relationship(self, edge: dict, source_id: str, target_id: str) -> Relationship:
        """Convert Neo4j relationship to Relationship model."""
        from datetime import datetime
        return Relationship(
            tenant_id=edge["tenant_id"],
            conversation_id=edge["conversation_id"],
            message_id=edge["message_id"],
            source_id=source_id,
            target_id=target_id,
            rel_type=RelationshipType(edge["type"]),
            confidence=edge["confidence"],
            evidence=edge.get("evidence", ""),
            valid_from=datetime.fromisoformat(edge["valid_from"]),
            valid_to=datetime.fromisoformat(edge["valid_to"]) if edge.get("valid_to") else None,
            recorded_from=datetime.fromisoformat(edge["recorded_from"]),
            recorded_to=datetime.fromisoformat(edge["recorded_to"]) if edge.get("recorded_to") else None,
            version=edge.get("version", 1),
            supersedes=edge.get("supersedes"),
            metadata=edge.get("metadata", {}),
        )

    # --- Other methods from GraphStore interface ---
    # (get_entity_by_name, query_evolution, get_recent_entities)
    # Follow same pattern: query string + params, call _execute_read/write, parse results
```

> **Note to implementer:** The code above shows complete implementations for the core methods. The remaining methods (get_entity_by_name, query_evolution, get_recent_entities) follow the same pattern: build Cypher query string, pass parameters, call `_execute_read` or `_execute_write`, parse results with helper functions.

**Step 4: Run integration tests**

Run: `uv run pytest tests/integration/ -v -m integration`
Expected: All passed (with Neo4j running)

**Step 5: Add Redis integration test**

Create: `tests/integration/test_redis_dedup.py`

```python
# tests/integration/test_redis_dedup.py
"""Integration tests for Redis deduplication.

Requires: docker-compose up redis -d
"""
import pytest
import redis.asyncio as aioredis

from engram.services.dedup import RedisDedup


@pytest.fixture
async def redis_client():
    """Connect to local Redis for testing."""
    client = await aioredis.from_url("redis://localhost:6379", decode_responses=True)
    yield client
    # Clean up test keys
    await client.flushdb()
    await client.close()


@pytest.fixture
async def redis_dedup(redis_client):
    return RedisDedup(redis_client, ttl_seconds=10)


@pytest.mark.integration
class TestRedisDedup:
    async def test_first_message_is_new(self, redis_dedup):
        assert await redis_dedup.check_and_mark("test-msg-1") is True

    async def test_duplicate_message_is_not_new(self, redis_dedup):
        await redis_dedup.check_and_mark("test-msg-1")
        assert await redis_dedup.check_and_mark("test-msg-1") is False

    async def test_ttl_expires(self, redis_dedup, redis_client):
        """After TTL, message should be treated as new again."""
        short_ttl_dedup = RedisDedup(redis_client, ttl_seconds=1)
        await short_ttl_dedup.check_and_mark("test-msg-ttl")
        
        # Wait for TTL to expire
        import asyncio
        await asyncio.sleep(1.5)
        
        # Should be treated as new again
        assert await short_ttl_dedup.check_and_mark("test-msg-ttl") is True

    async def test_concurrent_marking_is_atomic(self, redis_dedup):
        """Multiple concurrent attempts should only succeed once."""
        import asyncio
        
        results = await asyncio.gather(
            redis_dedup.check_and_mark("concurrent-msg"),
            redis_dedup.check_and_mark("concurrent-msg"),
            redis_dedup.check_and_mark("concurrent-msg"),
        )
        
        # Exactly one should succeed
        assert results.count(True) == 1
        assert results.count(False) == 2
```

Run: `uv run pytest tests/integration/test_redis_dedup.py -v -m integration`
Expected: All passed (with Redis running)

**Step 6: Update dedup.py to fix async Redis**

The `RedisDedup` class needs async Redis client. Update implementation:

```python
# src/engram/services/dedup.py (update RedisDedup class)
class RedisDedup(DedupService):
    """Redis-based deduplication (production).

    Uses SET NX with TTL for atomic idempotency checks.
    """

    def __init__(self, redis_client: object, ttl_seconds: int = 86400) -> None:
        self._redis = redis_client  # Expects redis.asyncio.Redis
        self._ttl = ttl_seconds

    async def check_and_mark(self, message_id: str) -> bool:
        key = f"engram:processed:{message_id}"
        # SET NX EX is atomic in Redis
        result = await self._redis.set(key, "1", nx=True, ex=self._ttl)
        return result is not None  # Redis returns None if key already exists

    async def close(self) -> None:
        pass  # Redis client lifecycle managed externally
```

**Step 7: Commit**

```bash
git add -A && git commit -m "feat: add Neo4j GraphStore implementation with schema and indexes"
git add tests/integration/test_redis_dedup.py && git commit -m "test: add Redis dedup integration tests"
```

---

### Task 9: Temporal Reasoning Service

**Files:**
- Create: `src/engram/services/temporal.py`
- Test: `tests/unit/test_temporal.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_temporal.py
import math
from datetime import datetime, timezone, timedelta

from engram.services.temporal import calculate_decayed_confidence, DECAY_RATES


def test_no_decay_at_time_zero():
    confidence = calculate_decayed_confidence(
        base_confidence=1.0,
        rel_type="prefers",
        last_mentioned=datetime(2024, 1, 1, tzinfo=timezone.utc),
        current_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    assert confidence == 1.0


def test_preference_decays_fast():
    """Preferences decay at 0.05/day. After 30 days: ~0.22"""
    confidence = calculate_decayed_confidence(
        base_confidence=1.0,
        rel_type="prefers",
        last_mentioned=datetime(2024, 1, 1, tzinfo=timezone.utc),
        current_time=datetime(2024, 1, 31, tzinfo=timezone.utc),
    )
    expected = math.exp(-0.05 * 30)
    assert abs(confidence - expected) < 0.01


def test_social_relationship_decays_slow():
    """Knows relationships decay at 0.005/day. After 30 days: ~0.86"""
    confidence = calculate_decayed_confidence(
        base_confidence=1.0,
        rel_type="knows",
        last_mentioned=datetime(2024, 1, 1, tzinfo=timezone.utc),
        current_time=datetime(2024, 1, 31, tzinfo=timezone.utc),
    )
    expected = math.exp(-0.005 * 30)
    assert abs(confidence - expected) < 0.01


def test_confidence_floors_at_minimum():
    """Confidence never drops below 0.1."""
    confidence = calculate_decayed_confidence(
        base_confidence=1.0,
        rel_type="prefers",
        last_mentioned=datetime(2024, 1, 1, tzinfo=timezone.utc),
        current_time=datetime(2024, 7, 1, tzinfo=timezone.utc),  # 6 months
    )
    assert confidence == 0.1


def test_reinforcement_boosts_confidence():
    from engram.services.temporal import calculate_reinforced_confidence

    result = calculate_reinforced_confidence(
        current_decayed=0.5,
        new_mention_confidence=0.9,
    )
    # 70% new + 30% old = 0.63 + 0.15 = 0.78
    assert abs(result - 0.78) < 0.01
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_temporal.py -v`
Expected: FAIL

**Step 3: Implement temporal service**

```python
# src/engram/services/temporal.py
"""Temporal reasoning: decay, reinforcement, point-in-time queries."""

from __future__ import annotations

import math
from datetime import datetime

from engram.config import Settings

CONFIDENCE_FLOOR = 0.1


def calculate_decayed_confidence(
    base_confidence: float,
    rel_type: str,
    last_mentioned: datetime,
    current_time: datetime,
    settings: Settings | None = None,
) -> float:
    """Exponential decay: confidence(t) = base * exp(-rate * days).
    
    Args:
        base_confidence: Initial confidence (0.0-1.0)
        rel_type: Relationship type (e.g., "prefers", "knows")
        last_mentioned: When relationship was last mentioned
        current_time: Current time for decay calculation
        settings: Optional Settings instance for configurable decay rates.
                 If None, uses default rates.
    
    Returns:
        Decayed confidence, floored at CONFIDENCE_FLOOR (0.1)
    
    Note: Open Question #5 (Decay Rate Tuning) — Now configurable via Settings.
    """
    days_elapsed = (current_time - last_mentioned).total_seconds() / 86400
    if days_elapsed <= 0:
        return base_confidence
    
    # Get decay rates from settings (configurable) or use defaults
    if settings:
        decay_rates = settings.get_decay_rates()
    else:
        # Fallback to hardcoded defaults if no settings provided
        decay_rates = {
            "prefers": 0.05, "avoids": 0.04, "knows": 0.005,
            "discussed": 0.03, "mentioned_with": 0.1, "has_goal": 0.02,
            "relates_to": 0.01, "default": 0.01,
        }
    
    decay_rate = decay_rates.get(rel_type, decay_rates["default"])
    decayed = base_confidence * math.exp(-decay_rate * days_elapsed)
    return max(decayed, CONFIDENCE_FLOOR)


def calculate_reinforced_confidence(
    current_decayed: float,
    new_mention_confidence: float,
) -> float:
    """Boost confidence on re-mention. 70% new, 30% old."""
    return min((new_mention_confidence * 0.7) + (current_decayed * 0.3), 1.0)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_temporal.py -v`
Expected: 5 passed

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add temporal reasoning (decay, reinforcement)"
```

---

### Task 10: Conflict Resolution Service

**Files:**
- Create: `src/engram/services/resolution.py`
- Test: `tests/unit/test_resolution.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_resolution.py
import pytest
from datetime import datetime, timezone

from engram.models.entity import Entity, EntityType
from engram.models.relationship import Relationship, RelationshipType
from engram.services.resolution import (
    ConflictResolver,
    EXCLUSIVITY_POLICIES,
)
from engram.storage.memory import MemoryStore


@pytest.fixture
async def resolver():
    store = MemoryStore()
    await store.initialize()
    return ConflictResolver(store)


@pytest.fixture
async def store_with_entities():
    store = MemoryStore()
    await store.initialize()
    for name, etype in [("kendra", EntityType.PERSON), ("nike", EntityType.PREFERENCE), ("adidas", EntityType.PREFERENCE)]:
        canonical = Entity.normalize_name(name)
        await store.upsert_entity(Entity(
            id=Entity.build_id("t1", "c1", etype, canonical),
            tenant_id="t1", conversation_id="c1",
            entity_type=etype, canonical_name=canonical,
        ))
    return store


async def test_prefers_enforces_single_active(store_with_entities):
    """When new "prefers" created, old one should be terminated."""
    store = store_with_entities
    resolver = ConflictResolver(store)
    now = datetime.now(timezone.utc)

    # Create initial preference: Kendra -> Nike
    nike_rel = Relationship(
        tenant_id="t1", conversation_id="c1", message_id="msg-1",
        source_id="t1:c1:PERSON:kendra", target_id="t1:c1:PREFERENCE:nike",
        rel_type=RelationshipType.PREFERS, confidence=0.95,
        evidence="I love Nike", valid_from=now, recorded_from=now,
    )
    await resolver.resolve_and_create(nike_rel)

    # Create conflicting preference: Kendra -> Adidas
    adidas_rel = Relationship(
        tenant_id="t1", conversation_id="c1", message_id="msg-2",
        source_id="t1:c1:PERSON:kendra", target_id="t1:c1:PREFERENCE:adidas",
        rel_type=RelationshipType.PREFERS, confidence=0.9,
        evidence="Switched to Adidas", valid_from=now, recorded_from=now,
    )
    await resolver.resolve_and_create(adidas_rel)

    # Verify: only Adidas is active
    active = await store.get_active_relationships("t1:c1:PERSON:kendra")
    assert len(active) == 1
    assert active[0].target_id.endswith("adidas")


async def test_knows_allows_multiple_active(store_with_entities):
    """'knows' relationships should allow multiple active."""
    store = store_with_entities
    # Add second person
    await store.upsert_entity(Entity(
        id="t1:c1:PERSON:sarah", tenant_id="t1", conversation_id="c1",
        entity_type=EntityType.PERSON, canonical_name="sarah",
    ))
    resolver = ConflictResolver(store)
    now = datetime.now(timezone.utc)

    for target, msg_id in [("nike", "msg-1"), ("sarah", "msg-2")]:
        await resolver.resolve_and_create(Relationship(
            tenant_id="t1", conversation_id="c1", message_id=msg_id,
            source_id="t1:c1:PERSON:kendra",
            target_id=f"t1:c1:PERSON:{target}" if target == "sarah" else f"t1:c1:PREFERENCE:{target}",
            rel_type=RelationshipType.KNOWS, confidence=0.9,
            evidence="test", valid_from=now, recorded_from=now,
        ))

    active = await store.get_active_relationships("t1:c1:PERSON:kendra", rel_type=RelationshipType.KNOWS)
    assert len(active) == 2
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_resolution.py -v`
Expected: FAIL

**Step 3: Implement conflict resolution**

```python
# src/engram/services/resolution.py
"""Conflict resolution with exclusivity enforcement.

See ARCHITECTURE.md section 2.4 for the full algorithm.
"""

from __future__ import annotations

import logging

from engram.models.relationship import ExclusivityPolicy, Relationship, RelationshipType
from engram.storage.base import GraphStore

logger = logging.getLogger(__name__)

EXCLUSIVITY_POLICIES: dict[str, ExclusivityPolicy] = {
    "prefers": ExclusivityPolicy(
        exclusivity_scope=("source",), max_active=1, close_on_new=True,
    ),
    "avoids": ExclusivityPolicy(
        exclusivity_scope=("source",), max_active=1, close_on_new=True,
        exclusive_with=["prefers"],
    ),
    "works_for": ExclusivityPolicy(
        exclusivity_scope=("source",), max_active=1, close_on_new=True,
    ),
    "has_goal": ExclusivityPolicy(close_on_new=False),
    "knows": ExclusivityPolicy(close_on_new=False),
    "discussed": ExclusivityPolicy(close_on_new=False),
    "mentioned_with": ExclusivityPolicy(close_on_new=False),
    "relates_to": ExclusivityPolicy(close_on_new=False),
}


class ConflictResolver:
    def __init__(self, store: GraphStore) -> None:
        self._store = store

    async def resolve_and_create(self, new_rel: Relationship) -> Relationship:
        """Apply exclusivity policies, terminate conflicts, create new relationship."""
        policy = EXCLUSIVITY_POLICIES.get(new_rel.rel_type, ExclusivityPolicy())

        if policy.close_on_new:
            terminated = await self._store.terminate_relationship(
                source_id=new_rel.source_id,
                rel_type=new_rel.rel_type,
                tenant_id=new_rel.tenant_id,
                conversation_id=new_rel.conversation_id,
                termination_time=new_rel.valid_from,
                exclude_target_id=new_rel.target_id,  # Don't close if same target (reinforcement)
            )
            if terminated > 0:
                new_rel.version = terminated + 1
                logger.info(f"Terminated {terminated} conflicting {new_rel.rel_type} relationships")

        return await self._store.create_relationship(new_rel)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_resolution.py -v`
Expected: All passed

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add conflict resolution with exclusivity enforcement"
```

---

## Phase 3: Extraction Pipeline (Tasks 11-14)

### Task 11: LLM Provider Wrapper

**Files:**
- Create: `src/engram/llm/provider.py`
- Create: `src/engram/llm/__init__.py`
- Test: `tests/unit/test_llm_provider.py`

**Step 1: Write the failing test (mock-based)**

```python
# tests/unit/test_llm_provider.py
from unittest.mock import AsyncMock, patch
import pytest

from engram.llm.provider import LLMProvider


async def test_complete_returns_parsed_json():
    provider = LLMProvider(model="gpt-4o-mini", temperature=0.1)
    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock(message=AsyncMock(content='{"entities": []}'))]

    with patch("engram.llm.provider.acompletion", return_value=mock_response):
        result = await provider.complete_json("Extract entities", system="You are a helper")
        assert result == {"entities": []}


async def test_complete_handles_invalid_json():
    provider = LLMProvider(model="gpt-4o-mini", temperature=0.1)
    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock(message=AsyncMock(content="not json"))]

    with patch("engram.llm.provider.acompletion", return_value=mock_response):
        with pytest.raises(ValueError, match="Failed to parse"):
            await provider.complete_json("Extract entities")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_llm_provider.py -v`
Expected: FAIL

**Step 3: Implement LLM provider**

```python
# src/engram/llm/provider.py
"""LLM provider wrapper using LiteLLM."""

from __future__ import annotations

import json
import logging
from typing import Any

from litellm import acompletion

logger = logging.getLogger(__name__)


class LLMProvider:
    """Async LLM wrapper with JSON mode and error handling."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.api_key = api_key

    async def complete_json(
        self,
        prompt: str,
        system: str = "You are a helpful assistant that outputs JSON.",
    ) -> dict[str, Any]:
        """Call LLM and parse JSON response."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        response = await acompletion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            api_key=self.api_key,
        )

        content = response.choices[0].message.content
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}\nContent: {content}")
```

```python
# src/engram/llm/__init__.py
"""LLM integration."""

from engram.llm.provider import LLMProvider

__all__ = ["LLMProvider"]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_llm_provider.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add LLM provider wrapper with JSON mode"
```

---

### Task 12: Extraction Prompts

**Files:**
- Create: `src/engram/llm/prompts.py`
- Test: `tests/unit/test_prompts.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_prompts.py
from engram.llm.prompts import build_entity_extraction_prompt, build_relationship_inference_prompt


def test_entity_prompt_includes_message():
    prompt = build_entity_extraction_prompt(
        message_text="Kendra loves Nike shoes",
        speaker="Kendra",
        timestamp="2024-01-15T10:00:00Z",
        context_entities=[],
    )
    assert "Kendra loves Nike shoes" in prompt
    assert "Kendra" in prompt
    assert "PERSON" in prompt  # Should mention entity types


def test_relationship_prompt_includes_entities():
    prompt = build_relationship_inference_prompt(
        message_text="Kendra loves Nike shoes",
        speaker="Kendra",
        timestamp="2024-01-15T10:00:00Z",
        entities=[{"name": "Kendra", "type": "PERSON"}, {"name": "Nike shoes", "type": "PREFERENCE"}],
        existing_relationships=[],
    )
    assert "Kendra" in prompt
    assert "Nike shoes" in prompt
    assert "prefers" in prompt  # Should mention relationship types
```

**Step 2: Implement prompts from ARCHITECTURE.md sections 2.2 and 2.3**

> Copy the exact prompts from ARCHITECTURE.md, parameterized with f-strings. Keep them in a dedicated module for easy iteration.

**IMPORTANT (Open Question #2 - Relationship Type Explosion)**: In `build_relationship_inference_prompt()`, add explicit constraint:

```python
# Add to relationship inference prompt
f"""
Use ONLY these relationship types:
- prefers: likes, dislikes, choices, opinions
- avoids: negative sentiment, rejection
- knows: social relationships, connections
- discussed: conversation participation
- mentioned_with: co-occurrence (fallback)
- has_goal: ownership of objective
- relates_to: generic connection (fallback if no better match)

NEVER create compound type names (e.g., "slightly_prefers", "knows_well").
Capture nuance via properties, not type names.

If relationship doesn't match any type, use "relates_to" and explain in evidence field.
"""
```

**Step 3: Commit**

```bash
git add -A && git commit -m "feat: add extraction prompt templates"
```

---

### Task 13: Extraction Pipeline (Full 3-Stage)

**Files:**
- Create: `src/engram/services/extraction.py`
- Test: `tests/unit/test_extraction.py`

**Step 1: Write tests with mocked LLM**

```python
# tests/unit/test_extraction.py
from unittest.mock import AsyncMock, patch
import pytest
from datetime import datetime, timezone

from engram.models.message import IngestRequest
from engram.services.extraction import ExtractionPipeline
from engram.storage.memory import MemoryStore
from engram.services.dedup import InMemoryDedup
from engram.services.resolution import ConflictResolver


@pytest.fixture
async def pipeline():
    store = MemoryStore()
    await store.initialize()
    dedup = InMemoryDedup()
    resolver = ConflictResolver(store)
    return ExtractionPipeline(store=store, dedup=dedup, resolver=resolver, llm=AsyncMock())


async def test_pipeline_extracts_entities_and_relationships(pipeline):
    """Full pipeline: message -> entities + relationships in graph."""
    # Mock LLM responses
    pipeline._llm.complete_json = AsyncMock(side_effect=[
        # Stage 1: Entity extraction
        {"entities": [
            {"name": "Kendra", "canonical": "kendra", "type": "PERSON", "confidence": 1.0},
            {"name": "Nike shoes", "canonical": "nike-shoes", "type": "PREFERENCE", "confidence": 1.0},
        ]},
        # Stage 2: Relationship inference
        {"relationships": [
            {"source_mention": "Kendra", "target_mention": "Nike shoes",
             "type": "prefers", "confidence": 0.95, "evidence": "loves Nike shoes",
             "temporal_marker": "now"},
        ]},
    ])

    request = IngestRequest(
        text="Kendra loves Nike shoes",
        speaker="Kendra",
        conversation_id="c1",
        tenant_id="t1",
    )
    response = await pipeline.process_message(request)

    assert response.entities_extracted == 2
    assert response.relationships_inferred == 1

    # Verify entities in store
    entities = await pipeline._store.list_entities("t1", "c1")
    assert len(entities) == 2

    # Verify relationships
    kendra_id = "t1:c1:PERSON:kendra"
    rels = await pipeline._store.get_active_relationships(kendra_id)
    assert len(rels) == 1
    assert rels[0].rel_type == "prefers"


async def test_pipeline_deduplicates_messages(pipeline):
    """Processing same message twice should be idempotent."""
    pipeline._llm.complete_json = AsyncMock(side_effect=[
        {"entities": [{"name": "Kendra", "canonical": "kendra", "type": "PERSON", "confidence": 1.0}]},
        {"relationships": []},
    ])

    request = IngestRequest(
        text="Kendra is here", speaker="Kendra",
        conversation_id="c1", tenant_id="t1", message_id="msg-fixed",
    )
    await pipeline.process_message(request)
    response2 = await pipeline.process_message(request)

    # Second call should be skipped (dedup)
    assert response2.entities_extracted == 0
    assert response2.relationships_inferred == 0


async def test_correction_detection(pipeline):
    """Test that 'Actually...' corrections retract old beliefs."""
    # First message: Kendra likes Nike
    pipeline._llm.complete_json = AsyncMock(side_effect=[
        {"entities": [
            {"name": "Kendra", "canonical": "kendra", "type": "PERSON", "confidence": 1.0},
            {"name": "Nike", "canonical": "nike", "type": "PREFERENCE", "confidence": 1.0},
        ]},
        {"relationships": [
            {"source_mention": "Kendra", "target_mention": "Nike",
             "type": "prefers", "confidence": 0.9, "evidence": "likes Nike",
             "temporal_marker": "now"},
        ]},
        # Correction: Actually Adidas
        {"entities": [
            {"name": "Kendra", "canonical": "kendra", "type": "PERSON", "confidence": 1.0},
            {"name": "Adidas", "canonical": "adidas", "type": "PREFERENCE", "confidence": 1.0},
        ]},
        {"relationships": [
            {"source_mention": "Kendra", "target_mention": "Adidas",
             "type": "prefers", "confidence": 0.95, "evidence": "correction to Adidas",
             "temporal_marker": "now"},
        ]},
    ])

    msg1 = IngestRequest(
        text="Kendra likes Nike", speaker="Kendra",
        conversation_id="c1", tenant_id="t1", message_id="msg-1",
    )
    await pipeline.process_message(msg1)

    # Correction message (note: "Actually" keyword)
    msg2 = IngestRequest(
        text="Actually, Kendra prefers Adidas", speaker="System",
        conversation_id="c1", tenant_id="t1", message_id="msg-2",
    )
    await pipeline.process_message(msg2)

    # Verify: Only Adidas is active (Nike retracted)
    kendra_id = "t1:c1:PERSON:kendra"
    active_prefs = await pipeline._store.get_active_relationships(kendra_id, rel_type="prefers")
    assert len(active_prefs) == 1
    assert "adidas" in active_prefs[0].target_id

    # Verify: Nike relationship exists but is terminated
    all_prefs = await pipeline._store.query_evolution("t1", "kendra", rel_type="prefers")
    assert len(all_prefs) >= 2  # Nike (terminated) + Adidas (active)
```

**Step 2: Implement ExtractionPipeline**

```python
# src/engram/services/extraction.py (outline)
"""3-stage extraction pipeline: Entity -> Relationship -> Conflict Resolution.

See ARCHITECTURE.md section 2 for full design.
"""

from __future__ import annotations

import time
import uuid
import logging
from datetime import datetime, timezone

from engram.llm.provider import LLMProvider
from engram.llm.prompts import build_entity_extraction_prompt, build_relationship_inference_prompt
from engram.models.entity import Entity, EntityType
from engram.models.message import IngestRequest, IngestResponse
from engram.models.relationship import Relationship, RelationshipType
from engram.services.dedup import DedupService
from engram.services.resolution import ConflictResolver
from engram.storage.base import GraphStore

logger = logging.getLogger(__name__)


class ExtractionPipeline:
    def __init__(
        self,
        store: GraphStore,
        dedup: DedupService,
        resolver: ConflictResolver,
        llm: LLMProvider,
    ) -> None:
        self._store = store
        self._dedup = dedup
        self._resolver = resolver
        self._llm = llm

    async def process_message(self, request: IngestRequest) -> IngestResponse:
        start = time.monotonic()
        message_id = request.message_id or str(uuid.uuid4())

        # Step 0: Idempotency check
        is_new = await self._dedup.check_and_mark(message_id)
        if not is_new:
            return IngestResponse(
                message_id=message_id, entities_extracted=0,
                relationships_inferred=0, conflicts_resolved=0,
                processing_time_ms=0,
            )

        # Step 1: Entity extraction (LLM)
        entities = await self._extract_entities(request)

        # Step 2: Relationship inference (LLM)
        relationships = await self._infer_relationships(request, entities)

        # Step 3: Conflict resolution + graph writes
        conflicts = 0
        for rel in relationships:
            await self._resolver.resolve_and_create(rel)

        elapsed_ms = (time.monotonic() - start) * 1000
        return IngestResponse(
            message_id=message_id,
            entities_extracted=len(entities),
            relationships_inferred=len(relationships),
            conflicts_resolved=conflicts,
            processing_time_ms=round(elapsed_ms, 2),
        )

    async def _extract_entities(self, request: IngestRequest) -> list[Entity]:
        # Build context, call LLM, parse response, upsert entities
        # See ARCHITECTURE.md section 2.2
        
        # NEW (Open Question #3): Pass group_id from request for cross-conversation linking
        group_id = request.group_id or request.conversation_id  # Default to conversation-scoped
        
        # ... LLM extraction logic ...
        # When building entity IDs, use group_id:
        # entity_id = Entity.build_id(
        #     tenant_id=request.tenant_id,
        #     entity_type=EntityType[item["type"]],
        #     canonical_name=canonical,
        #     group_id=group_id
        # )
        ...

    async def _infer_relationships(
        self, request: IngestRequest, entities: list[Entity]
    ) -> list[Relationship]:
        # Build context, call LLM, parse response, map mentions to IDs
        # See ARCHITECTURE.md section 2.3
        
        # ... LLM inference logic ...
        
        # NEW (Open Question #2 - Relationship Type Explosion): Validate types
        for item in inferred["relationships"]:
            # Validate against RelationshipType enum
            try:
                rel_type = RelationshipType(item["type"])
            except ValueError:
                # Unknown type -> fallback to relates_to, preserve original
                logger.warning(f"Unknown relationship type '{item['type']}', using 'relates_to'")
                rel_type = RelationshipType.RELATES_TO
                if "metadata" not in item:
                    item["metadata"] = {}
                item["metadata"]["original_type"] = item["type"]
            
            # NEW (Open Question #4 - Confidence Calibration): Snap to discrete levels
            confidence = snap_confidence(item.get("confidence", 0.8))
            
            # ... relationship creation logic ...
        ...


def snap_confidence(raw: float, levels: tuple[float, ...] = (1.0, 0.8, 0.6, 0.4)) -> float:
    """Snap raw LLM confidence to nearest defined level.
    
    Args:
        raw: Raw confidence from LLM (0.0-1.0)
        levels: Allowed confidence levels (default: 1.0, 0.8, 0.6, 0.4 from ARCHITECTURE.md)
    
    Returns:
        Confidence snapped to nearest level
    
    Note: Open Question #4 (Confidence Calibration) - LLM self-reported confidence is poorly
          calibrated. Snapping to discrete levels from ARCHITECTURE.md prompt ensures semantic
          consistency: 1.0=direct, 0.8=strong implication, 0.6=weak inference, 0.4=co-occurrence.
    """
    return min(levels, key=lambda level: abs(level - raw))
```

> **Note to implementer:** Fill in `_extract_entities` and `_infer_relationships` following the exact patterns from ARCHITECTURE.md sections 2.2 and 2.3. The LLM returns mentions (not IDs), the service maps mentions to deterministic IDs using `Entity.build_id()` and `Entity.normalize_name()`.

**Step 3: Run tests**

Run: `uv run pytest tests/unit/test_extraction.py -v`
Expected: All passed

**Step 4: Commit**

```bash
git add -A && git commit -m "feat: add 3-stage extraction pipeline (entity, relationship, conflict)"
```

---

### Task 14: Vector Index + Entity Resolution

**Files:**
- Modify: `src/engram/storage/neo4j.py` (add vector operations)
- Create: `src/engram/services/embeddings.py`
- Test: `tests/integration/test_vector.py`

**Step 1: Create embeddings service**

```python
# src/engram/services/embeddings.py
"""Embedding generation for entity resolution."""

from __future__ import annotations

from openai import AsyncOpenAI


class EmbeddingService:
    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None):
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def embed(self, text: str) -> list[float]:
        response = await self._client.embeddings.create(model=self._model, input=text)
        return response.data[0].embedding
```

**Step 2: Add vector index creation to Neo4jStore.initialize()**

```python
# Add to _create_indexes():
"CREATE VECTOR INDEX entity_embedding IF NOT EXISTS FOR (e:Entity) ON (e.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}"
```

**Step 3: Add vector search method to Neo4jStore**

```python
async def find_similar_entities(
    self, embedding: list[float], entity_type: EntityType, limit: int = 5, threshold: float = 0.85
) -> list[tuple[Entity, float]]:
    results = await self._execute_read("""
        CALL db.index.vector.queryNodes('entity_embedding', $limit, $embedding)
        YIELD node, score
        WHERE node.type = $type AND score > $threshold
        RETURN node, score
        ORDER BY score DESC
    """, embedding=embedding, limit=limit, type=entity_type.value, threshold=threshold)
    # Parse and return
```

**Step 4: Wire similarity check into extraction pipeline (Open Question #1 - Entity Merging)**

Add to `ExtractionPipeline._extract_entities()`:

```python
# After creating entity, check for potential duplicates
if entity.embedding:  # Only if embedding generated
    similar = await self._store.find_similar_entities(
        embedding=entity.embedding,
        entity_type=entity.entity_type,
        limit=3,
        threshold=0.85  # Graphiti uses 0.90, we use 0.85 for warnings
    )
    
    if similar:
        # Log warning for manual review (no auto-merge in MVP)
        logger.warning(
            f"Potential duplicate entity detected: '{entity.canonical_name}' "
            f"similar to {[(e.canonical_name, score) for e, score in similar]}. "
            f"Consider manual merge via POST /entities/{entity.id}/merge"
        )
```

**Note**: This is the MVP approach for Open Question #1. No auto-merge, just warnings. Manual merge API added in Task 16.

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add vector index, entity resolution, and duplicate detection warnings"
```

---

## Phase 4: API Layer (Tasks 15-18)

### Task 15: FastAPI App + Health Endpoint

**Files:**
- Create: `src/engram/main.py`
- Create: `src/engram/api/deps.py`
- Create: `src/engram/api/routes.py`
- Test: `tests/integration/test_api.py`

**Step 1: Write the failing test**

```python
# tests/integration/test_api.py
import pytest
from httpx import AsyncClient, ASGITransport

from engram.main import create_app


@pytest.fixture
async def client():
    app = create_app(use_memory_store=True)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def test_health_check(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


async def test_ingest_message(client):
    response = await client.post("/messages", json={
        "text": "Kendra loves Nike shoes",
        "speaker": "Kendra",
        "conversation_id": "c1",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["entities_extracted"] >= 0  # May be 0 without real LLM


async def test_list_entities(client):
    response = await client.get("/entities", params={"tenant_id": "default"})
    assert response.status_code == 200
    assert isinstance(response.json(), list)
```

**Step 2: Implement FastAPI app with lifespan**

```python
# src/engram/main.py
"""FastAPI application with lifespan management."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from engram.api.routes import router
from engram.config import Settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize and tear down services."""
    settings: Settings = app.state.settings
    # Initialize storage, Redis, LLM
    # Store in app.state for dependency injection
    yield
    # Cleanup


def create_app(use_memory_store: bool = False) -> FastAPI:
    settings = Settings(_env_file=None) if use_memory_store else Settings()
    app = FastAPI(
        title="Engram",
        description="Temporal knowledge graph engine for AI memory",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.use_memory_store = use_memory_store
    app.include_router(router)
    return app


app = create_app()
```

**Step 3: Implement routes**

```python
# src/engram/api/routes.py
"""API route definitions.

Endpoints from ARCHITECTURE.md section 6.2:
POST /messages
GET  /entities
GET  /entities/{id}
GET  /entities/{id}/relationships
GET  /query/point-in-time
GET  /query/evolution
GET  /search
GET  /health
"""

from fastapi import APIRouter, Depends, Query
from engram.models.message import IngestRequest, IngestResponse

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "healthy", "version": "0.1.0"}


@router.post("/messages", response_model=IngestResponse)
async def ingest_message(request: IngestRequest):
    # Get pipeline from app state, call process_message
    ...


@router.get("/entities")
async def list_entities(tenant_id: str = "default", conversation_id: str | None = None):
    ...


@router.get("/entities/{entity_id}")
async def get_entity(entity_id: str):
    ...


@router.get("/entities/{entity_id}/relationships")
async def get_relationships(entity_id: str, rel_type: str | None = None):
    ...


@router.get("/query/point-in-time")
async def point_in_time_query(entity: str, as_of: str, tenant_id: str = "default"):
    ...


@router.get("/query/evolution")
async def evolution_query(entity: str, rel_type: str | None = None, tenant_id: str = "default"):
    ...


@router.get("/search")
async def search_entities(q: str, tenant_id: str = "default"):
    ...


@router.post("/entities/{entity_id}/merge")
async def merge_entities(entity_id: str, duplicate_id: str):
    """Manual entity merge endpoint (Open Question #1 - Entity Merging).
    
    Merges duplicate_id into entity_id:
    1. Transfer all relationships from duplicate to primary
    2. Update relationship endpoints to point to primary
    3. Mark duplicate as merged (set metadata.merged_into = entity_id)
    4. Optionally delete duplicate node
    
    This is the MVP approach: manual merge via API, no auto-merge.
    v0.2 will add merge candidates queue with similarity scores.
    """
    ...
```

> **Note to implementer:** Each route delegates to the appropriate service (pipeline, store, temporal). Use FastAPI dependency injection via `app.state` or `Depends()`. Follow patterns from ARCHITECTURE.md section 6.2.
>
> **NEW (Open Question #1)**: The `POST /entities/{id}/merge` endpoint is the MVP approach for entity deduplication. It requires manual invocation. The extraction pipeline logs warnings when `find_similar_entities()` detects potential duplicates (see Task 14, Step 4).

**Step 4: Run tests**

Run: `uv run pytest tests/integration/test_api.py -v`
Expected: All passed

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add FastAPI app with 9 API endpoints (includes manual merge endpoint)"
```

---

### Task 16: CLI Commands

**Files:**
- Create: `src/engram/cli/main.py`
- Create: `src/engram/cli/__init__.py`

**Step 1: Implement CLI with typer**

```python
# src/engram/cli/main.py
"""Engram CLI commands."""

import typer
from rich.console import Console

app = typer.Typer(name="engram", help="Temporal knowledge graph engine for AI memory")
console = Console()


@app.command()
def init():
    """Initialize the Neo4j schema and indexes."""
    console.print("[green]Initializing Engram schema...[/green]")
    # Connect to Neo4j, run initialize()
    console.print("[green]Done! Schema created successfully.[/green]")


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Start the API server."""
    import uvicorn
    uvicorn.run("engram.main:app", host=host, port=port, reload=reload)


@app.command()
def ingest(file: str):
    """Ingest conversation messages from a JSON file."""
    console.print(f"[blue]Ingesting from {file}...[/blue]")
    # Read JSON, iterate messages, call POST /messages
    console.print("[green]Done![/green]")


@app.command()
def query(text: str, tenant_id: str = "default"):
    """Query the knowledge graph."""
    console.print(f"[blue]Querying: {text}[/blue]")
    # Call appropriate API endpoint
    ...


@app.command()
def export(output: str = "graph.json", tenant_id: str = "default"):
    """Export graph to JSON."""
    console.print(f"[blue]Exporting to {output}...[/blue]")
    ...
```

**Step 2: Verify CLI works**

Run: `uv run engram --help`
Expected: Shows help with init, serve, ingest, query, export commands

**Step 3: Commit**

```bash
git add -A && git commit -m "feat: add CLI commands (init, serve, ingest, query, export)"
```

---

## Phase 5: Demo & Integration (Tasks 17-19)

### Task 17: Coaching Demo Scenario

**Files:**
- Create: `examples/coaching-demo.json`

**Step 1: Create demo data (Kendra's shoe preference scenario)**

```json
{
  "conversation_id": "coaching-session-1",
  "messages": [
    {
      "text": "Hi, I'm Kendra. I've been training for the Boston Marathon.",
      "speaker": "Kendra",
      "timestamp": "2024-01-15T10:00:00Z"
    },
    {
      "text": "I love Nike running shoes, they're the best for long distance.",
      "speaker": "Kendra",
      "timestamp": "2024-01-15T10:05:00Z"
    },
    {
      "text": "My goal is to finish the Boston Marathon under 4 hours.",
      "speaker": "Kendra",
      "timestamp": "2024-01-15T10:10:00Z"
    },
    {
      "text": "I've been working with Coach Sarah on my training plan.",
      "speaker": "Kendra",
      "timestamp": "2024-02-01T14:00:00Z"
    },
    {
      "text": "Actually, I switched to Adidas. The arch support is much better for my feet.",
      "speaker": "Kendra",
      "timestamp": "2024-03-20T14:30:00Z"
    },
    {
      "text": "I completed the Boston Marathon! Finished in 3:45:00.",
      "speaker": "Kendra",
      "timestamp": "2024-04-15T18:00:00Z"
    }
  ]
}
```

**Step 2: Commit**

```bash
git add -A && git commit -m "feat: add coaching demo scenario"
```

---

### Task 18: End-to-End Test

**Files:**
- Create: `tests/e2e/test_coaching_demo.py`

**Step 1: Write E2E test**

```python
# tests/e2e/test_coaching_demo.py
"""E2E test: Ingest coaching demo -> verify temporal queries.

This test validates the FULL pipeline from message ingestion to temporal queries.
Uses MemoryStore (no Neo4j needed) with mocked LLM.
"""
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from engram.models.message import IngestRequest
from engram.services.extraction import ExtractionPipeline
from engram.services.dedup import InMemoryDedup
from engram.services.resolution import ConflictResolver
from engram.storage.memory import MemoryStore


@pytest.fixture
async def pipeline_with_mock_llm():
    store = MemoryStore()
    await store.initialize()
    dedup = InMemoryDedup()
    resolver = ConflictResolver(store)
    llm = AsyncMock()

    # Program LLM responses for each message in the coaching demo
    llm.complete_json = AsyncMock(side_effect=[
        # Message 1: "Hi, I'm Kendra. Training for Boston Marathon."
        {"entities": [
            {"name": "Kendra", "canonical": "kendra", "type": "PERSON", "confidence": 1.0},
            {"name": "Boston Marathon", "canonical": "boston-marathon", "type": "EVENT", "confidence": 1.0},
        ]},
        {"relationships": [
            {"source_mention": "Kendra", "target_mention": "Boston Marathon",
             "type": "has_goal", "confidence": 0.9, "evidence": "training for", "temporal_marker": "now"},
        ]},
        # Message 2: "I love Nike running shoes"
        {"entities": [
            {"name": "Kendra", "canonical": "kendra", "type": "PERSON", "confidence": 1.0},
            {"name": "Nike running shoes", "canonical": "nike-running-shoes", "type": "PREFERENCE", "confidence": 1.0},
        ]},
        {"relationships": [
            {"source_mention": "Kendra", "target_mention": "Nike running shoes",
             "type": "prefers", "confidence": 0.95, "evidence": "loves Nike", "temporal_marker": "now"},
        ]},
        # Message 5: "Switched to Adidas"
        {"entities": [
            {"name": "Kendra", "canonical": "kendra", "type": "PERSON", "confidence": 1.0},
            {"name": "Adidas", "canonical": "adidas", "type": "PREFERENCE", "confidence": 1.0},
        ]},
        {"relationships": [
            {"source_mention": "Kendra", "target_mention": "Adidas",
             "type": "prefers", "confidence": 0.9, "evidence": "switched to Adidas", "temporal_marker": "now"},
        ]},
    ])

    return ExtractionPipeline(store=store, dedup=dedup, resolver=resolver, llm=llm), store


async def test_coaching_demo_preference_evolution(pipeline_with_mock_llm):
    """Kendra's shoe preference changes from Nike to Adidas."""
    pipeline, store = pipeline_with_mock_llm

    # Ingest messages
    messages = [
        IngestRequest(text="Hi, I'm Kendra. Training for Boston Marathon.",
                     speaker="Kendra", conversation_id="c1", tenant_id="t1",
                     timestamp=datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc), message_id="msg-1"),
        IngestRequest(text="I love Nike running shoes",
                     speaker="Kendra", conversation_id="c1", tenant_id="t1",
                     timestamp=datetime(2024, 1, 15, 10, 5, tzinfo=timezone.utc), message_id="msg-2"),
        IngestRequest(text="Switched to Adidas, better arch support",
                     speaker="Kendra", conversation_id="c1", tenant_id="t1",
                     timestamp=datetime(2024, 3, 20, 14, 30, tzinfo=timezone.utc), message_id="msg-5"),
    ]

    for msg in messages:
        await pipeline.process_message(msg)

    # QUERY 1: Current preference = Adidas
    kendra_id = "t1:c1:PERSON:kendra"
    active_prefs = await store.get_active_relationships(kendra_id, rel_type="prefers")
    assert len(active_prefs) == 1
    assert "adidas" in active_prefs[0].target_id

    # QUERY 2: Week 2 preference = Nike (world state as of Feb 15)
    w2 = datetime(2024, 2, 15, tzinfo=timezone.utc)
    historical = await store.query_world_state_as_of("t1", "kendra", w2, rel_type="prefers")
    assert len(historical) == 1
    assert "nike" in historical[0].target_id

    # QUERY 3: Evolution shows Nike -> Adidas
    evolution = await store.query_evolution("t1", "kendra", rel_type="prefers")
    assert len(evolution) == 2  # Nike (terminated) + Adidas (active)
```

**Step 2: Run E2E test**

Run: `uv run pytest tests/e2e/test_coaching_demo.py -v`
Expected: All passed

**Step 3: Commit**

```bash
git add -A && git commit -m "test: add E2E coaching demo test with temporal queries"
```

---

### Task 19: Final Polish

**Step 1: Run full test suite**

Run: `uv run pytest --cov=engram --cov-report=term-missing`
Expected: > 80% coverage, all tests pass

**Step 2: Run linting**

Run: `uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/`
Expected: No errors

**Step 3: Run type checking**

Run: `uv run mypy src/engram/`
Expected: No errors (or only expected ignores on third-party libs)

**Step 4: Test Docker build**

Run: `docker-compose build api && docker-compose up -d && sleep 10 && curl http://localhost:8000/health`
Expected: `{"status": "healthy", "version": "0.1.0"}`

**Step 5: Run coaching demo end-to-end**

Run: `docker-compose exec api engram ingest examples/coaching-demo.json`
Expected: Messages ingested, entities and relationships created

Run: `curl "http://localhost:8000/entities?tenant_id=default"`
Expected: List of entities (Kendra, Nike, Adidas, etc.)

**Step 6: Final commit**

```bash
git add -A && git commit -m "chore: polish, lint, and verify full pipeline"
git tag v0.1.0
```

---

### Task 20: CI/CD Pipeline

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `.github/workflows/docker.yml`

**Step 1: Create GitHub Actions CI workflow**

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
        with:
          enable-cache: true
      - name: Lint with ruff
        run: |
          uv run ruff check src/ tests/
          uv run ruff format --check src/ tests/

  typecheck:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
        with:
          enable-cache: true
      - name: Type check with mypy
        run: uv run mypy src/engram/

  test:
    runs-on: ubuntu-latest
    services:
      neo4j:
        image: neo4j:5.26-community
        env:
          NEO4J_AUTH: neo4j/password
          NEO4J_PLUGINS: '["apoc"]'
        ports:
          - 7687:7687
        options: >-
          --health-cmd "cypher-shell -u neo4j -p password 'RETURN 1'"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
        with:
          enable-cache: true

      - name: Run unit tests
        run: uv run pytest tests/unit/ -v --cov=engram --cov-report=xml

      - name: Run integration tests
        env:
          NEO4J_URI: bolt://localhost:7687
          NEO4J_USER: neo4j
          NEO4J_PASSWORD: password
          REDIS_HOST: localhost
          REDIS_PORT: 6379
          OPENAI_API_KEY: sk-test-fake  # Not used in integration tests
        run: uv run pytest tests/integration/ -v -m integration

      - name: Run E2E tests
        run: uv run pytest tests/e2e/ -v

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
          fail_ci_if_error: false
```

**Step 2: Create Docker build workflow**

```yaml
# .github/workflows/docker.yml
name: Docker Build

on:
  push:
    tags:
      - 'v*'
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/${{ github.repository }}
          tags: |
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

**Step 3: Verify workflows**

Create a test commit to trigger CI:
```bash
git add .github/workflows/ && git commit -m "ci: add GitHub Actions workflows"
git push origin main
```

Check: GitHub Actions tab shows green checks

**Step 4: Commit**

```bash
git tag v0.1.0 && git push --tags
```

---

## MVP Scope Note: Web UI Deferred

**ARCHITECTURE.md Section 6.2 calls for a basic web UI** (graph visualization, entity browser, query interface).

**Decision:** Web UI is deferred to v0.2.0 for the following reasons:
1. **Backend completeness is MVP-critical** - API + CLI already provide full functionality
2. **Time constraint** - 3-week MVP timeline prioritizes working engine over UI polish
3. **Developer-first approach** - Target users (developers integrating Engram) will use API/CLI first
4. **v0.2 focus** - Graph visualization is better served with dedicated iteration (D3.js/react-force-graph requires significant effort)

**Workaround for MVP demo:**
- Neo4j Browser (localhost:7474) provides graph visualization for demos
- CLI `engram query` provides command-line queries
- API docs (localhost:8000/docs) provide interactive testing

**v0.2 Web UI plan:**
- React + TypeScript frontend
- react-force-graph for temporal graph visualization with time slider
- Entity search + detail pages
- Query builder UI for point-in-time and evolution queries

---

## Execution Summary

| Phase | Tasks | Estimated Time | Key Deliverable |
|-------|-------|---------------|-----------------|
| 1: Foundation | 1-6 | 2-3 days | Project scaffold, models, Docker, config |
| 2: Storage | 7-10 | 2-3 days | MemoryStore, Neo4jStore, temporal queries, conflict resolution |
| 3: Extraction | 11-14 | 3-4 days | LLM wrapper, prompts, 3-stage pipeline, vector search |
| 4: API | 15-16 | 2 days | FastAPI with 9 endpoints (incl. manual merge), CLI commands |
| 5: Demo & CI | 17-20 | 2-3 days | Coaching demo, E2E tests, Docker verification, GitHub Actions |

**Total: 20 tasks, ~12-15 working days (2.5-3 weeks of focused implementation)**

**Gap Fixes Applied:**
- ✅ .gitignore added (Task 1)
- ✅ Test coverage expanded (knowledge-as-of queries, Redis dedup integration, correction detection)
- ✅ Neo4jStore fully implemented (complete Cypher methods, not just comments)
- ✅ CI/CD pipeline added (Task 20, GitHub Actions)
- ✅ Web UI scope clarified (deferred to v0.2, rationale documented)

---

## Key Implementation Notes

1. **MemoryStore is the TDD reference**: Write all logic against MemoryStore first. Neo4jStore implements the same interface with real Cypher.

2. **LLM calls are always mocked in unit tests**: Only integration/E2E tests with real Neo4j. Never call real LLM in tests (expensive, nondeterministic).

3. **Bitemporal invariant**: Every relationship query MUST check BOTH `valid_to IS NULL` (truth) AND `recorded_to IS NULL` (knowledge). Missing either = bug.

4. **Entity IDs are deterministic**: `{tenant}:{group_id}:{type}:{canonical}` where `group_id` defaults to `conversation_id` if not provided. The LLM never generates IDs. Period. (Open Question #3: `group_id` enables cross-conversation entity linking when explicitly set.)

5. **Dedup check is FIRST**: Before any LLM call, before any graph write. Redis SET NX in production, InMemoryDedup in tests.

6. **Exclusivity enforcement is atomic**: In Neo4jStore, Steps 2+3 from ARCHITECTURE.md section 2.4 MUST be in a single transaction (`session.begin_transaction()`).

---

## Open Questions from ARCHITECTURE.md § 8.1: MVP Decisions

The following decisions address the 5 open questions documented in ARCHITECTURE.md § 8.1. Each decision is based on research across production systems (Graphiti/Zep, Mem0, GraphRAG, CortexGraph, OpenSanctions, HALO) and balances MVP simplicity with v0.2 extensibility.

### 1. Entity Merging → DEFER Auto-Merge, Add Manual Merge API

**Problem**: "Mike Johnson" vs "Michael Johnson" — same person or different?

**MVP Decision**: 
- **Keep deterministic normalization** (existing `Entity.normalize_name()`)
- **Add similarity warning**: Wire `find_similar_entities()` into extraction pipeline to LOG WARNING when match ≥ 0.85
- **Add manual merge endpoint**: `POST /entities/{id}/merge` for user-initiated merges
- **No auto-merge**: Avoids false positive risk

**v0.2 Scope**: Auto-merge with MinHash+LSH (Graphiti pattern) + merge candidates queue + auto-merge threshold 0.95

**Rationale**: 
- Deterministic normalization already handles 60-80% of duplicates
- OpenSanctions (high-stakes sanctions data) uses manual review, not automatic
- GraphRAG ships with zero dedup logic
- False merge risk ("Mike Johnson" = different people) outweighs inconvenience

**Code Changes**: See Task 14, Step 4 (new) and Task 16, Step 3 (new endpoint)

---

### 2. Relationship Type Explosion → SOLVE in MVP (~15 lines)

**Problem**: LLM can emit unbounded custom types ("slightly_prefers_sometimes") making graph unqueryable.

**MVP Decision**:
- **Validate against enum**: In `_infer_relationships()`, validate `item["type"]` against `RelationshipType`
- **Fallback to `relates_to`**: If unknown type, use `relates_to` + store original in `metadata.original_type`
- **Prompt constraint**: Update LLM prompt to explicitly list allowed types + instruct to use properties for nuance
- **Qualifier properties**: Add `intensity`, `context`, `sentiment` to metadata (not type names)

**v0.2 Scope**: Learnable type suggestion tracker (types used >50x get promoted to whitelist with human review)

**Rationale**:
- Zep uses explicit ontology with whitelist
- Neo4j best practice: 10-30 types with rich properties > hundreds of specific types
- Graphiti allows free-form but encourages constrained schemas
- Preserves queryability: `MATCH ()-[:prefers {intensity: "high"}]->()` works

**Code Changes**: See Task 12 (prompt update) and Task 13, Step 2 (validation logic)

---

### 3. Cross-Conversation Entity Resolution → Add `group_id` to MVP (~30 lines)

**Problem**: "Kendra" in conversation A is isolated from "Kendra" in conversation B (same person).

**MVP Decision**:
- **Add `group_id` field**: To Entity and Relationship models, defaults to `conversation_id`
- **Change Entity ID scheme**: `{tenant}:{group_id}:{type}:{canonical}` (not `conversation_id`)
- **Backward compatible**: Default behavior (no `group_id` passed) = current conversation isolation
- **Opt-in grouping**: User sets `group_id="kendra-coaching"` → entities shared across conversations in that group

**v0.2 Scope**: Two-tier pattern (global EntityNode tenant-scoped + EntityReference conversation-scoped) + cross-conversation query API

**Rationale**:
- Graphiti's proven production pattern
- Zero auto-merge risk — user explicitly opts into grouping
- Enables cross-conversation linking without false merge danger
- Mem0 scopes by `user_id`, not conversation — similar pattern

**Code Changes**: See Task 2, Step 4 (new), Task 8 (Neo4jStore scoping updates), Task 13 (extraction pipeline)

---

### 4. Confidence Calibration → SOLVE in MVP (~10 lines)

**Problem**: LLMs are poorly calibrated — self-reported confidence is overconfident/unreliable.

**MVP Decision**:
- **Enforce discrete levels**: Snap LLM confidence to 4 levels from ARCHITECTURE.md prompt (1.0, 0.8, 0.6, 0.4)
- **Add `snap_confidence()` function**: Post-processing after LLM returns
- **Semantic tiers**: Direct statement=1.0, strong implication=0.8, weak inference=0.6, co-occurrence=0.4

**v0.2 Scope**: Temperature scaling calibration (requires 50-100 labeled examples from production)

**Rationale**:
- Reddit/practitioners: "LLM self-reported confidence is gibberish"
- Zep avoids problem entirely — uses discrete categories (COMPLETE/PARTIAL/INSUFFICIENT)
- CortexGraph uses `strength` multiplier, not LLM confidence
- Discrete levels > continuous scores for downstream temporal reasoning

**Code Changes**: See Task 13, Step 2 (add snap_confidence function)

---

### 5. Decay Rate Tuning → SOLVE in MVP (~25 lines)

**Problem**: Hardcoded decay rates don't fit all domains (coaching vs sales vs support have different decay dynamics).

**MVP Decision**:
- **Move to settings**: `DECAY_RATES` → `EngineSettings` (Pydantic)
- **Env var overrides**: `ENGRAM_DECAY_RATE_PREFERS=0.05` per-type override
- **Preset modes**: `ENGRAM_DECAY_PRESET=balanced|fast|slow`
  - `balanced`: Current rates (prefers=0.05, knows=0.005)
  - `fast`: 2x rates (high-velocity conversations)
  - `slow`: 0.5x rates (long-term knowledge)

**v0.2 Scope**: YAML config file, per-tenant settings, learned rates from reinforcement patterns (HALO pattern)

**Rationale**:
- CortexGraph uses env var pattern with presets (battle-tested, 791 tests)
- HALO paper: different relationship types have significantly different half-lives
- Current hardcoded values align with CortexGraph production defaults
- Env vars sufficient for MVP — no need for complex config system

**Code Changes**: See Task 1, Step 3 (EngineSettings) and Task 10, Step 3 (update temporal.py)

---

### Summary: MVP Effort Impact

| # | Decision | Effort | Tasks Modified |
|---|----------|--------|----------------|
| 1 | Entity merging warning + manual API | ~1 hour | Task 14 (Step 4 new), Task 16 (endpoint) |
| 2 | Relationship type validation | ~30 min | Task 12 (prompt), Task 13 (validation) |
| 3 | `group_id` for cross-conversation | ~1 hour | Task 2 (models), Task 8 (Neo4j), Task 13 (extraction) |
| 4 | Confidence snapping | ~20 min | Task 13 (snap function) |
| 5 | Decay rate configurability | ~30 min | Task 1 (settings), Task 10 (temporal.py) |

**Total additional effort: ~3-4 hours** spread across existing tasks. No new tasks added. No timeline impact.

---
