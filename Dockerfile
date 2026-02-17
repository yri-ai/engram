# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install all dependencies (including dev for in-container testing)
RUN uv sync --frozen --no-install-project

# Copy source and tests
COPY src/ src/
COPY tests/ tests/

# Install project
RUN uv sync --frozen

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Install uv (needed for `uv run pytest` in container)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy project files (needed for uv run and test execution)
COPY --from=builder /app/pyproject.toml /app/uv.lock ./
COPY --from=builder /app/src/ src/
COPY --from=builder /app/tests/ tests/

# Set PATH to use venv
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "engram.main:app", "--host", "0.0.0.0", "--port", "8000"]
