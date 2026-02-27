# Contributing to Engram

Thanks for helping make Engram the best open-source temporal memory layer. This document explains how to set up your environment, run the required checks, and submit changes.

## Code of Conduct

Be respectful and collaborative. Assume good intent, give constructive feedback, and default to transparency. If something feels off, open a discussion or ping the maintainers before it escalates.

## Getting Started

1. **Fork + clone**
   ```bash
   git clone https://github.com/<your-username>/engram.git
   cd engram
   ```
2. **Install prerequisites**
   - Python 3.11+
   - [uv](https://docs.astral.sh/uv/) (used for dependency + virtualenv management)
   - Docker or Podman (for Neo4j/Redis in integration tests)
   - Optional: `mkcert`/OpenSSL for custom TLS setups
3. **Install dependencies**
   ```bash
   uv sync
   ```
4. **Environment variables**
   ```bash
   cp .env.example .env
   # edit .env with at least OPENAI_API_KEY (or configure Anthropic/Ollama)
   ```
5. **Start backing services**
   ```bash
   docker compose up  # or podman compose up
   ```
   This brings up Neo4j (bolt://localhost:7687), Redis, and the FastAPI server. For quick smoke tests you can run `uv run engram serve` with the in-memory store, but end-to-end tests expect Neo4j/Redis.

## Development Workflow

- Create a feature branch from `master`.
- Write tests first when feasible (the repo uses pytest + strict mypy + ruff).
- Keep commits focused; rebase before opening the PR.

## Testing & Linting

Always run these before pushing:

```bash
# Style + lint
uv run ruff check

# Fast unit suite
uv run pytest tests/unit -q

# Full suite (requires Neo4j & Redis running)
uv run pytest -q
```

Additional checks:
- `uv run ruff format` if you make formatting changes.
- `uv run mypy src/engram` when you touch type-heavy code.

CI runs lint, mypy, unit, integration, and e2e suites inside GitHub Actions; keeping local runs clean saves time.

## Documentation & Demo Updates

- Update `README.md` and `docs/onboarding.md` when user-facing behavior changes.
- If you add CLI commands or API endpoints, update `scripts/demo.sh` and `docs/plans/` when relevant.

## Pull Request Checklist

Before opening a PR:

- [ ] Tests pass locally (`uv run pytest -q`)
- [ ] `uv run ruff check` passes
- [ ] Added/updated tests cover the change
- [ ] Docs/README updated (if user-facing change)
- [ ] Linked the relevant issue (or describe motivation clearly)
- [ ] If the change touches deployment or infra, note required migrations/config updates

## Issue Reporting

When filing an issue, include:

- Summary of the problem or feature request
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Environment details (OS, Python, uv version, Neo4j/Redis versions)
- Logs or stack traces if available

## Questions & Discussions

Use GitHub Discussions for open-ended ideas, architecture proposals, or questions about roadmap priorities. For security issues, please email the maintainers instead of opening a public issue.

Thanks for contributing! 🚀
