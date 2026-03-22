"""Application configuration via environment variables."""

import hashlib
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, Template
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
    llm_provider: str = "openai"

    # Embedding
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Application
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Prompt Config
    prompt_dir: str = "config/prompts"

    # Decay Rates (Open Question #5: Configurable decay rates)
    decay_preset: str = "balanced"  # balanced | fast | slow
    decay_rate_prefers: float = 0.05  # 1.5-day half-life
    decay_rate_avoids: float = 0.04
    decay_rate_knows: float = 0.005  # 7-day half-life
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


settings = Settings()


class ConfigService:
    """Service for managing templates and dynamic configuration."""

    def __init__(self, prompt_dir: str | None = None) -> None:
        self.prompt_dir = Path(prompt_dir or settings.prompt_dir)
        # Ensure directory exists for FileSystemLoader
        self.prompt_dir.mkdir(parents=True, exist_ok=True)
        self.env = Environment(loader=FileSystemLoader(str(self.prompt_dir)))

    def get_template(self, name: str) -> Template:
        """Get a Jinja2 template by name."""
        return self.env.get_template(name)

    def get_template_sha256(self, name: str) -> str:
        """Calculate SHA256 of the template file for version tracking."""
        path = self.prompt_dir / name
        if not path.exists():
            return ""
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def render_prompt(self, template_name: str, **kwargs: Any) -> str:
        """Render a prompt template with provided context."""
        template = self.get_template(template_name)
        return template.render(**kwargs)
