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
