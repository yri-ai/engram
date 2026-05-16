from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest

if TYPE_CHECKING:
    from types import ModuleType


def _load_fetch_module() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "data_collection" / "fetch_edgar_cmbs.py"
    )
    spec = importlib.util.spec_from_file_location("fetch_edgar_cmbs", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeClient:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = responses

    async def get(self, url: str, **kwargs: object) -> httpx.Response:
        if not self._responses:
            raise AssertionError("No more fake responses configured")
        return self._responses.pop(0)


@pytest.mark.asyncio
async def test_rate_limited_get_retries_retryable_http_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_fetch_module()
    module._last_request_time = 0.0
    monkeypatch.setattr(module, "RATE_LIMIT_INTERVAL", 0.0)

    sleeps: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(module.asyncio, "sleep", _fake_sleep)

    request = httpx.Request("GET", "https://example.com")
    client = _FakeClient(
        [
            httpx.Response(status_code=429, headers={"Retry-After": "1.5"}, request=request),
            httpx.Response(status_code=200, text="ok", request=request),
        ]
    )

    response = await module.rate_limited_get(client, "https://example.com")

    assert response.status_code == 200
    assert sleeps == [1.5]


@pytest.mark.asyncio
async def test_rate_limited_get_raises_for_non_retryable_http_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_fetch_module()
    module._last_request_time = 0.0
    monkeypatch.setattr(module, "RATE_LIMIT_INTERVAL", 0.0)

    async def _fake_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(module.asyncio, "sleep", _fake_sleep)

    request = httpx.Request("GET", "https://example.com")
    client = _FakeClient([httpx.Response(status_code=404, request=request)])

    with pytest.raises(httpx.HTTPStatusError):
        await module.rate_limited_get(client, "https://example.com")
