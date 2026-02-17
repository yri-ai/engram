"""FastAPI dependency injection utilities."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from engram.config import Settings
from engram.services.dedup import DedupService
from engram.storage.base import GraphStore


async def get_settings(request: Request) -> Settings:
    """Get application settings from app state."""
    return request.app.state.settings


async def get_store(request: Request) -> GraphStore:
    """Get graph store from app state."""
    return request.app.state.store


async def get_dedup(request: Request) -> DedupService:
    """Get deduplication service from app state."""
    return request.app.state.dedup


# Type aliases for dependency injection
SettingsDep = Annotated[Settings, Depends(get_settings)]
StoreDep = Annotated[GraphStore, Depends(get_store)]
DedupDep = Annotated[DedupService, Depends(get_dedup)]
