"""Conflict resolution with exclusivity enforcement.

See ARCHITECTURE.md section 2.4 for the full algorithm.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from engram.models.relationship import ExclusivityPolicy, Relationship

if TYPE_CHECKING:
    from engram.storage.base import GraphStore

logger = logging.getLogger(__name__)

EXCLUSIVITY_POLICIES: dict[str, ExclusivityPolicy] = {
    "prefers": ExclusivityPolicy(
        exclusivity_scope=("source",),
        max_active=1,
        close_on_new=True,
    ),
    "avoids": ExclusivityPolicy(
        exclusivity_scope=("source",),
        max_active=1,
        close_on_new=True,
        exclusive_with=["prefers"],
    ),
    "works_for": ExclusivityPolicy(
        exclusivity_scope=("source",),
        max_active=1,
        close_on_new=True,
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
            max_version = await self._store.get_max_relationship_version(
                source_id=new_rel.source_id,
                rel_type=new_rel.rel_type,
                tenant_id=new_rel.tenant_id,
                conversation_id=new_rel.conversation_id,
            )
            new_rel.version = max_version + 1

            terminated = await self._store.terminate_relationship(
                source_id=new_rel.source_id,
                rel_type=new_rel.rel_type,
                tenant_id=new_rel.tenant_id,
                conversation_id=new_rel.conversation_id,
                termination_time=new_rel.valid_from,
                exclude_target_id=new_rel.target_id,
            )
            if terminated > 0:
                logger.info(f"Terminated {terminated} conflicting {new_rel.rel_type} relationships")

        return await self._store.create_relationship(new_rel)
