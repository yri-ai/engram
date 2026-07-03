"""Name-variant entity resolution.

Merges first-name / full-name / initial variants of the same person so that mentions like
"Caroline" and "Caroline Kim", or "Anna B" and "Anna Bethke", resolve to a single canonical
entity instead of creating duplicate nodes. Deterministic and dependency-free.

Rules (conservative, to avoid over-merging distinct people who share a first name):
- Both names must share the same first token.
- Every token of the shorter name must match a token of the longer name, in order, where
  "match" means equal, one is a prefix of the other, or one is a single-letter initial of the
  other (e.g. "b" ~ "bethke").
- If two *incompatible* fuller candidates match (e.g. "Mark Guzman" and "Mark Zschocke" for
  "Mark"), the result is ambiguous and no merge is performed.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from engram.models.entity import EntityType
    from engram.storage.base import GraphStore


def _tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t]


def _token_match(a: str, b: str) -> bool:
    if a == b:
        return True
    if len(a) == 1 or len(b) == 1:
        return a.startswith(b) or b.startswith(a)
    return a.startswith(b) or b.startswith(a)


def is_name_variant(a: str, b: str) -> bool:
    """True if a and b are plausibly the same person written at different specificity."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb or ta[0] != tb[0]:
        return False
    short, long = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    li = 0
    for st in short:
        matched = False
        while li < len(long):
            lt = long[li]
            li += 1
            if _token_match(st, lt):
                matched = True
                break
        if not matched:
            return False
    return True


def resolve_variant(canonical: str, candidates: list[str]) -> str | None:
    """Return the fullest existing candidate that is a name-variant of `canonical`.

    Returns None if there is no match, or if the match is ambiguous (two distinct fuller
    candidates that are not variants of each other).
    """
    matches = [c for c in candidates if c and c != canonical and is_name_variant(canonical, c)]
    if not matches:
        return None
    # Prefer the most complete (most tokens) name.
    matches.sort(key=lambda c: len(_tokens(c)), reverse=True)
    best = matches[0]
    # Ambiguity guard: if another match is NOT a variant of `best`, don't merge.
    for other in matches[1:]:
        if not is_name_variant(best, other):
            return None
    return best


async def consolidate_name_variants(
    store: GraphStore,
    tenant_id: str,
    group_id: str,
    entity_types: Sequence[EntityType],
) -> int:
    """Post-hoc: merge first-name/full-name duplicate person entities within a group.

    Order- and concurrency-independent (unlike creation-time resolution). For each entity type,
    clusters entities by first name and merges each shorter name into its UNIQUE fuller variant
    (skips ambiguous cases such as two distinct people sharing a first name). Returns the number
    of entities merged.
    """
    merged = 0
    for et in entity_types:
        ents = await store.list_entities(
            tenant_id, entity_type=et, group_id=group_id, limit=5000
        )
        # skip entities already merged away
        ents = [e for e in ents if "merged_into" not in str(e.metadata or {})]
        by_first: dict[str, list] = {}
        for e in ents:
            toks = _tokens(e.canonical_name)
            if toks:
                by_first.setdefault(toks[0], []).append(e)
        for group in by_first.values():
            if len(group) < 2:
                continue
            used: set[str] = set()
            # merge shorter names into their unique fuller variant
            for dup in sorted(group, key=lambda e: len(_tokens(e.canonical_name))):
                if dup.id in used:
                    continue
                cands = [
                    p
                    for p in group
                    if p.id != dup.id
                    and p.id not in used
                    and len(_tokens(p.canonical_name)) > len(_tokens(dup.canonical_name))
                    and is_name_variant(p.canonical_name, dup.canonical_name)
                ]
                if len(cands) != 1:
                    continue  # no match, or ambiguous
                primary = max(cands, key=lambda e: len(_tokens(e.canonical_name)))
                await store.merge_entity_into(primary.id, dup.id)
                used.add(dup.id)
                merged += 1
    return merged
