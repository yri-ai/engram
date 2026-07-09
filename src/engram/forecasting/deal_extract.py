"""Deal-document extraction seams with human verification output."""

from __future__ import annotations

import re
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from engram.models.deal import DealSpec


def emit_review_markdown(spec: DealSpec, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_deal_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", spec.deal_id).strip(".-") or "deal"
    path = (output_dir / f"{safe_deal_id}.md").resolve()
    root = output_dir.resolve()
    if root not in path.parents and path != root:
        raise ValueError("review path escapes output directory")
    path.write_text(
        f"# Deal Review: {spec.deal_id}\n\n- Spec ID: `{spec.spec_id}`\n- Verified: `{spec.verified}`\n"
    )
    return path


def approve_spec(spec: DealSpec, *, verified_by: str, verified_at: str) -> DealSpec:
    parsed_verified_at = datetime.fromisoformat(verified_at.replace("Z", "+00:00"))
    return spec.model_copy(
        update={"verified": True, "verified_by": verified_by, "verified_at": parsed_verified_at}
    )
