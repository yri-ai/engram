"""CI-safe ABS-EE fetch manifest writer.

This script intentionally does not download by default. It records supplied local
raw files into the manifest convention used by Phase 4 fetchers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def manifest_entry(path: Path, source_url: str) -> dict[str, str]:
    data = path.read_bytes()
    return {
        "source": "edgar-absee",
        "source_url": source_url,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "local_path": str(path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-file", type=Path, required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/edgar_absee.jsonl"))
    args = parser.parse_args(argv)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest_entry(args.raw_file, args.source_url), sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
