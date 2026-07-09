"""Emit calibration audit reports for forecast artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def write_audit(scoreboard_path: Path, output_path: Path) -> dict[str, object]:
    scoreboard = json.loads(scoreboard_path.read_text())
    rows = [
        {
            "model": model["model"],
            "ece": model["metrics"].get("ece", 0.0),
            "alarm": model["metrics"].get("ece", 0.0) > 0.15,
        }
        for model in scoreboard.get("models", [])
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Forecast Calibration Audit", "", "| Model | ECE | Alarm |", "|---|---:|---|"]
    lines.extend(f"| {row['model']} | {row['ece']:.6f} | {row['alarm']} |" for row in rows)
    output_path.write_text("\n".join(lines) + "\n")
    return {"rows": rows, "output_path": str(output_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scoreboard", type=Path, default=Path("outputs/results/forecast_scoreboard_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/plans/decisions/track-b-calibration-audit.md"))
    args = parser.parse_args(argv)
    print(json.dumps(write_audit(args.scoreboard, args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
