"""E2E smoke test for the public diligence forecast demo."""

from __future__ import annotations

import json
import subprocess


def test_diligence_demo_runs_green(tmp_path):
    result = subprocess.run(
        ["bash", "examples/diligence-demo/run_demo.sh", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads((tmp_path / "report.json").read_text())
    audit = json.loads((tmp_path / "audit.json").read_text())
    assert report["run_count"] == 1
    assert audit["status"] == "PASS"
    assert str(tmp_path) in result.stdout
