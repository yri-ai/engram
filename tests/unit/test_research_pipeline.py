from __future__ import annotations

import json
import zipfile
from pathlib import Path

from engram.services.research_pipeline import run_research_pipeline


def _write_zip(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, mode="w") as zf:
        zf.writestr("sample.txt", "ok")


def test_run_research_pipeline_generates_expected_outputs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_zip(data_dir / "fannie" / "2023_enterprise_pudb.zip")
    _write_zip(data_dir / "ginnie" / "202602" / "monthlySFS_202602.zip")

    edgar = data_dir / "edgar" / "cmbs" / "1713393"
    edgar.mkdir(parents=True, exist_ok=True)
    (edgar / "f.xml").write_text("<root />", encoding="utf-8")
    (edgar / "f.meta.json").write_text(
        json.dumps(
            {
                "cik": "1713393",
                "accession": "0000950131-17-000939",
                "filing_date": "2024-01-01",
                "source_url": "https://www.sec.gov/Archives/example.xml",
            }
        ),
        encoding="utf-8",
    )

    summary = run_research_pipeline(
        data_dir=data_dir,
        output_root=tmp_path / "outputs",
        fixture_per_split=5,
        sweep_budgets=[2, 4],
    )

    expected_files = {
        "snapshot_manifest",
        "normalized_scaffold",
        "split_manifest",
        "baseline_fixture",
        "reduced_fixture",
        "distractor_fixture",
        "thin_slice_result",
        "calibration_report",
        "sweep_report",
    }
    assert expected_files <= set(summary["artifacts"].keys())

    for path_str in summary["artifacts"].values():
        assert Path(path_str).exists()

    split_counts = summary["split_counts"]
    assert split_counts["train"] >= 1
    assert split_counts["eval"] >= 1
    assert split_counts["holdout"] >= 0
