from __future__ import annotations

import json
import zipfile
from typing import TYPE_CHECKING

from engram.services.research_data import (
    build_normalized_scaffold,
    build_research_fixtures,
    build_snapshot_manifest,
    build_time_split_manifest,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_zip(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, mode="w") as zf:
        zf.writestr("sample.txt", "ok")


def test_build_snapshot_manifest_counts_and_sizes(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_zip(data_dir / "fannie" / "2023_enterprise_pudb.zip")
    _write_zip(data_dir / "ginnie" / "202602" / "monthlySFS_202602.zip")

    edgar_dir = data_dir / "edgar" / "cmbs" / "1713393"
    edgar_dir.mkdir(parents=True, exist_ok=True)
    (edgar_dir / "filing.xml").write_text("<root />", encoding="utf-8")
    (edgar_dir / "filing.meta.json").write_text(
        json.dumps({"accession": "0000950131-17-000939", "filing_date": "2017-12-28"}),
        encoding="utf-8",
    )

    out_path = tmp_path / "manifests" / "research_snapshot.json"
    manifest = build_snapshot_manifest(data_dir, out_path)

    assert out_path.exists()
    assert manifest["sources"]["fannie"]["zip_files"] == 1
    assert manifest["sources"]["ginnie"]["zip_files"] == 1
    assert manifest["sources"]["edgar"]["xml_files"] == 1
    assert manifest["sources"]["edgar"]["meta_files"] == 1
    assert manifest["totals"]["bytes"] > 0
    assert manifest["totals"]["date_range"] == ["2017-12-28", "2026-02-28"]


def test_build_snapshot_manifest_uses_month_end_for_ginnie_range(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_zip(data_dir / "ginnie" / "202602" / "monthlySFS_202602.zip")
    _write_zip(data_dir / "ginnie" / "202604" / "monthlySFS_202604.zip")

    out_path = tmp_path / "manifests" / "research_snapshot.json"
    manifest = build_snapshot_manifest(data_dir, out_path)

    assert manifest["totals"]["date_range"] == ["2026-02-01", "2026-04-30"]


def test_build_normalized_scaffold_outputs_ndjson(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"

    _write_zip(data_dir / "fannie" / "2023_enterprise_pudb.zip")
    _write_zip(data_dir / "ginnie" / "202602" / "monthlySFS_202602.zip")

    edgar_dir = data_dir / "edgar" / "cmbs" / "1713393"
    edgar_dir.mkdir(parents=True, exist_ok=True)
    meta_path = edgar_dir / "000095013117000939_2017-12-28.meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "cik": "1713393",
                "accession": "0000950131-17-000939",
                "filing_date": "2017-12-28",
                "source_url": "https://www.sec.gov/Archives/example.xml",
            }
        ),
        encoding="utf-8",
    )

    out_path = tmp_path / "normalized" / "research_scaffold.ndjson"
    record_count = build_normalized_scaffold(data_dir, out_path)

    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert record_count == 3
    assert len(lines) == 3

    records = [json.loads(line) for line in lines]
    sources = {r["source"] for r in records}
    assert sources == {"fannie", "ginnie", "edgar"}

    edgar_record = next(r for r in records if r["source"] == "edgar")
    assert edgar_record["record_id"] == "0000950131-17-000939"
    assert edgar_record["event_date"] == "2017-12-28"


def test_build_normalized_scaffold_extracts_ginnie_period_from_directory(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_zip(data_dir / "ginnie" / "202602" / "disclosure_data_bulk.zip")

    out_path = tmp_path / "normalized" / "research_scaffold.ndjson"
    build_normalized_scaffold(data_dir, out_path)

    records = [
        json.loads(line) for line in out_path.read_text(encoding="utf-8").strip().splitlines()
    ]
    assert len(records) == 1
    assert records[0]["source"] == "ginnie"
    assert records[0]["event_date"] == "2026-02-01"
    assert records[0]["metadata"]["period"] == "202602"


def test_build_time_split_manifest_counts(tmp_path: Path) -> None:
    scaffold_path = tmp_path / "research_scaffold.ndjson"
    records = [
        {
            "source": "fannie",
            "dataset": "fhfa_enterprise_pudb",
            "record_id": "f2019",
            "event_date": "2019-06-01",
            "file_path": "data/fannie/2019.zip",
            "metadata": {},
        },
        {
            "source": "ginnie",
            "dataset": "disclosure_data_bulk",
            "record_id": "g2024",
            "event_date": "2024-04-01",
            "file_path": "data/ginnie/202404/file.zip",
            "metadata": {},
        },
        {
            "source": "edgar",
            "dataset": "cmbs_abs_ee",
            "record_id": "e2026",
            "event_date": "2026-02-15",
            "file_path": "data/edgar/cmbs/x.xml",
            "metadata": {},
        },
        {
            "source": "edgar",
            "dataset": "cmbs_abs_ee",
            "record_id": "missing-date",
            "event_date": None,
            "file_path": "data/edgar/cmbs/y.xml",
            "metadata": {},
        },
    ]
    scaffold_path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    out_path = tmp_path / "manifests" / "research_splits.json"
    splits = build_time_split_manifest(scaffold_path, out_path)

    assert out_path.exists()
    assert splits["counts"] == {"train": 1, "eval": 1, "holdout": 1, "unspecified": 1}
    assert splits["by_source"]["train"]["fannie"] == 1
    assert splits["by_source"]["eval"]["ginnie"] == 1
    assert splits["by_source"]["holdout"]["edgar"] == 1
    assert splits["by_source"]["unspecified"]["edgar"] == 1


def test_build_time_split_manifest_boundaries(tmp_path: Path) -> None:
    scaffold_path = tmp_path / "research_scaffold.ndjson"
    records = [
        {"source": "edgar", "record_id": "a", "event_date": "2023-12-31"},
        {"source": "edgar", "record_id": "b", "event_date": "2024-01-01"},
        {"source": "edgar", "record_id": "c", "event_date": "2025-12-31"},
        {"source": "edgar", "record_id": "d", "event_date": "2026-01-01"},
    ]
    scaffold_path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    out_path = tmp_path / "manifests" / "research_splits.json"
    splits = build_time_split_manifest(scaffold_path, out_path)

    assert splits["membership"]["a"] == "train"
    assert splits["membership"]["b"] == "eval"
    assert splits["membership"]["c"] == "eval"
    assert splits["membership"]["d"] == "holdout"


def test_build_research_fixtures_writes_three_profiles(tmp_path: Path) -> None:
    scaffold_path = tmp_path / "research_scaffold.ndjson"
    records = [
        {
            "source": "edgar",
            "dataset": "cmbs_abs_ee",
            "record_id": "id-1",
            "event_date": "2022-01-01",
            "file_path": "a.xml",
            "metadata": {"foo": "bar", "baz": "qux"},
        },
        {
            "source": "ginnie",
            "dataset": "disclosure_data_bulk",
            "record_id": "id-2",
            "event_date": "2026-01-02",
            "file_path": "b.zip",
            "metadata": {"k": "v"},
        },
    ]
    scaffold_path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    splits = {
        "membership": {"id-1": "train", "id-2": "holdout"},
        "counts": {"train": 1, "eval": 0, "holdout": 1, "unspecified": 0},
    }
    split_path = tmp_path / "research_splits.json"
    split_path.write_text(json.dumps(splits), encoding="utf-8")

    out_dir = tmp_path / "fixtures"
    outputs = build_research_fixtures(scaffold_path, split_path, out_dir, per_split=10)

    assert set(outputs.keys()) == {"baseline", "reduced_context", "distractor"}
    baseline_lines = outputs["baseline"].read_text(encoding="utf-8").strip().splitlines()
    reduced_lines = outputs["reduced_context"].read_text(encoding="utf-8").strip().splitlines()
    distractor_lines = outputs["distractor"].read_text(encoding="utf-8").strip().splitlines()
    assert len(baseline_lines) == 2
    assert len(reduced_lines) == 2
    assert len(distractor_lines) == 2

    baseline_record = json.loads(baseline_lines[0])
    reduced_record = json.loads(reduced_lines[0])
    distractor_record = json.loads(distractor_lines[0])

    assert baseline_record["profile"] == "baseline"
    assert reduced_record["profile"] == "reduced_context"
    assert distractor_record["profile"] == "distractor"
    assert reduced_record["metadata"] == {"baz": "qux"}
    assert distractor_record["distractor"] is True
