from __future__ import annotations

import json
import re
from calendar import monthrange
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def _extract_year(name: str) -> str | None:
    match = re.search(r"(20\d{2})", name)
    return match.group(1) if match else None


def _extract_yyyymm(name: str) -> str | None:
    match = re.search(r"(?<!\d)(20\d{2}(0[1-9]|1[0-2]))(?!\d)", name)
    return match.group(1) if match else None


def _read_meta_date(path: Path) -> str | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    value = data.get("filing_date")
    return value if isinstance(value, str) else None


def build_snapshot_manifest(data_dir: Path, output_path: Path) -> dict[str, object]:
    fannie_root = data_dir / "fannie"
    ginnie_root = data_dir / "ginnie"
    edgar_root = data_dir / "edgar" / "cmbs"

    fannie_zips = sorted(fannie_root.glob("*.zip")) if fannie_root.exists() else []
    ginnie_zips = sorted(ginnie_root.rglob("*.zip")) if ginnie_root.exists() else []
    edgar_xml = sorted(edgar_root.rglob("*.xml")) if edgar_root.exists() else []
    edgar_meta = sorted(edgar_root.rglob("*.meta.json")) if edgar_root.exists() else []
    tracked_paths = [*fannie_zips, *ginnie_zips, *edgar_xml, *edgar_meta]

    fannie_years = sorted({year for path in fannie_zips if (year := _extract_year(path.name))})
    ginnie_periods = sorted(
        {
            period
            for path in ginnie_zips
            if (period := _extract_yyyymm(path.name) or _extract_yyyymm(path.parent.name))
        }
    )
    edgar_dates = sorted({d for path in edgar_meta if (d := _read_meta_date(path))})

    date_start: str | None = None
    date_end: str | None = None

    all_dates: list[str] = []
    if fannie_years:
        all_dates.append(f"{fannie_years[0]}-01-01")
        all_dates.append(f"{fannie_years[-1]}-12-31")
    if ginnie_periods:
        start_period = ginnie_periods[0]
        end_period = ginnie_periods[-1]
        end_year = int(end_period[:4])
        end_month = int(end_period[4:])
        end_day = monthrange(end_year, end_month)[1]
        all_dates.append(f"{start_period[:4]}-{start_period[4:]}-01")
        all_dates.append(f"{end_period[:4]}-{end_period[4:]}-{end_day:02d}")
    all_dates.extend(edgar_dates)

    if all_dates:
        date_start = min(all_dates)
        date_end = max(all_dates)

    manifest: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "data_root": str(data_dir),
        "sources": {
            "fannie": {
                "zip_files": len(fannie_zips),
                "bytes": sum(path.stat().st_size for path in fannie_zips),
                "year_range": [fannie_years[0], fannie_years[-1]] if fannie_years else None,
            },
            "ginnie": {
                "zip_files": len(ginnie_zips),
                "bytes": sum(path.stat().st_size for path in ginnie_zips),
                "period_range": [ginnie_periods[0], ginnie_periods[-1]] if ginnie_periods else None,
            },
            "edgar": {
                "xml_files": len(edgar_xml),
                "meta_files": len(edgar_meta),
                "bytes": sum(path.stat().st_size for path in edgar_xml + edgar_meta),
                "date_range": [edgar_dates[0], edgar_dates[-1]] if edgar_dates else None,
            },
        },
        "totals": {
            "files": len(fannie_zips) + len(ginnie_zips) + len(edgar_xml) + len(edgar_meta),
            "bytes": sum(path.stat().st_size for path in tracked_paths),
            "date_range": [date_start, date_end] if date_start and date_end else None,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _edgar_records(edgar_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(edgar_root.rglob("*.meta.json")):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        accession = meta.get("accession")
        filing_date = meta.get("filing_date")
        if not isinstance(accession, str) or not isinstance(filing_date, str):
            continue

        records.append(
            {
                "source": "edgar",
                "dataset": "cmbs_abs_ee",
                "record_id": accession,
                "event_date": filing_date,
                "file_path": str(path),
                "metadata": {
                    "cik": meta.get("cik"),
                    "entity_name": meta.get("entity_name"),
                    "source_url": meta.get("source_url"),
                },
            }
        )
    return records


def _fannie_records(fannie_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(fannie_root.glob("*.zip")):
        year = _extract_year(path.name)
        event_date = f"{year}-01-01" if year else None
        records.append(
            {
                "source": "fannie",
                "dataset": "fhfa_enterprise_pudb",
                "record_id": path.stem,
                "event_date": event_date,
                "file_path": str(path),
                "metadata": {"file_name": path.name, "size_bytes": path.stat().st_size},
            }
        )
    return records


def _ginnie_records(ginnie_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(ginnie_root.rglob("*.zip")):
        period = _extract_yyyymm(path.name) or _extract_yyyymm(path.parent.name)
        event_date = f"{period[:4]}-{period[4:]}-01" if period and len(period) == 6 else None
        records.append(
            {
                "source": "ginnie",
                "dataset": "disclosure_data_bulk",
                "record_id": path.stem,
                "event_date": event_date,
                "file_path": str(path),
                "metadata": {
                    "file_name": path.name,
                    "period": period,
                    "size_bytes": path.stat().st_size,
                },
            }
        )
    return records


def build_normalized_scaffold(data_dir: Path, output_path: Path) -> int:
    records: list[dict[str, object]] = []

    fannie_root = data_dir / "fannie"
    ginnie_root = data_dir / "ginnie"
    edgar_root = data_dir / "edgar" / "cmbs"

    if fannie_root.exists():
        records.extend(_fannie_records(fannie_root))
    if ginnie_root.exists():
        records.extend(_ginnie_records(ginnie_root))
    if edgar_root.exists():
        records.extend(_edgar_records(edgar_root))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    return len(records)


def _assign_split(event_date: str | None) -> str:
    if event_date is None:
        return "unspecified"
    if event_date <= "2023-12-31":
        return "train"
    if event_date <= "2025-12-31":
        return "eval"
    return "holdout"


def build_time_split_manifest(scaffold_path: Path, output_path: Path) -> dict[str, object]:
    counts: dict[str, int] = {"train": 0, "eval": 0, "holdout": 0, "unspecified": 0}
    by_source: dict[str, dict[str, int]] = {
        "train": {},
        "eval": {},
        "holdout": {},
        "unspecified": {},
    }
    membership: dict[str, str] = {}

    with scaffold_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            source = record.get("source")
            source_name = source if isinstance(source, str) else "unknown"
            record_id = record.get("record_id")
            event_date = record.get("event_date")
            split = _assign_split(event_date if isinstance(event_date, str) else None)

            counts[split] += 1
            by_source[split][source_name] = by_source[split].get(source_name, 0) + 1
            if isinstance(record_id, str):
                membership[record_id] = split

    manifest: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "scaffold_path": str(scaffold_path),
        "policy": {
            "train_lte": "2023-12-31",
            "eval_from": "2024-01-01",
            "eval_lte": "2025-12-31",
            "holdout_gte": "2026-01-01",
            "unspecified": "records without event_date",
        },
        "counts": counts,
        "by_source": by_source,
        "membership": membership,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _select_records_by_split(
    records: list[dict[str, object]],
    membership: dict[str, str],
    per_split: int,
) -> list[dict[str, object]]:
    split_order = ["train", "eval", "holdout", "unspecified"]
    selected: list[dict[str, object]] = []
    by_split: dict[str, list[dict[str, object]]] = {k: [] for k in split_order}

    for record in records:
        record_id = record.get("record_id")
        if not isinstance(record_id, str):
            continue
        split = membership.get(record_id, "unspecified")
        if split not in by_split:
            split = "unspecified"
        by_split[split].append(record)

    for split in split_order:
        selected.extend(by_split[split][:per_split])
    return selected


def build_research_fixtures(
    scaffold_path: Path,
    split_manifest_path: Path,
    output_dir: Path,
    per_split: int = 200,
) -> dict[str, Path]:
    if not isinstance(per_split, int) or isinstance(per_split, bool) or per_split < 0:
        raise ValueError("per_split must be a non-negative integer")

    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    membership_raw = split_manifest.get("membership")
    membership: dict[str, str] = {}
    if isinstance(membership_raw, dict):
        membership = {
            key: value
            for key, value in membership_raw.items()
            if isinstance(key, str) and isinstance(value, str)
        }

    records: list[dict[str, object]] = []
    with scaffold_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))

    selected = _select_records_by_split(records, membership, per_split)

    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = output_dir / "baseline.ndjson"
    reduced_path = output_dir / "reduced_context.ndjson"
    distractor_path = output_dir / "distractor.ndjson"

    with (
        baseline_path.open("w", encoding="utf-8") as baseline,
        reduced_path.open("w", encoding="utf-8") as reduced,
        distractor_path.open("w", encoding="utf-8") as distractor,
    ):
        for record in selected:
            baseline_record = dict(record)
            baseline_record["profile"] = "baseline"
            baseline.write(json.dumps(baseline_record, separators=(",", ":")) + "\n")

            reduced_record = dict(record)
            reduced_record["profile"] = "reduced_context"
            metadata = reduced_record.get("metadata")
            if isinstance(metadata, dict):
                string_keys = sorted(key for key in metadata if isinstance(key, str))
                if string_keys:
                    selected_key = string_keys[0]
                    reduced_record["metadata"] = {selected_key: metadata[selected_key]}
                else:
                    reduced_record["metadata"] = {}
            reduced.write(json.dumps(reduced_record, separators=(",", ":")) + "\n")

            distractor_record = dict(record)
            distractor_record["profile"] = "distractor"
            distractor_record["distractor"] = True
            distractor_record["distractor_note"] = "synthetic distractor context placeholder"
            distractor.write(json.dumps(distractor_record, separators=(",", ":")) + "\n")

    return {
        "baseline": baseline_path,
        "reduced_context": reduced_path,
        "distractor": distractor_path,
    }
