"""Deterministic baseline extraction for structured deal-room documents."""

from __future__ import annotations

from typing import TYPE_CHECKING
from xml.etree import ElementTree
from zipfile import ZipFile

from engram.models.structured import (
    ExtractedEvidence,
    StructuredDocument,
    StructuredIngestionResult,
)

if TYPE_CHECKING:
    from pathlib import Path


def extract_structured_directory(directory: Path) -> StructuredIngestionResult:
    documents: list[StructuredDocument] = []
    evidence: list[ExtractedEvidence] = []

    for file_path in sorted(directory.rglob("*")):
        if not file_path.is_file() or file_path.name.startswith("."):
            continue
        relative_path = str(file_path.relative_to(directory))
        suffix = file_path.suffix.casefold()
        if suffix == ".docx":
            document = _extract_docx(file_path, relative_path)
        elif suffix in {".xlsx", ".xls"}:
            document = _extract_xlsx(file_path, relative_path)
        else:
            document = _inventory_document(file_path, relative_path)

        documents.append(document)
        evidence.append(
            ExtractedEvidence(
                source_path=document.source_path,
                relative_path=document.relative_path,
                document_type=document.document_type,
                extractor=document.extractor,
                snippet=document.text or document.relative_path,
            )
        )

    return StructuredIngestionResult(documents=documents, evidence=evidence)


def _inventory_document(file_path: Path, relative_path: str) -> StructuredDocument:
    return StructuredDocument(
        source_path=str(file_path),
        relative_path=relative_path,
        document_type=file_path.suffix.casefold().lstrip(".") or "unknown",
        extractor="inventory",
        text=f"Structured diligence file available: {relative_path}",
    )


def _extract_docx(file_path: Path, relative_path: str) -> StructuredDocument:
    with ZipFile(file_path) as archive:
        xml_text = archive.read("word/document.xml").decode("utf-8")
    root = ElementTree.fromstring(xml_text)
    text = " ".join(node.text for node in root.iter() if node.text).strip()
    return StructuredDocument(
        source_path=str(file_path),
        relative_path=relative_path,
        document_type="docx",
        extractor="docx",
        text=text,
    )


def _extract_xlsx(file_path: Path, relative_path: str) -> StructuredDocument:
    with ZipFile(file_path) as archive:
        workbook_xml = archive.read("xl/workbook.xml").decode("utf-8")
        shared_strings = _read_shared_strings(archive)
        sheet_names = _read_sheet_names(workbook_xml)
        worksheet_text = _read_first_sheet_text(archive, shared_strings)

    return StructuredDocument(
        source_path=str(file_path),
        relative_path=relative_path,
        document_type="xlsx",
        extractor="xlsx",
        text=worksheet_text,
        sheet_names=sheet_names,
    )


def _read_shared_strings(archive: ZipFile) -> list[str]:
    try:
        root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml").decode("utf-8"))
    except KeyError:
        return []
    return [node.text or "" for node in root.iter() if node.tag.endswith("}t")]


def _read_sheet_names(workbook_xml: str) -> list[str]:
    root = ElementTree.fromstring(workbook_xml)
    return [node.attrib.get("name", "") for node in root.iter() if node.tag.endswith("}sheet")]


def _read_first_sheet_text(archive: ZipFile, shared_strings: list[str]) -> str:
    root = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml").decode("utf-8"))
    values: list[str] = []
    for cell in root.iter():
        if not cell.tag.endswith("}c"):
            continue
        value_node = next((child for child in cell if child.tag.endswith("}v")), None)
        if value_node is None or value_node.text is None:
            continue
        if cell.attrib.get("t") == "s":
            index = int(value_node.text)
            values.append(shared_strings[index])
        else:
            values.append(value_node.text)
    return " ".join(value for value in values if value).strip()
