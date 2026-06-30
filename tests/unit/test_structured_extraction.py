"""Tests for structured extraction baseline."""

from __future__ import annotations

from zipfile import ZIP_DEFLATED, ZipFile

from engram.services.structured_extraction import extract_structured_directory


def _write_docx(path, text: str) -> None:  # type: ignore[no-untyped-def]
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "word/document.xml",
            (
                "<?xml version='1.0' encoding='UTF-8'?>"
                "<w:document xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'>"
                f"<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>"
            ),
        )


def _write_xlsx(path) -> None:  # type: ignore[no-untyped-def]
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "xl/workbook.xml",
            (
                "<?xml version='1.0' encoding='UTF-8'?>"
                "<workbook xmlns='http://schemas.openxmlformats.org/spreadsheetml/2006/main' "
                "xmlns:r='http://schemas.openxmlformats.org/officeDocument/2006/relationships'>"
                "<sheets><sheet name='Rent Roll' sheetId='1' r:id='rId1'/></sheets></workbook>"
            ),
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            (
                "<?xml version='1.0' encoding='UTF-8'?>"
                "<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'>"
                "<Relationship Id='rId1' "
                "Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet' "
                "Target='worksheets/sheet1.xml'/></Relationships>"
            ),
        )
        archive.writestr(
            "xl/sharedStrings.xml",
            (
                "<?xml version='1.0' encoding='UTF-8'?>"
                "<sst xmlns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'>"
                "<si><t>Rent Roll Summary</t></si><si><t>Lease Charges</t></si></sst>"
            ),
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            (
                "<?xml version='1.0' encoding='UTF-8'?>"
                "<worksheet xmlns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'>"
                "<sheetData><row r='1'><c t='s'><v>0</v></c><c t='s'><v>1</v></c></row></sheetData>"
                "</worksheet>"
            ),
        )


def test_extract_structured_directory_reads_docx_and_xlsx(tmp_path) -> None:
    docx_path = tmp_path / "deal-summary.docx"
    xlsx_path = tmp_path / "rent-roll.xlsx"
    _write_docx(docx_path, "Purchase and sale agreement summary")
    _write_xlsx(xlsx_path)

    result = extract_structured_directory(tmp_path)

    assert len(result.documents) == 2
    assert {doc.document_type for doc in result.documents} == {"docx", "xlsx"}
    assert any(evidence.document_type == "docx" for evidence in result.evidence)
    assert any(evidence.document_type == "xlsx" for evidence in result.evidence)


def test_extract_structured_directory_preserves_provenance_fields(tmp_path) -> None:
    pdf_path = tmp_path / "2022 Property Tax Bill.pdf"
    pdf_path.write_text("placeholder")

    result = extract_structured_directory(tmp_path)

    evidence = result.evidence[0]
    assert evidence.source_path.endswith("2022 Property Tax Bill.pdf")
    assert evidence.relative_path == "2022 Property Tax Bill.pdf"
    assert evidence.document_type == "pdf"
    assert evidence.extractor in {"inventory", "docx", "xlsx"}
    assert evidence.snippet
