"""Structured extraction contracts for deal-room documents."""

from __future__ import annotations

from pydantic import BaseModel, Field


class StructuredDocument(BaseModel):
    """A discovered structured document with extracted text."""

    source_path: str
    relative_path: str
    document_type: str
    extractor: str
    text: str = ""
    sheet_names: list[str] = Field(default_factory=list)


class ExtractedEvidence(BaseModel):
    """A provenance-preserving evidence snippet extracted from one document."""

    source_path: str
    relative_path: str
    document_type: str
    extractor: str
    snippet: str


class StructuredIngestionResult(BaseModel):
    """Structured extraction output for a directory."""

    documents: list[StructuredDocument] = Field(default_factory=list)
    evidence: list[ExtractedEvidence] = Field(default_factory=list)
