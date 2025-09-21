"""Utilities for extracting and chunking text from various document formats."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List
import importlib

from . import config


class UnsupportedDocumentTypeError(ValueError):
    """Raised when attempting to parse an unsupported document type."""


def extract_text_from_pdf(path: Path) -> str:
    """Extract text content from a PDF file using ``PyPDF2``.

    Parameters
    ----------
    path:
        Path to the PDF file to extract.
    """
    pypdf2 = importlib.import_module("PyPDF2")
    reader = pypdf2.PdfReader(str(path))
    parts: List[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        parts.append(page_text)
    return "\n".join(parts)


def extract_text_from_docx(path: Path) -> str:
    """Extract text from a Microsoft Word document using ``python-docx``."""
    docx = importlib.import_module("docx")
    document = docx.Document(str(path))
    parts = [paragraph.text for paragraph in document.paragraphs if paragraph.text]
    return "\n".join(parts)


def extract_text_from_txt(path: Path) -> str:
    """Read text content from a plain text file."""
    return path.read_text(encoding="utf-8", errors="ignore")


EXTRACTION_FUNCTIONS = {
    ".pdf": extract_text_from_pdf,
    ".docx": extract_text_from_docx,
    ".txt": extract_text_from_txt,
}


def extract_text(path: Path) -> str:
    """Dispatch to the appropriate extractor based on file extension."""
    suffix = path.suffix.lower()
    if suffix not in EXTRACTION_FUNCTIONS:
        raise UnsupportedDocumentTypeError(f"Unsupported document type: {suffix}")
    return EXTRACTION_FUNCTIONS[suffix](path)


def chunk_text(
    text: str,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> Iterable[str]:
    """Yield text chunks of approximately ``chunk_size`` characters.

    Parameters
    ----------
    text:
        Input text to chunk.
    chunk_size:
        Maximum number of characters per chunk. Defaults to ``config.CHUNK_SIZE``.
    chunk_overlap:
        Number of overlapping characters between consecutive chunks. Defaults to
        ``config.CHUNK_OVERLAP``.
    """
    if not text:
        return []

    size = chunk_size or config.CHUNK_SIZE
    overlap = chunk_overlap or config.CHUNK_OVERLAP
    if size <= overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap")

    clean_text = text.replace("\r", " ").strip()
    if not clean_text:
        return []

    chunks: List[str] = []
    start = 0
    text_length = len(clean_text)
    while start < text_length:
        end = min(start + size, text_length)
        chunk = clean_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks
