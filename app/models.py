"""Data models for the research assistant application."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


@dataclass
class DocumentChunk:
    """A chunk of text extracted from a source document."""

    content: str
    metadata: Dict[str, Any]


class GoogleDriveIngestRequest(BaseModel):
    """Request payload for Google Drive ingestion."""

    folder_id: str = Field(..., description="Google Drive folder ID to ingest")


class NotionIngestRequest(BaseModel):
    """Placeholder request payload for Notion ingestion."""

    workspace_id: Optional[str] = Field(
        None,
        description="Notion workspace identifier (placeholder).",
    )


class CanvaIngestRequest(BaseModel):
    """Placeholder request payload for Canva ingestion."""

    brand_kit_id: Optional[str] = Field(
        None,
        description="Canva brand kit identifier (placeholder).",
    )


class QueryRequest(BaseModel):
    """Request payload for similarity search queries."""

    query: str = Field(..., description="Natural language question to search against the knowledge base.")
    top_k: int = Field(5, description="Number of results to return.")


class QueryResult(BaseModel):
    """Response schema for similarity search results."""

    content: str
    score: float
    metadata: Dict[str, Any]


class QueryResponse(BaseModel):
    """Response payload containing ranked document chunks."""

    results: List[QueryResult]
