"""FastAPI application exposing research assistant capabilities."""
from __future__ import annotations

from typing import Dict

from fastapi import FastAPI, HTTPException

from . import config
from .embedding import get_embedding_service
from .ingestion import (
    GoogleDriveIngestionError,
    ingest_from_canva_placeholder,
    ingest_from_google_drive,
    ingest_from_notion_placeholder,
)
from .models import (
    CanvaIngestRequest,
    GoogleDriveIngestRequest,
    NotionIngestRequest,
    QueryRequest,
    QueryResponse,
    QueryResult,
)
from .vector_store import FaissVectorStore, VectorStoreError, get_vector_store

app = FastAPI(
    title="Research Assistant API",
    version="0.1.0",
    description="Ingests documents into a FAISS vector store and provides semantic search.",
)

_embedding_service = None
_vector_store: FaissVectorStore | None = None


@app.on_event("startup")
def startup_event() -> None:
    """Initialize the embedding service and FAISS vector store."""
    global _embedding_service, _vector_store  # noqa: PLW0603
    _embedding_service = get_embedding_service(config.EMBEDDING_MODEL_NAME)
    _vector_store = get_vector_store(_embedding_service.dimension)


@app.post("/ingest/google-drive")
def ingest_google_drive_endpoint(request: GoogleDriveIngestRequest) -> Dict[str, int]:
    """Ingest documents from a Google Drive folder into the vector store."""
    if _embedding_service is None or _vector_store is None:
        raise HTTPException(status_code=503, detail="Embedding service not initialized.")

    try:
        chunks = ingest_from_google_drive(request.folder_id)
    except GoogleDriveIngestionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    embeddings = _embedding_service.embed_documents(chunk.content for chunk in chunks)
    try:
        _vector_store.add_documents(embeddings, chunks)
    except VectorStoreError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"chunks_ingested": len(chunks)}


@app.post("/ingest/notion")
def ingest_notion_endpoint(_: NotionIngestRequest) -> Dict[str, str]:
    """Placeholder endpoint for Notion ingestion."""
    ingest_from_notion_placeholder()
    return {"status": "Notion ingestion is not yet implemented."}


@app.post("/ingest/canva")
def ingest_canva_endpoint(_: CanvaIngestRequest) -> Dict[str, str]:
    """Placeholder endpoint for Canva ingestion."""
    ingest_from_canva_placeholder()
    return {"status": "Canva ingestion is not yet implemented."}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest) -> QueryResponse:
    """Perform a semantic search against the FAISS vector store."""
    if _embedding_service is None or _vector_store is None:
        raise HTTPException(status_code=503, detail="Embedding service not initialized.")

    if not request.query:
        raise HTTPException(status_code=400, detail="Query text must be provided.")

    query_embedding = _embedding_service.embed_query(request.query)
    records = _vector_store.search(query_embedding, request.top_k)
    results = [
        QueryResult(
            content=record.get("content", ""),
            score=record.get("score", 0.0),
            metadata=record.get("metadata", {}),
        )
        for record in records
    ]
    return QueryResponse(results=results)


@app.get("/")
def root() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}
