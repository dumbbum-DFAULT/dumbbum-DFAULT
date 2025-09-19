"""FAISS-based vector store management."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

import faiss
import numpy as np

from .models import DocumentChunk


class VectorStoreError(RuntimeError):
    """Raised when interacting with the FAISS vector store fails."""


class FaissVectorStore:
    """Wrapper around a FAISS index with JSON metadata persistence."""

    def __init__(self, index_path: Path, metadata_path: Path, dimension: int) -> None:
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.dimension = dimension
        self.index = self._load_index()
        self.metadata: List[dict] = self._load_metadata()
        if self.index.ntotal != len(self.metadata):
            # Align metadata with index size by trimming excess entries.
            min_size = min(self.index.ntotal, len(self.metadata))
            if self.index.ntotal > min_size:
                self._shrink_index(min_size)
            self.metadata = self.metadata[:min_size]
            self._save_metadata()

    def _load_index(self) -> faiss.Index:
        if self.index_path.exists():
            return faiss.read_index(str(self.index_path))
        return faiss.IndexFlatIP(self.dimension)

    def _shrink_index(self, size: int) -> None:
        if size == self.index.ntotal:
            return
        xb = faiss.vector_to_array(self.index.xb)
        xb = xb.reshape(self.index.ntotal, self.dimension)[:size]
        new_index = faiss.IndexFlatIP(self.dimension)
        new_index.add(xb)
        self.index = new_index
        self._save_index()

    def _load_metadata(self) -> List[dict]:
        if not self.metadata_path.exists():
            return []
        with self.metadata_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _save_index(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))

    def _save_metadata(self) -> None:
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metadata_path.open("w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def add_documents(self, embeddings: np.ndarray, chunks: Sequence[DocumentChunk]) -> None:
        if embeddings.size == 0 or not chunks:
            return
        if embeddings.shape[1] != self.dimension:
            raise VectorStoreError("Embedding dimensionality does not match the index.")
        if embeddings.shape[0] != len(chunks):
            raise VectorStoreError("Number of embeddings must equal number of document chunks.")

        records = [
            {
                "content": chunk.content,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ]

        self.index.add(embeddings)
        self.metadata.extend(records)
        self._save_index()
        self._save_metadata()

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[dict]:
        if self.index.ntotal == 0:
            return []
        query = np.asarray([query_embedding], dtype="float32")
        top_k = max(1, min(top_k, self.index.ntotal))
        scores, indices = self.index.search(query, top_k)
        results: List[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            record = dict(self.metadata[idx])
            record["score"] = float(score)
            results.append(record)
        return results


def get_vector_store(dimension: int) -> FaissVectorStore:
    from . import config

    return FaissVectorStore(config.FAISS_INDEX_PATH, config.METADATA_PATH, dimension)
