"""Embedding utilities powered by Sentence Transformers."""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Wrapper around a ``SentenceTransformer`` model."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        """Return the dimensionality of the embeddings produced by the model."""
        return int(self._model.get_sentence_embedding_dimension())

    @property
    def model(self) -> SentenceTransformer:
        return self._model

    def embed_documents(self, texts: Iterable[str]) -> np.ndarray:
        """Compute embeddings for an iterable of texts."""
        texts = list(texts)
        if not texts:
            return np.empty((0, self.dimension), dtype="float32")
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype="float32")

    def embed_query(self, query: str) -> np.ndarray:
        """Compute an embedding vector for a single query."""
        vector = self._model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(vector[0], dtype="float32")


@lru_cache(maxsize=1)
def get_embedding_service(model_name: str) -> EmbeddingService:
    """Cached factory for :class:`EmbeddingService` instances."""
    return EmbeddingService(model_name)
