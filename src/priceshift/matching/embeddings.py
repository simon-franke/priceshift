"""Sentence embeddings with disk cache using all-MiniLM-L6-v2."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Lazy import to avoid slow startup when embeddings not needed
_model: Optional[object] = None


def _get_model(model_name: str = "all-MiniLM-L6-v2") -> object:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        logger.info("Loading embedding model %s ...", model_name)
        _model = SentenceTransformer(model_name)
        logger.info("Model loaded.")
    return _model


class EmbeddingCache:
    """Disk-backed embedding cache keyed by (model, text) hash."""

    def __init__(self, cache_dir: str = ".cache/embeddings", model_name: str = "all-MiniLM-L6-v2") -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._model_name = model_name
        self._index_path = self._cache_dir / "index.json"
        self._index: dict[str, str] = self._load_index()

    def _load_index(self) -> dict[str, str]:
        if self._index_path.exists():
            with open(self._index_path) as f:
                return json.load(f)
        return {}

    def _save_index(self) -> None:
        with open(self._index_path, "w") as f:
            json.dump(self._index, f)

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self._model_name}:{text}".encode()).hexdigest()

    def _embedding_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.npy"

    def get(self, text: str) -> Optional[np.ndarray]:
        key = self._key(text)
        if key in self._index:
            path = self._embedding_path(key)
            if path.exists():
                return np.load(str(path))
        return None

    def put(self, text: str, embedding: np.ndarray) -> None:
        key = self._key(text)
        np.save(str(self._embedding_path(key)), embedding)
        self._index[key] = text[:100]  # store snippet for debugging
        self._save_index()

    def encode(self, text: str) -> np.ndarray:
        cached = self.get(text)
        if cached is not None:
            return cached
        model = _get_model(self._model_name)
        embedding = model.encode(text, normalize_embeddings=True)  # type: ignore
        self.put(text, embedding)
        return embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two normalized vectors."""
    return float(np.dot(a, b))
