"""Sentence embeddings with an on-disk cache."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = Path("data/cache")

EncodeFn = Callable[[Sequence[str]], np.ndarray]


def _sentence_transformer_encode(model_id: str) -> EncodeFn:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id)
    return lambda texts: model.encode(list(texts), batch_size=64, show_progress_bar=False)


class Embedder:
    """Embeds texts; whole-call results are cached under cache_dir keyed by content hash."""

    def __init__(
        self,
        encode_fn: EncodeFn | None = None,
        cache_dir: Path = CACHE_DIR,
        model_id: str = MODEL_ID,
    ) -> None:
        self._encode_fn = encode_fn
        self.cache_dir = Path(cache_dir)
        self.model_id = model_id

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._encode_fn is None:
            self._encode_fn = _sentence_transformer_encode(self.model_id)
        return np.asarray(self._encode_fn(texts), dtype=np.float32)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        h = hashlib.sha256(self.model_id.encode())
        for t in texts:
            h.update(t.encode())
            h.update(b"\x00")
        path = self.cache_dir / f"emb_{h.hexdigest()[:24]}.npy"
        if path.exists():
            return np.load(path)
        out = self._encode(texts)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, out)
        return out
