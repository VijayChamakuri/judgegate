"""Retrieve similar *past* bookings to give the agent and judge context.

Embedding: `BAAI/bge-small-en-v1.5` via fastembed (local ONNX, cached by text hash), the same
model TrustGate uses. For a line in month m, neighbours come only from months strictly before m,
so no future label can leak into a prompt (tested in tests/test_retrieval.py).
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from .data import DATA

MODEL = "BAAI/bge-small-en-v1.5"


def line_text(row: pd.Series | dict) -> str:
    return f"{row['merchant_raw']} | {row['department']} | {row['service_category']}"


class Embedder:
    def __init__(self, model: str = MODEL):
        self.model_name = model
        self.cache_file = DATA / f"embeddings_{model.replace('/', '__')}.npz"
        self._cache: dict[str, np.ndarray] = {}
        if self.cache_file.exists():
            z = np.load(self.cache_file)
            self._cache = dict(zip(z["keys"].tolist(), z["vecs"], strict=True))
        self._model: Any = None

    def encode(self, texts: list[str]) -> np.ndarray:
        keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
        missing = {k: t for k, t in zip(keys, texts, strict=True) if k not in self._cache}
        if missing:
            if self._model is None:
                from fastembed import TextEmbedding

                self._model = TextEmbedding(self.model_name)
            items = sorted(missing.items())
            for (k, _), v in zip(items, self._model.embed([t for _, t in items], batch_size=256), strict=True):
                self._cache[k] = np.asarray(v, dtype=np.float32)
            np.savez(self.cache_file, keys=np.array(list(self._cache)), vecs=np.stack(list(self._cache.values())))
        out = np.stack([self._cache[k] for k in keys])
        return out / np.linalg.norm(out, axis=1, keepdims=True).clip(min=1e-9)


def past_neighbours(df: pd.DataFrame, targets: pd.DataFrame, k: int = 8, emb: Embedder | None = None) -> dict[str, list[int]]:
    """For each target line: row indices (into df) of its k most similar lines from earlier months."""
    emb = emb or Embedder()
    vecs = emb.encode([line_text(r) for r in df.to_dict("records")])
    out: dict[str, list[int]] = {}
    for month, grp in targets.groupby("statement_month"):
        past_idx = np.flatnonzero((df.statement_month < month).to_numpy())
        if len(past_idx) == 0:
            continue
        P = vecs[past_idx]
        Q = vecs[grp.index.to_numpy()]
        sims = Q @ P.T
        top = np.argsort(-sims, axis=1)[:, :k]
        for row_id, idxs in zip(grp.card_line_id, top, strict=True):
            out[row_id] = [int(past_idx[i]) for i in idxs]
    return out
