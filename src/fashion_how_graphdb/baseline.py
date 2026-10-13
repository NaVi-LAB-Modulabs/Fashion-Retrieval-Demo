"""CLIP text-to-image baseline: the raw query against stored item image embeddings."""

from __future__ import annotations

import importlib.util
import os
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

# Item.image_embedding was produced by this sentence-transformers model (fp32,
# normalize_embeddings=True); the query must come from the same model's text tower.
CLIP_MODEL = "clip-ViT-L-14"

# That text tower exported by scripts/export_clip_text_onnx.py: fp16 weights, fp32 arithmetic.
# Query embeddings match the torch model at cosine >= 0.9999998, without torch at runtime.
TEXT_MODEL_PATH = Path(os.getenv("CLIP_TEXT_MODEL_PATH") or ROOT / "models" / "clip_text_fp16w.onnx")
TOKENIZER_PATH = ROOT / "models" / "clip_tokenizer.json"
MAX_TOKENS = 77

# Exact scan over every item (no ANN index, no filters). Neo4j returns
# (1 + cosine) / 2, so the score is rescaled to raw cosine in [-1, 1].
BASELINE_CYPHER = (
    "MATCH (item:Item)\n"
    "WHERE item.image_embedding_model = $embedding_model\n"
    "  AND item.image_embedding IS NOT NULL\n"
    "WITH item, 2 * vector.similarity.cosine(item.image_embedding, $query_embedding) - 1 AS score\n"
    "RETURN item.id AS id, item.item_ID AS item_ID, item.category AS category,\n"
    "  item.category_name AS category_name, score\n"
    "ORDER BY score DESC, id ASC\n"
    "LIMIT $limit"
)

_encoder_lock = Lock()


def encoder_available() -> bool:
    return (
        TEXT_MODEL_PATH.is_file() and TOKENIZER_PATH.is_file()
        and _runtime_installed()
    )


@lru_cache(maxsize=1)
def _runtime_installed() -> bool:
    return all(importlib.util.find_spec(name) for name in ("onnxruntime", "tokenizers", "numpy"))


class ClipTextEncoder:
    def __init__(self, model_path: Path, tokenizer_path: Path):
        import onnxruntime
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        # CLIP's context is 77 tokens; longer queries are cut, keeping the end-of-text token.
        self.tokenizer.enable_truncation(max_length=MAX_TOKENS)
        self.session = onnxruntime.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    def encode(self, text: str) -> list[float]:
        import numpy as np

        input_ids = np.asarray([self.tokenizer.encode(text).ids], dtype=np.int64)
        vector = self.session.run(None, {
            "input_ids": input_ids, "attention_mask": np.ones_like(input_ids),
        })[0][0]
        return [float(value) for value in vector / np.linalg.norm(vector)]


@lru_cache(maxsize=1)
def _load_encoder() -> ClipTextEncoder:
    return ClipTextEncoder(TEXT_MODEL_PATH, TOKENIZER_PATH)


def clip_encoder() -> ClipTextEncoder:
    # Loading takes seconds; the lock keeps concurrent first calls from loading twice.
    with _encoder_lock:
        return _load_encoder()


def text_embedding(encoder: Any, text: str) -> list[float]:
    return encoder.encode(text)


def search(session: Any, embedding: list[float], *, limit: int) -> list[dict[str, Any]]:
    return [dict(row) for row in session.run(
        BASELINE_CYPHER, embedding_model=CLIP_MODEL, query_embedding=embedding, limit=limit,
    )]
