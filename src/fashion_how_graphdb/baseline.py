"""CLIP text-to-image baseline: the raw query against stored item image embeddings."""

from __future__ import annotations

import importlib.util
import os
from functools import lru_cache
from pathlib import Path
import tempfile
from threading import Lock
from typing import Any

from . import clip_onnx

# Item.image_embedding was produced by this sentence-transformers model (fp32,
# normalize_embeddings=True); the query must come from the same model's text tower.
CLIP_MODEL = "clip-ViT-L-14"

# The same OpenAI ViT-L/14 text tower as an fp16 ONNX export on Hugging Face, downloaded on
# first use like the CLIP comparison's model. Against the sentence-transformers fp32 model its
# query embeddings agree at cosine 1.0000 (the int8 export drifts to 0.83-0.91), and its
# tokenizer gives the same ids.
TEXT_MODEL_REPO = "Xenova/clip-vit-large-patch14"
TEXT_MODEL_REVISION = "c307790166907339eed5a9a53a249af534102536"
TEXT_MODEL_FILES = {
    "model": ("onnx/text_model_fp16.onnx", "sha256", "643d385d6adbc4b9067f3f94384cc63a8409accb1bfd414496d17df84b161032"),
    "tokenizer": ("tokenizer.json", "git", "f9d4455c06f24d31797f7cd3e480a83d7a84cb97"),
}
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
    # The model is downloaded on first use, so only the runtime packages are checked here.
    return _runtime_installed()


@lru_cache(maxsize=1)
def _runtime_installed() -> bool:
    return all(importlib.util.find_spec(name) for name in ("onnxruntime", "tokenizers", "numpy", "httpx"))


def text_model_file(kind: str) -> Path:
    # Its own revision directory: the CLIP comparison's files have the same names.
    directory = Path(os.getenv("FASHION_CLIP_MODEL_DIR") or
                     Path(tempfile.gettempdir()) / "fashion-clip-onnx") / TEXT_MODEL_REVISION
    return clip_onnx.hub_file(TEXT_MODEL_REPO, TEXT_MODEL_REVISION, *TEXT_MODEL_FILES[kind], directory)


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
        feed = {"input_ids": input_ids, "attention_mask": np.ones_like(input_ids)}
        # The Hugging Face export takes input_ids only.
        vector = self.session.run(None, {entry.name: feed[entry.name] for entry in self.session.get_inputs()})[0][0]
        vector = np.asarray(vector, dtype=np.float32)
        return [float(value) for value in vector / np.linalg.norm(vector)]


@lru_cache(maxsize=1)
def _load_encoder() -> ClipTextEncoder:
    return ClipTextEncoder(text_model_file("model"), text_model_file("tokenizer"))


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
