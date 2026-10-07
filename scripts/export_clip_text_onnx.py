"""Export the CLIP ViT-L/14 text tower used by the baseline to ONNX.

Writes models/clip_text_fp16w.onnx (fp16 weights, fp32 arithmetic) and
models/clip_tokenizer.json, then checks the app's ONNX encoder against the
sentence-transformers model that produced Item.image_embedding.
Needs requirements-export.txt (PyTorch); the app itself does not.

    python scripts/export_clip_text_onnx.py
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import torch
from huggingface_hub import hf_hub_download
from onnx import TensorProto, helper, numpy_helper
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fashion_how_graphdb import baseline  # noqa: E402

MIN_COSINE = 0.9999
CHECK_QUERIES = [
    "A casual blue shirt, not white, for daily wear",
    "A romantic floral blouse for a spring date",
    "Minimal black outerwear that feels formal",
    "dress 👗 with flowers 🌸",
    "파란색 셔츠",
    "!!!",
    "a " * 60 + "very long query about a blue shirt that keeps going past the token limit",
]


class TextTower(torch.nn.Module):
    """Same path as CLIPModel.get_text_features: pooled end-of-text state -> projection."""

    def __init__(self, clip):
        super().__init__()
        self.clip = clip

    def forward(self, input_ids, attention_mask):
        pooled = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        return self.clip.text_projection(pooled)


def store_weights_as_fp16(source: Path, target: Path) -> None:
    """Halve the file: store float weights as fp16 and Cast them back, so arithmetic stays fp32."""
    model = onnx.load(str(source))
    graph = model.graph
    initializers, casts = [], []
    for initializer in graph.initializer:
        array = numpy_helper.to_array(initializer)
        if initializer.data_type == TensorProto.FLOAT and array.size >= 1024:
            half = numpy_helper.from_array(array.astype(np.float16), initializer.name + "__fp16")
            initializers.append(half)
            casts.append(helper.make_node(
                "Cast", [half.name], [initializer.name], to=TensorProto.FLOAT, name=initializer.name + "__cast",
            ))
        else:
            initializers.append(initializer)
    del graph.initializer[:]
    graph.initializer.extend(initializers)
    nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(casts + nodes)
    onnx.save(model, str(target))


def main() -> None:
    model = SentenceTransformer(baseline.CLIP_MODEL, device="cpu")
    clip, tokenizer = model[0].model.eval(), model[0].tokenizer
    baseline.TEXT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as directory:
        fp32_path = Path(directory) / "clip_text_fp32.onnx"
        sample = tokenizer(["a casual blue shirt"], return_tensors="pt")
        torch.onnx.export(
            TextTower(clip), (sample["input_ids"], sample["attention_mask"]), str(fp32_path),
            input_names=["input_ids", "attention_mask"], output_names=["text_embeds"],
            dynamic_axes={"input_ids": {0: "batch", 1: "sequence"},
                          "attention_mask": {0: "batch", 1: "sequence"},
                          "text_embeds": {0: "batch"}},
            opset_version=17, dynamo=False,
        )
        store_weights_as_fp16(fp32_path, baseline.TEXT_MODEL_PATH)
    shutil.copyfile(
        hf_hub_download(f"sentence-transformers/{baseline.CLIP_MODEL}", "0_CLIPModel/tokenizer.json"),
        baseline.TOKENIZER_PATH,
    )

    # Check the exact runtime path (standalone tokenizer + ONNX) against the torch model.
    encoder = baseline.ClipTextEncoder(baseline.TEXT_MODEL_PATH, baseline.TOKENIZER_PATH)
    expected = model.encode(CHECK_QUERIES, convert_to_numpy=True, normalize_embeddings=True)
    cosines = [float(np.dot(encoder.encode(query), vector)) for query, vector in zip(CHECK_QUERIES, expected)]
    digest = hashlib.sha256(baseline.TEXT_MODEL_PATH.read_bytes()).hexdigest()
    print(f"{baseline.TEXT_MODEL_PATH.relative_to(ROOT)}: {baseline.TEXT_MODEL_PATH.stat().st_size / 1e6:.0f} MB")
    print(f"sha256: {digest}")
    print(f"min cosine vs sentence-transformers fp32: {min(cosines):.7f}")
    if min(cosines) < MIN_COSINE:
        raise SystemExit(f"Exported model differs from the source model (cosine < {MIN_COSINE}).")


if __name__ == "__main__":
    main()
