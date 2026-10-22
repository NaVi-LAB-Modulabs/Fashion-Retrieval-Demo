"""CPU-only ONNX CLIP text/image encoders and exact catalog retrieval."""

from __future__ import annotations

from functools import lru_cache
from importlib.metadata import version
import hashlib
import json
import os
import platform
from pathlib import Path
import tempfile
from threading import Lock
from typing import Any

from .reproducibility import cached, fingerprint, normalize_query

ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = ROOT / "data/clip"
MODEL_REPO = "Xenova/clip-vit-base-patch32"
MODEL_REVISION = "d15189d7028b43f1d3e65039190477f6af591c2a"
DIMENSION = 512
TOKEN_LIMIT = 77
ASSETS = {
    "text": ("onnx/text_model_quantized.onnx", "sha256", "73baab855d406190da9faa498cfedf65f15cf309f4cc7385b7b032e6d08e5c3a"),
    "vision": ("onnx/vision_model_quantized.onnx", "sha256", "583fd1110a514667812fee7d684952aaf82a99b959760c8d7dca7e0ab9839299"),
    "tokenizer": ("tokenizer.json", "git", "bc1f77d20440541dd073ebae6f6c401087c7d34e"),
    "preprocessor": ("preprocessor_config.json", "git", "e0675735fd0052745f2f7b7291e4aa7859389998"),
}
_asset_lock = Lock()


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.sha1() if algorithm == "git" else hashlib.sha256()
    if algorithm == "git":
        digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_asset(kind: str) -> Path:
    filename, algorithm, expected = ASSETS[kind]
    directory = Path(os.getenv("FASHION_CLIP_MODEL_DIR") or
                     Path(tempfile.gettempdir()) / "fashion-clip-onnx" / MODEL_REVISION)
    return hub_file(MODEL_REPO, MODEL_REVISION, filename, algorithm, expected, directory)


def hub_file(repo: str, revision: str, filename: str, algorithm: str, expected: str, directory: Path) -> Path:
    """Download one pinned Hugging Face file once and verify it before use."""
    path = directory / Path(filename).name
    with _asset_lock:
        if path.is_file() and file_digest(path, algorithm) == expected:
            return path
        import httpx
        directory.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
                temporary = Path(stream.name)
                url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
                with httpx.stream("GET", url, follow_redirects=True,
                                  timeout=httpx.Timeout(120, connect=15)) as response:
                    response.raise_for_status()
                    for chunk in response.iter_bytes(1024 * 1024):
                        stream.write(chunk)
            if file_digest(temporary, algorithm) != expected:
                raise ValueError("Downloaded CLIP asset failed integrity verification")
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return path


@lru_cache(maxsize=2)
def encoder_session(kind: str) -> Any:
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = int(os.getenv("FASHION_CLIP_THREADS", "2"))
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(model_asset(kind)), sess_options=options,
                                providers=["CPUExecutionProvider"])


@lru_cache(maxsize=1)
def text_tokenizer() -> Any:
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(model_asset("tokenizer")))


def normalized_vectors(values: Any) -> Any:
    import numpy as np
    vectors = np.asarray(values, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != DIMENSION or not np.isfinite(vectors).all():
        raise ValueError("CLIP embeddings must be finite 512-dimensional vectors")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("CLIP embeddings must have nonzero norm")
    return vectors / norms


@lru_cache(maxsize=1)
def runtime_provenance() -> dict:
    return {"onnxruntime": version("onnxruntime"), "tokenizers": version("tokenizers"),
            "numpy": version("numpy"), "platform": platform.system(),
            "machine": platform.machine(), "provider": "CPUExecutionProvider"}


def encode_text(query: str) -> tuple[Any, int, bool]:
    import numpy as np
    query = normalize_query(query)
    tokens = text_tokenizer().encode(query).ids
    count = len(tokens)
    if count > TOKEN_LIMIT:
        tokens = tokens[:TOKEN_LIMIT - 1] + [49407]
    # OpenAI CLIP pads with its EOS token; retain the real sequence mask.
    input_ids = np.asarray([tokens + [49407] * (TOKEN_LIMIT - len(tokens))], dtype=np.int64)
    attention_mask = np.asarray([[1] * len(tokens) + [0] * (TOKEN_LIMIT - len(tokens))], dtype=np.int64)

    def compute():
        session = encoder_session("text")
        feed = {"input_ids": input_ids, "attention_mask": attention_mask}
        output = session.run(["text_embeds"], {entry.name: feed[entry.name] for entry in session.get_inputs()})[0]
        return normalized_vectors(output)[0].tolist()

    embedding = cached("clip_text", {"query": query, "model": MODEL_REPO, "revision": MODEL_REVISION,
                                    "encoder_hash": ASSETS["text"][2], "tokenizer_hash": ASSETS["tokenizer"][2],
                                    "runtime": runtime_provenance(),
                                    "preprocessing": "eos-padding-77-v1"}, compute).value
    return np.asarray(embedding, dtype=np.float32), count, count > TOKEN_LIMIT


@lru_cache(maxsize=1)
def image_preprocessor() -> dict:
    return json.loads(model_asset("preprocessor").read_text(encoding="utf-8"))


def image_tensor(path: Path) -> Any:
    import numpy as np
    from PIL import Image, ImageOps
    config = image_preprocessor()
    shortest = config["size"]["shortest_edge"]
    crop = config["crop_size"]
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        scale = shortest / min(width, height)
        image = image.resize((int(width * scale), int(height * scale)), resample=Image.Resampling.BICUBIC)
        left, top = (image.width - crop["width"]) // 2, (image.height - crop["height"]) // 2
        image = image.crop((left, top, left + crop["width"], top + crop["height"]))
        pixels = np.asarray(image, dtype=np.float32) * config["rescale_factor"]
    return ((pixels - np.asarray(config["image_mean"], dtype=np.float32)) /
            np.asarray(config["image_std"], dtype=np.float32)).transpose(2, 0, 1)


def encode_images(paths: list[Path]) -> Any:
    import numpy as np
    session = encoder_session("vision")
    pixels = np.stack([image_tensor(path) for path in paths])
    output = session.run(["image_embeds"], {"pixel_values": pixels})[0]
    return normalized_vectors(output)


@lru_cache(maxsize=1)
def index_manifest() -> dict:
    manifest = json.loads((INDEX_DIR / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["model"] != MODEL_REPO or manifest["model_revision"] != MODEL_REVISION
            or manifest["dimension"] != DIMENSION or manifest["vision_sha256"] != ASSETS["vision"][2]
            or manifest["text_sha256"] != ASSETS["text"][2]):
        raise ValueError("CLIP index and encoder must share the same model and revision")
    return manifest


def index_revision() -> str:
    return fingerprint(index_manifest())


def configuration() -> dict:
    available = (INDEX_DIR / "manifest.json").is_file() and (INDEX_DIR / "image_embeddings.npy").is_file()
    count = 0
    if available:
        try:
            count = len(index_manifest()["items"])
        except (OSError, ValueError, KeyError, TypeError):
            available = False
    return {"available": available, "model": "OpenAI CLIP ViT-B/32 (ONNX int8)",
            "model_repo": MODEL_REPO, "model_revision": MODEL_REVISION,
            "dimension": DIMENSION, "provider": "CPUExecutionProvider", "token_limit": TOKEN_LIMIT,
            "catalog_count": count if available else 0}


@lru_cache(maxsize=1)
def gallery() -> tuple[Any, list[dict]]:
    import numpy as np
    manifest = index_manifest()
    path = INDEX_DIR / "image_embeddings.npy"
    if file_digest(path) != manifest["embeddings_sha256"]:
        raise ValueError("CLIP gallery failed integrity verification")
    vectors = np.load(path, allow_pickle=False, mmap_mode="r")
    if vectors.shape != (len(manifest["items"]), DIMENSION):
        raise ValueError("CLIP gallery shape does not match its manifest")
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
        raise ValueError("CLIP gallery vectors must be finite and L2-normalized")
    items = manifest["items"]
    if len({item["id"] for item in items}) != len(items):
        raise ValueError("CLIP gallery contains duplicate item IDs")
    return vectors, items


def rank_vector(vector: Any, *, limit: int, candidate_ids: list[str] | None = None,
                categories: list[str] | None = None) -> dict:
    import numpy as np
    from .web_service import image_url
    vectors, items = gallery()
    allowed = set(candidate_ids) if candidate_ids is not None else None
    category_set = set(categories or [])
    positions = [index for index, item in enumerate(items)
                 if (allowed is None or item["id"] in allowed) and (not category_set or item["category"] in category_set)]
    query = normalized_vectors(np.asarray(vector, dtype=np.float32).reshape(1, -1))[0]
    scores = vectors[positions] @ query
    ordered = sorted(zip(positions, scores), key=lambda pair: (-float(pair[1]), items[pair[0]]["id"]))[:limit]
    results = []
    for rank, (index, cosine) in enumerate(ordered, 1):
        item = items[index]
        value = round(float(np.clip(cosine, -1, 1)), 6)
        results.append({"id": item["id"], "image_id": item["id"], "category": item["category"],
                        "type_code": item["category"], "type_name": item["category"],
                        "image_url": image_url(item["id"]), "rank": rank, "score": value,
                        "clip_score": value, "score_components": ["clip"]})
    return {"items": results, "result_count": len(results), "candidate_count": len(positions),
            "scope": "same_candidates" if allowed is not None else "catalog",
            "missing_candidate_ids": sorted(allowed - {items[position]["id"] for position in positions}) if allowed is not None else [],
            "score_type": "raw_cosine", "model": MODEL_REPO, "model_revision": MODEL_REVISION,
            "runtime": runtime_provenance(),
            "index_revision": index_revision(), "reranked": allowed is not None}


def validate_request(payload: dict) -> dict:
    query, limit = payload.get("query"), payload.get("limit", 12)
    if not isinstance(query, str) or not query.strip() or len(query) > 2000:
        raise ValueError("Provide a CLIP query between 1 and 2,000 characters.")
    if type(limit) is not int or not 1 <= limit <= 50:
        raise ValueError("CLIP result limit must be between 1 and 50.")
    categories = payload.get("categories", [])
    if not isinstance(categories, list) or any(category not in ("dresses", "jackets", "pants", "skirts", "tops") for category in categories):
        raise ValueError("Choose valid CLIP catalog categories.")
    return {"query": normalize_query(query), "limit": limit, "categories": sorted(set(categories))}


def retrieve(payload: dict) -> dict:
    from time import perf_counter
    request = validate_request(payload)
    started = perf_counter()
    vector, count, truncated = encode_text(request["query"])
    encoded = perf_counter()
    result = rank_vector(vector, limit=request["limit"], categories=request["categories"])
    result.update(query=request["query"], settings=request, token_count=count, truncated=truncated,
                  timings={"encode_ms": round((encoded - started) * 1000),
                           "total_ms": round((perf_counter() - started) * 1000)})
    return result
