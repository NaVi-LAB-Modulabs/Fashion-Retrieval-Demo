"""Precompute the saved catalog's CLIP image vectors using ONNX Runtime only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    import numpy as np
    from fashion_how_graphdb import clip_onnx as clip
    from fashion_how_graphdb.web_service import image_index, sample_ids
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=clip.INDEX_DIR)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    images = {path.stem: path for path in image_index().values()}
    items = sorted([{"id": item_id, "category": category} for category, ids in sample_ids().items()
                    for item_id in ids], key=lambda item: item["id"])
    if any(item["id"] not in images for item in items):
        raise ValueError("Every catalog item needs a saved image before CLIP indexing")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = ROOT / ".cache/clip-image-vectors" / clip.MODEL_REVISION
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    vectors, image_hashes = [], {}
    started = perf_counter()
    for start in range(0, len(items), args.batch_size):
        rows = items[start:start + args.batch_size]
        batch = []
        missing = []
        for index, item in enumerate(rows):
            image_hash = clip.file_digest(images[item["id"]])
            image_hashes[item["id"]] = image_hash
            checkpoint = checkpoint_dir / f"{item['id']}-{image_hash}.npy"
            if checkpoint.is_file():
                batch.append(clip.normalized_vectors(np.load(checkpoint, allow_pickle=False).reshape(1, -1))[0])
            else:
                batch.append(None)
                missing.append((index, item, checkpoint))
        if missing:
            encoded = clip.encode_images([images[item["id"]] for _, item, _ in missing])
            for (index, item, checkpoint), vector in zip(missing, encoded):
                batch[index] = vector
                np.save(checkpoint, vector, allow_pickle=False)
        vectors.extend(batch)
        if start == 0 or (start + len(rows)) % 200 == 0 or start + len(rows) == len(items):
            print(f"Encoded {start + len(rows)}/{len(items)} images in {perf_counter() - started:.1f}s", flush=True)
    embedding_path = args.output_dir / "image_embeddings.npy"
    np.save(embedding_path, clip.normalized_vectors(vectors), allow_pickle=False)
    manifest = {"version": "onnx-clip-catalog-v1", "model": clip.MODEL_REPO,
                "model_revision": clip.MODEL_REVISION, "dimension": clip.DIMENSION,
                "text_sha256": clip.ASSETS["text"][2], "vision_sha256": clip.ASSETS["vision"][2],
                "embeddings_sha256": clip.file_digest(embedding_path),
                "image_preprocessing": "rgb-bicubic-shortest224-center224-clip-normalization-v1",
                "catalog_image_revision": clip.fingerprint(image_hashes), "items": items}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(items)} normalized CLIP image vectors to {args.output_dir}")


if __name__ == "__main__":
    main()
