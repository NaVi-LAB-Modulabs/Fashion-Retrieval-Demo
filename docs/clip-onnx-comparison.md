# CLIP ONNX Baseline

## Model and Runtime

This experiment runs **OpenAI CLIP ViT-B/32**, converted to quantized ONNX by
[Xenova](https://huggingface.co/Xenova/clip-vit-base-patch32), on
[ONNX Runtime CPU](https://onnxruntime.ai/docs/api/python/api_summary.html).
It does not import PyTorch, OpenCLIP or sentence-transformers in the web server.
The source implementation and contrastive retrieval method are documented by
[OpenAI CLIP](https://github.com/openai/CLIP).

The checkpoint is pinned to revision
`d15189d7028b43f1d3e65039190477f6af591c2a`. Downloads are verified against the
committed hashes in `clip_onnx.py`. Text and image towers are from the **same**
checkpoint. Vectors from another model, even with 512 dimensions, must not be mixed.
The sibling project's `clip_similarity.py` and `clip_retrieval_baseline.py`
informed the normalization, product ranking and Recall@K evaluation. This is
not an exact reproduction of its OpenCLIP `laion2b_s34b_b79k` baseline, nor of
unquantized OpenAI CLIP. Quantization and training checkpoints affect rankings.

## Search Calculation

1. Offline, process the 5,000 saved catalog images: EXIF transpose, RGB, bicubic
   resize with shortest side 224, centered 224x224 crop, rescale by 1/255, then
   the checkpoint's channel means and standard deviations. The ONNX vision
   tower produces 512-dimensional embeddings, each L2-normalized.
2. At search time, the pinned CLIP tokenizer encodes the **original query** with
   BOS/EOS. Pad to 77 tokens; longer queries retain the first 76 tokens and EOS.
   The response explicitly reports original token count and truncation. The
   ONNX text tower returns a 512-dimensional embedding, also L2-normalized.
3. Score every eligible image with `image_vector @ query_vector`. This is exact
   cosine, not ANN, a probability, or a calibrated style-axis target. Sort by
   descending unrounded cosine, then ascending item ID; serialize six decimals.
   Values remain in [-1, 1], with no logit scaling or softmax applied.

`data/clip/image_embeddings.npy` stores float32 vectors (about 10 MB).
`manifest.json` records IDs, categories, encoder checksums, revision, preprocessing
and index checksum. The loader rejects incompatible checkpoints, corrupt files,
duplicate IDs, incorrect shapes, non-finite vectors and non-unit norms.

## Three Comparisons

- **Hybrid** keeps the existing category/attribute hard filters, description and
  graph top-10 union, and graph/text/requested-style weighted ranking unchanged.
  Image-reference vs legacy target comparison remains available.
- **CLIP / same candidates** scores all fetched hybrid candidates, **before**
  the final hybrid score threshold and result limit. This isolates the reranker;
  it cannot recover a relevant image absent from the hybrid candidate pool.
  Missing gallery IDs are explicitly reported, not silently counted as scored.
- **CLIP / full catalog** searches all 5,000 saved images without graph filters,
  reference calibration, stored axes or hybrid score thresholds. This compares
  the whole retrieval pipeline; category mismatches are valid observed errors.

Do not compare the numerical size of CLIP cosine with the hybrid weighted score
as evidence of quality. Inspect product ranks or evaluate against the same labels.
OpenAI CLIP is primarily an English-language model. Queries are not translated:
use the same English query for controlled comparisons. Negation and fine-grained
style distinctions can be difficult. This is text-to-image retrieval, not an
uploaded-reference-image CIR composition model.

## API and UI

The UI offers `Retrieval: CLIP ONNX` for independent search. It works without
OpenAI or Neo4j credentials when the packaged catalog index is present. Hybrid
search with `Compare CLIP` checked exposes the two CLIP rankings under `Ranking`.
JSON export preserves every comparison and checkpoint revision.

```json
POST /api/clip/search
{"query": "a formal black dress", "limit": 12}

POST /api/search
{"query": "a formal black dress", "style_mode": "reference",
 "compare_styles": true, "compare_clip": true, "limit": 12}
```

The CLIP-only API additionally supports explicit `categories`, e.g. `["dresses"]`.
The UI's full-catalog baseline uses no category filter. `/api/config` reports
CLIP index availability, checkpoint, token limit and catalog size; availability
does not assert model download connectivity. Dependency-free `preview.py`
continues to disable both retrieval paths.

## Build and Evaluate

The checked-in gallery makes rebuilding unnecessary for this catalog. For a
changed image catalog, install runtime dependencies plus Pillow, then run:

```sh
python -m pip install -r requirements.txt Pillow
python scripts/build_clip_index.py --batch-size 8
python web_app.py
```

The builder downloads both towers and keeps resumable checkpoints in `.cache`.
Regenerate the index after changing images or the model; keep both the NPY file
and manifest together. Restart/redeploy after rebuilding, and increment
`FASHION_DATASET_REVISION` when changing the graph or saved data.

```sh
# Qualitative comparison, without claiming retrieval accuracy
python scripts/evaluate_clip.py --base-url https://fashion-retrieval-demo.vercel.app

# Independent ONNX baseline, no graph/provider needed
python scripts/evaluate_clip.py --clip-only

# Ground-truth evaluation across the same catalog
python scripts/evaluate_clip.py --labels evaluation.jsonl --base-url https://fashion-retrieval-demo.vercel.app
```

Labels use JSONL, e.g. `{"query":"a black dress","relevant_ids":["73141854_0"]}`.
Existing rows with `shopping_search_label` and `product_id` are also accepted.
Supply genuinely annotated relevant products, not this illustrative example.
Each method retrieves up to 20; the evaluator reports mean Recall@1/5/10
(fraction of in-gallery relevant IDs retrieved) and MRR@20. Queries whose labels
are entirely outside this saved gallery are excluded and explicitly listed.
In multi-label rows, out-of-gallery positives are removed from the denominator.
Unlabeled runs save rankings only, with `quality_metrics: null`.

Local execution can use `--env-file /path/to/.env`; credentials are never written
to reports. Outputs contain queries and product IDs and are git-ignored.
`check_clip_ui.py` verifies real CLIP searches and recorded hybrid comparisons
across desktop/mobile, including independent readiness, images, score table,
details, switching, export and horizontal overflow.

## Vercel Deployment

Model weights are not checked into git or the deployment bundle. On first use,
the server downloads the pinned text ONNX tower (64.5 MB) and tokenizer (2.2 MB)
to the instance's temporary directory; subsequent requests reuse its session.
The vision tower (89.1 MB) is only required for offline indexing. No GPU or new
API key is needed. Allow outbound HTTPS to Hugging Face; cold starts incur
download/session initialization latency. Temporary model files and text caches
are per-instance, not durable or shared. `FASHION_CLIP_MODEL_DIR` can specify
a pre-provisioned directory on a local server; `FASHION_CLIP_THREADS` controls
CPU inference threads (default 2). Do not set a read-only download directory.
Review the project's current [function limits](https://vercel.com/docs/functions/limitations).

The original hybrid baseline remains recoverable from git tag
`style-targets-baseline-20261011`; reference calibration and CLIP are opt-in API
additions and do not rewrite graph properties or stored embeddings.
