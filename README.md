# FashioNavi

**FashioNavi: Fashion Navigator with Attribute-Enriched Graphs and Hybrid Retrieval**

[Jiyoon Myung†](https://www.linkedin.com/in/jiyoon-myung), [Yujin Jung†](https://www.linkedin.com/in/유진-정-59b296258), [Mia Lim†](https://www.linkedin.com/in/seoyounglim), [Youjin Jeon†](https://www.linkedin.com/in/youjinjeon), and Jaein Kim†  

*NaVi LAB, Seoul, Republic of Korea · † Equal contribution*

<p align="center">
  <strong>📄 Paper (link pending) | <a href="https://github.com/NaVi-LAB-Modulabs/Fashion-Retrieval-Demo">💻 Code</a> | <a href="https://fashion-retrieval-demo.vercel.app/">🚀 Live Demo</a></strong>
</p>

## 📝 Abstract

Natural-language fashion queries often combine explicit product requirements with subjective style preferences, making them difficult to satisfy with a single similarity score. We present **FashioNavi**, a hybrid fashion retrieval system that combines an attribute-enriched product graph, vision-language model (VLM)-generated descriptions, and fine-grained style axes. FashioNavi decomposes each query into attribute constraints, a visual description, and style targets. It then applies graph-based filtering and semantic retrieval, followed by hybrid reranking. The interactive demo makes the retrieval process transparent by showing parsed queries, item-level evidence, and component scores. Adjustable settings and an unfiltered CLIP baseline allow users to explore how each retrieval signal affects the results.

## 🔎 Overview

FashioNavi consists of two stages:

- **Offline graph construction:** Extract typed product attributes, seven reference-grounded style scores, and VLM-generated descriptions designed to distinguish each item from visually similar garments retrieved by CLIP.
- **Online retrieval:** Parse the user's query, apply hard constraints, combine the top 10 candidates from attribute matching with the top 10 from description similarity, and rerank the merged candidates using attribute, semantic, and style evidence.

![FashioNavi offline graph construction and online hybrid retrieval pipeline](docs/figures/metadata-aware-fashion-graph-methodology.png)

The default scoring weights are **55% description · 35% graph attributes · 10% style**. Weights are configurable and are renormalized over the available scoring components.

> **Scope:** This repository contains the online retrieval service and interactive demo. The offline pipeline is maintained in [Fashion-Retrieval](https://github.com/NaVi-LAB-Modulabs/Fashion-Retrieval), which currently requires repository access. A downloadable snapshot of the deployed graph has not been released.

## 📂 Repository Structure

```text
web_app.py                  # FastAPI application and API
preview.py                  # Catalog preview server
web/                        # Frontend and style references
src/fashion_how_graphdb/    # Query parsing, retrieval, and reranking
public/images/              # Product images
data/clip/                  # Standalone CLIP index
tests/                      # Automated tests
.github/                    # CI workflow and offline test runner
```

## ✨ Demo

Try a natural-language query such as:

> A casual day dress with long sleeves, but not a black one.

The demo lets you:

- Inspect parsed constraints, item-level supporting evidence, and component scores.
- Adjust scoring weights, thresholds, and style-target modes.
- Compare FashioNavi results with the **unfiltered CLIP ViT-L/14 baseline** used in the paper.
- Inspect generated Cypher queries and execution timings, then export complete search runs as JSON.

The live demo covers **15,000 Fashion200K images** across five garment categories.

## 🚀 Quick Start

### 1. 🌐 Try It Online

Open the **[Live Demo](https://fashion-retrieval-demo.vercel.app/)**. No installation is required.

Example queries:

- `A soft, elegant maxi dress with a V-neck and short sleeves`
- `A navy jacket with waist pockets, without a zipper`
- `A minimal black midi skirt with a slit, not pleated`

### 2. 💻 Run Locally (No API Keys)

**Requirements:** Python 3.11–3.13. The repository includes approximately 280 MB of catalog images, so cloning may take some time.

```bash
git clone https://github.com/NaVi-LAB-Modulabs/Fashion-Retrieval-Demo.git
cd Fashion-Retrieval-Demo
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python web_app.py
```

Open <http://127.0.0.1:7860> in your browser.

Without OpenAI and Neo4j credentials, the application automatically switches to **CLIP ONNX mode**: CPU-based text-to-image retrieval using **CLIP ViT-B/32** over **5,000 bundled images**. On the first search, it downloads a pinned model file (approximately 64 MB) from Hugging Face, so an internet connection is required. Subsequent searches are faster.

To preview the interface without installing the full application, run `python preview.py`. This launches an unranked catalog preview with search disabled.

### 3. 🗃️ Run the Full FashioNavi Pipeline

Full hybrid retrieval requires an **OpenAI API key** and a **Neo4j database populated with the FashioNavi fashion graph**.

> **Graph data release pending.** The public repository includes the image catalog and standalone CLIP index, but not the deployed Neo4j graph. OpenAI and Neo4j credentials alone are insufficient to reproduce hybrid retrieval against an empty database. Download and restore instructions will be added with a versioned graph snapshot. Until then, explore the full pipeline through the [live demo](https://fashion-retrieval-demo.vercel.app/); collaborators with access to the construction repository can follow its [CLI guide](https://github.com/NaVi-LAB-Modulabs/Fashion-Retrieval/blob/main/src/fashion_200k_graphdb/README.md).

Once the graph data is released:

1. Install **Neo4j 5.23 or later** using [Neo4j Desktop](https://neo4j.com/download/) or the [official Docker image](https://hub.docker.com/_/neo4j).
2. Restore the released graph data into your Neo4j database.
3. Copy `.env.example` to `.env` and configure the variables below.
4. Start the application:

```bash
cp .env.example .env               # Windows: copy .env.example .env
python web_app.py
```

| Variable | Required | Default or example |
| --- | --- | --- |
| `OPENAI_API_KEY` | Yes | — |
| `NEO4J_URI` | Yes | `bolt://localhost:7687` for a local installation |
| `NEO4J_PASSWORD` | Yes | — |
| `NEO4J_USERNAME` | No | `neo4j` |
| `NEO4J_DATABASE` | No | Server default |
| `OPENAI_MODEL` | No | `gpt-5.4-mini` (pinned to `gpt-5.4-mini-2026-03-17`) |
| `OPENAI_EMBEDDING_MODEL` | No | `text-embedding-3-large` |
| `FASHION_VECTOR_SEARCH_MODE` | No | `exact`; optional `hnsw` has additional requirements below |
| `NEO4J_DESCRIPTION_VECTOR_INDEX` | HNSW only | `item_description_embedding_filtered` |
| `FASHION_DATASET_REVISION` | No | `1`; change when graph data, embeddings, or indexes change |

When the required environment variables are present, the application header displays **Live search configured**, and **Hybrid** becomes the default retrieval mode. This indicator checks configuration presence; it does not verify database connectivity or graph completeness.

### Graph requirements and retrieval modes

The graph must use English taxonomy IDs and `(Item)-[:IS_CATEGORY]->(Category)`
relationships. Description retrieval requires `Item.description_embedding` and
`Item.description_embedding_model` to match the query embedding model and vector
dimensions (3,072 for the default `text-embedding-3-large`). Style reranking uses
the seven stored axis properties: `visual_trend`, `thermal_impression`,
`design_expression`, `minimal_maximal`, `casual_formal`, `soft_sharp`, and
`young_mature`. The ViT-L/14 baseline additionally requires compatible
`image_embedding` values with `image_embedding_model = 'clip-ViT-L-14'`.

The default **`FASHION_VECTOR_SEARCH_MODE=exact`** computes description similarity
over all model-compatible items that pass the hard filters, then selects the top
10. It does not require a description vector index. Neo4j 5.23 or later supports
the [scoped subqueries](https://neo4j.com/docs/cypher-manual/5/subqueries/call-subquery/)
used by retrieval.

Optional **`FASHION_VECTOR_SEARCH_MODE=hnsw`** uses approximate filtered vector
search. It requires **Neo4j 2026.06 or later**, Cypher 25, and an online vector index
on `Item.description_embedding` with `id` included as an additional filter property.
This requirement comes from the [`SEARCH ... WHERE item.id IN ...` syntax](https://neo4j.com/docs/cypher-manual/25/clauses/search/).
Set `NEO4J_DESCRIPTION_VECTOR_INDEX` to the index name. An unsupported query or
missing index raises an error; the service does not automatically switch to an
exact scan. See the [Neo4j vector-index guide](https://neo4j.com/docs/cypher-manual/current/indexes/search-indexes/vector-indexes/)
for index creation and status checks.

## 🎛️ Using the Interface

- **Search bar:** Describe an item in plain English. Negations such as “not white” and “without a zipper” are converted into exclusion filters.
- **Search settings (collapsible):**
  - **Retrieval mode:** Choose *Hybrid* (full FashioNavi) or *CLIP ONNX* (standalone CLIP search).
  - **Style targets:** Choose *Image references* to calibrate style targets using labeled reference images, or *Legacy* for text-only targets.
  - **Scoring weights:** Adjust description, graph-attribute, and style weights (default: **55 : 35 : 10**).
  - **Thresholds:** Set the minimum attribute match and minimum final score.
- **Results:** Each card displays its final score and component scores (**G** = graph attributes, **T** = text description, **S** = style). Open a card to inspect matched attributes and style-axis values.
- **How this search was ranked:** Examine extracted constraints, generated Cypher and its parameters, and stage-by-stage timings.
- **Show CLIP results:** Expand a separate **CLIP ViT-L/14** baseline. It is collapsed by default to distinguish baseline output from FashioNavi results; overlapping items are marked.
- **Export:** Download the complete search run as JSON.

## 🧠 How It Works

1. **Parse:** An LLM extracts the product category, required and excluded attributes, a visual description, and requested style-axis targets.
2. **Filter:** Neo4j applies hard constraints on category and required/excluded attributes before ranking.
3. **Select candidates:** Retrieve the top 10 filtered items by graph-attribute score and the top 10 by description-embedding similarity. Merge and deduplicate the two sets to obtain at most 20 candidates.
4. **Rerank:** Compute a weighted mean of description similarity, graph-attribute match, and style-axis closeness. Missing components are excluded, and the remaining weights are renormalized.

## 🔌 API

The FastAPI server exposes a JSON API, with interactive documentation at **`/docs`**.

| Endpoint | Requirements | Purpose |
| --- | --- | --- |
| `POST /api/clip/search` | No API keys or database | CLIP ViT-B/32 search over 5,000 bundled images |
| `POST /api/search` | OpenAI + Neo4j | Full FashioNavi hybrid retrieval |
| `POST /api/baseline` | Neo4j | CLIP ViT-L/14 baseline using graph image embeddings |
| `GET /api/config` | None | Server readiness and default settings |

Example request for local CLIP search:

```bash
curl -X POST http://127.0.0.1:7860/api/clip/search \
  -H "Content-Type: application/json" \
  -d '{"query": "a formal black dress", "limit": 5}'
```

The `/api/search` endpoint accepts the following fields:

- `query`: Natural-language search request.
- `limit`: Number of returned results (1–50).
- `model`: Model selection.
- `style_mode`: `legacy` or `reference` (**API default: `legacy`**).
- `weights`: Object containing `text`, `graph`, and `style` weights.
- `min_confidence`: Minimum attribute-match threshold.
- `min_score`: Minimum final-score threshold.

## Reproducing and checking a run

The hybrid demo uses **15,000 images**. The credential-free **ViT-B/32 ONNX** mode
uses a **5,000-image subset**, while the paper's **ViT-L/14 baseline** searches
compatible image embeddings in Neo4j. These modes use different models and
catalogs; record the selected mode when comparing results.

[`sample_ids.json`](sample_ids.json) identifies the image catalog.
[`data/clip/manifest.json`](data/clip/manifest.json) records the standalone index's
item IDs, pinned model revision, preprocessing, embedding checksum, and build
runtime. The service validates its index against this manifest.

For hybrid runs, keep the code commit, graph snapshot/revision, actual parser and
embedding models, vector-search mode, style mode, weights, and thresholds with the
exported search JSON. The UI defaults to `reference` style targets; direct API
requests default to `legacy`. Deployment environment variables can override model
defaults, so use the exported run's settings rather than assuming that the live
demo uses the README defaults. Increment `FASHION_DATASET_REVISION` when changing
the graph or stored embeddings to invalidate cached results.

To run the same offline checks used by CI, activate the virtual environment and run:

```bash
python -m pip install -r requirements-dev.txt
python .github/scripts/run_tests.py
```

These tests use provider and database doubles. They check retrieval behavior
without establishing connectivity to the live services or measuring retrieval
quality. The optional model-download test is skipped by default.

## ⚖️ License

The project code is licensed under the **[MIT License](LICENSE)**. Catalog images
come from [Fashion200K via Marqo](https://huggingface.co/datasets/Marqo/fashion200k);
refer to the original dataset for image attribution and usage terms.
