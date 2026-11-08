# FashioNavi

**Fashion Navigator with Attribute-Enriched Graphs and Hybrid Retrieval**

[Jiyoon Myung†](https://www.linkedin.com/in/jiyoon-myung), [Yujin Jung†](https://www.linkedin.com/in/유진-정-59b296258), [Mia Lim†](https://www.linkedin.com/in/seoyounglim), [Youjin Jeon†](https://www.linkedin.com/in/youjinjeon), and Jaein Kim†

NaVi LAB, Seoul, Republic of Korea · † Equal contribution.

<p align="center">
   <b>[ 📄 Paper (link not yet available) | <a href="https://github.com/NaVi-LAB-Modulabs/Fashion-Retrieval-Demo">💻 Code</a> | <a href="https://fashion-retrieval-demo.vercel.app/">🚀 Live Demo</a> ]</b>
</p>

## 📝 Abstract

Natural-language fashion queries combine explicit requirements with subjective style preferences, which are difficult to satisfy with a single similarity score. We present FashioNavi, a hybrid retrieval system that integrates an attribute-enriched product graph, VLM-generated descriptions, and fine-grained style axes. It decomposes each query into attribute filters, a visual description, and style targets, then applies graph filtering and semantic retrieval before hybrid reranking. The interactive demo exposes the parsed request, supporting evidence, and component scores, while adjustable settings and an unfiltered CLIP baseline show how different signals affect the results.

## 🔎 Overview

- **Offline graph construction:** extract typed attributes, seven reference-grounded style scores, and descriptions that distinguish each item from CLIP-retrieved similar garments.
- **Online retrieval:** parse the query, apply hard constraints, merge the attribute top 10 and description top 10, and rerank using attribute, semantic, and style evidence.

![Online retrieval: query parsing, hard filters, candidate selection, and hybrid reranking](web/search-flow.svg)

Default scoring weights are **description 55% · attributes 35% · style 10%**, adjustable and normalized over available components. This repository contains the online retrieval service and demo; graph construction is separate.

## ✨ Demo

> A casual day dress with long sleeves, but not a black one.

- Inspect parsed constraints, item-level evidence, and component scores.
- Adjust scoring weights, thresholds, and style-target modes.
- Compare results with the unfiltered **CLIP ViT-L/14** baseline used in the paper.
- View Cypher and timings, and export runs as JSON.

The demo includes **15,000 Fashion200K images** across five garment categories.

## 🚀 Quick start

Use **Python 3.11–3.13** in a virtual environment. No frontend build or local GPU is required.

For hybrid search, copy [`.env.example`](.env.example) to `.env` and set `OPENAI_API_KEY`, `NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD`. A populated Neo4j graph with compatible attributes and embeddings is required and is **not bundled**.

```sh
python -m pip install -r requirements.txt
python web_app.py
```

Open **http://127.0.0.1:7860**. API documentation: **http://127.0.0.1:7860/docs**.

- **UI preview:** run `python preview.py` without dependencies or credentials; live search is disabled.
- **Search without credentials:** select **CLIP ONNX**, an additional ViT-B/32 mode using the packaged 5,000-image index. It downloads its text encoder on first use.

## 📂 Repository structure

```text
web_app.py                 # FastAPI application and API
preview.py                 # Catalog preview server
web/                       # Frontend and style references
src/fashion_how_graphdb/    # Query parsing, retrieval, and reranking
public/images/             # Product images
data/clip/                 # Standalone CLIP index
scripts/                   # Index building and evaluation
tests/                     # Automated tests
docs/                      # Experiment details
```

## 🧪 Development

```sh
python -m pip install -r requirements-dev.txt
python scripts/run_tests.py
```

See the [CLIP comparison](docs/clip-onnx-comparison.md) and [style-reference experiment](docs/reference-style-experiment.md) for implementation and evaluation details. Vercel deployment is configured in `pyproject.toml` and `vercel.json`.

## ⚖️ License

This project is licensed under the MIT License.
