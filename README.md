# FashioNavi

*Navigating Fashion Retrieval through Knowledge Graphs and Interpretable Style Axes*

**[Live demo](https://fashion-retrieval-demo.vercel.app/)**

FashioNavi turns a natural-language clothing request such as *"a casual blue shirt,
not white, for daily wear"* into ranked garment images. Unlike plain embedding search,
the request is first parsed into hard constraints, so exclusions like "not white"
are actually enforced. Every result also shows why it ranked where it did.

![Retrieval flow](web/search-flow.svg)

## Quick start

### 1. Try it online

Open the **[live demo](https://fashion-retrieval-demo.vercel.app/)**. No setup is
needed. Some queries to try:

- `A soft, elegant maxi dress with a V-neck and short sleeves`
- `A navy jacket with waist pockets, without a zipper`
- `A minimal black midi skirt with a slit, not pleated`

### 2. Run locally (no API keys)

You need Python 3.11–3.13. The repository includes about 280 MB of catalog images,
so the clone takes a while.

```bash
git clone https://github.com/NaVi-LAB-Modulabs/Fashion-Retrieval-Demo.git
cd Fashion-Retrieval-Demo
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python web_app.py
```

Open <http://127.0.0.1:7860>. Without credentials, the app switches to **CLIP ONNX**
mode: plain CLIP ViT-B/32 text-to-image search over 5,000 bundled images, running
on the CPU. The first search downloads a pinned 64 MB model from Hugging Face and
takes a few seconds. Later searches are fast.

To look at the interface without installing anything, run `python preview.py`. It
shows an unranked catalog preview, and search is disabled.

### 3. Run the full FashioNavi pipeline

Full search needs an **OpenAI API key** and a **Neo4j database loaded with the
FashioNavi fashion graph**.

> **The Neo4j graph data will be released soon.** Download and restore instructions
> will be added here at that time. Until then, you can try the full pipeline in the
> [live demo](https://fashion-retrieval-demo.vercel.app/).

Once the data is available:

1. Install Neo4j 5.23 or later, either [Neo4j Desktop](https://neo4j.com/download/)
   or the [Docker image](https://hub.docker.com/_/neo4j).
2. Load the released graph data into a database.
3. Create `.env` and fill in the values below. Then start the app:

```bash
cp .env.example .env               # Windows: copy .env.example .env
python web_app.py
```

| Variable | Required | Default or example |
| --- | --- | --- |
| `OPENAI_API_KEY` | yes | |
| `NEO4J_URI` | yes | `bolt://localhost:7687` for a local install |
| `NEO4J_PASSWORD` | yes | |
| `NEO4J_USERNAME` | | `neo4j` |
| `NEO4J_DATABASE` | | server default |
| `OPENAI_MODEL` | | `gpt-5.4-mini` (pinned to `gpt-5.4-mini-2026-03-17`) |
| `OPENAI_EMBEDDING_MODEL` | | `text-embedding-3-large` |

Once these are set, the header shows **Live search configured** and searches use the
**Hybrid** retrieval mode by default.

## Using the interface

- **Search bar.** Describe what you want in plain English. Negations ("not white",
  "without a zipper") become exclusion filters.
- **Search settings** (collapsible):
  - **Retrieval**: *Hybrid* (full FashioNavi) or *CLIP ONNX* (plain CLIP).
  - **Style targets**: *Image references* calibrates style targets against labeled
    reference images. *Legacy* uses text-only targets.
  - **Scoring weights**: description, graph-attribute and style weights. The
    default is 55 : 35 : 10.
  - **Min. attribute match** and **minimum score** thresholds.
- **Results.** Each card shows its final score and component scores
  (G = graph, T = text description, S = style). Click a card for its matched
  attributes and style-axis values.
- **How this search was ranked.** Shows the extracted constraints, the generated
  Cypher and its parameters, and the time each stage took.
- **Show CLIP results.** A separate CLIP ViT-L/14 baseline for comparison. It is
  collapsed by default so it is not mistaken for FashioNavi results. Items that
  appear in both lists are marked.
- **Export.** Downloads the complete run as JSON.

## How it works

1. **Parse.** An LLM extracts the category, required and excluded attributes, a
   visual description, and any requested style axes.
2. **Filter.** Hard filters on the category and on required and excluded attributes
   run in Neo4j before any ranking.
3. **Select candidates.** From the filtered items, the top 10 by graph-attribute
   score and the top 10 by description-embedding similarity are merged (at most
   20 candidates).
4. **Rerank.** The final score is a weighted mean of description similarity,
   graph-attribute match and style-axis closeness. Components that are missing
   are left out.

## API

The server exposes a JSON API. Interactive docs are at `/docs`.

| Endpoint | Needs | Purpose |
| --- | --- | --- |
| `POST /api/clip/search` | nothing | CLIP ViT-B/32 search over the bundled 5,000 images |
| `POST /api/search` | OpenAI + Neo4j | Full FashioNavi search |
| `POST /api/baseline` | Neo4j | CLIP ViT-L/14 baseline over graph image embeddings |
| `GET /api/config` | | Server readiness and defaults |

```bash
curl -X POST http://127.0.0.1:7860/api/clip/search \
  -H "Content-Type: application/json" \
  -d '{"query": "a formal black dress", "limit": 5}'
```

`/api/search` accepts the fields `query`, `limit` (1–50), `model`, `style_mode`
(`legacy` or `reference`; the API defaults to `legacy`), `weights`
(`{"text", "graph", "style"}`), `min_confidence` and `min_score`.

## Data and license

The catalog images and IDs come from the
[`Marqo/fashion200k`](https://huggingface.co/datasets/Marqo/fashion200k) distribution
of [Fashion200K](https://github.com/xthan/fashion-200k): 15,000 images, 3,000 for each
of five categories. That distribution is listed as Apache-2.0, but Marqo does not own
the original dataset. If you redistribute or publicly deploy the images, check the
original dataset terms and image rights first. The license for the software in this
repository is separate from the license for this third-party data.
