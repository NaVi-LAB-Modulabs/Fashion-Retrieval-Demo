# Fashion Search

## Web experience

The web UI uses the dataset-independent name **Fashion Search**. Fashion-How is
temporary development data; the intended research demo dataset is **Fashion200K**.
Dataset names belong in experiment documentation and provenance rather than the
main interface. The initial preview uses bundled Fashion-How images. Search results
can resolve Fashion200K images by their original Hugging Face item ID; graph
ingestion and schema mapping remain a separate step. Catalog preview labels distinguish unranked browsing from
actual search results, without presenting a dataset as the product name.

The web UI is a retrieval workspace: query and settings at the top, followed by
extracted constraints, executed Cypher and ranked results in three adjacent panes.
Cypher stays visible without opening an accordion. Parameter values, applied
settings and measured stage timings are shown alongside the query. Image cards
include component scores; a table and item details provide more ranking evidence.
Raw parser/parameter JSON and run export remain available for inspection.
Before a search, analysis panes show empty states rather than fabricated examples.

It uses HTML, CSS and JavaScript served by FastAPI; no Node.js
or frontend build is required. The application entrypoint is `web_app.py`.

Features include responsive image cards, an optional ranking table, natural
language examples, parser/threshold controls, extracted and excluded constraints,
style-axis targets, per-item score breakdowns and mapped attributes, exact Cypher
and parameters, measured stage timings, and downloadable JSON search runs.
The initial gallery is explicitly an **unranked catalog preview**, with no invented
scores or attributes. Live results always come from the existing retrieval code.

### Run the web UI

Use Python 3.11–3.13 and Windows Command Prompt:

```cmd
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe web_app.py
```

Open `http://127.0.0.1:7860`. Configure the variables below in a root `.env` file
or the server environment to enable live search. Credentials stay on the server.
The status indicator reports configuration presence, not verified connectivity.
No local GPU is required by the web retrieval path.

To view the design without installing packages or calling external services:

```cmd
python preview.py
```

This uses the same UI and local images on port 7860, with live search explicitly
disabled. Stop it with Ctrl+C before starting `web_app.py` on the same port.

### Deploy the web UI to Vercel

Import this repository into Vercel and select the **FastAPI** framework preset.
`pyproject.toml` specifies `web_app:app` as the entrypoint and lists the application
dependencies. `.vercelignore` excludes local environment files, development
artifacts and the preview server from uploads. There is no frontend build command. Set
`OPENAI_API_KEY`, `NEO4J_URI`, `NEO4J_USERNAME` and `NEO4J_PASSWORD` in the project
environment, plus any optional model/database settings listed below. Redeploy
after changing environment variables. Use a Neo4j endpoint reachable from the
deployment. API reference: `/docs`.

Static UI assets are mounted at `/assets`; `/images/{filename}` serves only indexed
catalog images. The nine bundled sample images total about 0.5 MB. Items whose
local images are absent are resolved by ID through the image lookup API below.
Review the function duration available to the project before a live demo: parser
and embedding calls can take time, especially when provider retries are needed.

Vercel configuration follows the [official FastAPI deployment guide](https://vercel.com/docs/frameworks/backend/fastapi).

### Verification

```cmd
python -m unittest discover -s tests -v
```

The service tests use provider doubles to verify real Cypher generation,
reranking, thresholds, safe response fields and preview semantics without API
charges. They do not establish connectivity to a deployed database or provider.

## Environment variables

Set these in the server environment or a local `.env` file. For deployment,
register the values in the Vercel project's environment variables:

```text
OPENAI_API_KEY
NEO4J_URI
NEO4J_USERNAME
NEO4J_PASSWORD
```

Optional:

```text
NEO4J_DATABASE
OPENAI_MODEL
OPENAI_EMBEDDING_MODEL
```

## Fashion200K images from Hugging Face

The live web app does not need a local copy of Fashion200K images. It returns
`image_id` with each result lacking a bundled image, using the Neo4j `item_ID`
property if present, otherwise `id`. Preserve the original ID, including its
image suffix (for example, `51727804_0`).

After rendering the search evidence, the browser sends up to 50 IDs per request
to `POST /api/images`. The server uses the Dataset Viewer `/filter` endpoint with
OR predicates and maps returned rows by ID, independently of their order. Images
are loaded directly from Hugging Face; no image files or image bytes are stored
in the repository, database or Vercel function. Requests for image metadata do not
change the retrieval ranking or its measured stage timings.

Defaults (no extra Vercel configuration is required for this public dataset):

| Variable | Default |
| --- | --- |
| `HF_IMAGE_DATASET` | `Marqo/fashion200k` |
| `HF_IMAGE_CONFIG` | `default` |
| `HF_IMAGE_SPLIT` | `data` |
| `HF_IMAGE_ID_COLUMN` | `item_ID` |
| `HF_TOKEN` | Unset; optional server-side read token |

Signed image URLs are cached in memory for at most 60 seconds (shorter when a
known expiry approaches). Missing IDs are cached for 30 seconds. If an image
fails to load, the browser requests a fresh URL once. Provider errors, missing
IDs and failed images produce an image-unavailable placeholder while retaining
all search results and evidence. Exported runs retain image IDs, not the fetched
temporary URLs. Lookup speed and availability depend on the external service.
`preview.py` stays offline and uses bundled sample images only.

API reference: [filter predicates](https://huggingface.co/docs/dataset-viewer/filter)
and [temporary image URLs](https://huggingface.co/docs/dataset-viewer/rows).

## Demo Structure

```text
.
|-- web_app.py                     # FastAPI application entrypoint
|-- preview.py                     # dependency-free, offline UI preview
|-- web/                           # HTML, CSS, JavaScript and favicon
|-- requirements.txt              # application dependencies for local install
|-- pyproject.toml                 # project dependencies and Vercel entrypoint
|-- vercel.json                    # FastAPI deployment preset
|-- tests/                         # offline service and preview tests
|-- fashion-how/
|   `-- image/                     # sample image assets
`-- src/
    `-- fashion_how_graphdb/
        |-- web_service.py         # retrieval API adapter and image catalog
        |-- search.py              # query parsing, Cypher build, reranking
        |-- cypher.py              # Neo4j defaults and import helpers
        |-- taxonomy.py
        |-- category_taxonomy.py
        `-- vlm.py                 # OpenAI Responses API helpers
```

## Neo4j Item Properties

For Fashion200K images, each `Item` should preserve the Hugging Face `item_ID`
as `id` (or supply a separate `item_ID` property). For bundled local images, an
optional `image_file` property can match a file under `fashion-how/image`.

For description-vector reranking, each `Item` can include one of:

```text
description_embedding
image_description_embedding
text_description_embedding
text_embedding
embedding
```

For style-axis reranking, each `Item` can include:

```text
trendy_classic
cool_warm
feminine_mannish
minimal_maximal
casual_formal
soft_sharp
young_mature
```

Each axis value is a float from `0.0` to `1.0`, where `0.0` is the left pole and
`1.0` is the right pole.
