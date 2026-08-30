---
title: Fashion-How Graph Search
colorFrom: teal
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# Fashion-How Graph Search

Demo for natural-language fashion retrieval over a Neo4j attribute graph. The
system parses a user query into structured graph constraints, executes the
generated Cypher query, and optionally reranks candidates with text and style
signals.

This Space is prepared for an ECIR demo-track style walkthrough: the UI shows
ranked fashion items, extracted filters, graph candidates, score components, and
the final Cypher query used for retrieval.

## Required Space Secrets

Set these in the Hugging Face Space settings:

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

## Local Run

Use Windows Command Prompt:

```cmd
cd /d C:\Users\jiyoo\GithubRepo\Fashion-Retrieval-Demo
set PYTHONPATH=src
python app.py
```

Open:

```text
http://127.0.0.1:7860
```

## Demo Structure

```text
.
|-- app.py                         # Gradio Space entry point
|-- requirements.txt
|-- fashion-how/
|   `-- image/                     # sample image assets
`-- src/
    `-- fashion_how_graphdb/
        |-- search.py              # query parsing, Cypher build, reranking
        |-- cypher.py              # Neo4j defaults and import helpers
        |-- taxonomy.py
        |-- category_taxonomy.py
        `-- vlm.py                 # OpenAI Responses API helpers
```

## Neo4j Item Properties

For image display, each `Item` should include an `image_file` property that
matches a file under `fashion-how/image`.

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
