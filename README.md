---
title: Fashion How Graph Search
sdk: docker
app_port: 7860
---

# Fashion-How Graph Search Demo

Fashion-How 상품을 Neo4j GraphDB와 OpenAI 기반 query parser로 검색하는 데모입니다.

이 데모는 자연어 검색어를 다음 신호로 분해합니다.

- Hard filters: item type, color, material, style, occasion, pattern, season, category attributes
- Description query: hard filter로 설명되지 않은 남은 표현만 semantic vector score에 사용
- Style axis targets: query에 드러난 축만 선택해서 item axis score와의 거리로 점수화

최종 점수는 활성화된 component만 평균냅니다.

```text
final_score = average(graph_score, text_score, style_score)
```

모든 query 내용이 hard filter로 해결되면 `text_score`와 `style_score`는 계산하지 않습니다.

## Demo Structure

```text
.
├── Dockerfile
├── README.md
├── requirements.txt
├── fashion-how/
│   └── image/                  # small image sample only
└── src/
    └── fashion_how_graphdb/
        ├── search_server.py     # local HTTP server
        ├── search_viewer.html   # browser UI
        ├── search.py            # extraction, Cypher build, reranking
        ├── cypher.py            # Neo4j defaults/helpers
        ├── category_taxonomy.py
        ├── taxonomy.py
        ├── prompts.py
        └── vlm.py
```

## Required Space Secrets

Set these in Hugging Face Space Settings.

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

## Neo4j Item Properties

For image display, each `Item` should include an `image_file` property that matches a file under `fashion-how/image`.

For description vector reranking, each `Item` can include one of:

```text
description_embedding
image_description_embedding
text_description_embedding
text_embedding
embedding
```

For style axis reranking, each `Item` can include:

```text
trendy_classic
cool_warm
feminine_mannish
minimal_maximal
casual_formal
soft_sharp
young_mature
```

Each axis value is a float from `0.0` to `1.0`, where `0.0` is the left pole and `1.0` is the right pole.

## Local Run

Use Windows Command Prompt:

```cmd
cd /d C:\Users\jiyoo\GithubRepo\Fashion-Retrieval-Demo
set PYTHONPATH=src
python -m fashion_how_graphdb.search_server --host 127.0.0.1 --port 7860
```

Open:

```text
http://127.0.0.1:7860
```

## Hugging Face Spaces

Create a new Space with Docker SDK, then push this folder.

The container starts with:

```cmd
python -m fashion_how_graphdb.search_server --host 0.0.0.0 --port 7860
```
