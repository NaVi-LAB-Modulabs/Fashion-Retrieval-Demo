"""Framework-independent retrieval service shared by the web API and preview."""

from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import quote

from . import search
from .diagnostics import retrieval_stage
from .neo4j_config import (
    DEFAULT_NEO4J_DATABASE, DEFAULT_NEO4J_PASSWORD,
    DEFAULT_NEO4J_URI, DEFAULT_NEO4J_USER,
)
from .taxonomy import CATEGORY_NAMES
from .llm import default_model

ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT / "fashion-how" / "image"


@lru_cache(maxsize=1)
def image_index() -> dict[str, Path]:
    return {
        path.name: path for path in sorted(IMAGE_DIR.rglob("*"))
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        and path.resolve().is_relative_to(IMAGE_DIR.resolve())
    }


def image_url(filename: Any) -> str | None:
    name = str(filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    return f"/images/{quote(name)}" if name in image_index() else None


def configuration(*, preview: bool = False) -> dict[str, Any]:
    configured = bool(os.getenv("OPENAI_API_KEY") and DEFAULT_NEO4J_URI and DEFAULT_NEO4J_PASSWORD)
    catalog = search.search_catalog()
    return {
        "ready": configured and not preview,
        "preview": preview,
        "default_model": default_model(),
        "models": list(dict.fromkeys([default_model(), "gpt-4.1-mini", "gpt-4.1"])),
        "image_count": len(image_index()),
        "type_count": len(catalog["item_types"]),
        "style_axis_count": len(catalog["style_axes"]),
        "type_names": CATEGORY_NAMES,
        "style_axes": search.STYLE_AXES,
    }


def catalog_preview() -> dict[str, Any]:
    from .hf_images import sample_catalog
    return sample_catalog()


@lru_cache(maxsize=1)
def openai_client() -> Any:
    from openai import OpenAI
    return OpenAI(timeout=60.0, max_retries=1)


@lru_cache(maxsize=1)
def neo4j_driver() -> Any:
    from neo4j import GraphDatabase
    return GraphDatabase.driver(
        DEFAULT_NEO4J_URI, auth=(DEFAULT_NEO4J_USER, DEFAULT_NEO4J_PASSWORD),
        connection_timeout=15.0, connection_acquisition_timeout=20.0,
    )


def validate_request(payload: dict[str, Any]) -> dict[str, Any]:
    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Describe the fashion items you are looking for.")
    if len(query) > 2000:
        raise ValueError("Keep your query under 2,000 characters.")
    limit = payload.get("limit", 12)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 50:
        raise ValueError("Results must be a whole number between 1 and 50.")
    model = payload.get("model") or default_model()
    if not isinstance(model, str) or len(model) > 100 or not model.strip():
        raise ValueError("Choose a valid parser model.")
    values = {"query": query.strip(), "limit": limit, "model": model.strip()}
    for name in ("min_confidence", "min_score"):
        value = payload.get(name, 0.0)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1
        ):
            raise ValueError("Score and edge confidence must be between 0 and 1.")
        values[name] = value
    return values


def retrieve(payload: dict[str, Any]) -> dict[str, Any]:
    request = validate_request(payload)
    if not configuration()["ready"]:
        raise RuntimeError("Search is not configured. Set the OpenAI and Neo4j environment variables on the server.")
    started = perf_counter()
    with retrieval_stage("query parsing"):
        extraction = search.extract_search_filters(
            request["query"], client=openai_client(), model=request["model"],
        )
    parsed = perf_counter()
    with retrieval_stage("query construction"):
        soft = search.needs_soft_rerank(extraction)
        cypher, params = search.build_search_cypher(
            extraction,
            limit=search.soft_candidate_limit(request["limit"]) if soft else request["limit"],
            min_confidence=request["min_confidence"],
            min_score=None if soft else request["min_score"],
        )
    with retrieval_stage("Neo4j retrieval"):
        driver = neo4j_driver()
        with driver.session(**({"database": DEFAULT_NEO4J_DATABASE} if DEFAULT_NEO4J_DATABASE else {})) as session:
            from neo4j import Query
            raw_results = [dict(row["result"]) for row in session.run(Query(cypher, timeout=30.0), **params)]
    retrieved = perf_counter()
    with retrieval_stage("reranking"):
        items = search.rerank_search_results(
            raw_results, extraction, client=openai_client() if soft else None,
            embedding_model=search.DEFAULT_EMBEDDING_MODEL, limit=request["limit"],
            min_score=request["min_score"] if soft else None,
        )
    # Only expose fields needed to explain retrieval; never serialize arbitrary DB properties.
    fields = ("id", "category", "category_name", "type_code", "type_name", "score", "graph_score", "text_score",
              "style_score", "score_components", "matched_filters", "mapped_attributes", "style_axis_matches")
    results = []
    for rank, item in enumerate(items, 1):
        public = {key: item.get(key) for key in fields}
        local_url = image_url(item.get("image_file"))
        item_id = item.get("item_ID") or item.get("id")
        remote_id = str(item_id) if item_id is not None and not local_url else None
        remote_url = f"/images/hf/{quote(remote_id, safe='')}.jpg" if remote_id else None
        public["id"] = item.get("id") or item_id
        public.update(rank=rank, image_url=local_url or remote_url, image_id=remote_id)
        public["type_code"] = item.get("category") or item.get("type_code")
        public["type_name"] = item.get("category_name") or item.get("type_name") or CATEGORY_NAMES.get(public["type_code"], "Garment")
        results.append(public)
    finished = perf_counter()
    return {
        "query": request["query"], "settings": request, "items": results,
        "extraction": extraction, "cypher": cypher, "params": params,
        "candidate_count": len(raw_results), "result_count": len(results),
        "reranked": soft, "embedding_model": search.DEFAULT_EMBEDDING_MODEL,
        "timings": {"parse_ms": round((parsed - started) * 1000),
                    "graph_ms": round((retrieved - parsed) * 1000),
                    "rerank_ms": round((finished - retrieved) * 1000),
                    "total_ms": round((finished - started) * 1000)},
    }
