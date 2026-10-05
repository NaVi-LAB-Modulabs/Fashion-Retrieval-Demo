"""Framework-independent retrieval service shared by the web API and preview."""

from __future__ import annotations

import math
import json
import re
import os
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import quote

from . import hybrid_search, search
from .diagnostics import retrieval_stage
from .neo4j_config import (
    DEFAULT_NEO4J_DATABASE, DEFAULT_NEO4J_PASSWORD,
    DEFAULT_NEO4J_URI, DEFAULT_NEO4J_USER,
)
from .taxonomy import CATEGORY_NAMES
from .llm import default_model

ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT / "sample_images"
SAMPLE_IDS_FILE = ROOT / "sample_ids.json"


@lru_cache(maxsize=1)
def sample_ids() -> dict[str, list[str]]:
    data = json.loads(SAMPLE_IDS_FILE.read_text(encoding="utf-8"))
    samples = data["samples"]
    if not isinstance(samples, dict) or any(
        not isinstance(category, str) or not isinstance(ids, list)
        or any(not isinstance(item_id, str) or not re.fullmatch(r"[0-9]+_0", item_id) for item_id in ids)
        for category, ids in samples.items()
    ):
        raise ValueError("Invalid sample_ids.json")
    return samples


@lru_cache(maxsize=1)
def image_index() -> dict[str, Path]:
    allowed = {item_id for ids in sample_ids().values() for item_id in ids}
    return {
        path.name: path for path in sorted(IMAGE_DIR.rglob("*"))
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        and path.stem in allowed
        and path.resolve().is_relative_to(IMAGE_DIR.resolve())
    }


def image_url(item_id: Any) -> str | None:
    if not isinstance(item_id, str):
        return None
    for extension in (".jpg", ".jpeg", ".png", ".webp"):
        name = item_id + extension
        if name in image_index():
            return f"/images/{quote(name)}"
    return None


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
        "default_weights": hybrid_search.DEFAULT_WEIGHTS,
    }


def catalog_preview() -> dict[str, Any]:
    items = []
    for category, ids in sample_ids().items():
        for item_id in ids:
            url = image_url(item_id)
            if url:
                items.append({"id": item_id, "image_id": item_id,
                              "image_url": url, "type_name": category})
            if len(items) == 12:
                break
        if len(items) == 12:
            break
    return {"items": items, "source": "local", "ranked": False,
            "status": "ok" if items else "unavailable"}


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
            raise ValueError("Final score and attribute match threshold must be between 0 and 1.")
        values[name] = value
    weights = payload.get("weights", hybrid_search.DEFAULT_WEIGHTS)
    if not isinstance(weights, dict) or set(weights) != set(hybrid_search.DEFAULT_WEIGHTS):
        raise ValueError("Provide text, graph, and style weights.")
    if any(
        isinstance(weight, bool) or not isinstance(weight, (int, float))
        or not math.isfinite(weight) or not 0 <= weight <= 1
        for weight in weights.values()
    ) or not any(weight > 0 for weight in weights.values()):
        raise ValueError("Weights must be between 0 and 1, with at least one above zero.")
    values["weights"] = {name: float(weights[name]) for name in hybrid_search.DEFAULT_WEIGHTS}
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
    driver = neo4j_driver()
    with driver.session(**({"database": DEFAULT_NEO4J_DATABASE} if DEFAULT_NEO4J_DATABASE else {})) as session:
        with retrieval_stage("candidate selection"):
            candidate_ids, embedding = hybrid_search.select_candidates(
                session, extraction, request["query"], client=openai_client(),
                embedding_model=search.DEFAULT_EMBEDDING_MODEL,
                limit=request["limit"], min_confidence=request["min_confidence"],
            )
        with retrieval_stage("Neo4j retrieval"):
            cypher = hybrid_search.result_cypher(extraction)
            query_params = hybrid_search.result_params(
                extraction, candidate_ids, embedding, search.DEFAULT_EMBEDDING_MODEL,
            )
            raw_results = [
                dict(row["result"]) for row in session.run(cypher, **query_params)
            ] if candidate_ids else []
    retrieved = perf_counter()
    with retrieval_stage("reranking"):
        items = hybrid_search.rerank(
            raw_results, extraction, limit=request["limit"],
            min_confidence=request["min_confidence"], min_score=request["min_score"],
            weights=request["weights"],
        )
    params = {
        **hybrid_search.candidate_cypher(extraction)[1],
        "candidate_ids": candidate_ids,
        "item_type_codes": query_params["item_type_codes"],
        "candidate_branch_limit": hybrid_search.CANDIDATE_BRANCH_SIZE,
        "vector_index": hybrid_search.DESCRIPTION_VECTOR_INDEX,
        "filtered_ids": "<all hard-filtered IDs with matching embedding model; omitted>",
        "embedding_model": search.DEFAULT_EMBEDDING_MODEL,
        "query_embedding": f"<{len(embedding)} dimensions; omitted from response>",
        "min_confidence": request["min_confidence"],
        "min_score": request["min_score"],
        "weights": request["weights"],
    }
    # Only expose fields needed to explain retrieval; never serialize arbitrary DB properties.
    fields = ("id", "category", "category_name", "type_code", "type_name", "score", "graph_score", "text_score",
              "style_score", "score_components", "matched_filters", "mapped_attributes", "style_axis_matches")
    results = []
    for rank, item in enumerate(items, 1):
        public = {key: item.get(key) for key in fields}
        item_id = item.get("item_ID") or item.get("id")
        image_id = str(item_id) if item_id is not None else None
        public["id"] = item.get("id") or item_id
        public.update(rank=rank, image_url=image_url(image_id), image_id=image_id)
        public["type_code"] = item.get("category") or item.get("type_code")
        public["type_name"] = item.get("category_name") or item.get("type_name") or CATEGORY_NAMES.get(public["type_code"], "Garment")
        results.append(public)
    finished = perf_counter()
    return {
        "query": request["query"], "settings": request, "items": results,
        "extraction": extraction, "cypher": cypher, "params": params,
        "candidate_cypher": hybrid_search.candidate_cypher(extraction)[0],
        "description_candidate_cypher": hybrid_search.description_candidate_cypher(),
        "candidate_count": len(raw_results), "result_count": len(results),
        "reranked": True, "embedding_model": search.DEFAULT_EMBEDDING_MODEL,
        "timings": {"parse_ms": round((parsed - started) * 1000),
                    "graph_ms": round((retrieved - parsed) * 1000),
                    "rerank_ms": round((finished - retrieved) * 1000),
                    "total_ms": round((finished - started) * 1000)},
    }
