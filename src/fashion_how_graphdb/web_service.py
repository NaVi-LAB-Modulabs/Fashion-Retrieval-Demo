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

from . import baseline, hybrid_search, search
from .diagnostics import RetrievalFailure, retrieval_stage
from .neo4j_config import (
    DEFAULT_NEO4J_DATABASE, DEFAULT_NEO4J_PASSWORD,
    DEFAULT_NEO4J_URI, DEFAULT_NEO4J_USER,
)
from .taxonomy import CATEGORY_NAMES
from .llm import default_model

ROOT = Path(__file__).resolve().parents[2]
# Served from the CDN on Vercel (public/ is static there) and by the /images route locally.
IMAGE_DIR = ROOT / "public" / "images"
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
def manifest_ids() -> frozenset[str]:
    return frozenset(item_id for ids in sample_ids().values() for item_id in ids)


@lru_cache(maxsize=1)
def image_index() -> dict[str, Path]:
    allowed = manifest_ids()
    return {
        path.name: path for path in sorted(IMAGE_DIR.rglob("*"))
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        and path.stem in allowed
        and path.resolve().is_relative_to(IMAGE_DIR.resolve())
    }


def image_url(item_id: Any) -> str | None:
    # Every manifest ID has public/images/<id>.jpg. The manifest, not the filesystem, decides,
    # because the deployed function bundle excludes public/.
    if not isinstance(item_id, str) or item_id not in manifest_ids():
        return None
    return f"/images/{quote(item_id)}.jpg"


def configuration(*, preview: bool = False) -> dict[str, Any]:
    neo4j_configured = bool(DEFAULT_NEO4J_URI and DEFAULT_NEO4J_PASSWORD)
    configured = bool(os.getenv("OPENAI_API_KEY") and neo4j_configured)
    catalog = search.search_catalog()
    return {
        "ready": configured and not preview,
        "preview": preview,
        # The baseline needs only Neo4j and the exported CLIP text model, not OpenAI.
        "baseline_ready": neo4j_configured and baseline.encoder_available() and not preview,
        "baseline_model": baseline.CLIP_MODEL,
        "default_model": default_model(),
        "models": list(dict.fromkeys([default_model(), "gpt-4.1-mini", "gpt-4.1"])),
        "image_count": len(manifest_ids()),
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
        connection_timeout=15.0, connection_acquisition_timeout=10.0,
        # Pooled connections can die while idle (network change, server-side idle close);
        # check ones idle for 30 s before reuse so a dead one is replaced, not used.
        # A check against an unreachable peer waits out the acquisition timeout.
        liveness_check_timeout=30.0,
    )


def neo4j_read(work: Any) -> Any:
    """Run read-only `work(session)`, retrying once if a pooled connection was dead.

    The failed attempt discards the dead connections, so the retry gets a fresh one.
    """
    from neo4j import exceptions
    retryable = tuple(error for error in (
        exceptions.ServiceUnavailable, exceptions.SessionExpired,
        getattr(exceptions, "ConnectionAcquisitionTimeoutError", None),  # neo4j >= 6
    ) if error)
    for attempt in (1, 2):
        try:
            with neo4j_driver().session(**({"database": DEFAULT_NEO4J_DATABASE} if DEFAULT_NEO4J_DATABASE else {})) as session:
                return work(session)
        except RetrievalFailure as exc:
            if attempt == 2 or not isinstance(exc.cause, retryable):
                raise
        except retryable:
            if attempt == 2:
                raise


def item_identity(item: dict[str, Any]) -> dict[str, Any]:
    item_id = item.get("item_ID") or item.get("id")
    image_id = str(item_id) if item_id is not None else None
    type_code = item.get("category") or item.get("type_code")
    return {
        "id": item.get("id") or item_id, "image_id": image_id, "image_url": image_url(image_id),
        "type_code": type_code,
        "type_name": item.get("category_name") or item.get("type_name") or CATEGORY_NAMES.get(type_code, "Garment"),
    }


def validate_baseline_request(payload: dict[str, Any]) -> dict[str, Any]:
    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Describe the fashion items you are looking for.")
    if len(query) > 2000:
        raise ValueError("Keep your query under 2,000 characters.")
    limit = payload.get("limit", 12)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 50:
        raise ValueError("Results must be a whole number between 1 and 50.")
    return {"query": query.strip(), "limit": limit}


def validate_request(payload: dict[str, Any]) -> dict[str, Any]:
    values = validate_baseline_request(payload)
    model = payload.get("model") or default_model()
    if not isinstance(model, str) or len(model) > 100 or not model.strip():
        raise ValueError("Choose a valid parser model.")
    values["model"] = model.strip()
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

    def select_and_fetch(session: Any) -> tuple[Any, ...]:
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
        return candidate_ids, embedding, cypher, query_params, raw_results

    candidate_ids, embedding, cypher, query_params, raw_results = neo4j_read(select_and_fetch)
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
        public.update(rank=rank, **item_identity(item))
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


def retrieve_baseline(payload: dict[str, Any]) -> dict[str, Any]:
    """Rank every item by CLIP cosine between the raw query and its image embedding."""
    request = validate_baseline_request(payload)
    if not configuration()["baseline_ready"]:
        raise RuntimeError("The CLIP baseline is not available. Provide the CLIP text model (models/) and configure Neo4j.")
    started = perf_counter()
    with retrieval_stage("CLIP query encoding"):
        embedding = baseline.text_embedding(baseline.clip_encoder(), request["query"])
    encoded = perf_counter()
    with retrieval_stage("baseline similarity search"):
        rows = neo4j_read(lambda session: baseline.search(session, embedding, limit=request["limit"]))
    finished = perf_counter()
    items = [
        {"rank": rank, **item_identity(row), "score": round(float(row["score"]), 6)}
        for rank, row in enumerate(rows, 1)
    ]
    return {
        "query": request["query"], "settings": request, "items": items,
        "model": baseline.CLIP_MODEL, "cypher": baseline.BASELINE_CYPHER,
        "params": {
            "embedding_model": baseline.CLIP_MODEL,
            "query_embedding": f"<{len(embedding)} dimensions; omitted from response>",
            "limit": request["limit"],
        },
        "result_count": len(items),
        "timings": {"encode_ms": round((encoded - started) * 1000),
                    "search_ms": round((finished - encoded) * 1000),
                    "total_ms": round((finished - started) * 1000)},
    }
