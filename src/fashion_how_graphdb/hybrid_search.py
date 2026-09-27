"""Candidate union and reranking for Fashion-200K text search."""

from __future__ import annotations

import os
from typing import Any

from . import search
from .category_taxonomy import category_attribute_id

VECTOR_INDEX_NAME = os.getenv("NEO4J_DESCRIPTION_VECTOR_INDEX", "item_description_embedding")
VECTOR_POOL_SIZE = 500
ATTRIBUTE_POOL_SIZE = 150
TEXT_WEIGHT = 0.55
ATTRIBUTE_WEIGHT = 0.35
STYLE_WEIGHT = 0.10
DEFAULT_WEIGHTS = {"text": TEXT_WEIGHT, "graph": ATTRIBUTE_WEIGHT, "style": STYLE_WEIGHT}

VECTOR_CYPHER = """
CALL db.index.vector.queryNodes($index_name, $pool_size, $query_embedding)
YIELD node, score
WHERE node.description_embedding_model = $embedding_model
  AND ($item_type_codes = [] OR EXISTS {
    MATCH (node)-[:IS_CATEGORY]->(category:Category)
    WHERE category.id IN $item_type_codes
  })
RETURN node.id AS id, score
"""


def query_embedding(client: Any, text: str, model: str) -> list[float]:
    response = client.embeddings.create(model=model, input=text)
    return [float(value) for value in response.data[0].embedding]


def attribute_candidate_cypher(search_filter: dict[str, Any]) -> tuple[str, str]:
    if search_filter["scope"] == "common":
        spec = search.COMMON_FILTER_GROUPS[search_filter["group"]]
        relation, label = spec["rel_type"], spec["node_label"]
        value_id = search_filter["value"]
    else:
        relation, label = "HAS_ATTRIBUTE", "Attribute"
        value_id = category_attribute_id(search_filter["group"], search_filter["value"])
    cypher = (
        f"MATCH (item:Item)-[edge:{relation}]->(:{label} {{id: $value_id}})\n"
        "WHERE coalesce(edge.confidence, 0.0) >= $min_confidence\n"
        "AND ($item_type_codes = [] OR EXISTS {\n"
        "  MATCH (item)-[:IS_CATEGORY]->(category:Category)\n"
        "  WHERE category.id IN $item_type_codes\n"
        "})\n"
        "RETURN item.id AS id\n"
        "ORDER BY coalesce(edge.confidence, 0.0) DESC, item.id ASC\n"
        "LIMIT $pool_size"
    )
    return cypher, value_id


def required_attribute_filters(extraction: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        search_filter
        for search_filter in [
            *(extraction.get("common_filters") or []),
            *(extraction.get("category_filters") or []),
        ]
        if search_filter.get("hard") is True
    ]


def select_candidates(
    session: Any,
    extraction: dict[str, Any],
    query: str,
    *,
    client: Any,
    embedding_model: str,
    limit: int,
    min_confidence: float | None,
) -> tuple[list[str], list[float]]:
    # description_query keeps visual terms not represented by graph attributes.
    semantic_query = search._normalize_description_query(extraction.get("description_query")) or query
    embedding = query_embedding(client, semantic_query, embedding_model)
    vector_rows = session.run(
        VECTOR_CYPHER,
        index_name=VECTOR_INDEX_NAME,
        pool_size=max(VECTOR_POOL_SIZE, limit * 30),
        query_embedding=embedding,
        embedding_model=embedding_model,
        item_type_codes=extraction.get("item_type_codes") or [],
    )
    ids = [row["id"] for row in vector_rows]
    for search_filter in required_attribute_filters(extraction):
        cypher, value_id = attribute_candidate_cypher(search_filter)
        rows = session.run(
            cypher,
            value_id=value_id,
            item_type_codes=extraction.get("item_type_codes") or [],
            min_confidence=min_confidence or 0.0,
            pool_size=max(ATTRIBUTE_POOL_SIZE, limit * 5),
        )
        ids.extend(row["id"] for row in rows)
    return list(dict.fromkeys(ids)), embedding


def result_cypher(extraction: dict[str, Any]) -> str:
    axis_properties = ", ".join(f".{axis}" for axis in search.STYLE_AXES)
    return (
        "MATCH (item:Item)-[:IS_CATEGORY]->(item_category:Category)\n"
        "WHERE item.id IN $candidate_ids\n"
        "AND ($item_type_codes = [] OR item_category.id IN $item_type_codes)\n"
        "CALL (item) {\n"
        "  OPTIONAL MATCH (item)-[mapped_rel]->(mapped_node)\n"
        "  WHERE type(mapped_rel) IN [\n"
        "    'HAS_COLOR', 'HAS_MATERIAL', 'HAS_PATTERN', 'HAS_SEASON', 'HAS_ATTRIBUTE'\n"
        "  ]\n"
        "  WITH collect({\n"
        "    scope: CASE type(mapped_rel) WHEN 'HAS_ATTRIBUTE' THEN 'category' ELSE 'common' END,\n"
        "    rel_type: type(mapped_rel),\n"
        "    group: coalesce(mapped_node.group, mapped_rel.group, CASE type(mapped_rel)\n"
        "      WHEN 'HAS_COLOR' THEN 'colors'\n"
        "      WHEN 'HAS_MATERIAL' THEN 'materials'\n"
        "      WHEN 'HAS_PATTERN' THEN 'patterns'\n"
        "      WHEN 'HAS_SEASON' THEN 'seasons'\n"
        "      ELSE null END),\n"
        "    type_code: coalesce(mapped_node.type_code, mapped_node.category),\n"
        "    value: mapped_node.name,\n"
        "    confidence: mapped_rel.confidence,\n"
        "    coverage: mapped_rel.coverage,\n"
        "    prominence: mapped_rel.prominence,\n"
        "    strength: mapped_rel.score,\n"
        "    score: coalesce(mapped_rel.score, mapped_rel.prominence,\n"
        "                    mapped_rel.coverage, mapped_rel.confidence)\n"
        "  }) AS raw_attributes\n"
        "  RETURN [attr IN raw_attributes WHERE attr.value IS NOT NULL] AS mapped_attributes\n"
        "}\n"
        "WITH item, mapped_attributes,\n"
        "  CASE WHEN item.description_embedding_model = $embedding_model\n"
        "    AND item.description_embedding IS NOT NULL\n"
        "  THEN vector.similarity.cosine(item.description_embedding, $query_embedding)\n"
        "  ELSE null END AS text_score\n"
        f"RETURN item {{.id, .item_ID, .category, .category_name, .type_code, .type_name, {axis_properties},\n"
        "  text_score: text_score, mapped_attributes: mapped_attributes} AS result"
    )


def result_params(
    extraction: dict[str, Any],
    candidate_ids: list[str],
    embedding: list[float],
    embedding_model: str,
) -> dict[str, Any]:
    return {
        "candidate_ids": candidate_ids,
        "item_type_codes": extraction.get("item_type_codes") or [],
        "query_embedding": embedding,
        "embedding_model": embedding_model,
    }


def _number(value: Any) -> float | None:
    return search._to_float(value)


def attribute_matches(
    search_filter: dict[str, Any],
    attribute: dict[str, Any],
    min_confidence: float | None,
) -> bool:
    if any(
        attribute.get(key) != search_filter[key]
        for key in ("scope", "group", "value")
    ):
        return False
    if search_filter["scope"] == "category" and (
        attribute.get("type_code") != search_filter["type_code"]
    ):
        return False
    confidence = _number(attribute.get("confidence"))
    return min_confidence is None or (
        confidence is not None and confidence >= min_confidence
    )


def passes_hard_filters(
    attributes: list[dict[str, Any]],
    extraction: dict[str, Any],
    min_confidence: float | None,
) -> bool:
    for search_filter in required_attribute_filters(extraction):
        if not any(
            attribute_matches(search_filter, attribute, min_confidence)
            for attribute in attributes
        ):
            return False
    excluded = [
        *(extraction.get("excluded_common_filters") or []),
        *(extraction.get("excluded_category_filters") or []),
    ]
    return not any(
        attribute_matches(search_filter, attribute, min_confidence)
        for search_filter in excluded
        for attribute in attributes
    )


def attribute_strength(attribute: dict[str, Any]) -> float:
    confidence = _number(attribute.get("confidence"))
    if confidence is None:
        return 0.0
    group = attribute.get("group")
    metric_name = (
        "coverage" if group == "colors" else
        "prominence" if group == "patterns" or attribute.get("scope") == "category" else
        "strength" if group == "seasons" else None
    )
    metric = _number(attribute.get(metric_name)) if metric_name else None
    confidence = max(0.0, min(1.0, confidence))
    if metric is None:
        return confidence
    return confidence * (0.5 + 0.5 * max(0.0, min(1.0, metric)))


def graph_score(
    attributes: list[dict[str, Any]],
    extraction: dict[str, Any],
    min_confidence: float | None,
) -> tuple[float | None, list[dict[str, Any]]]:
    filters = required_attribute_filters(extraction)
    if not filters:
        return None, []
    weights = {"materials": 0.5, "seasons": 0.3}
    total = 0.0
    denominator = 0.0
    matches = []
    for search_filter in filters:
        weight = weights.get(search_filter["group"], 1.0)
        denominator += weight
        found = [
            attribute for attribute in attributes
            if attribute_matches(search_filter, attribute, min_confidence)
        ]
        if not found:
            continue
        best = max(found, key=attribute_strength)
        strength = attribute_strength(best)
        total += weight * strength
        match = {
            "scope": search_filter["scope"],
            "group": search_filter["group"],
            "value": search_filter["value"],
            "score": round(strength, 6),
        }
        if search_filter["scope"] == "category":
            match["type_code"] = search_filter["type_code"]
        matches.append(match)
    return total / denominator, matches


def style_score(
    item: dict[str, Any],
    targets: list[dict[str, Any]],
) -> tuple[float | None, list[dict[str, Any]]]:
    scores = []
    matches = []
    for target in targets:
        axis = target["axis"]
        value = _number(item.get(axis))
        if value is None:
            continue
        value = max(0.0, min(1.0, value))
        target_value = float(target["target"])
        score = 1.0 - abs(value - target_value)
        scores.append(score)
        matches.append({
            "axis": axis,
            "target": target_value,
            "value": value,
            "score": round(score, 6),
            "evidence": target.get("evidence", ""),
        })
    return (sum(scores) / len(scores) if scores else None), matches


def rerank(
    results: list[dict[str, Any]],
    extraction: dict[str, Any],
    *,
    limit: int,
    min_confidence: float | None,
    min_score: float | None,
    weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    selected_weights = DEFAULT_WEIGHTS if weights is None else weights
    ranked = []
    for raw_item in results:
        item = dict(raw_item)
        attributes = item.get("mapped_attributes") or []
        if not passes_hard_filters(attributes, extraction, min_confidence):
            continue
        graph, matched_filters = graph_score(attributes, extraction, min_confidence)
        text = _number(item.get("text_score"))
        style, style_matches = style_score(
            item, extraction.get("style_axis_targets") or []
        )
        components = [
            (selected_weights["graph"], graph),
            (selected_weights["text"], text),
            (selected_weights["style"], style),
        ]
        active = [(weight, score) for weight, score in components
                  if weight > 0 and score is not None]
        final = (
            sum(weight * score for weight, score in active) /
            sum(weight for weight, _ in active)
            if active else 0.0
        )
        if min_score is not None and final < min_score:
            continue
        item["graph_score"] = round(graph, 6) if graph is not None else None
        item["text_score"] = round(text, 6) if text is not None else None
        item["style_score"] = round(style, 6) if style is not None else None
        item["matched_filters"] = matched_filters
        item["style_axis_matches"] = style_matches
        item["score_components"] = [
            name for name, weight, score in (
                ("graph", selected_weights["graph"], graph),
                ("text", selected_weights["text"], text),
                ("style", selected_weights["style"], style),
            ) if weight > 0 and score is not None
        ]
        item["score"] = round(final, 6)
        ranked.append(item)
    ranked.sort(key=lambda item: (-item["score"], str(item.get("id") or "")))
    return ranked[:limit]


def retrieve_parsed(
    session: Any,
    extraction: dict[str, Any],
    query: str,
    *,
    client: Any,
    embedding_model: str,
    limit: int,
    min_confidence: float | None,
    min_score: float | None,
) -> dict[str, Any]:
    candidate_ids, embedding = select_candidates(
        session, extraction, query, client=client,
        embedding_model=embedding_model, limit=limit,
        min_confidence=min_confidence,
    )
    cypher = result_cypher(extraction)
    params = result_params(extraction, candidate_ids, embedding, embedding_model)
    raw_results = [
        dict(row["result"]) for row in session.run(cypher, **params)
    ] if candidate_ids else []
    items = rerank(
        raw_results, extraction, limit=limit,
        min_confidence=min_confidence, min_score=min_score,
    )
    return {
        "cypher": cypher,
        "params": {
            "candidate_ids": candidate_ids,
            "item_type_codes": params["item_type_codes"],
            "vector_index": VECTOR_INDEX_NAME,
            "embedding_model": embedding_model,
            "query_embedding": f"<{len(embedding)} dimensions>",
            "min_confidence": min_confidence,
            "min_score": min_score,
        },
        "candidate_count": len(raw_results),
        "items": items,
    }
