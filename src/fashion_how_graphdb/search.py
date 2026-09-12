"""Natural-language search over the Fashion-200K Neo4j graph."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

from .category_taxonomy import CATEGORY_ATTRIBUTES, category_attribute_id
from .neo4j_config import (
    DEFAULT_NEO4J_DATABASE,
    DEFAULT_NEO4J_PASSWORD,
    DEFAULT_NEO4J_URI,
    DEFAULT_NEO4J_USER,
)
from .taxonomy import (
    COLORS,
    MATERIALS,
    PATTERNS,
    SEASONS,
    CATEGORY_NAMES,
)
from .llm import call_text_json_with_error, default_model

COMMON_FILTER_GROUPS = {
    "colors": {
        "label": "Color",
        "rel_type": "HAS_COLOR",
        "node_label": "Color",
        "values": COLORS,
    },
    "materials": {
        "label": "Material",
        "rel_type": "HAS_MATERIAL",
        "node_label": "Material",
        "values": MATERIALS,
    },
    "patterns": {
        "label": "Pattern",
        "rel_type": "HAS_PATTERN",
        "node_label": "Pattern",
        "values": PATTERNS,
    },
    "seasons": {
        "label": "Season",
        "rel_type": "HAS_SEASON",
        "node_label": "Season",
        "values": SEASONS,
    },
}

STYLE_AXES = {
    "trendy_classic": {
        "label": "Trendy <-> Classic",
        "left": "trendy",
        "right": "classic",
        "meaning": "trend-sensitive, current, novelty-driven vs timeless, enduring, long-wearing",
    },
    "cool_warm": {
        "label": "Cool <-> Warm",
        "left": "cool",
        "right": "warm",
        "meaning": "cool, crisp, refreshing visual temperature vs warm, cozy, heated visual temperature",
    },
    "feminine_mannish": {
        "label": "Feminine <-> Mannish",
        "left": "feminine",
        "right": "mannish",
        "meaning": "feminine, delicate, romantic vs mannish, tailored, masculine-coded",
    },
    "minimal_maximal": {
        "label": "Minimal <-> Maximal",
        "left": "minimal",
        "right": "maximal",
        "meaning": "few elements, restrained styling vs many elements, decorative or visually busy styling",
    },
    "casual_formal": {
        "label": "Casual <-> Formal",
        "left": "casual",
        "right": "formal",
        "meaning": "relaxed daily wear vs appropriate for formal or polished situations",
    },
    "soft_sharp": {
        "label": "Soft <-> Sharp",
        "left": "soft",
        "right": "sharp",
        "meaning": "soft, rounded, gentle shape language vs sharp, crisp, angular shape language",
    },
    "young_mature": {
        "label": "Young <-> Mature",
        "left": "young",
        "right": "mature",
        "meaning": "cute, youthful impression vs mature, adult, composed impression",
    },
}

STYLE_AXIS_PROPERTY_CANDIDATES = {
    axis_id: [
        axis_id,
        f"style_{axis_id}",
        f"{axis_id}_score",
    ]
    for axis_id in STYLE_AXES
}

DESCRIPTION_EMBEDDING_PROPERTIES = (
    "description_embedding",
    "image_description_embedding",
    "text_description_embedding",
    "text_embedding",
    "embedding",
)

DEFAULT_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

SYSTEM_PROMPT = """You convert Korean or English fashion search text into Fashion-200K graph filters.
Map either input language to the exact English identifiers in the catalog.
item_type_codes are Fashion-200K category IDs: dresses, jackets, pants, skirts, tops.
Only colors, materials, patterns, and seasons are common hard-filter groups.
Use description_query for wearing occasions and style_axis_targets for supported
style intent; there are no styles or occasions graph filters.

Use only values that appear in the provided catalog. Never invent values.
Return extracted values exactly as they appear in the provided catalog.
If the user asks for a garment category, select matching item_type codes.

Extract filters that are explicitly stated or reasonably implied by the query.
Reasonable inference is allowed, but each inference must stay focused on the
attribute being extracted. Do not convert a clue for one attribute into another
attribute unless the query clearly supports that attribute.

Attribute guidance:
- colors, materials, patterns, and category details should describe the garment
  itself. A place, event, activity, or mood is not a fabric pattern, material,
  color, sleeve, collar, pocket, or fit unless the garment property is described.
- seasons may be inferred from explicit season words or strong seasonal wearing
  context, but avoid weak associations.

Minimal examples:
- "?? ????" or "blue blouse": item_type_codes ["tops"], color "blue".
- "???? ???" is wearing-context meaning for description_query.
- "???" is an activity clue; do not extract "floral" unless the query
  describes the garment as ???/???/floral.

For negated conditions such as "흰색이 아닌", "화이트 제외", or "not white",
put the value in excluded_common_filters or excluded_category_filters instead of
common_filters/category_filters.
After extracting hard filters, identify query meaning that is still not covered
by those filters:
- Put leftover descriptive meaning for semantic description matching in
  description_query. Keep it short. If all meaningful query content is already
  represented by item_type_codes and hard/excluded filters, return an empty
  string.
- Put leftover style intent in style_axis_targets only for axes clearly implied
  by the query. Use 0.0 for the left pole and 1.0 for the right pole. Do not
  choose every axis by default. If a style clue is already fully represented by
  a hard filter, omit the duplicate style axis.
- Valid style axes are trendy_classic, cool_warm, feminine_mannish,
  minimal_maximal, casual_formal, soft_sharp, young_mature.

When uncertain between extracting and omitting a filter, omit it.
Return strict JSON only."""

@dataclass(frozen=True)
class SearchFilter:
    scope: str
    group: str
    value: str
    type_code: str | None = None


def search_catalog() -> dict[str, Any]:
    category_groups: dict[str, Any] = {}
    for type_code, config in CATEGORY_ATTRIBUTES.items():
        category_groups[type_code] = {
            "name": CATEGORY_NAMES.get(type_code, type_code),
            "domain": config["domain"],
            "groups": {
                group_id: {
                    "name": group["name"],
                    "cardinality": group["cardinality"],
                    "values": group["values"],
                }
                for group_id, group in config["groups"].items()
            },
        }
    return {
        "item_types": CATEGORY_NAMES,
        "common_groups": {
            group_id: {
                "name": spec["label"],
                "values": spec["values"],
            }
            for group_id, spec in COMMON_FILTER_GROUPS.items()
        },
        "category_groups": category_groups,
        "style_axes": STYLE_AXES,
    }


def extraction_schema() -> dict[str, Any]:
    common_filter_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["group", "value"],
            "properties": {
                "group": {
                    "type": "string",
                    "enum": list(COMMON_FILTER_GROUPS),
                },
                "value": {"type": "string"},
            },
        },
    }
    category_filter_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["type_code", "group", "value"],
            "properties": {
                "type_code": {
                    "type": "string",
                    "enum": list(CATEGORY_ATTRIBUTES),
                },
                "group": {"type": "string"},
                "value": {"type": "string"},
            },
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "item_type_codes",
            "common_filters",
            "category_filters",
            "excluded_common_filters",
            "excluded_category_filters",
            "description_query",
            "style_axis_targets",
        ],
        "properties": {
            "item_type_codes": {
                "type": "array",
                "items": {"type": "string", "enum": list(CATEGORY_NAMES)},
            },
            "common_filters": common_filter_schema,
            "category_filters": category_filter_schema,
            "excluded_common_filters": common_filter_schema,
            "excluded_category_filters": category_filter_schema,
            "description_query": {"type": "string"},
            "style_axis_targets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["axis", "target", "evidence"],
                    "properties": {
                        "axis": {"type": "string", "enum": list(STYLE_AXES)},
                        "target": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                        "evidence": {"type": "string"},
                    },
                },
            },
        },
    }


def extraction_prompt(query: str) -> str:
    return json.dumps(
        {
            "query": query,
            "catalog": search_catalog(),
        },
        ensure_ascii=False,
        indent=2,
    )


def extract_search_filters(
    query: str,
    *,
    client: Any,
    model: str,
) -> dict[str, Any]:
    raw, error = call_text_json_with_error(
        client=client,
        model=model,
        user_prompt=extraction_prompt(query),
        system_prompt_text=SYSTEM_PROMPT,
        json_schema=extraction_schema(),
        json_schema_name="fashion_search_filters",
    )
    if raw is None:
        failure = RuntimeError("Search filter extraction failed")
        failure.provider_error = error
        raise failure
    return normalize_extraction(raw)


def normalize_extraction(raw: dict[str, Any]) -> dict[str, Any]:
    type_codes = [
        code
        for code in dict.fromkeys(raw.get("item_type_codes") or [])
        if code in CATEGORY_NAMES
    ]

    common_filters = _normalize_common_filters(raw.get("common_filters"))
    excluded_common_filters = _normalize_common_filters(
        raw.get("excluded_common_filters"),
    )
    category_filters = _normalize_category_filters(
        raw.get("category_filters"),
        type_codes,
    )
    excluded_category_filters = _normalize_category_filters(
        raw.get("excluded_category_filters"),
        type_codes,
    )

    return {
        "item_type_codes": type_codes,
        "common_filters": [filter_to_dict(item) for item in common_filters],
        "category_filters": [filter_to_dict(item) for item in category_filters],
        "excluded_common_filters": [
            filter_to_dict(item) for item in excluded_common_filters
        ],
        "excluded_category_filters": [
            filter_to_dict(item) for item in excluded_category_filters
        ],
        "description_query": _normalize_description_query(
            raw.get("description_query"),
        ),
        "style_axis_targets": _normalize_style_axis_targets(
            raw.get("style_axis_targets"),
        ),
    }


def _normalize_common_filters(
    raw_items: Any,
) -> list[SearchFilter]:
    filters: list[SearchFilter] = []
    for item in raw_items or []:
        if not isinstance(item, dict):
            continue
        group = item.get("group")
        value = item.get("value")
        if (
            group in COMMON_FILTER_GROUPS
            and value in COMMON_FILTER_GROUPS[group]["values"]
        ):
            filters.append(SearchFilter("common", group, value))
    return filters


def _normalize_category_filters(
    raw_items: Any,
    type_codes: list[str],
) -> list[SearchFilter]:
    filters: list[SearchFilter] = []
    for item in raw_items or []:
        if not isinstance(item, dict):
            continue
        type_code = item.get("type_code")
        group = item.get("group")
        value = item.get("value")
        groups = CATEGORY_ATTRIBUTES.get(type_code, {}).get("groups", {})
        if (
            _category_type_is_compatible(type_code, type_codes)
            and group in groups
            and value in groups[group]["values"]
        ):
            filters.append(SearchFilter("category", group, value, type_code=type_code))
    return filters


def _category_type_is_compatible(
    category_type_code: str,
    item_type_codes: list[str],
) -> bool:
    if not item_type_codes:
        return True
    return category_type_code in item_type_codes


def _to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _normalize_description_query(raw: Any) -> str:
    if not isinstance(raw, str):
        return ""
    return " ".join(raw.split())


def _normalize_style_axis_targets(raw_items: Any) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    seen = set()
    for item in raw_items or []:
        if not isinstance(item, dict):
            continue
        axis = item.get("axis")
        if axis not in STYLE_AXES or axis in seen:
            continue
        target = _to_float(item.get("target"))
        if target is None:
            continue
        seen.add(axis)
        targets.append(
            {
                "axis": axis,
                "target": max(0.0, min(1.0, target)),
                "evidence": _normalize_description_query(item.get("evidence")),
            }
        )
    return targets


def filter_to_dict(search_filter: SearchFilter) -> dict[str, Any]:
    payload = {
        "scope": search_filter.scope,
        "group": search_filter.group,
        "value": search_filter.value,
    }
    if search_filter.type_code:
        payload["type_code"] = search_filter.type_code
    return payload


def _rank_expr(rel_var: str) -> str:
    return (
        f"coalesce({rel_var}.score, {rel_var}.prominence, "
        f"{rel_var}.coverage, {rel_var}.confidence, 0.0)"
    )


def _has_graph_score(extraction: dict[str, Any]) -> bool:
    return bool(extraction.get("common_filters") or extraction.get("category_filters"))


def needs_soft_rerank(extraction: dict[str, Any]) -> bool:
    return bool(
        _normalize_description_query(extraction.get("description_query"))
        or extraction.get("style_axis_targets")
    )


def soft_candidate_limit(limit: int) -> int:
    return max(limit * 10, 100)


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(left * right for left, right in zip(a, b))
    norm_a = math.sqrt(sum(value * value for value in a))
    norm_b = math.sqrt(sum(value * value for value in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _scale_cosine(score: float) -> float:
    return max(0.0, min(1.0, (score + 1.0) / 2.0))


def _embedding_from_item(item: dict[str, Any]) -> list[float]:
    for property_name in DESCRIPTION_EMBEDDING_PROPERTIES:
        raw = item.get(property_name)
        if isinstance(raw, list) and raw:
            values = []
            for value in raw:
                number = _to_float(value)
                if number is None:
                    return []
                values.append(number)
            return values
    return []


def _item_axis_value(item: dict[str, Any], axis_id: str) -> float | None:
    style_axes = item.get("style_axes")
    if isinstance(style_axes, dict):
        value = _to_float(style_axes.get(axis_id))
        if value is not None:
            return max(0.0, min(1.0, value))
    for property_name in STYLE_AXIS_PROPERTY_CANDIDATES[axis_id]:
        value = _to_float(item.get(property_name))
        if value is not None:
            return max(0.0, min(1.0, value))
    return None


def _style_score(
    item: dict[str, Any],
    style_axis_targets: list[dict[str, Any]],
) -> tuple[float | None, list[dict[str, Any]]]:
    scores = []
    matches = []
    for target in style_axis_targets:
        axis_id = target["axis"]
        item_value = _item_axis_value(item, axis_id)
        if item_value is None:
            continue
        target_value = float(target["target"])
        score = 1.0 - abs(item_value - target_value)
        scores.append(score)
        matches.append(
            {
                "axis": axis_id,
                "target": target_value,
                "value": item_value,
                "score": round(score, 6),
                "evidence": target.get("evidence", ""),
            }
        )
    if not scores:
        return None, matches
    return sum(scores) / len(scores), matches


def _get_text_embedding(client: Any, text: str, model: str) -> list[float]:
    response = client.embeddings.create(model=model, input=text)
    return [float(value) for value in response.data[0].embedding]


def strip_large_vector_properties(item: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(item)
    for property_name in DESCRIPTION_EMBEDDING_PROPERTIES:
        cleaned.pop(property_name, None)
    cleaned.pop("image_embedding", None)
    return cleaned


def rerank_search_results(
    results: list[dict[str, Any]],
    extraction: dict[str, Any],
    *,
    client: Any | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    limit: int = 20,
    min_score: float | None = None,
) -> list[dict[str, Any]]:
    if not results:
        return []
    description_query = _normalize_description_query(extraction.get("description_query"))
    style_axis_targets = extraction.get("style_axis_targets") or []
    graph_score_active = _has_graph_score(extraction)
    query_embedding = []
    if description_query:
        if client is None:
            raise RuntimeError("OpenAI client is required for description vector scoring.")
        query_embedding = _get_text_embedding(client, description_query, embedding_model)

    reranked = []
    for raw_item in results:
        item = dict(raw_item)
        components = []
        graph_score = _to_float(item.get("score")) or 0.0
        if graph_score_active:
            components.append(graph_score)

        text_score = None
        if query_embedding:
            item_embedding = _embedding_from_item(item)
            if item_embedding:
                text_score = _scale_cosine(_cosine(query_embedding, item_embedding))
                components.append(text_score)

        style_score = None
        style_matches = []
        if style_axis_targets:
            style_score, style_matches = _style_score(item, style_axis_targets)
            if style_score is not None:
                components.append(style_score)

        if components:
            final_score = sum(components) / len(components)
        else:
            final_score = graph_score

        item = strip_large_vector_properties(item)
        item["graph_score"] = round(graph_score, 6)
        item["text_score"] = round(text_score, 6) if text_score is not None else None
        item["style_score"] = round(style_score, 6) if style_score is not None else None
        item["style_axis_matches"] = style_matches
        item["score_components"] = [
            name
            for name, active in (
                ("graph", graph_score_active),
                ("text", text_score is not None),
                ("style", style_score is not None),
            )
            if active
        ]
        item["score"] = round(final_score, 6)
        if min_score is None or final_score >= min_score:
            reranked.append(item)

    reranked.sort(key=lambda item: (-float(item.get("score") or 0.0), str(item.get("id") or "")))
    return reranked[:limit]


def build_search_cypher(
    extraction: dict[str, Any],
    *,
    limit: int = 20,
    min_confidence: float | None = None,
    min_score: float | None = None,
) -> tuple[str, dict[str, Any]]:
    params: dict[str, Any] = {
        "item_type_codes": extraction.get("item_type_codes") or [],
        "limit": limit,
    }
    lines = ["MATCH (item:Item)"]
    where_lines = []
    score_terms = []
    matched_terms = []

    if params["item_type_codes"]:
        lines.append("MATCH (item)-[:IS_CATEGORY]->(item_category:Category)")
        where_lines.append("item_category.id IN $item_type_codes")

    filter_index = 0
    for item in extraction.get("common_filters") or []:
        group = item["group"]
        spec = COMMON_FILTER_GROUPS[group]
        value_param = f"value_{filter_index}"
        params[value_param] = item["value"]
        rel_var = f"r{filter_index}"
        node_var = f"n{filter_index}"
        lines.append(
            f"MATCH (item)-[{rel_var}:{spec['rel_type']}]->"
            f"({node_var}:{spec['node_label']} {{id: ${value_param}}})"
        )
        if min_confidence is not None:
            where_lines.append(f"{rel_var}.confidence >= $min_confidence")
        score_terms.append(_rank_expr(rel_var))
        matched_terms.append(
            "{scope: 'common', group: '"
            + group
            + f"', value: {node_var}.name, score: {score_terms[-1]}}}"
        )
        filter_index += 1

    for item in extraction.get("category_filters") or []:
        value_param = f"value_{filter_index}"
        params[value_param] = category_attribute_id(item["group"], item["value"])
        rel_var = f"r{filter_index}"
        node_var = f"n{filter_index}"
        lines.append(
            f"MATCH (item)-[{rel_var}:HAS_ATTRIBUTE]->"
            f"({node_var}:Attribute {{id: ${value_param}}})"
        )
        if min_confidence is not None:
            where_lines.append(f"{rel_var}.confidence >= $min_confidence")
        score_terms.append(_rank_expr(rel_var))
        matched_terms.append(
            "{scope: 'category', type_code: '"
            + item["type_code"]
            + "', group: '"
            + item["group"]
            + f"', value: {node_var}.name, score: {score_terms[-1]}}}"
        )
        filter_index += 1

    exclude_index = 0
    for item in extraction.get("excluded_common_filters") or []:
        group = item["group"]
        spec = COMMON_FILTER_GROUPS[group]
        value_param = f"excluded_value_{exclude_index}"
        params[value_param] = item["value"]
        where_lines.append(
            "NOT EXISTS { "
            f"MATCH (item)-[:{spec['rel_type']}]->"
            f"(:{spec['node_label']} {{id: ${value_param}}}) "
            "}"
        )
        exclude_index += 1

    for item in extraction.get("excluded_category_filters") or []:
        value_param = f"excluded_value_{exclude_index}"
        params[value_param] = category_attribute_id(item["group"], item["value"])
        where_lines.append(
            "NOT EXISTS { "
            f"MATCH (item)-[:HAS_ATTRIBUTE]->(:Attribute {{id: ${value_param}}}) "
            "}"
        )
        exclude_index += 1

    if min_confidence is not None:
        params["min_confidence"] = min_confidence
    if min_score is not None:
        params["min_score"] = min_score
    if where_lines:
        lines.append("WHERE " + " AND ".join(where_lines))

    matched_expr = "[" + ", ".join(matched_terms) + "]"
    if score_terms:
        score_expr = " + ".join(score_terms)
        score_count = len(score_terms)
        lines.append(
            f"WITH item, ({score_expr}) / {score_count}.0 AS score, "
            f"{matched_expr} AS matched_filters"
        )
    else:
        lines.append("WITH item, 0.0 AS score, [] AS matched_filters")
    if min_score is not None:
        lines.append("WHERE score >= $min_score")

    lines.extend(
        [
            "CALL (item) {",
            "  OPTIONAL MATCH (item)-[mapped_rel]->(mapped_node)",
            "  WHERE type(mapped_rel) IN [",
            "    'HAS_COLOR', 'HAS_MATERIAL',",
            "    'HAS_PATTERN', 'HAS_SEASON', 'HAS_ATTRIBUTE'",
            "  ]",
            "  WITH collect({",
            "    scope: CASE type(mapped_rel)",
            "      WHEN 'HAS_ATTRIBUTE' THEN 'category'",
            "      ELSE 'common'",
            "    END,",
            "    rel_type: type(mapped_rel),",
            "    group: coalesce(mapped_node.group, CASE type(mapped_rel)",
            "      WHEN 'HAS_COLOR' THEN 'colors'",
            "      WHEN 'HAS_MATERIAL' THEN 'materials'",
            "      WHEN 'HAS_PATTERN' THEN 'patterns'",
            "      WHEN 'HAS_SEASON' THEN 'seasons'",
            "      ELSE null",
            "    END),",
            "    type_code: mapped_node.type_code,",
            "    value: mapped_node.name,",
            f"    score: {_rank_expr('mapped_rel')}",
            "  }) AS raw_mapped_attributes",
            "  RETURN [attr IN raw_mapped_attributes WHERE attr.value IS NOT NULL] AS mapped_attributes",
            "}",
            "RETURN item {.*, score: score, matched_filters: matched_filters, mapped_attributes: mapped_attributes} AS result",
            "ORDER BY result.score DESC, result.id ASC",
            "LIMIT $limit",
        ]
    )
    return "\n".join(lines), params


def run_search(
    query: str,
    *,
    client: Any,
    driver: Any,
    model: str,
    database: str | None = None,
    limit: int = 20,
    min_confidence: float | None = None,
    min_score: float | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    log_query: bool = False,
) -> dict[str, Any]:
    extraction = extract_search_filters(query, client=client, model=model)
    soft_rerank = needs_soft_rerank(extraction)
    cypher, params = build_search_cypher(
        extraction,
        limit=soft_candidate_limit(limit) if soft_rerank else limit,
        min_confidence=min_confidence,
        min_score=None if soft_rerank else min_score,
    )
    if log_query:
        log_search_plan(extraction, cypher, params)
    with driver.session(database=database) if database else driver.session() as session:
        rows = session.run(cypher, **params)
        raw_results = [dict(row["result"]) for row in rows]
    results = rerank_search_results(
        raw_results,
        extraction,
        client=client if soft_rerank else None,
        embedding_model=embedding_model,
        limit=limit,
        min_score=min_score if soft_rerank else None,
    )
    return {
        "query": query,
        "extraction": extraction,
        "cypher": cypher,
        "params": params,
        "results": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search Fashion-200K Neo4j GraphDB with a natural-language query.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("query", nargs="?", help="Natural-language fashion query.")
    parser.add_argument(
        "--query", dest="query_option", help="Natural-language fashion query."
    )
    parser.add_argument("--model", default=default_model())
    parser.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--min_confidence", type=float, default=None)
    parser.add_argument("--min_score", type=float, default=None)
    parser.add_argument("--neo4j_uri", default=DEFAULT_NEO4J_URI)
    parser.add_argument("--neo4j_user", default=DEFAULT_NEO4J_USER)
    parser.add_argument("--neo4j_password", default=DEFAULT_NEO4J_PASSWORD)
    parser.add_argument("--neo4j_database", default=DEFAULT_NEO4J_DATABASE)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Extract filters and print the generated Cypher without querying Neo4j.",
    )
    parser.add_argument(
        "--print_catalog",
        action="store_true",
        help="Print the available item/common/category options and exit.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print extracted filters or generated Cypher to stderr.",
    )
    return parser


def _build_openai_client() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install openai") from exc
    return OpenAI()


def _build_neo4j_driver(uri: str | None, user: str, password: str | None) -> Any:
    if not uri:
        raise SystemExit("NEO4J_URI is not set.")
    if not password:
        raise SystemExit("NEO4J_PASSWORD is not set.")
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install neo4j") from exc
    return GraphDatabase.driver(uri, auth=(user, password))


def log_search_plan(
    extraction: dict[str, Any],
    cypher: str,
    params: dict[str, Any],
) -> None:
    print("[search] extracted filters:", file=sys.stderr)
    print(json.dumps(extraction, ensure_ascii=False, indent=2), file=sys.stderr)
    print("[search] cypher:", file=sys.stderr)
    print(cypher, file=sys.stderr)
    print("[search] params:", file=sys.stderr)
    print(json.dumps(params, ensure_ascii=False, indent=2), file=sys.stderr)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.print_catalog:
        print(json.dumps(search_catalog(), ensure_ascii=False, indent=2))
        return

    query = args.query_option or args.query
    if not query:
        raise SystemExit("Query is required.")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")

    client = _build_openai_client()
    extraction = extract_search_filters(query, client=client, model=args.model)
    soft_rerank = needs_soft_rerank(extraction)
    cypher, params = build_search_cypher(
        extraction,
        limit=soft_candidate_limit(args.limit) if soft_rerank else args.limit,
        min_confidence=args.min_confidence,
        min_score=None if soft_rerank else args.min_score,
    )
    if not args.quiet:
        log_search_plan(extraction, cypher, params)

    if args.dry_run:
        print(
            json.dumps(
                {"extraction": extraction, "cypher": cypher, "params": params},
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    driver = _build_neo4j_driver(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
    try:
        with (
            driver.session(database=args.neo4j_database)
            if args.neo4j_database
            else driver.session()
        ) as session:
            rows = session.run(cypher, **params)
            raw_results = [dict(row["result"]) for row in rows]
    finally:
        driver.close()

    results = rerank_search_results(
        raw_results,
        extraction,
        client=client if soft_rerank else None,
        embedding_model=args.embedding_model,
        limit=args.limit,
        min_score=args.min_score if soft_rerank else None,
    )

    print(
        json.dumps(
            {"extraction": extraction, "results": results}, ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
