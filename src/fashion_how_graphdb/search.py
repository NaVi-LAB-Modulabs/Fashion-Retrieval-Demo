"""Natural-language search over the Fashion-200K Neo4j graph."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

from .category_taxonomy import CATEGORY_ATTRIBUTES
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
    "visual_trend": {
        "label": "Statement-led <-> Timeless",
        "left": "statement-led",
        "right": "timeless",
        "meaning": "experimental or statement design vs restrained and enduring design; not current popularity",
    },
    "thermal_impression": {
        "label": "Airy <-> Warm-looking",
        "left": "airy",
        "right": "warm-looking",
        "meaning": "light, open construction vs thick, enclosing construction; not color temperature",
    },
    "design_expression": {
        "label": "Soft-romantic <-> Mannish",
        "left": "soft-romantic",
        "right": "mannish",
        "meaning": "soft, draped or romantic construction vs angular, tailored or menswear-inspired construction",
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

DEFAULT_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

SYSTEM_PROMPT = """You convert English fashion search text into Fashion-200K graph filters.
Use only exact English identifiers from the provided catalog.
item_type_codes are the five Fashion-200K categories. Select a category when
the query names it. Only colors, materials, patterns, and seasons are common
attribute groups. Category-specific details use their listed group IDs.

For every positive common or category filter, set hard=true when the user
directly names that attribute. "White jacket" requires a white color edge;
"floral dress" requires a floral pattern edge. Explicit words such as "must"
or "only" are not necessary. Inferred attributes have hard=false. A hard
filter excludes items without a matching graph edge meeting the configured
attribute match threshold (confidence times coverage, prominence, or season
score; confidence alone when no metric is available).
Do not invent values.

Place explicit negative attributes such as "not white" in excluded filters.
Do not infer that a missing graph edge proves an attribute is absent.
Extract only garment properties. Occasions, moods, and activities are not
colors, materials, patterns, or category details.

Write description_query as a short positive visual description of the garment,
including its category and mentioned colors, patterns, materials, and details.
Omit negated attributes, mandatory wording, and search instructions. Put style
intent in style_axis_targets only when the query
clearly implies a supported axis. Use 0.0 for its left pole and 1.0 for its
right pole. Supported axes are visual_trend, thermal_impression,
design_expression, minimal_maximal, casual_formal, soft_sharp, young_mature.
thermal_impression describes apparent insulation, not color temperature.
visual_trend describes statement-led versus timeless design, not popularity.
Do not choose all style axes by default.

When uncertain about a filter or style axis, omit it. Return strict JSON only."""
@dataclass(frozen=True)
class SearchFilter:
    scope: str
    group: str
    value: str
    type_code: str | None = None
    hard: bool = False


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
            "required": ["group", "value", "hard"],
            "properties": {
                "group": {
                    "type": "string",
                    "enum": list(COMMON_FILTER_GROUPS),
                },
                "value": {"type": "string"},
                "hard": {"type": "boolean"},
            },
        },
    }
    category_filter_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["type_code", "group", "value", "hard"],
            "properties": {
                "type_code": {
                    "type": "string",
                    "enum": list(CATEGORY_ATTRIBUTES),
                },
                "group": {"type": "string"},
                "value": {"type": "string"},
                "hard": {"type": "boolean"},
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
            "excluded_common_filters": {
                **common_filter_schema,
                "items": {
                    **common_filter_schema["items"],
                    "required": ["group", "value"],
                    "properties": {
                        key: value for key, value in common_filter_schema["items"]["properties"].items()
                        if key != "hard"
                    },
                },
            },
            "excluded_category_filters": {
                **category_filter_schema,
                "items": {
                    **category_filter_schema["items"],
                    "required": ["type_code", "group", "value"],
                    "properties": {
                        key: value for key, value in category_filter_schema["items"]["properties"].items()
                        if key != "hard"
                    },
                },
            },
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
    return normalize_extraction(raw, query=query)


def normalize_extraction(raw: dict[str, Any], *, query: str | None = None) -> dict[str, Any]:
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
    if query:
        common_filters = [
            replace(item, hard=True) if _explicitly_mentions(query, item.value) else item
            for item in common_filters
        ]
        category_filters = [
            replace(item, hard=True) if _explicitly_mentions(query, item.value) else item
            for item in category_filters
        ]
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


def _explicitly_mentions(query: str, value: str) -> bool:
    words = [re.escape(part) for part in re.split(r"[_\s-]+", value) if part]
    if not words:
        return False
    phrase = r"[\s_-]+".join(words)
    return re.search(rf"(?<!\w){phrase}(?!\w)", query, re.IGNORECASE) is not None


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
            filters.append(SearchFilter("common", group, value, hard=item.get("hard") is True))
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
            filters.append(SearchFilter("category", group, value, type_code=type_code,
                                        hard=item.get("hard") is True))
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
    if search_filter.hard:
        payload["hard"] = True
    return payload


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
    from . import hybrid_search

    extraction = extract_search_filters(query, client=client, model=model)
    with driver.session(database=database) if database else driver.session() as session:
        retrieval = hybrid_search.retrieve_parsed(
            session, extraction, query, client=client,
            embedding_model=embedding_model, limit=limit,
            min_confidence=min_confidence, min_score=min_score,
        )
    if log_query:
        log_search_plan(extraction, retrieval["cypher"], retrieval["params"])
    return {
        "query": query,
        "extraction": extraction,
        "cypher": retrieval["cypher"],
        "params": retrieval["params"],
        "candidate_count": retrieval["candidate_count"],
        "results": retrieval["items"],
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
    parser.add_argument(
        "--min_confidence", type=float, default=None,
        help="Minimum attribute match score (confidence * metric; confidence if no metric).",
    )
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
    from . import hybrid_search

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
    if args.dry_run:
        extraction = extract_search_filters(query, client=client, model=args.model)
        print(
            json.dumps(
                {
                    "extraction": extraction,
                    "vector_cypher": hybrid_search.VECTOR_CYPHER,
                    "result_cypher": hybrid_search.result_cypher(extraction),
                    "vector_index": hybrid_search.VECTOR_INDEX_NAME,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    driver = _build_neo4j_driver(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
    try:
        result = run_search(
            query, client=client, driver=driver, model=args.model,
            database=args.neo4j_database, limit=args.limit,
            min_confidence=args.min_confidence, min_score=args.min_score,
            embedding_model=args.embedding_model, log_query=not args.quiet,
        )
    finally:
        driver.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
