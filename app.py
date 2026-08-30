"""Gradio demo for Fashion-How graph-based fashion retrieval."""

from __future__ import annotations

import os
import sys
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

# Hugging Face Spaces enables Gradio SSR by default. This demo uses standard
# client-side rendering to avoid the Node SSR proxy shutting down the app.
os.environ["GRADIO_SSR_MODE"] = "False"

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import gradio as gr
import spaces

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from fashion_how_graphdb.cypher import (
    DEFAULT_NEO4J_DATABASE,
    DEFAULT_NEO4J_PASSWORD,
    DEFAULT_NEO4J_URI,
    DEFAULT_NEO4J_USER,
)
from fashion_how_graphdb.search import (
    DEFAULT_EMBEDDING_MODEL,
    build_search_cypher,
    extract_search_filters,
    needs_soft_rerank,
    rerank_search_results,
    search_catalog,
    soft_candidate_limit,
)
from fashion_how_graphdb.vlm import default_model, default_top_detail_model

APP_TITLE = "Fashion-How Graph Search"
DEFAULT_IMAGE_DIR = ROOT / "fashion-how" / "image"
DEFAULT_MODELS = [
    default_model(),
    "gpt-5.4-mini",
    default_top_detail_model(),
    "gpt-4.1-mini",
    "gpt-4.1",
]
EXAMPLE_QUERIES = [
    ["A romantic floral blouse for a spring date", default_model(), 12, 0.0, 0.0],
    ["Minimal black outerwear that feels formal", default_model(), 12, 0.0, 0.2],
    ["A casual blue shirt, not white, for daily wear", default_model(), 12, 0.0, 0.0],
]


@lru_cache(maxsize=1)
def openai_client() -> Any:
    from openai import OpenAI

    return OpenAI()


@lru_cache(maxsize=1)
def neo4j_driver() -> Any:
    if not DEFAULT_NEO4J_URI:
        raise RuntimeError("NEO4J_URI is not set.")
    if not DEFAULT_NEO4J_PASSWORD:
        raise RuntimeError("NEO4J_PASSWORD is not set.")

    from neo4j import GraphDatabase

    return GraphDatabase.driver(
        DEFAULT_NEO4J_URI,
        auth=(DEFAULT_NEO4J_USER, DEFAULT_NEO4J_PASSWORD),
    )


@lru_cache(maxsize=1)
def image_index() -> dict[str, Path]:
    if not DEFAULT_IMAGE_DIR.exists():
        return {}
    return {
        path.name: path
        for path in DEFAULT_IMAGE_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    }


def runtime_status() -> str:
    checks = [
        ("OpenAI key", bool(os.getenv("OPENAI_API_KEY"))),
        ("Neo4j URI", bool(DEFAULT_NEO4J_URI)),
        ("Neo4j password", bool(DEFAULT_NEO4J_PASSWORD)),
        ("Sample images", DEFAULT_IMAGE_DIR.exists()),
    ]
    rendered = " | ".join(f"{name}: {'ready' if ok else 'missing'}" for name, ok in checks)
    return f"**Runtime**  {rendered}"


def catalog_overview() -> str:
    catalog = search_catalog()
    item_count = len(catalog["item_types"])
    common_count = sum(len(group["values"]) for group in catalog["common_groups"].values())
    category_count = sum(
        len(group["values"])
        for type_data in catalog["category_groups"].values()
        for group in type_data["groups"].values()
    )
    axis_count = len(catalog["style_axes"])
    return (
        f"**Catalog**  {item_count} item types | "
        f"{common_count} shared attributes | "
        f"{category_count} category attributes | "
        f"{axis_count} style axes"
    )


def _safe_optional_float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    return float(value)


def _summarize_extraction(extraction: dict[str, Any]) -> str:
    type_codes = extraction.get("item_type_codes") or []
    common = extraction.get("common_filters") or []
    category = extraction.get("category_filters") or []
    excluded_common = extraction.get("excluded_common_filters") or []
    excluded_category = extraction.get("excluded_category_filters") or []
    style_targets = extraction.get("style_axis_targets") or []
    description_query = extraction.get("description_query") or ""

    lines = [
        f"**Item types**: {', '.join(type_codes) if type_codes else 'any'}",
        f"**Hard filters**: {len(common) + len(category)}",
        f"**Excluded filters**: {len(excluded_common) + len(excluded_category)}",
        f"**Style axes**: {len(style_targets)}",
        f"**Semantic remainder**: {description_query or 'none'}",
    ]
    return "\n\n".join(lines)


def _result_rows(results: list[dict[str, Any]]) -> list[list[Any]]:
    rows = []
    for rank, item in enumerate(results, start=1):
        matched = item.get("matched_filters") or []
        matched_text = "; ".join(
            f"{entry.get('group')}: {entry.get('value')}" for entry in matched[:4]
        )
        rows.append(
            [
                rank,
                item.get("id") or "",
                item.get("type_name") or item.get("type_code") or "",
                item.get("score"),
                item.get("graph_score"),
                item.get("text_score"),
                item.get("style_score"),
                matched_text,
            ]
        )
    return rows


def _gallery_items(results: list[dict[str, Any]]) -> list[tuple[str, str]]:
    images = image_index()
    gallery = []
    for rank, item in enumerate(results, start=1):
        image_name = Path(str(item.get("image_file") or "")).name
        image_path = images.get(image_name)
        if image_path is None:
            continue
        score = item.get("score")
        score_text = f"{float(score):.3f}" if isinstance(score, (int, float)) else "n/a"
        caption = f"#{rank} {item.get('id', '')} | score {score_text}"
        gallery.append((str(image_path), caption))
    return gallery


@spaces.GPU(duration=30)
def search_demo(
    query: str,
    model: str,
    limit: int,
    min_confidence: float | None,
    min_score: float | None,
) -> tuple[
    list[tuple[str, str]],
    list[list[Any]],
    dict[str, Any],
    str,
    str,
    str,
]:
    query = (query or "").strip()
    if not query:
        raise gr.Error("Enter a fashion search query.")
    if not os.getenv("OPENAI_API_KEY"):
        raise gr.Error("OPENAI_API_KEY is not set in Space Secrets.")

    limit = max(1, min(int(limit or 12), 50))
    min_confidence_value = _safe_optional_float(min_confidence)
    min_score_value = _safe_optional_float(min_score)

    extraction = extract_search_filters(
        query,
        client=openai_client(),
        model=model or default_model(),
    )
    soft_rerank = needs_soft_rerank(extraction)
    cypher, params = build_search_cypher(
        extraction,
        limit=soft_candidate_limit(limit) if soft_rerank else limit,
        min_confidence=min_confidence_value,
        min_score=None if soft_rerank else min_score_value,
    )

    driver = neo4j_driver()
    with (
        driver.session(database=DEFAULT_NEO4J_DATABASE)
        if DEFAULT_NEO4J_DATABASE
        else driver.session()
    ) as session:
        rows = session.run(cypher, **params)
        raw_results = [dict(row["result"]) for row in rows]

    results = rerank_search_results(
        raw_results,
        extraction,
        client=openai_client() if soft_rerank else None,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        limit=limit,
        min_score=min_score_value if soft_rerank else None,
    )

    gallery = _gallery_items(results)
    table = _result_rows(results)
    cypher_text = f"{cypher}\n\nparams = {json.dumps(params, ensure_ascii=False, indent=2)}"
    status = (
        f"Retrieved {len(results)} ranked items "
        f"from {len(raw_results)} graph candidates."
    )
    return gallery, table, extraction, cypher_text, _summarize_extraction(extraction), status


CSS = """
:root {
  --demo-accent: #166c72;
  --demo-ink: #1f2933;
}
.gradio-container {
  max-width: 1440px !important;
}
.demo-title h1 {
  margin-bottom: 0.25rem;
  color: var(--demo-ink);
}
.demo-subtitle {
  color: #52606d;
  font-size: 1rem;
}
.runtime-line {
  color: #52606d;
  font-size: 0.92rem;
}
button.primary {
  background: var(--demo-accent) !important;
}
"""


with gr.Blocks(
    title=APP_TITLE,
) as demo:
    gr.Markdown(
        """
        # Fashion-How Graph Search
        <div class="demo-subtitle">
        Natural-language fashion retrieval over a Neo4j attribute graph with LLM query parsing,
        hard constraints, semantic reranking, and interpretable Cypher.
        </div>
        """,
        elem_classes=["demo-title"],
    )

    with gr.Row():
        runtime = gr.Markdown(runtime_status(), elem_classes=["runtime-line"])
        gr.Markdown(catalog_overview(), elem_classes=["runtime-line"])

    with gr.Row():
        with gr.Column(scale=4):
            query_input = gr.Textbox(
                label="Search query",
                placeholder="e.g., A romantic floral blouse for a spring date",
                value="A romantic floral blouse for a spring date",
                lines=3,
                max_lines=5,
            )
        with gr.Column(scale=2):
            model_input = gr.Dropdown(
                label="LLM parser",
                choices=list(dict.fromkeys(DEFAULT_MODELS)),
                value=default_model(),
                allow_custom_value=True,
            )
            with gr.Row():
                limit_input = gr.Slider(
                    label="Results",
                    minimum=4,
                    maximum=50,
                    value=12,
                    step=1,
                )
                min_score_input = gr.Slider(
                    label="Min score",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.05,
                )
            min_confidence_input = gr.Slider(
                label="Min edge confidence",
                minimum=0.0,
                maximum=1.0,
                value=0.0,
                step=0.05,
            )

    with gr.Row():
        search_button = gr.Button("Run retrieval", variant="primary")
        clear_button = gr.ClearButton(value="Clear", components=[query_input])

    status_output = gr.Markdown("Ready.")

    with gr.Row():
        with gr.Column(scale=3):
            gallery_output = gr.Gallery(
                label="Ranked fashion items",
                columns=[2, 3, 4, 5],
                rows=[1, 2, 2],
                height=560,
                object_fit="cover",
                show_label=True,
            )
        with gr.Column(scale=2):
            summary_output = gr.Markdown("Run a query to inspect extracted constraints.")
            table_output = gr.Dataframe(
                label="Ranking table",
                headers=[
                    "rank",
                    "item_id",
                    "type",
                    "score",
                    "graph",
                    "text",
                    "style",
                    "matched_filters",
                ],
                datatype=["number", "str", "str", "number", "number", "number", "number", "str"],
                interactive=False,
                wrap=True,
            )

    with gr.Accordion("Generated query plan", open=False):
        with gr.Row():
            extraction_output = gr.JSON(label="Extracted graph filters")
            cypher_output = gr.Code(label="Cypher sent to Neo4j", language="sql", lines=18)

    gr.Examples(
        examples=EXAMPLE_QUERIES,
        inputs=[
            query_input,
            model_input,
            limit_input,
            min_confidence_input,
            min_score_input,
        ],
    )

    search_button.click(
        fn=search_demo,
        inputs=[
            query_input,
            model_input,
            limit_input,
            min_confidence_input,
            min_score_input,
        ],
        outputs=[
            gallery_output,
            table_output,
            extraction_output,
            cypher_output,
            summary_output,
            status_output,
        ],
    )

    demo.load(runtime_status, outputs=runtime)

demo.queue()


if __name__ == "__main__":
    demo.launch(
        show_error=True,
        debug=True,
        ssr_mode=False,
        css=CSS,
        theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"),
    )
