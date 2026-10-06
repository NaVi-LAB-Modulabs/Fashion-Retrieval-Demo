# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary: external, general visitors who arrive through the public (Vercel) link and try fashion search for themselves. They are not researchers. They type what they want in natural language ("a casual blue shirt, not white, for daily wear") and judge the product by whether the clothes that come back match what they meant.

Secondary (inferred from the existing UI, not the design target): the research team and technically curious visitors who want to inspect why a result ranked where it did.

## Product Purpose

Fashion Search turns a natural-language clothing request into ranked garment images from a fashion knowledge graph. Success is felt search quality: a visitor writes a request in plain words and immediately sees clothes that match it, including exclusions ("not white"), named attributes, and style feel.

## Positioning

Unlike pure embedding search, a request is parsed into hard constraints (category, required and excluded attributes) that filter the graph before any similarity ranking. Two candidate lists, graph-attribute matches and description similarity (filtered HNSW), are merged and reranked with description, attribute and style-axis scores. So "not white" really excludes white, and every result has an explainable reason.

## Operating Context

- Visitors use a browser, on desktop and mobile web, through a public deployment.
- Live search depends on server-side OpenAI (query parsing and embeddings) and Neo4j. Parsing and embedding can take noticeable time, especially with provider retries.
- `preview.py` serves the same UI with live search disabled, showing an unranked catalog preview of local images.

## Capabilities and Constraints

- Natural-language query parsing extracts category, positive, excluded and directly named attributes, a visual description, and style-axis targets.
- Ranking evidence is available for every result: component scores, mapped attributes, the generated Cypher (embedding parameters redacted), stage timings, and a JSON run export. For external users this evidence is **secondary**. Results come first, and evidence stays available on demand without dominating the screen.
- Adjustable settings: parser and threshold controls, "Min. attribute match", and scoring weights (default 55:35:10 for description, graph attributes and style).
- Images are served only from the local `sample_images/` catalog (5,000 IDs across five categories, listed in `sample_ids.json`). Missing images show a placeholder.
- Current stack: FastAPI serving static HTML/CSS/JS with no frontend build. This is the current state, not a binding constraint; a framework may be introduced if needed.
- UI language is currently English. Korean UI is under consideration (open decision).

## Brand Commitments

- Product name is **Fashion Search**. It is dataset-independent.
- Dataset names (Fashion200K, Fashion-How) must not appear in the main interface as product identity. They belong in experiment documentation and provenance only.

## Evidence on Hand

- 5,000 real garment images in `sample_images/` (Fashion200K samples).
- Retrieval flow diagram: `web/search-flow.svg`, `web/search-flow.png`.
- No user testimonials, usage metrics, benchmarks or quality evaluations exist. Do not fabricate any.

## Product Principles

1. **Results are the product.** Lead with what the visitor asked for: their query and the matching clothes. Machinery comes second.
2. **Real or nothing.** Scores, attributes and results always come from the actual retrieval. Before a search, show honest empty or unranked-preview states, never invented examples or scores.
3. **Explainable on demand.** Every result can answer "why this?", but explanation is opt-in for the general visitor, not the default view.
4. **Constraints are honored visibly.** When a request says "not X", the visitor should be able to see that X was understood and excluded.
5. **Patience is designed for.** Live search has real latency; waiting states should feel intentional and informative, not broken.
