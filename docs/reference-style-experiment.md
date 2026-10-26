# Query-time style reference experiment

## Preserved baseline

The complete original algorithm is commit `eebf43f43baf0d2dc7cd236e47743535b3379780`.
It is preserved as Git tag `style-targets-baseline-20261011`.
The `legacy` mode still uses that parser's numeric targets and the existing
`1 - abs(item_value - target)` style scoring, candidate selection and final
weighted mean. API requests without `style_mode` retain legacy behavior.

## Reference mode

The initial parser selects requested axes and garment constraints. A second,
multimodal call receives the original query and labeled image sheets for only
those axes. It does not receive the baseline numbers. It interpolates a
continuous target on the supplied global scale and selects one supporting reference.
The resulting targets replace the parser's style targets before reranking.
Stored item values, graph filters, vectors and scoring weights are unchanged.

The images and rubrics come from the sibling Fashion-Retrieval project's
`src/fashion_200k_graphdb/fashion200k_global_axes.py` and human-selected
`axis_calibration/anchors.json`, which also define the item annotation scales.
Three axes have 40 human references each at 0.25/0.50/0.75/1.00; four axes have
20 atlas crops each at 0.00/0.25/0.50/0.75/1.00. All 200 examples are packaged
in `web/style-references/`, with IDs, image hashes and provenance in the manifest.
No new or generated reference garments are substituted.

The model response is an object keyed by the requested axes, with every key
required and additional keys forbidden. Each axis has its own numeric range and
`supporting_reference_id` enum, drawn directly from the validated manifest. A
single citation avoids duplicate-reference arrays; fixed keys avoid duplicate
axis rows. Public responses still expose an `assessments` array with
`supporting_reference_ids` for compatibility with the inspector and exports.

| Axis | Valid target range | Interpretation |
| --- | --- | --- |
| `visual_trend` | 0.25–1.00 | Statement-led to timeless design, not current popularity |
| `thermal_impression` | 0.25–1.00 | Airy to warm-looking construction, not color temperature |
| `design_expression` | 0.25–1.00 | Soft-romantic to menswear-inspired construction, not wearer identity |
| `young_mature` | 0.00–1.00 | Playful to composed design, not wearer age |
| `soft_sharp` | 0.00–1.00 | Rounded to angular construction |
| `casual_formal` | 0.00–1.00 | Casual to formal construction |
| `minimal_maximal` | 0.00–1.00 | Restrained to decorative detail |

These retain the catalog's original scales. Axes are independent; garment type
alone does not determine a style position. Ambiguous or below-0.75-confidence
assessments are omitted from scoring and retained in the evidence.

The server validates outputs even with a strict schema. Invalid output gets one
repair attempt with the same images and rubric. Calibration requests use a
30-second SDK timeout, no SDK/rate-limit/JSON retries, and at most two calls for
semantic validation. If calibration still fails (including missing assets or
provider errors), retrieval continues with the original parser targets.
`style_calibration.status` is `fallback`, `effective_mode` is `legacy`, and a
visible notice explains the change. No fabricated reference evidence is shown.
Successful calibration uses status `calibrated`; no requested axes uses `skipped`.
The degraded search is not cached, so the next request can recover. Successful
calibrations and searches retain the existing revision-aware caching.

Vercel's `web_app.py` function allows 300 seconds and retains the reference assets
in its bundle; only `public/**` is excluded because the CDN serves product images.
SQLite uses the writable temporary directory when `VERCEL` is set. Its contents
remain instance-local and ephemeral. Calibration resilience does not guarantee
availability of the parser, database, network, or hosting service.

## Comparison

The web UI defaults to Image references. With `compare_styles` in the API, a
single search parses and retrieves once, then reranks the same raw candidates
using both target sets. The Compare legacy checkbox and Reference/Legacy buttons
were removed from the UI on 2026-10-15; the comparison remains available via the API.
The inspector records baseline targets, calibrated targets, confidence,
reference IDs and omission reasons. Reference images opens the actual sheets.
Export run includes both rankings and the reference revision.
Stored-axis coverage is reported per requested axis. The connected database
checked on 2026-10-11 has axes on dresses, skirts and pants, but none on jackets
or tops (1,000 items per category). Missing item scores remain unavailable;
this experiment does not write or fabricate those values.

```json
{
  "query": "A mostly formal jacket with slightly mannish tailoring",
  "style_mode": "reference",
  "compare_styles": true,
  "weights": {"text": 0.55, "graph": 0.35, "style": 0.10}
}
```

The candidate pool remains graph top 10 union text top 10. This experiment
isolates target calibration; it does not improve style candidate recall.
Increasing the style weight helps inspect its effect, but changes the overall
retrieval tradeoff. Target changes alone do not prove better relevance. Judge
the same query pairs blindly, mark preferred results, and include repeated and
degree-sensitive queries before adopting the experiment as the API default.

## References

- Existing implementation: `Fashion-Retrieval/fashion200k_global_axes.py`,
  `load_scoring_axes`, `style_reference_cells`, `normalize_score` and its global
  calibration/independent-axis rules. The manifest records the source hash.
- [LLM-Rubric (ACL 2024)](https://arxiv.org/abs/2501.00274) motivates explicit
  dimension-specific rubrics and calibration. Its learned calibration network
  is not implemented here; this experiment uses existing visual landmarks.
- [OpenAI image input documentation](https://developers.openai.com/api/docs/guides/images-vision)
  specifies the Responses API `input_image` and base64 data URL mechanism.

## Local verification on 2026-10-11

All 70 offline tests passed under Python 3.11. Browser checks passed at
1440x1100 and 390x844 using recorded live responses, including all seven
reference sheets, comparison switching, score table, item details and overflow.
Live OpenAI/Neo4j comparisons used `gpt-5.4-mini-2026-03-17`:

| Query | Legacy targets | Reference targets |
| --- | --- | --- |
| A slightly formal dress | formal 0.70 | formal 0.62 |
| A very formal dress | formal 1.00 | formal 0.95 |
| A soft-romantic feminine dress | design 0.00 | design 0.25 |
| A mostly formal, slightly mannish skirt | formal 1.00, design 1.00 | formal 0.75, design 0.75 |

All four searches had stored requested axes on all ten candidates. Style and
final scores changed, but the top-six ordering was unchanged at default weights.
These are functional results, not a measured relevance improvement. A repeat
of the feminine query produced 0.33 instead of 0.25; visual anchoring and model
snapshots do not eliminate model variability. Successful-result caching makes
identical queries repeatable for a fixed pipeline/reference/dataset revision.

## Repackage

Pillow is a development-only dependency. Runtime deployment reads packaged JPEGs.

```sh
python scripts/package_style_references.py /path/to/Fashion-Retrieval
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python scripts/evaluate_style_references.py --env-file /path/to/local/.env
python scripts/evaluate_style_references.py --base-url https://your-site.vercel.app
```
