"""Record hybrid/CLIP comparisons; evaluate Recall@K and MRR with labeled queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def metrics(rows: list[dict], method: str) -> dict:
    ranks, recalls = [], {k: [] for k in [1, 5, 10]}
    for row in rows:
        relevant = set(row["relevant_ids"])
        items = row["rankings"][method]["items"][:20]
        ranks.append(next((index for index, item in enumerate(items, 1) if item["image_id"] in relevant), None))
        for k in recalls:
            recalls[k].append(len({item["image_id"] for item in items[:k]} & relevant) / len(relevant))
    return {"queries": len(ranks), **{f"recall@{k}": sum(values) / len(values) for k, values in recalls.items()},
            "mrr@20": sum(1 / rank for rank in ranks if rank is not None) / len(ranks)} if ranks else {"queries": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--query", action="append")
    parser.add_argument("--labels", type=Path, help='JSONL: {"query": "...", "relevant_ids": ["123_0"]}')
    parser.add_argument("--clip-only", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/clip-comparison.json")
    args = parser.parse_args()
    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file)
    from fashion_how_graphdb import clip_onnx as clip
    from fashion_how_graphdb.web_service import retrieve
    catalog_ids = {item["id"] for item in clip.index_manifest()["items"]}
    examples = [json.loads(line) for line in args.labels.read_text(encoding="utf-8").splitlines() if line.strip()] if args.labels else [
        {"query": query} for query in (args.query or ["A very formal black dress", "A soft romantic feminine dress", "A casual blue shirt"])]
    reports, excluded = [], []
    for example in examples:
        query = example.get("query") or example.get("shopping_search_label")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Every evaluation example requires a query")
        relevant = example.get("relevant_ids")
        if relevant is None and example.get("product_id") is not None:
            product = str(example["product_id"])
            relevant = [product if product.endswith("_0") else product + "_0"]
        if args.labels and (not isinstance(relevant, list) or not relevant or any(not isinstance(item, str) for item in relevant)):
            raise ValueError("Labeled queries require nonempty relevant_ids or product_id")
        relevant_in_catalog = sorted(set(relevant or []) & catalog_ids)
        if args.labels and not relevant_in_catalog:
            excluded.append({"query": query, "reason": "No ground truth in the saved 5,000-image gallery", "relevant_ids": relevant})
            continue
        payload = {"query": query, "limit": 20, "style_mode": "reference", "compare_styles": True, "compare_clip": True}
        if args.base_url:
            route = "/api/clip/search" if args.clip_only else "/api/search"
            request = Request(args.base_url.rstrip("/") + route, data=json.dumps(payload).encode(),
                              headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=240) as response:
                run = json.load(response)
        else:
            run = clip.retrieve(payload) if args.clip_only else retrieve(payload)
        rankings = {"clip_catalog": run} if args.clip_only else {
            "hybrid_reference": run, "hybrid_legacy": run["comparison"],
            "clip_same_candidates": run["clip_comparisons"]["same_candidates"],
            "clip_catalog": run["clip_comparisons"]["catalog"]}
        report = {"query": query, "relevant_ids": relevant_in_catalog,
                  "out_of_gallery_relevant_ids": sorted(set(relevant or []) - catalog_ids), "rankings": rankings}
        reports.append(report)
        print(json.dumps({"query": query, "top3": {method: [item["image_id"] for item in ranking["items"][:3]]
                                                    for method, ranking in rankings.items()}}, ensure_ascii=False), flush=True)
    summary = {"input_queries": len(examples), "evaluated_queries": len(reports), "excluded_queries": excluded,
               "quality_metrics": {method: metrics(reports, method) for method in reports[0]["rankings"]} if args.labels and reports else None,
               "runs": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {len(reports)} comparisons; {len(excluded)} out-of-gallery labels excluded: {args.output}")
    if args.labels and not reports:
        raise SystemExit("No evaluable labels in the saved gallery")


if __name__ == "__main__":
    main()
