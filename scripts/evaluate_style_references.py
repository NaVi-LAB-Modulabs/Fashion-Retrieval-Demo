"""Record live, same-candidate legacy/reference comparisons without judging quality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--query", action="append")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/style-reference-comparison.json")
    args = parser.parse_args()
    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file)
    from fashion_how_graphdb.web_service import retrieve
    from fashion_how_graphdb.diagnostics import failure_details
    queries = args.query or ["A slightly formal dress", "A very formal dress",
                             "A soft-romantic feminine dress", "A mostly formal, slightly mannish skirt"]
    reports = []
    for query in queries:
        payload = {"query": query, "style_mode": "reference", "compare_styles": True, "limit": 6}
        if args.model:
            payload["model"] = args.model
        try:
            if args.base_url:
                request = Request(args.base_url.rstrip("/") + "/api/search",
                                  data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
                with urlopen(request, timeout=240) as response:
                    result = json.load(response)
            else:
                result = retrieve(payload)
            reports.append(result)
            print(json.dumps({"query": query,
                              "legacy": result["comparison"]["extraction"]["style_axis_targets"],
                              "reference": result["extraction"]["style_axis_targets"],
                              "candidate_count": result["candidate_count"],
                              "style_coverage": result["style_coverage"],
                              "legacy_top": [item["id"] for item in result["comparison"]["items"][:3]],
                              "reference_top": [item["id"] for item in result["items"][:3]]}), flush=True)
        except Exception as exc:
            report = {"query": query, "failure": failure_details(exc)}
            reports.append(report)
            print(json.dumps(report), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded {len(reports)} comparisons in {args.output}")
    if any("failure" in report for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
