"""Check real ONNX retrieval and recorded hybrid comparisons on desktop/mobile."""

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8027")
    parser.add_argument("--run-file", type=Path, default=Path("outputs/clip-comparison.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/clip-ui"))
    args = parser.parse_args()
    comparison = json.loads(args.run_file.read_text())["runs"][0]["rankings"]["hybrid_reference"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        for name, width, height in [("desktop", 1440, 1100), ("mobile", 390, 844)]:
            page = browser.new_page(viewport={"width": width, "height": height})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            def local_config(route):
                config = route.fetch().json()
                route.fulfill(json={**config, "ready": False})
            page.route("**/api/config", local_config)
            page.goto(args.base_url, wait_until="domcontentloaded")
            page.wait_for_function("document.querySelector('#retrieval-mode').value === 'clip'")
            assert page.locator("#style-mode").is_disabled()
            page.locator("#query").fill("a formal black dress")
            page.locator("#search-button").click()
            page.wait_for_function("document.querySelector('#inspector-title').textContent === 'CLIP retrieval'", timeout=180000)
            assert page.locator(".item-card").count() == 12
            assert "5000 candidates" in page.locator("#results-caption").inner_text()
            assert page.locator("#query-plan").is_hidden()
            assert page.locator("#copy-cypher").is_disabled()
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.locator(".item-card img").evaluate_all("images => images.forEach(image => image.loading = 'eager')")
            page.wait_for_function("Array.from(document.querySelectorAll('.item-card img')).every(image => image.complete && image.naturalWidth > 0)")
            page.screenshot(path=str(args.output_dir / f"{name}-clip.png"), full_page=True, animations="disabled")
            page.locator("#table-view").click()
            assert page.locator("#ranking-table th").all_text_contents() == ["Rank", "Item", "Type", "CLIP cosine"]
            page.locator("#ranking-table [data-item]").first.click()
            assert "Raw text" in page.locator("#item-dialog").inner_text()
            assert "Matched graph attributes" not in page.locator("#item-dialog").inner_text()
            page.keyboard.press("Escape")
            with page.expect_download() as download:
                page.locator("#export-run").click()
            data = json.loads(Path(download.value.path()).read_text())
            assert data["score_type"] == "raw_cosine" and data["candidate_count"] == 5000
            page.unroute("**/api/config", local_config)
            def ready_config(route):
                route.fulfill(json={**route.fetch().json(), "ready": True})
            page.route("**/api/config", ready_config)
            def recorded_search(route):
                assert route.request.post_data_json["compare_clip"] is True
                route.fulfill(json=comparison)
            page.route("**/api/search", recorded_search)
            page.reload(wait_until="domcontentloaded")
            page.wait_for_function("document.querySelector('#runtime-label').textContent.includes('Live search')")
            page.locator("#compare-clip").check()
            page.locator("#query").fill(comparison["query"])
            page.locator("#search-button").click()
            page.wait_for_function("!document.querySelector('#ranking-method-label').hidden")
            for method in ["same_candidates", "catalog", "hybrid"]:
                page.locator("#ranking-method").select_option(method)
                expected = comparison["items"] if method == "hybrid" else comparison["clip_comparisons"][method]["items"]
                assert page.locator(".item-card .card-id").all_text_contents() == [item["id"] for item in expected]
                assert page.locator("#query-plan").is_visible() == (method == "hybrid")
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.locator("#legacy-results").click()
            assert page.locator(".item-card .card-id").all_text_contents() == [item["id"] for item in comparison["comparison"]["items"]]
            page.locator("#ranking-method").select_option("catalog")
            page.screenshot(path=str(args.output_dir / f"{name}-comparison.png"), full_page=True, animations="disabled")
            assert not errors, errors
            print(f"{name}: real CLIP, independent readiness, comparisons, scores, export, images and layout passed")
            page.close()
        browser.close()


if __name__ == "__main__":
    main()
