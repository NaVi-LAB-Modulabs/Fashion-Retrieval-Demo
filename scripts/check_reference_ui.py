"""Browser checks with a recorded live search; no additional provider calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

from playwright.sync_api import sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:7861")
    parser.add_argument("--run-file", type=Path, default=Path("outputs/style-reference-comparison.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ui-check"))
    parser.add_argument("--fallback", action="store_true", help="Verify a recorded calibration fallback run")
    args = parser.parse_args()
    runs = json.loads(args.run_file.read_text(encoding="utf-8"))
    run = next(row for row in runs if row["items"] and
               (row["extraction"].get("style_calibration", {}).get("status") == "fallback") == args.fallback)
    with urlopen(args.base_url + "/api/config") as response:
        config = {**json.load(response), "ready": True, "preview": False}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        for name, width, height in [("desktop", 1440, 1100), ("mobile", 390, 844)]:
            page = browser.new_page(viewport={"width": width, "height": height})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.route("**/api/config", lambda route: route.fulfill(json=config))
            def recorded_search(route):
                payload = route.request.post_data_json
                assert payload["style_mode"] == "reference" and "compare_styles" not in payload
                route.fulfill(json=run)
            page.route("**/api/search", recorded_search)
            page.goto(args.base_url, wait_until="domcontentloaded")
            page.wait_for_function("document.querySelectorAll('.item-card').length > 0")
            page.locator("#browse-references").click()
            page.wait_for_function("document.querySelector('#reference-dialog').open")
            options = page.locator("#reference-axis option").evaluate_all("options => options.map(option => option.value)")
            assert len(options) == 7
            for axis in options:
                page.locator("#reference-axis").select_option(axis)
                page.wait_for_function("document.querySelector('#reference-image').complete && document.querySelector('#reference-image').naturalWidth > 0")
            page.locator("#reference-axis").select_option("casual_formal")
            page.wait_for_function("document.querySelector('#reference-image').complete && document.querySelector('#reference-image').naturalWidth > 0")
            page.screenshot(path=str(args.output_dir / f"{name}-references.png"), animations="disabled")
            page.keyboard.press("Escape")
            page.locator("#query").fill(run["query"])
            page.locator("#search-button").click()
            page.wait_for_function("document.querySelector('#results-eyebrow').textContent === 'RANKED RESULTS'")
            assert page.locator(".reference-assessment").count() == len(run["extraction"]["style_calibration"]["assessments"])
            if args.fallback:
                warning = run["extraction"]["style_calibration"]["warning"]
                assert warning in page.locator("#notice").inner_text()
                assert warning in page.locator("#reference-evidence").inner_text()
                assert "No requested style axes" not in page.locator("#reference-evidence").inner_text()
                assert "Legacy (image references unavailable)" in page.locator("#run-summary").inner_text()
            reference_ids = page.locator("#results-grid .card-id").all_text_contents()
            assert reference_ids == [item["id"] for item in run["items"]]
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
            page.locator(".item-card img").evaluate_all("images => images.forEach(image => image.loading = 'eager')")
            page.wait_for_function("Array.from(document.querySelectorAll('.item-card img')).every(image => image.complete && image.naturalWidth > 0)")
            page.screenshot(path=str(args.output_dir / f"{name}-results.png"), full_page=True, animations="disabled")
            page.locator("#table-view").click()
            assert page.locator("#ranking-table").is_visible()
            page.locator("#grid-view").click()
            page.locator(".item-card").first.click()
            assert page.locator("#item-dialog").is_visible()
            page.keyboard.press("Escape")
            assert not errors, errors
            print(f"{name}: reference assets, results, score table, details and layout passed")
            page.close()
        browser.close()


if __name__ == "__main__":
    main()
