"""Offline checks for Hugging Face image lookup and temporary URL caching."""

import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fashion_how_graphdb import hf_images as images


URL = "https://datasets-server.huggingface.co/assets/image.jpg"


def row(item_id, url=URL):
    return {"row": {"item_ID": item_id, "image": {"src": url}}}


class ImageTests(unittest.TestCase):
    def setUp(self):
        images._cache.clear()
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_filter_request_escapes_ids_and_keeps_token_in_header(self):
        response = io.BytesIO(json.dumps({"rows": [row("a'b")]}).encode())
        with patch.dict(os.environ, {"HF_TOKEN": "test-token"}), patch.object(images, "urlopen", return_value=response) as fetch:
            result = images.resolve_images({"item_ids": ["a'b", "second", "a'b"]})
        request = fetch.call_args.args[0]
        params = parse_qs(urlsplit(request.full_url).query)
        self.assertEqual(params["where"], ['"item_ID" = \'a\'\'b\' OR "item_ID" = \'second\''])
        self.assertEqual(params["split"], ["data"])
        self.assertEqual(request.get_header("Authorization"), "Bearer test-token")
        self.assertNotIn("test-token", request.full_url)
        self.assertEqual(fetch.call_args.kwargs["timeout"], 8)
        self.assertEqual(result["missing_ids"], ["second"])
        self.assertEqual(result["images"]["a'b"]["url"], URL)

    def test_maps_by_id_not_row_order_and_rejects_foreign_urls(self):
        with patch.object(images, "_request_rows", return_value=[row("b"), row("other"), row("a", "https://evil.example/image"), None]):
            result = images.resolve_images({"item_ids": ["a", "b"]})
        self.assertEqual(list(result["images"]), ["b"])
        self.assertEqual(result["missing_ids"], ["a"])

    def test_cache_expiry_refresh_and_missing_cache(self):
        with patch.object(images, "time", return_value=1000) as clock, patch.object(images, "_request_rows", return_value=[row("a")]) as fetch:
            payload = {"item_ids": ["a", "missing"]}
            images.resolve_images(payload)
            images.resolve_images(payload)
            self.assertEqual(fetch.call_count, 1)
            clock.return_value = 1031
            images.resolve_images(payload)
            self.assertEqual(fetch.call_args.args[0], ["missing"])
            images.resolve_images({**payload, "refresh": True})
            self.assertEqual(fetch.call_args.args[0], ["a", "missing"])
            clock.return_value = 1092
            images.resolve_images(payload)
            self.assertEqual(fetch.call_count, 4)

    def test_signed_url_near_expiry_is_not_returned(self):
        with patch.object(images, "time", return_value=1000), patch.object(images, "_request_rows", return_value=[row("a", URL + "?Expires=1020")]):
            self.assertEqual(images.resolve_images({"item_ids": ["a"]})["images"], {})

    def test_outage_preserves_cached_images(self):
        with patch.object(images, "_request_rows", return_value=[row("a")]):
            images.resolve_images({"item_ids": ["a"]})
        with patch.object(images, "_request_rows", side_effect=TimeoutError), self.assertLogs(images.__name__, level="WARNING"):
            result = images.resolve_images({"item_ids": ["a", "b"]})
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["unavailable_ids"], ["b"])
        self.assertIn("a", result["images"])

    def test_malformed_response_is_graceful(self):
        with patch.object(images, "urlopen", return_value=io.BytesIO(b"not json")), self.assertLogs(images.__name__, level="WARNING"):
            result = images.resolve_images({"item_ids": ["a"]})
        self.assertEqual(result["status"], "unavailable")

    def test_invalid_inputs_rejected_before_network(self):
        with patch.object(images, "urlopen") as fetch:
            for payload in [{}, {"item_ids": []}, {"item_ids": ["a"] * 51}, {"item_ids": [1]}, {"item_ids": ["\n"]}, {"item_ids": ["a"], "refresh": "true"}]:
                with self.subTest(payload=payload), self.assertRaises(ValueError):
                    images.resolve_images(payload)
            fetch.assert_not_called()

    def test_image_url_allowlist(self):
        for url in ["http://huggingface.co/a", "https://huggingface.co.evil.example/a", "https://user:pass@hf.co/a", "https://hf.co:80/a", "javascript:alert(1)"]:
            self.assertFalse(images.is_image_url(url), url)
        self.assertTrue(images.is_image_url("https://cdn-lfs.hf.co/image"))


if __name__ == "__main__":
    unittest.main()
