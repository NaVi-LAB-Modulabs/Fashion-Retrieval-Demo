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
        images._preview_cache.clear()
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_filter_request_escapes_ids_and_keeps_token_in_header(self):
        response = io.BytesIO(json.dumps({"rows": [row("a'b")]}).encode())
        with patch.dict(os.environ, {"HF_TOKEN": "test-token"}), patch.object(images, "urlopen", return_value=response) as fetch:
            result = images._request_rows(["a'b"], images.image_source())
        request = fetch.call_args.args[0]
        params = parse_qs(urlsplit(request.full_url).query)
        self.assertEqual(params["where"], ['"item_ID" = \'a\'\'b\''])
        self.assertEqual(params["split"], ["data"])
        self.assertEqual(request.get_header("Authorization"), "Bearer test-token")
        self.assertNotIn("test-token", request.full_url)
        self.assertEqual(fetch.call_args.kwargs["timeout"], images.IMAGE_REQUEST_TIMEOUT_SECONDS)
        self.assertEqual(result, [row("a'b")])

    def test_manifest_returns_same_origin_proxy_urls_without_hf_lookup(self):
        with patch.object(images, "_request_rows") as fetch, patch.object(images, "time", return_value=1000):
            result = images.resolve_images({"item_ids": ["a b", "second", "a b"]})
        fetch.assert_not_called()
        self.assertEqual(result["images"]["a b"]["url"], "/images/hf/a%20b.jpg")
        self.assertEqual(result["images"]["second"]["expires_at"], 4600)
        self.assertEqual(result["missing_ids"], [])

    def test_fetches_one_image_and_caches_bytes(self):
        jpeg = b"\xff\xd8\xffpayload"
        with patch.object(images, "_request_rows", return_value=[row("a")]) as fetch, patch.object(images, "urlopen", return_value=io.BytesIO(jpeg)) as download:
            self.assertEqual(images.fetch_image("a"), (jpeg, "image/jpeg"))
            self.assertEqual(images.fetch_image("a"), (jpeg, "image/jpeg"))
            self.assertEqual(fetch.call_count, 1)
            self.assertEqual(fetch.call_args.args[0], ["a"])
            self.assertEqual(download.call_args.kwargs["timeout"], images.IMAGE_REQUEST_TIMEOUT_SECONDS)

    def test_refresh_bypasses_image_byte_cache(self):
        jpeg = b"\xff\xd8\xffpayload"
        with patch.object(images, "_request_rows", return_value=[row("a")]) as fetch, patch.object(images, "urlopen", side_effect=lambda *args, **kwargs: io.BytesIO(jpeg)):
            images.fetch_image("a")
            images.fetch_image("a", refresh=True)
        self.assertEqual(fetch.call_count, 2)

    def test_refresh_manifest_uses_boolean_cache_buster(self):
        result = images.resolve_images({"item_ids": ["a"], "refresh": True})
        self.assertEqual(result["images"]["a"]["url"], "/images/hf/a.jpg?refresh=true")

    def test_missing_or_foreign_image_is_rejected(self):
        with patch.object(images, "_request_rows", return_value=[row("other")]):
            with self.assertRaises(KeyError):
                images.fetch_image("a")
        with patch.object(images, "_request_rows", return_value=[row("a", "https://evil.example/image")]):
            with self.assertRaises(ValueError):
                images.fetch_image("a")

    def test_timeout_propagates_from_single_image_lookup(self):
        with patch.object(images, "_request_rows", side_effect=TimeoutError):
            with self.assertRaises(TimeoutError):
                images.fetch_image("a")

    def test_invalid_inputs_rejected_before_network(self):
        with patch.object(images, "urlopen") as fetch:
            for payload in [{}, {"item_ids": []}, {"item_ids": ["a"] * 51}, {"item_ids": [1]}, {"item_ids": ["\n"]}, {"item_ids": ["a"], "refresh": "true"}]:
                with self.subTest(payload=payload), self.assertRaises(ValueError):
                    images.resolve_images(payload)
            fetch.assert_not_called()

    def test_preview_uses_rows_without_filter_or_graph_and_caches(self):
        payload = {"rows": [row("a_1"), row("a_0"), row("a_0"), row("bad_0", "https://evil.example/a"), row("b_0"), row("c_00"), row("d0")]}
        with patch.object(images, "urlopen", return_value=io.BytesIO(json.dumps(payload).encode())) as fetch:
            result = images.sample_catalog()
            self.assertEqual(images.sample_catalog(), result)
            self.assertEqual(fetch.call_count, 1)
            request = fetch.call_args.args[0]
            self.assertEqual(urlsplit(request.full_url).path, "/rows")
            self.assertEqual(parse_qs(urlsplit(request.full_url).query)["length"], ["100"])
        self.assertEqual([item["id"] for item in result["items"]], ["a_0", "b_0"])
        self.assertFalse(result["ranked"])
        self.assertNotIn("score", result["items"][0])

    def test_preview_expiry_and_failure_do_not_restore_local_samples(self):
        with patch.object(images, "time", return_value=1000) as clock, patch.object(images, "_fetch_rows", return_value=[row("a_0")]) as fetch:
            images.sample_catalog()
            clock.return_value = 1061
            fetch.side_effect = TimeoutError
            with self.assertLogs(images.__name__, level="WARNING"):
                result = images.sample_catalog()
        self.assertEqual(result["items"], [])
        self.assertEqual(result["status"], "unavailable")

    def test_image_url_allowlist(self):
        for url in ["http://huggingface.co/a", "https://huggingface.co.evil.example/a", "https://user:pass@hf.co/a", "https://hf.co:80/a", "javascript:alert(1)"]:
            self.assertFalse(images.is_image_url(url), url)
        self.assertTrue(images.is_image_url("https://cdn-lfs.hf.co/image"))


if __name__ == "__main__":
    unittest.main()
