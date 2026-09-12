"""Resolve Fashion200K image URLs without downloading the dataset or image bytes."""

from __future__ import annotations

from collections import OrderedDict
import json
import logging
import math
import os
import re
from threading import BoundedSemaphore, Lock
from time import time
from typing import Any
from urllib.parse import parse_qs, quote, urlencode, urlsplit
from urllib.request import Request, urlopen

API_URL = "https://datasets-server.huggingface.co/filter"
MAX_IDS = 50
MAX_RESPONSE_BYTES = 2_000_000
MAX_IMAGE_BYTES = 12_000_000
IMAGE_REQUEST_TIMEOUT_SECONDS = 30
IMAGE_CACHE_SIZE = 256
_cache: OrderedDict[tuple[str, ...], tuple[bytes, str]] = OrderedDict()
_cache_lock = Lock()
_image_requests = BoundedSemaphore(4)
_preview_cache: dict[tuple[str, ...], tuple[float, list[dict[str, Any]]]] = {}


def image_source() -> tuple[str, str, str, str]:
    return (os.getenv("HF_IMAGE_DATASET", "Marqo/fashion200k"),
            os.getenv("HF_IMAGE_CONFIG", "default"),
            os.getenv("HF_IMAGE_SPLIT", "data"),
            os.getenv("HF_IMAGE_ID_COLUMN", "item_ID"))


def validate_image_request(payload: dict[str, Any]) -> tuple[list[str], bool]:
    values = payload.get("item_ids")
    if not isinstance(values, list) or not 1 <= len(values) <= MAX_IDS:
        raise ValueError("Provide between 1 and 50 item IDs.")
    if any(not isinstance(value, str) or not value.strip() or len(value) > 128
           or any(ord(char) < 32 for char in value) for value in values):
        raise ValueError("Item IDs must be non-empty strings of at most 128 characters.")
    refresh = payload.get("refresh", False)
    if not isinstance(refresh, bool):
        raise ValueError("refresh must be a boolean.")
    return list(dict.fromkeys(values)), refresh


def is_image_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = urlsplit(value)
        host = parsed.hostname or ""
        return (parsed.scheme == "https" and not parsed.username and not parsed.password
                and parsed.port in (None, 443)
                and any(host == domain or host.endswith("." + domain)
                        for domain in ("huggingface.co", "hf.co")))
    except ValueError:
        return False


def _expires_at(url: str, now: float) -> float:
    """Keep Dataset Viewer URLs only while their signature remains usable."""
    expiry = now + 60
    raw_expiry = parse_qs(urlsplit(url).query).get("Expires", [None])[0]
    if raw_expiry is not None:
        try:
            value = float(raw_expiry)
            if math.isfinite(value):
                expiry = min(expiry, value - 30)
        except ValueError:
            pass
    return expiry


def _request_rows(item_ids: list[str], source: tuple[str, str, str, str]) -> list[Any]:
    dataset, config, split, column = source
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", column):
        raise ValueError("Invalid image ID column configuration.")
    # /filter accepts SQL-style predicates, not a Cypher query. Escape string literals.
    predicates = [f'"{column}" = \'' + value.replace("'", "''") + "'" for value in item_ids]
    params = {"dataset": dataset, "config": config, "split": split,
              "where": " OR ".join(predicates), "offset": 0, "length": 100}
    return _fetch_rows(API_URL, params)


def _fetch_rows(endpoint: str, params: dict[str, Any]) -> list[Any]:
    headers = {"Accept": "application/json", "User-Agent": "FashionSearch/1.0"}
    token = os.getenv("HF_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(endpoint + "?" + urlencode(params), headers=headers)
    with urlopen(request, timeout=IMAGE_REQUEST_TIMEOUT_SECONDS) as response:
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("Image metadata response is too large.")
    payload = json.loads(raw)
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        raise ValueError("Unexpected image metadata response.")
    return payload["rows"]


def resolve_images(payload: dict[str, Any]) -> dict[str, Any]:
    """Return stable same-origin proxy URLs without running a batch HF filter."""
    item_ids, refresh = validate_image_request(payload)
    suffix = "?refresh=true" if refresh else ""
    expires_at = time() + 3600
    return {
        "images": {
            item_id: {
                "url": f"/api/image/{quote(item_id, safe='')}.jpg{suffix}",
                "expires_at": expires_at,
            }
            for item_id in item_ids
        },
        "missing_ids": [],
        "status": "ok",
    }


def _image_content_type(payload: bytes) -> str:
    if payload.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if payload.startswith(b"RIFF") and payload[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("Unsupported image response.")


def fetch_image(item_id: str, *, refresh: bool = False) -> tuple[bytes, str]:
    """Fetch one Fashion200K image through a simple equality filter and cache bytes."""
    validated, _ = validate_image_request({"item_ids": [item_id], "refresh": refresh})
    item_id = validated[0]
    source = image_source()
    key = (*source, item_id)
    with _cache_lock:
        cached = _cache.get(key)
        if cached is not None and not refresh:
            _cache.move_to_end(key)
            return cached

    with _image_requests:
        rows = _request_rows([item_id], source)
        row = next(
            (
                entry["row"]
                for entry in rows
                if isinstance(entry, dict)
                and isinstance(entry.get("row"), dict)
                and str(entry["row"].get(source[3])) == item_id
            ),
            None,
        )
        if row is None:
            raise KeyError(item_id)
        image = row.get("image")
        image_url = image.get("src") if isinstance(image, dict) else None
        if not is_image_url(image_url):
            raise ValueError("Invalid image URL from Hugging Face.")
        request = Request(
            image_url,
            headers={"Accept": "image/*", "User-Agent": "FashionSearch/1.0"},
        )
        with urlopen(request, timeout=IMAGE_REQUEST_TIMEOUT_SECONDS) as response:
            payload = response.read(MAX_IMAGE_BYTES + 1)
        if len(payload) > MAX_IMAGE_BYTES:
            raise ValueError("Image response is too large.")
        result = (payload, _image_content_type(payload))

    with _cache_lock:
        _cache[key] = result
        _cache.move_to_end(key)
        while len(_cache) > IMAGE_CACHE_SIZE:
            _cache.popitem(last=False)
    return result


def sample_catalog() -> dict[str, Any]:
    """Read a small unranked slice directly from HF, without graph or LLM calls."""
    source = image_source()
    now = time()
    with _cache_lock:
        cached = _preview_cache.get(source)
        if cached and cached[0] > now:
            return {"items": cached[1], "source": "huggingface", "ranked": False, "status": "ok"}
    try:
        rows = _fetch_rows("https://datasets-server.huggingface.co/rows", {
            "dataset": source[0], "config": source[1], "split": source[2],
            "offset": 0, "length": 100,
        })
        now = time()
        items = []
        seen = set()
        for entry in rows:
            row = entry.get("row") if isinstance(entry, dict) else None
            if not isinstance(row, dict):
                continue
            item_id = row.get(source[3])
            image = row.get("image")
            url = image.get("src") if isinstance(image, dict) else None
            if not isinstance(item_id, str) or not item_id.endswith("_0") or len(item_id) > 128 or item_id in seen or not is_image_url(url):
                continue
            expiry = _expires_at(url, now)
            if expiry <= now:
                continue
            seen.add(item_id)
            category = next((row[key] for key in ("category3", "category2", "category1")
                             if isinstance(row.get(key), str) and row[key].strip()), "Garment")
            items.append({"id": item_id, "image_id": item_id, "image_url": url,
                          "image_expires_at": expiry, "type_name": category})
            if len(items) == 12:
                break
        with _cache_lock:
            if len(_preview_cache) >= 8:
                _preview_cache.clear()
            _preview_cache[source] = (min((item["image_expires_at"] for item in items), default=now + 15), items)
        return {"items": items, "source": "huggingface", "ranked": False, "status": "ok"}
    except Exception as exc:
        logging.getLogger(__name__).warning("Catalog preview unavailable (%s)", type(exc).__name__)
        return {"items": [], "source": "huggingface", "ranked": False, "status": "unavailable"}
