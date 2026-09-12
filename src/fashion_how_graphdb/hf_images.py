"""Resolve Fashion200K image URLs without downloading the dataset or image bytes."""

from __future__ import annotations

from collections import OrderedDict
import json
import logging
import math
import os
import re
from threading import Lock
from time import time
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import Request, urlopen

API_URL = "https://datasets-server.huggingface.co/filter"
MAX_IDS = 50
MAX_RESPONSE_BYTES = 2_000_000
CACHE_SIZE = 2048
_cache: OrderedDict[tuple[str, ...], tuple[float, str | None]] = OrderedDict()
_cache_lock = Lock()


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
    # Viewer URLs are signed, temporary URLs. Never persist them in the graph.
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
    headers = {"Accept": "application/json", "User-Agent": "FashionSearch/1.0"}
    token = os.getenv("HF_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(API_URL + "?" + urlencode(params), headers=headers)
    with urlopen(request, timeout=8) as response:
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("Image metadata response is too large.")
    payload = json.loads(raw)
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        raise ValueError("Unexpected image metadata response.")
    return payload["rows"]


def resolve_images(payload: dict[str, Any]) -> dict[str, Any]:
    item_ids, refresh = validate_image_request(payload)
    source = (os.getenv("HF_IMAGE_DATASET", "Marqo/fashion200k"),
              os.getenv("HF_IMAGE_CONFIG", "default"),
              os.getenv("HF_IMAGE_SPLIT", "data"),
              os.getenv("HF_IMAGE_ID_COLUMN", "item_ID"))
    now = time()
    images: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    pending: list[str] = []
    with _cache_lock:
        for item_id in item_ids:
            key = (*source, item_id)
            entry = _cache.get(key)
            if entry and not refresh and entry[0] > now:
                _cache.move_to_end(key)
                if entry[1]:
                    images[item_id] = {"url": entry[1], "expires_at": entry[0]}
                else:
                    missing.append(item_id)
            else:
                _cache.pop(key, None)
                pending.append(item_id)
    if not pending:
        return {"images": images, "missing_ids": missing, "status": "ok"}
    try:
        rows = _request_rows(pending, source)
        found: dict[str, str] = {}
        for entry in rows:
            row = entry.get("row") if isinstance(entry, dict) else None
            if not isinstance(row, dict):
                continue
            item_id = row.get(source[3])
            image = row.get("image")
            url = image.get("src") if isinstance(image, dict) else None
            if isinstance(item_id, str) and item_id in pending and is_image_url(url):
                found.setdefault(item_id, url)
        now = time()
        with _cache_lock:
            for item_id in pending:
                url = found.get(item_id)
                expiry = _expires_at(url, now) if url else now + 30
                if expiry <= now:
                    url, expiry = None, now  # Do not reuse an already expired viewer response.
                _cache[(*source, item_id)] = (expiry, url)
                if url:
                    images[item_id] = {"url": url, "expires_at": expiry}
                else:
                    missing.append(item_id)
            while len(_cache) > CACHE_SIZE:
                _cache.popitem(last=False)
        return {"images": images, "missing_ids": missing, "status": "ok"}
    except Exception as exc:
        # Image failure must not discard the search results. Do not log URLs or tokens.
        logging.getLogger(__name__).warning("Image lookup unavailable (%s)", type(exc).__name__)
        return {"images": images, "missing_ids": missing, "unavailable_ids": pending,
                "status": "unavailable"}
