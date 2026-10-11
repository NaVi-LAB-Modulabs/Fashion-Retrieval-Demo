"""Persistent successful-result caching and reproducible retrieval configuration."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time
from typing import Any, Callable
import unicodedata
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
MODEL_SNAPSHOTS = {
    "gpt-5.4-mini": "gpt-5.4-mini-2026-03-17",
    "gpt-4.1": "gpt-4.1-2025-04-14",
    "gpt-4.1-mini": "gpt-4.1-mini-2025-04-14",
}


def normalize_query(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def resolve_model(model: str) -> str:
    return MODEL_SNAPSHOTS.get(model, model)


def fingerprint(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@lru_cache(maxsize=1)
def pipeline_revision() -> str:
    digest = hashlib.sha256()
    for name in ("reproducibility.py", "llm.py", "search.py", "hybrid_search.py",
                 "web_service.py", "style_references.py", "taxonomy.py", "category_taxonomy.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    digest.update((ROOT / "sample_ids.json").read_bytes())
    return digest.hexdigest()


def dataset_revision() -> str:
    # Change this after modifying the graph, stored vectors, or vector index.
    return os.getenv("FASHION_DATASET_REVISION", "1")


def vector_search_mode() -> str:
    mode = os.getenv("FASHION_VECTOR_SEARCH_MODE", "exact").strip().lower()
    if mode not in {"exact", "hnsw"}:
        raise ValueError("FASHION_VECTOR_SEARCH_MODE must be exact or hnsw")
    return mode


def cache_enabled() -> bool:
    return os.getenv("FASHION_CACHE_ENABLED", "1").strip().lower() not in {"0", "false", "no"}


@dataclass(frozen=True)
class CacheResult:
    value: Any
    hit: bool
    key: str
    created_at: float


class PersistentCache:
    """SQLite cache with per-key leases shared by threads and local workers.

    No provider call runs inside a database transaction. Errors are never cached.
    The first successful value is reused until expiry, eviction, or revision change.
    """

    def __init__(self, path: Path, *, ttl: float = 0, max_entries: int = 4096,
                 lease_seconds: float = 600, wait_seconds: float = 660):
        self.path = path
        self.ttl = ttl
        self.max_entries = max_entries
        self.lease_seconds = lease_seconds
        self.wait_seconds = wait_seconds
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection, connection:
            connection.execute("CREATE TABLE IF NOT EXISTS entries ("
                               "namespace TEXT, key TEXT, value TEXT NOT NULL, "
                               "created REAL NOT NULL, expires REAL NOT NULL, "
                               "PRIMARY KEY (namespace, key))")
            connection.execute("CREATE TABLE IF NOT EXISTS leases ("
                               "namespace TEXT, key TEXT, owner TEXT NOT NULL, "
                               "expires REAL NOT NULL, PRIMARY KEY (namespace, key))")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def get_or_compute(self, namespace: str, signature: Any,
                       compute: Callable[[], Any]) -> CacheResult:
        key = fingerprint(signature)
        owner = uuid4().hex
        deadline = time.monotonic() + self.wait_seconds
        while True:
            now = time.time()
            with closing(self._connect()) as connection, connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT value, created, expires FROM entries WHERE namespace=? AND key=?",
                    (namespace, key),
                ).fetchone()
                if row and (row[2] == 0 or row[2] > now):
                    return CacheResult(json.loads(row[0]), True, key, row[1])
                connection.execute("DELETE FROM entries WHERE namespace=? AND key=?",
                                   (namespace, key))
                connection.execute("DELETE FROM leases WHERE expires <= ?", (now,))
                claimed = connection.execute(
                    "INSERT OR IGNORE INTO leases VALUES (?, ?, ?, ?)",
                    (namespace, key, owner, now + self.lease_seconds),
                ).rowcount == 1
            if claimed:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for the identical cached request")
            time.sleep(0.1)
        try:
            value = compute()
            encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
            created = time.time()
            with closing(self._connect()) as connection, connection:
                connection.execute("BEGIN IMMEDIATE")
                # A replacement worker may have completed after this lease expired.
                row = connection.execute(
                    "SELECT value, created, expires FROM entries WHERE namespace=? AND key=?",
                    (namespace, key),
                ).fetchone()
                if row and (row[2] == 0 or row[2] > created):
                    return CacheResult(json.loads(row[0]), True, key, row[1])
                connection.execute("INSERT OR REPLACE INTO entries VALUES (?, ?, ?, ?, ?)",
                                   (namespace, key, encoded, created,
                                    created + self.ttl if self.ttl else 0))
                connection.execute("DELETE FROM entries WHERE expires > 0 AND expires <= ?", (created,))
                connection.execute(
                    "DELETE FROM entries WHERE namespace=? AND key IN "
                    "(SELECT key FROM entries WHERE namespace=? "
                    "ORDER BY created DESC, key ASC LIMIT -1 OFFSET ?)",
                    (namespace, namespace, self.max_entries),
                )
            # Return a fresh JSON copy on both paths so callers cannot mutate the cache.
            return CacheResult(json.loads(encoded), False, key, created)
        finally:
            with closing(self._connect()) as connection, connection:
                connection.execute("DELETE FROM leases WHERE namespace=? AND key=? AND owner=?",
                                   (namespace, key, owner))


@lru_cache(maxsize=8)
def _cache(path: str, ttl: float, max_entries: int) -> PersistentCache:
    return PersistentCache(Path(path), ttl=ttl, max_entries=max_entries)


def cached(namespace: str, signature: Any, compute: Callable[[], Any]) -> CacheResult:
    signature = {"pipeline": pipeline_revision(), "dataset": dataset_revision(),
                 "input": signature}
    if not cache_enabled():
        return CacheResult(compute(), False, fingerprint(signature), time.time())
    default_path = ((Path(tempfile.gettempdir()) / "fashion-search" / "retrieval.sqlite3")
                    if os.getenv("VERCEL") else ROOT / ".cache" / "retrieval.sqlite3")
    path = Path(os.getenv("FASHION_CACHE_PATH") or default_path).resolve()
    ttl = float(os.getenv("FASHION_CACHE_TTL_SECONDS", "0"))
    max_entries = int(os.getenv("FASHION_CACHE_MAX_ENTRIES", "4096"))
    if not 0 <= ttl < float("inf") or max_entries < 1:
        raise ValueError("Cache TTL must be finite and nonnegative; max entries must be positive")
    return _cache(str(path), ttl, max_entries).get_or_compute(namespace, signature, compute)
