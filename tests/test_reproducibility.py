"""Offline checks for persistence, concurrent misses and deterministic retrieval."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fashion_how_graphdb import hybrid_search, search, web_service
from fashion_how_graphdb import reproducibility as repro
from fashion_how_graphdb.llm import _create_kwargs


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "retrieval.sqlite3"
        self.cache = repro.PersistentCache(self.path)

    def test_reopening_preserves_success_and_caller_mutation_is_isolated(self):
        compute = MagicMock(return_value={"ids": ["first"]})
        first = self.cache.get_or_compute("search", {"query": "blue"}, compute)
        first.value["ids"].append("mutation")
        reopened = repro.PersistentCache(self.path)
        second = reopened.get_or_compute("search", {"query": "blue"}, compute)
        self.assertFalse(first.hit)
        self.assertTrue(second.hit)
        self.assertEqual(second.value, {"ids": ["first"]})
        self.assertEqual(first.created_at, second.created_at)
        compute.assert_called_once()

    def test_failure_releases_lease_and_is_not_cached(self):
        compute = MagicMock(side_effect=[RuntimeError("provider failed"), ["success"]])
        with self.assertRaises(RuntimeError):
            self.cache.get_or_compute("parser", "query", compute)
        recovered = self.cache.get_or_compute("parser", "query", compute)
        self.assertFalse(recovered.hit)
        self.assertEqual(recovered.value, ["success"])
        self.assertEqual(compute.call_count, 2)

    def test_expiry_eviction_and_namespaces(self):
        cache = repro.PersistentCache(self.path, ttl=5, max_entries=1)
        cache.get_or_compute("embedding", "a", lambda: [1])
        cache.get_or_compute("search", "a", lambda: [2])
        cache.get_or_compute("search", "b", lambda: [3])
        self.assertTrue(cache.get_or_compute("embedding", "a", lambda: [9]).hit)
        self.assertFalse(cache.get_or_compute("search", "a", lambda: [4]).hit)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("UPDATE entries SET expires=1 WHERE namespace='embedding'")
        self.assertEqual(cache.get_or_compute("embedding", "a", lambda: [5]).value, [5])

    def test_shared_file_single_flight_across_independent_cache_instances(self):
        other_worker = repro.PersistentCache(self.path)
        entered, release = Event(), Event()

        def compute():
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test did not release provider")
            return ["one computation"]

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.cache.get_or_compute, "search", "same", compute)
            self.assertTrue(entered.wait(5))
            duplicate_compute = MagicMock(return_value=["different"])
            second = pool.submit(other_worker.get_or_compute, "search", "same", duplicate_compute)
            # The lease already exists in SQLite when the callback starts.
            release.set()
            a, b = first.result(timeout=5), second.result(timeout=5)
        duplicate_compute.assert_not_called()
        self.assertEqual(a.value, b.value)
        self.assertTrue(b.hit)

    def test_expired_worker_lease_is_reclaimed(self):
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("INSERT INTO leases VALUES (?, ?, ?, ?)",
                               ("search", repro.fingerprint("query"), "dead-worker", 1))
        self.assertEqual(self.cache.get_or_compute("search", "query", lambda: [1]).value, [1])


class ReproducibleRetrievalTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        environment = patch.dict("os.environ", {
            "FASHION_CACHE_ENABLED": "1",
            "FASHION_CACHE_PATH": str(Path(directory.name) / "retrieval.sqlite3"),
            "FASHION_CACHE_TTL_SECONDS": "0", "FASHION_CACHE_MAX_ENTRIES": "4096",
            "FASHION_DATASET_REVISION": "test-v1", "FASHION_VECTOR_SEARCH_MODE": "exact",
        })
        environment.start()
        self.addCleanup(environment.stop)
        self.client = MagicMock()
        self.client.base_url = "https://api.openai.com/v1"

    def test_parser_caches_normalized_query_and_uses_snapshot_without_seed(self):
        with patch.object(search, "call_text_json_with_error", return_value=(
            {"common_filters": [{"group": "colors", "value": "blue", "hard": True}]}, None
        )) as provider:
            first = search.extract_search_filters(" blue\n shirt ", client=self.client, model="gpt-5.4-mini")
            second = search.extract_search_filters("blue shirt", client=self.client, model="gpt-5.4-mini")
        self.assertEqual(first, second)
        provider.assert_called_once()
        self.assertEqual(provider.call_args.kwargs["model"], "gpt-5.4-mini-2026-03-17")
        self.assertEqual(_create_kwargs("gpt-4.1-mini-2025-04-14")["temperature"], 0)
        for model in ("gpt-5.4-mini-2026-03-17", "gpt-4.1-mini-2025-04-14"):
            self.assertNotIn("seed", _create_kwargs(model))

    def test_embedding_is_reused_across_queries_and_separated_by_model(self):
        self.client.embeddings.create.side_effect = [
            SimpleNamespace(data=[SimpleNamespace(embedding=[.1, .2])]),
            SimpleNamespace(data=[SimpleNamespace(embedding=[.3, .4])]),
        ]
        first = hybrid_search.query_embedding(self.client, " blue\nshirt ", "model-a")
        second = hybrid_search.query_embedding(self.client, "blue shirt", "model-a")
        third = hybrid_search.query_embedding(self.client, "blue shirt", "model-b")
        self.assertEqual(first, second)
        self.assertNotEqual(second, third)
        self.assertEqual(self.client.embeddings.create.call_count, 2)

    def test_cache_revision_and_disable_force_recomputation(self):
        compute = MagicMock(side_effect=[1, 2, 3, 4])
        self.assertEqual(repro.cached("test", "query", compute).value, 1)
        self.assertTrue(repro.cached("test", "query", compute).hit)
        with patch.dict("os.environ", {"FASHION_DATASET_REVISION": "test-v2"}):
            self.assertEqual(repro.cached("test", "query", compute).value, 2)
        with patch.dict("os.environ", {"FASHION_CACHE_ENABLED": "0"}):
            self.assertEqual(repro.cached("test", "query", compute).value, 3)
            self.assertEqual(repro.cached("test", "query", compute).value, 4)

    def test_filter_order_duplicates_and_attribute_order_do_not_change_scores(self):
        filters = [{"group": "colors", "value": "blue", "hard": False},
                   {"group": "colors", "value": "blue", "hard": True},
                   {"group": "materials", "value": "cotton", "hard": True}]
        first = search.normalize_extraction({"common_filters": filters})
        second = search.normalize_extraction({"common_filters": list(reversed(filters))})
        self.assertEqual(first, second)
        self.assertEqual(len(first["common_filters"]), 2)
        attributes = [{"scope": "common", "group": "colors", "value": "blue", "confidence": .8},
                      {"scope": "common", "group": "materials", "value": "cotton", "confidence": .9}]
        results = []
        for ordered in (attributes, list(reversed(attributes))):
            results.append(hybrid_search.rerank(
                [{"id": "b", "text_score": .7, "mapped_attributes": ordered},
                 {"id": "a", "text_score": .7, "mapped_attributes": ordered}],
                first, limit=12, min_confidence=None, min_score=None,
            ))
        self.assertEqual(results[0], results[1])
        self.assertEqual([row["id"] for row in results[0]], ["a", "b"])

    def test_exact_mode_searches_entire_filtered_population_and_breaks_ties_before_limit(self):
        session = MagicMock()
        session.run.side_effect = [
            [{"id": item_id, "graph_score": None, "description_eligible": True}
             for item_id in ("z", "b", "a")],
            [{"id": "a", "text_score": .9}, {"id": "b", "text_score": .9}],
        ]
        with patch.object(hybrid_search, "query_embedding", return_value=[1, 0]):
            ids, _ = hybrid_search.select_candidates(
                session, {}, "blue shirt", client=self.client,
                embedding_model="model", limit=1, min_confidence=None,
            )
        self.assertEqual(ids, ["a", "b"])
        query = session.run.call_args.args[0]
        self.assertNotIn("SEARCH item IN", query)
        self.assertIn("vector.similarity.cosine", query)
        self.assertLess(query.index("ORDER BY text_score DESC, id ASC"), query.index("LIMIT"))
        self.assertEqual(session.run.call_args.kwargs["filtered_ids"], ["a", "b", "z"])
        self.assertEqual(session.run.call_args.kwargs["embedding_model"], "model")

    def test_full_search_cache_skips_providers_and_varies_with_settings_and_mode(self):
        with ExitStack() as stack:
            stack.enter_context(patch.object(web_service, "configuration", return_value={"ready": True}))
            stack.enter_context(patch.object(web_service, "openai_client", side_effect=AssertionError("cache hit called provider")))
            stack.enter_context(patch.object(web_service, "neo4j_driver", side_effect=AssertionError("cache hit called DB")))
            compute = stack.enter_context(patch.object(web_service, "_retrieve_uncached", return_value={
                "items": [{"id": "first", "score": .9}], "extraction": {"description_query": "blue"},
                "timings": {"parse_ms": 12, "graph_ms": 23, "rerank_ms": 1, "total_ms": 36},
            }))
            first = web_service.retrieve({"query": " blue\nshirt "})
            first["items"][0]["id"] = "caller mutation"
            second = web_service.retrieve({"query": "blue shirt"})
            self.assertTrue(second["cache"]["hit"])
            self.assertEqual(second["items"][0]["id"], "first")
            self.assertEqual(second["timings"]["parse_ms"], 0)
            self.assertEqual(second["cache"]["original_timings"]["parse_ms"], 12)
            compute.assert_called_once()
            changes = [{"limit": 5}, {"min_confidence": .5}, {"min_score": .5},
                       {"model": "gpt-4.1"}, {"weights": {"text": 1, "graph": 0, "style": 0}}]
            for settings in changes:
                self.assertFalse(web_service.retrieve({"query": "blue shirt", **settings})["cache"]["hit"])
            with patch.dict("os.environ", {"FASHION_VECTOR_SEARCH_MODE": "hnsw"}):
                self.assertFalse(web_service.retrieve({"query": "blue shirt"})["cache"]["hit"])
            with patch.dict("os.environ", {"FASHION_DATASET_REVISION": "test-v2"}):
                self.assertFalse(web_service.retrieve({"query": "blue shirt"})["cache"]["hit"])
            self.assertEqual(compute.call_count, 8)


if __name__ == "__main__":
    unittest.main()
