"""Offline checks for vector and graph candidate union and ranking."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fashion_how_graphdb import hybrid_search, search


class HybridSearchTests(unittest.TestCase):
    def test_explicit_white_jacket_requires_white_edge(self):
        extraction = search.normalize_extraction({
            "item_type_codes": ["jackets"],
            "common_filters": [
                {"group": "colors", "value": "white", "hard": False},
                {"group": "patterns", "value": "stripes", "hard": False},
            ],
        }, query="white jacket")
        self.assertEqual(extraction["common_filters"][0]["hard"], True)
        self.assertNotIn("hard", extraction["common_filters"][1])
        white = {"scope": "common", "group": "colors", "value": "white",
                 "confidence": .8, "coverage": 1.0}
        ranked = hybrid_search.rerank([
            {"id": "white", "text_score": .7, "mapped_attributes": [white]},
            {"id": "not-white", "text_score": .99, "mapped_attributes": []},
        ], extraction, limit=10, min_confidence=.5, min_score=None)
        self.assertEqual([item["id"] for item in ranked], ["white"])

    def test_candidate_union_uses_vector_index_and_attribute_edges(self):
        extraction = search.normalize_extraction({
            "item_type_codes": ["jackets"],
            "common_filters": [{"group": "colors", "value": "white", "hard": True}],
            "category_filters": [{"type_code": "jackets", "group": "outer_closure",
                                  "value": "zipper", "hard": True}],
        })
        session = MagicMock()
        session.run.side_effect = [
            [{"id": "vector-1"}, {"id": "both"}],
            [{"id": "both"}, {"id": "graph-1"}],
            [{"id": "graph-1"}],
        ]
        client = MagicMock()
        client.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1, 0.2])]
        )

        ids, embedding = hybrid_search.select_candidates(
            session, extraction, "white zippered jacket", client=client,
            embedding_model="text-embedding-3-large", limit=12, min_confidence=.5,
        )

        self.assertEqual(ids, ["vector-1", "both", "graph-1"])
        self.assertEqual(embedding, [0.1, 0.2])
        vector_query = session.run.call_args_list[0]
        self.assertIn("db.index.vector.queryNodes", vector_query.args[0])
        self.assertEqual(vector_query.kwargs["index_name"], "item_description_embedding")
        self.assertEqual(vector_query.kwargs["item_type_codes"], ["jackets"])
        self.assertEqual(session.run.call_args_list[2].kwargs["value_id"],
                         "outer_closure:zipper")

    def test_hard_filters_exclude_only_required_or_forbidden_attributes(self):
        extraction = search.normalize_extraction({
            "common_filters": [
                {"group": "colors", "value": "white", "hard": True},
                {"group": "patterns", "value": "stripes", "hard": False},
            ],
            "excluded_common_filters": [{"group": "colors", "value": "red"}],
            "style_axis_targets": [
                {"axis": "casual_formal", "target": .2, "evidence": "casual"}
            ],
        })
        white = {"scope": "common", "group": "colors", "value": "white",
                 "confidence": .8, "coverage": 1.0}
        red = {"scope": "common", "group": "colors", "value": "red",
               "confidence": .9}
        rows = [
            {"id": "keep", "text_score": .9, "casual_formal": .2,
             "mapped_attributes": [white]},
            {"id": "no-white", "text_score": 1.0, "casual_formal": .2,
             "mapped_attributes": []},
            {"id": "red", "text_score": 1.0, "casual_formal": .2,
             "mapped_attributes": [white, red]},
        ]

        ranked = hybrid_search.rerank(
            rows, extraction, limit=10, min_confidence=.5, min_score=None,
        )

        self.assertEqual([row["id"] for row in ranked], ["keep"])
        self.assertEqual(ranked[0]["matched_filters"][0]["value"], "white")
        self.assertEqual(ranked[0]["score_components"], ["graph", "text", "style"])
        self.assertAlmostEqual(
            ranked[0]["score"],
            .35 * .8 + .55 * .9 + .10 * 1.0,
        )

    def test_inferred_attribute_is_neither_graph_candidate_nor_score(self):
        extraction = search.normalize_extraction({
            "common_filters": [{"group": "patterns", "value": "stripes", "hard": False}],
        })
        session = MagicMock()
        session.run.return_value = [{"id": "vector-1"}]
        client = MagicMock()
        client.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1, 0.2])]
        )
        ids, _ = hybrid_search.select_candidates(
            session, extraction, "shirt", client=client,
            embedding_model="text-embedding-3-large", limit=12, min_confidence=0.5,
        )
        self.assertEqual(ids, ["vector-1"])
        self.assertEqual(session.run.call_count, 1)
        ranked = hybrid_search.rerank([
            {"id": "vector-1", "text_score": .9, "mapped_attributes": []},
        ], extraction, limit=10, min_confidence=.5, min_score=None)
        self.assertEqual(ranked[0]["score_components"], ["text"])
        self.assertIsNone(ranked[0]["graph_score"])

    def test_schema_and_style_axes_match_loaded_graph(self):
        schema = search.extraction_schema()
        positive = schema["properties"]["common_filters"]["items"]
        excluded = schema["properties"]["excluded_common_filters"]["items"]
        self.assertIn("hard", positive["required"])
        self.assertNotIn("hard", excluded["properties"])
        self.assertIn("visual_trend", search.STYLE_AXES)
        self.assertIn("thermal_impression", search.STYLE_AXES)
        self.assertIn("design_expression", search.STYLE_AXES)
        self.assertNotIn("cool_warm", search.STYLE_AXES)
        cypher = hybrid_search.result_cypher({})
        self.assertIn("vector.similarity.cosine", cypher)
        self.assertIn(".thermal_impression", cypher)


if __name__ == "__main__":
    unittest.main()
