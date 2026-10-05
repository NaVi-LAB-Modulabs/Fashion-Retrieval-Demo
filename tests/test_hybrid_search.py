"""Offline checks for vector and graph candidate union and ranking."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fashion_how_graphdb import hybrid_search, search


class HybridSearchTests(unittest.TestCase):
    def test_brown_accent_is_scored_by_coverage_and_filtered_by_product(self):
        extraction = search.normalize_extraction({
            "item_type_codes": ["jackets"],
            "common_filters": [{"group": "colors", "value": "brown", "hard": True}],
            "category_filters": [{"type_code": "jackets", "group": "outer_type",
                                  "value": "coat", "hard": True}],
        })
        brown = {"scope": "common", "group": "colors", "value": "brown",
                 "confidence": .96, "coverage": .08}
        coat = {"scope": "category", "type_code": "jackets", "group": "outer_type",
                "value": "coat", "confidence": .99}
        rows = [
            {"id": "accent", "text_score": .99, "mapped_attributes": [brown, coat]},
            {"id": "main-color", "text_score": .6,
             "mapped_attributes": [{**brown, "coverage": .92}, coat]},
        ]
        ranked = hybrid_search.rerank(
            rows, extraction, limit=10, min_confidence=0, min_score=None,
            weights={"graph": 1, "text": 0, "style": 0},
        )
        self.assertEqual([item["id"] for item in ranked], ["main-color", "accent"])
        self.assertEqual(ranked[1]["matched_filters"][0]["score"], .0768)
        self.assertEqual(ranked[1]["graph_score"], .5334)
        filtered = hybrid_search.rerank(
            rows, extraction, limit=10, min_confidence=.1, min_score=None,
        )
        self.assertEqual([item["id"] for item in filtered], ["main-color"])

    def test_excluded_color_uses_product_threshold(self):
        extraction = search.normalize_extraction({
            "excluded_common_filters": [{"group": "colors", "value": "brown"}],
        })
        brown = {"scope": "common", "group": "colors", "value": "brown",
                 "confidence": .96, "coverage": .08}
        self.assertTrue(hybrid_search.passes_hard_filters([brown], extraction, .1))
        self.assertFalse(hybrid_search.passes_hard_filters([brown], extraction, .05))

    def test_attribute_metrics_and_candidate_thresholds(self):
        cases = [
            ("common", "colors", "coverage", "coverage"),
            ("common", "patterns", "prominence", "prominence"),
            ("category", "outer_closure", "prominence", "prominence"),
            ("common", "seasons", "strength", "score"),
            ("common", "materials", None, None),
        ]
        for scope, group, metric, edge_property in cases:
            with self.subTest(group=group):
                search_filter = {"scope": scope, "group": group, "value": "zipper",
                                 "type_code": "jackets"}
                attribute = {**search_filter, "confidence": .8}
                # Unrelated fields must not become the group's metric.
                attribute.update(coverage=.1, prominence=.2, strength=.3)
                if metric:
                    attribute[metric] = .5
                expected = .4 if metric else .8
                self.assertAlmostEqual(hybrid_search.attribute_strength(attribute), expected)
                self.assertTrue(hybrid_search.attribute_matches(search_filter, attribute, expected))
                self.assertFalse(hybrid_search.attribute_matches(search_filter, attribute, expected + .01))
                if metric:
                    del attribute[metric]
                    self.assertEqual(hybrid_search.attribute_strength(attribute), .8)
                cypher, _ = hybrid_search.attribute_candidate_cypher(search_filter)
                expression = (f"coalesce(toFloat(edge.{edge_property}), 1.0)"
                              if edge_property else "1.0")
                self.assertIn(expression + " AS metric", cypher)
                self.assertIn("WHERE attribute_score >= $min_confidence", cypher)
                self.assertIn("ORDER BY attribute_score DESC", cypher)

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
