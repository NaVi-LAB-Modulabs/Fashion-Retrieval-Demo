"""Offline checks for vector and graph candidate union and ranking."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

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
                cypher, params = hybrid_search.candidate_cypher({
                    "common_filters": [{**search_filter, "hard": True}],
                })
                expression = (f"coalesce(toFloat(required_edge_0.{edge_property}), 1.0)"
                              if edge_property else "1.0")
                self.assertIn(expression, cypher)
                self.assertIn(">= $min_confidence", cypher)
                self.assertIn("AS graph_score", cypher)
                self.assertEqual(params["branch_limit"], 10)

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

    def test_candidate_union_uses_two_filtered_top_ten_lists(self):
        extraction = search.normalize_extraction({
            "item_type_codes": ["jackets"],
            "common_filters": [{"group": "colors", "value": "white", "hard": True}],
            "category_filters": [{"type_code": "jackets", "group": "outer_closure",
                                  "value": "zipper", "hard": True}],
        })
        session = MagicMock()
        session.run.side_effect = [
            [{"id": "graph-1", "graph_score": .9, "description_eligible": False},
             {"id": "both", "graph_score": .8, "description_eligible": True},
             *[{"id": f"other-{i}", "graph_score": .7, "description_eligible": True}
               for i in range(8)],
             {"id": "vector-1", "graph_score": .1, "description_eligible": True}],
            [{"id": "vector-1", "text_score": .95}, {"id": "both", "text_score": .9},
             {"id": "outside-filter", "text_score": 1}],
        ]
        client = MagicMock()
        client.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1, 0.2])]
        )

        ids, embedding = hybrid_search.select_candidates(
            session, extraction, "white zippered jacket", client=client,
            embedding_model="text-embedding-3-large", limit=12, min_confidence=.5,
        )

        self.assertEqual(ids, ["graph-1", "both", *[f"other-{i}" for i in range(8)], "vector-1"])
        self.assertEqual(embedding, [0.1, 0.2])
        filter_query, vector_query = session.run.call_args_list
        self.assertNotIn("vector.similarity.cosine", filter_query.args[0])
        self.assertNotIn("LIMIT", filter_query.args[0])
        self.assertIn("CYPHER 25", vector_query.args[0])
        self.assertIn("VECTOR INDEX `item_description_embedding_filtered`", vector_query.args[0])
        self.assertIn("WHERE item.id IN $filtered_ids", vector_query.args[0])
        self.assertNotIn("vector.similarity.cosine", vector_query.args[0])
        self.assertNotIn("graph-1", vector_query.kwargs["filtered_ids"])
        self.assertIn("vector-1", vector_query.kwargs["filtered_ids"])
        self.assertEqual(len(vector_query.kwargs["filtered_ids"]), 10)
        self.assertEqual(vector_query.kwargs["branch_limit"], 10)
        self.assertEqual(filter_query.kwargs["item_type_codes"], ["jackets"])
        self.assertEqual(filter_query.kwargs["required_0"], "white")
        self.assertEqual(filter_query.kwargs["required_1"], "outer_closure:zipper")
        self.assertEqual(session.run.call_count, 2)

    def test_both_lists_apply_required_and_excluded_filters_before_limit(self):
        extraction = search.normalize_extraction({
            "item_type_codes": ["jackets"],
            "common_filters": [
                {"group": "colors", "value": "white", "hard": True},
                {"group": "materials", "value": "denim", "hard": True},
            ],
            "excluded_category_filters": [
                {"type_code": "jackets", "group": "outer_closure", "value": "zipper"},
            ],
        })
        cypher, params = hybrid_search.candidate_cypher(extraction)
        self.assertEqual(params["excluded_0"], "outer_closure:zipper")
        self.assertEqual(params["excluded_0_type"], "jackets")
        self.assertEqual(params["branch_limit"], 10)
        self.assertIn("NOT EXISTS", cypher)
        self.assertIn("id: $required_1", cypher)
        self.assertNotIn("LIMIT", cypher)
        self.assertIn("(1.0 * attribute_score_0 + 0.5 * attribute_score_1) / 1.5", cypher)
        ann = hybrid_search.description_candidate_cypher()
        self.assertLess(ann.index("WHERE item.id IN $filtered_ids"), ann.index("LIMIT $branch_limit"))
        self.assertLess(ann.index("LIMIT $branch_limit"), ann.index(") SCORE AS"))


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
        session.run.side_effect = [
            [{"id": "vector-1", "graph_score": None, "description_eligible": True}],
            [{"id": "vector-1", "text_score": .9}],
        ]
        client = MagicMock()
        client.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1, 0.2])]
        )
        ids, _ = hybrid_search.select_candidates(
            session, extraction, "shirt", client=client,
            embedding_model="text-embedding-3-large", limit=12, min_confidence=0.5,
        )
        self.assertEqual(ids, ["vector-1"])
        self.assertEqual(session.run.call_count, 2)
        ranked = hybrid_search.rerank([
            {"id": "vector-1", "text_score": .9, "mapped_attributes": []},
        ], extraction, limit=10, min_confidence=.5, min_score=None)
        self.assertEqual(ranked[0]["score_components"], ["text"])
        self.assertIsNone(ranked[0]["graph_score"])

    def test_empty_filter_skips_embedding_and_ann(self):
        session, client = MagicMock(), MagicMock()
        session.run.return_value = []
        self.assertEqual(hybrid_search.select_candidates(
            session, {}, "white jacket", client=client,
            embedding_model="text-embedding-3-large", limit=20, min_confidence=.5,
        ), ([], []))
        session.run.assert_called_once()
        client.embeddings.create.assert_not_called()

    def test_items_without_compatible_embeddings_use_graph_only(self):
        session, client = MagicMock(), MagicMock()
        session.run.return_value = [
            {"id": "graph", "graph_score": .8, "description_eligible": False},
        ]
        self.assertEqual(hybrid_search.select_candidates(
            session, {}, "white jacket", client=client,
            embedding_model="text-embedding-3-large", limit=20, min_confidence=.5,
        ), (["graph"], []))
        session.run.assert_called_once()
        client.embeddings.create.assert_not_called()

    def test_index_configuration_is_quoted_as_one_identifier(self):
        with patch.object(hybrid_search, "DESCRIPTION_VECTOR_INDEX", "custom`index"):
            self.assertIn("VECTOR INDEX `custom``index`", hybrid_search.description_candidate_cypher())

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
