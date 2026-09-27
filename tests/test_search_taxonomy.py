"""Regression checks for Fashion-200K extraction and graph identifiers."""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fashion_how_graphdb import hybrid_search, search


class Fashion200KSearchTests(unittest.TestCase):
    def test_catalog_and_schema_use_fashion200k_categories(self):
        catalog = search.search_catalog()
        categories = {"dresses", "jackets", "pants", "skirts", "tops"}
        self.assertEqual(set(catalog["item_types"]), categories)
        self.assertEqual(set(catalog["category_groups"]), categories)
        self.assertEqual(set(catalog["common_groups"]),
                         {"colors", "materials", "patterns", "seasons"})
        self.assertEqual(set(search.extraction_schema()["properties"]
                             ["item_type_codes"]["items"]["enum"]), categories)

    def test_extraction_to_cypher_keeps_english_ids_and_exclusions(self):
        raw = {
            "item_type_codes": ["pants"],
            "common_filters": [{"group": "colors", "value": "navy"}],
            "category_filters": [{"type_code": "pants", "group": "pants_fit",
                                  "value": "wide_leg"}],
            "excluded_common_filters": [{"group": "materials", "value": "denim"}],
            "excluded_category_filters": [{"type_code": "pants", "group": "pants_rise",
                                           "value": "low_rise"}],
        }
        for query in ("데님 아닌 네이비 와이드 팬츠, 로우라이즈 제외",
                      "navy wide-leg pants, no denim or low rise"):
            with self.subTest(query=query), patch.object(
                search, "call_text_json_with_error", return_value=(raw, None)
            ) as provider:
                extraction = search.extract_search_filters(query, client=object(), model="test")
                prompt = json.loads(provider.call_args.kwargs["user_prompt"])
                self.assertEqual(prompt["query"], query)
                self.assertIn("navy", prompt["catalog"]["common_groups"]["colors"]["values"])
                cypher = hybrid_search.result_cypher(extraction)
                self.assertIn("[:IS_CATEGORY]->(item_category:Category)", cypher)
                self.assertIn("vector.similarity.cosine", cypher)
                self.assertIn("CALL (item) {", cypher)
                self.assertNotIn("IS_TYPE", cypher)
                self.assertEqual(extraction["item_type_codes"], ["pants"])
                self.assertEqual(
                    hybrid_search.attribute_candidate_cypher(extraction["category_filters"][0])[1],
                    "pants_fit:wide_leg",
                )

    def test_legacy_values_and_incompatible_category_details_are_rejected(self):
        result = search.normalize_extraction({
            "item_type_codes": ["BL", "tops"],
            "common_filters": [{"group": "colors", "value": "파란색"},
                               {"group": "styles", "value": "캐주얼한"},
                               {"group": "colors", "value": "blue"}],
            "category_filters": [{"type_code": "pants", "group": "pants_fit",
                                  "value": "wide_leg"}],
        })
        self.assertEqual(result["item_type_codes"], ["tops"])
        self.assertEqual([item["value"] for item in result["common_filters"]], ["blue"])
        self.assertEqual(result["category_filters"], [])

    def test_updated_taxonomy_accepts_new_graph_ids_and_rejects_removed_values(self):
        result = search.normalize_extraction({
            "item_type_codes": ["tops", "dresses"],
            "common_filters": [
                {"group": "colors", "value": "burgundy"},
                {"group": "colors", "value": "cream"},
            ],
            "category_filters": [
                {"type_code": "tops", "group": "top_type", "value": "blouse"},
                {"type_code": "dresses", "group": "dress_silhouette", "value": "flared"},
                {"type_code": "dresses", "group": "dress_silhouette", "value": "fit_and_flare"},
            ],
        })
        self.assertEqual([item["value"] for item in result["common_filters"]], ["burgundy"])
        self.assertEqual(
            [(item["group"], item["value"]) for item in result["category_filters"]],
            [("top_type", "blouse"), ("dress_silhouette", "flared")],
        )


if __name__ == "__main__":
    unittest.main()
