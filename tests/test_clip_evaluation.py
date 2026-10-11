"""Ranking metrics count misses and multiple relevant products correctly."""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_clip import metrics


class MetricsTests(unittest.TestCase):
    def row(self, relevant, ranked):
        return {"relevant_ids": relevant, "rankings": {"clip": {"items": [{"image_id": item} for item in ranked]}}}

    def test_miss_is_zero_and_reciprocal_rank_uses_first_match(self):
        result = metrics([self.row(["a"], ["x", "a"]), self.row(["b"], ["x"])], "clip")
        self.assertEqual(result["recall@1"], 0)
        self.assertEqual(result["recall@5"], .5)
        self.assertEqual(result["mrr@20"], .25)

    def test_multiple_relevant_products_use_fractional_recall(self):
        result = metrics([self.row(["a", "b"], ["a", "x", "b"])], "clip")
        self.assertEqual(result["recall@1"], .5)
        self.assertEqual(result["recall@5"], 1)
        self.assertEqual(result["mrr@20"], 1)

    def test_empty_evaluation_does_not_claim_accuracy(self):
        self.assertEqual(metrics([], "clip"), {"queries": 0})


if __name__ == "__main__":
    unittest.main()
