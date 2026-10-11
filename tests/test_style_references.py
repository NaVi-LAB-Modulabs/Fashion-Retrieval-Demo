"""Reference grounding, abstention, cache isolation and fair ranking comparison."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from fashion_how_graphdb import llm, style_references as refs, web_service as service
import test_web_service as integration


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.axis = next(axis for axis in refs.reference_manifest()["axes"] if axis["id"] == "casual_formal")
        self.row = {"axis": "casual_formal", "target": .78, "confidence": .9,
                    "evidence": "Tailored construction fits the mostly formal references.",
                    "supporting_reference_ids": [self.axis["references"][-5]["id"]],
                    "is_ambiguous": False, "ambiguity_reason": ""}
        self.extraction = {"style_axis_targets": [{"axis": "casual_formal", "target": 1., "evidence": "formal"}]}

    def test_assets_match_all_seven_axes_and_original_landmark_ranges(self):
        manifest = refs.reference_manifest()
        self.assertEqual(len(manifest["axes"]), 7)
        self.assertEqual(sum(len(axis["references"]) for axis in manifest["axes"]), 200)
        design = next(axis for axis in manifest["axes"] if axis["id"] == "design_expression")
        self.assertEqual(min(design["levels"]), .25)
        self.assertEqual(min(self.axis["levels"]), 0.)

    def test_calibration_sends_real_images_and_preserves_non_style_filters(self):
        extraction = {**self.extraction, "common_filters": [{"value": "white", "hard": True}]}
        with patch.dict("os.environ", {"FASHION_CACHE_ENABLED": "0"}), \
                patch.object(refs, "call_text_json_with_error", return_value=({"assessments": [self.row]}, None)) as call:
            result = refs.calibrate_query("a formal white jacket", extraction, client=object(), model="gpt-4.1")
        content = call.call_args.kwargs["user_content"]
        self.assertEqual(len(content), 2)
        self.assertEqual(content[1]["type"], "input_image")
        self.assertTrue(content[1]["image_url"].startswith("data:image/jpeg;base64,/9j/"))
        self.assertNotIn('"target": 1.0', call.call_args.kwargs["user_prompt"])
        self.assertIn("DESIRED style position", call.call_args.kwargs["system_prompt_text"])
        self.assertEqual(result["style_axis_targets"][0]["target"], .78)
        self.assertEqual(result["common_filters"], extraction["common_filters"])
        self.assertEqual(extraction["style_axis_targets"][0]["target"], 1.)

    def test_uncertain_axes_are_omitted_and_evidence_retained(self):
        for change in ({"confidence": .74}, {"is_ambiguous": True}, {"ambiguity_reason": "Conflicting intent"}):
            with self.subTest(change=change):
                targets, assessments = refs.normalize_assessments({"assessments": [{**self.row, **change}]}, [self.axis])
                self.assertEqual(targets, [])
                self.assertFalse(assessments[0]["active"])

    def test_wrong_reference_ids_duplicates_and_out_of_range_are_rejected(self):
        for change in ({"supporting_reference_ids": ["invented"]}, {"target": float("nan")},
                       {"target": 1.1}, {"confidence": True}, {"axis": "design_expression"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                refs.normalize_assessments({"assessments": [{**self.row, **change}]}, [self.axis])
        with self.assertRaises(ValueError):
            refs.normalize_assessments({"assessments": [self.row, self.row]}, [self.axis])

    def test_unrequested_axes_do_not_trigger_a_visual_call(self):
        with patch.object(refs, "call_text_json_with_error") as call:
            result = refs.calibrate_query("a white jacket", {"style_axis_targets": []}, client=object(), model="test")
        call.assert_not_called()
        self.assertEqual(result["style_calibration"]["assessments"], [])

    def test_model_failure_is_not_silently_replaced_by_legacy_targets(self):
        with patch.dict("os.environ", {"FASHION_CACHE_ENABLED": "0"}), \
                patch.object(refs, "call_text_json_with_error", return_value=(None, {"type": "APIError"})), \
                self.assertRaises(RuntimeError):
            refs.calibrate_query("formal", self.extraction, client=object(), model="test")

    def test_reference_revision_is_part_of_successful_result_cache(self):
        with TemporaryDirectory() as directory, patch.dict("os.environ", {
                "FASHION_CACHE_ENABLED": "1", "FASHION_CACHE_PATH": str(Path(directory) / "cache.sqlite3")}), \
                patch.object(refs, "call_text_json_with_error", return_value=({"assessments": [self.row]}, None)) as call:
            first = refs.calibrate_query("formal", self.extraction, client=object(), model="test")
            second = refs.calibrate_query("formal", self.extraction, client=object(), model="test")
            self.assertEqual(first, second)
            self.assertEqual(call.call_count, 1)
            with patch.object(refs, "reference_revision", return_value="changed-reference-images"):
                refs.calibrate_query("formal", self.extraction, client=object(), model="test")
            self.assertEqual(call.call_count, 2)

    def test_json_retry_keeps_reference_images(self):
        client = MagicMock()
        client.responses.create.side_effect = [SimpleNamespace(output_text="invalid"), SimpleNamespace(output_text='{"ok": true}')]
        content = [{"type": "input_image", "image_url": "data:image/jpeg;base64,test", "detail": "high"}]
        result, error = llm.call_text_json_with_error(client=client, model="gpt-4.1", user_prompt="formal",
                                                     system_prompt_text="calibrate", user_content=content)
        self.assertEqual(result, {"ok": True})
        self.assertIsNone(error)
        for call in client.responses.create.call_args_list:
            self.assertIn(content[0], call.kwargs["input"][1]["content"])


class ComparisonTests(unittest.TestCase):
    def test_reference_mode_reranks_the_identical_candidate_pool_against_legacy(self):
        helper = integration.RetrievalTests()
        helper.setUp()
        try:
            helper.extraction["style_axis_targets"] = [{"axis": "casual_formal", "target": 1., "evidence": "formal"}]
            def calibrate(query, extraction, **kwargs):
                return {**extraction, "style_axis_targets": [{"axis": "casual_formal", "target": .7, "evidence": "references"}]}
            rows = [{"id": item_id, "text_score": .8, "casual_formal": value,
                     "internal_note": "private", "mapped_attributes": [
                         {"scope": "common", "group": "colors", "value": "blue", "confidence": .8}]}
                    for item_id, value in [("a", .7), ("b", 1.)]]
            with patch.object(service.style_references, "calibrate_query", side_effect=calibrate) as call:
                result, session, client = helper.run_search(rows, style_mode="reference", compare_styles=True)
            self.assertEqual([item["id"] for item in result["items"]], ["a", "b"])
            self.assertEqual([item["id"] for item in result["comparison"]["items"]], ["b", "a"])
            self.assertTrue(result["comparison"]["same_candidate_pool"])
            self.assertEqual(result["style_coverage"], [{"axis": "casual_formal", "available_candidates": 2,
                                                        "total_candidates": 2}])
            self.assertEqual(session.run.call_count, 3)
            call.assert_called_once()
            client.embeddings.create.assert_called_once()
            self.assertNotIn("internal_note", result["comparison"]["items"][0])
        finally:
            helper.doCleanups()

    def test_legacy_mode_remains_default_and_invalid_comparison_is_rejected(self):
        self.assertEqual(service.validate_request({"query": "formal"})["style_mode"], "legacy")
        for values in ({"style_mode": "other"}, {"compare_styles": True},
                       {"style_mode": "reference", "compare_styles": "yes"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                service.validate_request({"query": "formal", **values})


if __name__ == "__main__":
    unittest.main()
