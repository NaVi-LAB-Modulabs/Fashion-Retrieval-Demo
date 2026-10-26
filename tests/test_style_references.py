"""Reference grounding, abstention, cache isolation and fair ranking comparison."""

from copy import deepcopy
from itertools import combinations
from pathlib import Path

from jsonschema import Draft202012Validator
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from fashion_how_graphdb import llm, style_references as refs, web_service as service
from fashion_how_graphdb.diagnostics import failure_details
import test_web_service as integration


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.axis = next(axis for axis in refs.reference_manifest()["axes"] if axis["id"] == "casual_formal")
        self.row = {"target": .78, "confidence": .9,
                    "evidence": "Tailored construction fits the mostly formal references.",
                    "supporting_reference_id": self.axis["references"][-5]["id"],
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
                patch.object(refs, "call_text_json_with_error", return_value=({"assessments": {"casual_formal": self.row}}, None)) as call:
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
                targets, assessments = refs.normalize_assessments({"assessments": {"casual_formal": {**self.row, **change}}}, [self.axis])
                self.assertEqual(targets, [])
                self.assertFalse(assessments[0]["active"])

    def test_wrong_reference_ids_duplicates_and_out_of_range_are_rejected(self):
        for change in ({"supporting_reference_id": "invented"}, {"target": float("nan")},
                       {"target": 1.1}, {"confidence": True}, {"confidence": float("inf")}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                refs.normalize_assessments({"assessments": {"casual_formal": {**self.row, **change}}}, [self.axis])
        with self.assertRaises(ValueError):
            refs.normalize_assessments({"assessments": [self.row, self.row]}, [self.axis])

    def test_unrequested_axes_do_not_trigger_a_visual_call(self):
        with patch.object(refs, "call_text_json_with_error") as call:
            result = refs.calibrate_query("a white jacket", {"style_axis_targets": []}, client=object(), model="test")
        call.assert_not_called()
        self.assertEqual(result["style_calibration"]["assessments"], [])

    def test_schema_validates_every_axis_subset_and_rejects_invalid_outputs(self):
        axes = refs.reference_manifest()["axes"]
        for size in range(1, len(axes) + 1):
            for subset in combinations(axes, size):
                schema = refs.calibration_schema(list(subset))
                Draft202012Validator.check_schema(schema)
                validator = Draft202012Validator(schema)
                raw = {"assessments": {axis["id"]: {
                    **self.row, "target": min(axis["levels"]),
                    "supporting_reference_id": axis["references"][0]["id"],
                } for axis in subset}}
                self.assertTrue(validator.is_valid(raw), [e.message for e in validator.iter_errors(raw)])
                targets, assessments = refs.normalize_assessments(raw, list(subset))
                self.assertEqual(len(targets), size)
                self.assertEqual(len(assessments), size)
                axis = subset[0]
                for field, value in (("target", min(axis["levels"]) - .01),
                                     ("target", max(axis["levels"]) + .01),
                                     ("supporting_reference_id", "invented"),
                                     ("supporting_reference_id", ["duplicate", "duplicate"]),
                                     ("confidence", True)):
                    bad = deepcopy(raw)
                    bad["assessments"][axis["id"]][field] = value
                    self.assertFalse(validator.is_valid(bad))
                bad = deepcopy(raw)
                del bad["assessments"][axis["id"]]
                self.assertFalse(validator.is_valid(bad))
                bad = deepcopy(raw)
                bad["assessments"]["invented_axis"] = self.row
                self.assertFalse(validator.is_valid(bad))
                self.assertFalse(validator.is_valid({"assessments": [self.row, self.row]}))

    def test_invalid_references_retry_once_with_images_and_cache_only_success(self):
        bad = {"assessments": {"casual_formal": {**self.row, "supporting_reference_id": "invented"}}}
        good = {"assessments": {"casual_formal": self.row}}
        with TemporaryDirectory() as directory, patch.dict("os.environ", {
                "FASHION_CACHE_ENABLED": "1", "FASHION_CACHE_PATH": str(Path(directory) / "cache.sqlite3")}), \
                patch.object(refs, "call_text_json_with_error", side_effect=[(bad, None), (good, None)]) as call:
            first = refs.calibrate_query("formal", self.extraction, client=object(), model="test")
            second = refs.calibrate_query("formal", self.extraction, client=object(), model="test")
        self.assertEqual(first, second)
        self.assertEqual(first["style_axis_targets"][0]["source"], "image_reference")
        self.assertEqual(call.call_count, 2)
        original, retry = [entry.kwargs for entry in call.call_args_list]
        self.assertEqual(original["user_content"], retry["user_content"][:-1])
        self.assertEqual(original["json_schema"], retry["json_schema"])
        self.assertEqual(original["user_prompt"], retry["user_prompt"])
        self.assertIn("calibration_reference_unknown", retry["user_content"][-1]["text"])
        self.assertNotIn("invented", retry["user_content"][-1]["text"])

    def test_exhausted_retry_returns_explicit_fallback_and_is_not_cached(self):
        bad = {"assessments": {"casual_formal": {**self.row, "supporting_reference_id": "SECRET"}}}
        with TemporaryDirectory() as directory, patch.dict("os.environ", {
                "FASHION_CACHE_ENABLED": "1", "FASHION_CACHE_PATH": str(Path(directory) / "cache.sqlite3")}), \
                patch.object(refs, "call_text_json_with_error", return_value=(bad, None)) as call, \
                self.assertLogs(refs.__name__, level="WARNING") as logs:
            for _ in range(2):
                result = refs.calibrate_query("formal", self.extraction, client=object(), model="test")
                self.assertEqual(result["style_axis_targets"], self.extraction["style_axis_targets"])
                self.assertEqual(result["style_calibration"]["status"], "fallback")
                self.assertEqual(result["style_calibration"]["effective_mode"], "legacy")
                self.assertEqual(result["style_calibration"]["reason"], "calibration_reference_unknown")
                self.assertTrue(result["style_calibration"]["warning"])
                self.assertEqual(result["style_calibration"]["assessments"], [])
            self.assertEqual(call.call_count, 4)
        self.assertNotIn("SECRET", str(result) + str(logs.output))
        self.assertIn("calibration_reference_unknown", str(logs.output))
        self.assertNotIn("style_calibration", self.extraction)

    def test_foreign_reference_ids_and_malformed_citations_are_rejected(self):
        valid_id = self.row["supporting_reference_id"]
        other_id = next(ref["id"] for axis in refs.reference_manifest()["axes"]
                        for ref in axis["references"]
                        if ref["id"] not in {entry["id"] for entry in self.axis["references"]})
        for ref_id in (None, [], {}, [valid_id, valid_id], other_id, valid_id + ".jpg"):
            with self.subTest(ref_id=ref_id):
                with self.assertRaises(refs.CalibrationValidationError) as caught:
                    refs.normalize_assessments({"assessments": {"casual_formal": {
                        **self.row, "supporting_reference_id": ref_id}}}, [self.axis])
                self.assertEqual(caught.exception.code, "calibration_reference_unknown")
                self.assertEqual(failure_details(caught.exception)["param"], "casual_formal")

    def test_malformed_rows_produce_validation_errors(self):
        for raw in ([], {"assessments": [None]}, {"assessments": {"casual_formal": None}},
                    {"assessments": {}}, {"assessments": {"design_expression": self.row}}):
            with self.subTest(raw=raw), self.assertRaises(refs.CalibrationValidationError):
                refs.normalize_assessments(raw, [self.axis])

    def test_provider_and_asset_failures_return_explicit_legacy_fallback(self):
        cases = [patch.object(refs, "call_text_json_with_error", return_value=(None, {"type": "APIError"})),
                 patch.object(refs, "call_text_json_with_error", side_effect=TimeoutError("SECRET")),
                 patch.object(refs, "reference_manifest", side_effect=FileNotFoundError("SECRET")),
                 patch.object(refs, "cached", side_effect=OSError("SECRET"))]
        with patch.dict("os.environ", {"FASHION_CACHE_ENABLED": "0"}):
            for failure in cases:
                with failure, self.assertLogs(refs.__name__, level="WARNING") as logs:
                    result = refs.calibrate_query("formal", self.extraction, client=object(), model="test")
                self.assertEqual(result["style_axis_targets"], self.extraction["style_axis_targets"])
                self.assertEqual(result["style_calibration"]["status"], "fallback")
                self.assertNotIn("SECRET", str(result) + str(logs.output))

    def test_calibration_call_has_no_nested_retries(self):
        client = MagicMock()
        bounded = client.with_options.return_value
        bounded.responses.create.return_value = SimpleNamespace(output_text="invalid")
        raw, error = llm.call_text_json_with_error(
            client=client, model="test", user_prompt="formal", system_prompt_text="calibrate",
            request_timeout=30, retry_delays=(), retry_invalid_json=False)
        self.assertIsNone(raw)
        self.assertEqual(error["stage"], "json_parse")
        client.with_options.assert_called_once_with(timeout=30, max_retries=0)
        bounded.responses.create.assert_called_once()

    def test_reference_revision_is_part_of_successful_result_cache(self):
        with TemporaryDirectory() as directory, patch.dict("os.environ", {
                "FASHION_CACHE_ENABLED": "1", "FASHION_CACHE_PATH": str(Path(directory) / "cache.sqlite3")}), \
                patch.object(refs, "call_text_json_with_error", return_value=({"assessments": {"casual_formal": self.row}}, None)) as call:
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
