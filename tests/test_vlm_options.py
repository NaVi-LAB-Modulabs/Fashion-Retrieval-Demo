"""Check actual parser request options without calling the OpenAI API."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fashion_how_graphdb.vlm import call_text_json_with_error


class ParserOptionsTests(unittest.TestCase):
    def test_reasoning_models_omit_temperature_on_initial_and_json_retry(self):
        for model in ("gpt-5", "gpt-5-mini", "gpt-5.4-mini", "gpt-5.5", "o3", "o4-mini"):
            with self.subTest(model=model):
                client = MagicMock()
                client.responses.create.side_effect = [
                    SimpleNamespace(output_text="not json"),
                    SimpleNamespace(output_text='{"item_type_codes": []}'),
                ]
                data, error = call_text_json_with_error(
                    client=client, model=model, user_prompt="blue shirt", system_prompt_text="Parse query",
                    json_schema={"type": "object", "properties": {}},
                )
                self.assertIsNone(error)
                self.assertEqual(data, {"item_type_codes": []})
                self.assertEqual(client.responses.create.call_count, 2)
                for call in client.responses.create.call_args_list:
                    self.assertNotIn("temperature", call.kwargs)
                    self.assertEqual(call.kwargs["model"], model)
                    self.assertTrue(call.kwargs["text"]["format"]["strict"])

    def test_existing_non_reasoning_models_keep_sampling_option(self):
        for model in ("gpt-4.1-mini", "gpt-4.1", "gpt-4o-2024-08-06"):
            with self.subTest(model=model):
                client = MagicMock()
                client.responses.create.return_value = SimpleNamespace(output_text="{}")
                call_text_json_with_error(client=client, model=model, user_prompt="blue shirt", system_prompt_text="Parse query")
                self.assertEqual(client.responses.create.call_args.kwargs["temperature"], 0)


if __name__ == "__main__":
    unittest.main()
