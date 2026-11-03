import unittest
from unittest.mock import patch

from backend import main
from backend.model_service import SQLModelService


class FakeService:
    def generate_sql(self, question, schema, model_type):
        return {
            "sql": "SELECT * FROM employees;",
            "raw_output": "```sql\nSELECT * FROM employees;\n```",
            "extracted_sql": "SELECT * FROM employees;",
            "model": model_type,
            "status": "success",
            "inference_time_seconds": 0.125,
            "queue_wait_seconds": 0.001,
            "input_tokens": 12,
            "generated_tokens": 6,
            "device": "cpu",
            "generation_config": {
                "max_new_tokens": 160,
                "do_sample": False,
            },
            "prompt_version": "v1",
            "postprocessed": False,
        }


class BackendContractTests(unittest.TestCase):
    def test_extract_sql_does_not_rewrite_table_names(self):
        raw_output = "SELECT a.id FROM source_a AS a JOIN source_b AS b ON a.id = b.id;"

        extracted = SQLModelService.extract_sql(raw_output)

        self.assertEqual(extracted, raw_output)

    def test_extract_sql_is_separate_from_raw_output(self):
        raw_output = "```sql\nSELECT * FROM employees;\n```\n"

        extracted = SQLModelService.extract_sql(raw_output)

        self.assertEqual(extracted, "SELECT * FROM employees;")
        self.assertNotEqual(extracted, raw_output)

    def test_api_contract_preserves_sql_and_metadata_for_both_models(self):
        with patch.object(main, "service", FakeService()):
            for model in ("base", "finetuned"):
                request = main.SQLRequest(
                    question="List employees",
                    schema="CREATE TABLE employees (id INTEGER);",
                    model=model,
                )
                response = main.generate_sql(request)
                payload = response.model_dump()

                self.assertEqual(payload["sql"], payload["extracted_sql"])
                self.assertEqual(payload["raw_output"], "```sql\nSELECT * FROM employees;\n```")
                self.assertEqual(payload["model"], model)
                self.assertGreater(payload["generated_tokens"], 0)
                self.assertGreaterEqual(payload["inference_time_seconds"], 0)
                self.assertGreaterEqual(payload["queue_wait_seconds"], 0)
                self.assertEqual(payload["generation_config"]["max_new_tokens"], 160)
                self.assertFalse(payload["postprocessed"])


if __name__ == "__main__":
    unittest.main()
