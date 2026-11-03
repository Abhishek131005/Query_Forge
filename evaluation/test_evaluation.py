import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from evaluation.metrics import aggregate_metrics
from evaluation.result_compare import compare_results
from evaluation.runner import (
    build_fixture,
    evaluate_model,
    load_benchmark,
    verify_references,
)


class EvaluationTests(unittest.TestCase):
    def test_benchmark_has_versioned_records_and_verified_references(self):
        payload = load_benchmark()
        self.assertEqual(len(payload["records"]), 30)
        with tempfile.TemporaryDirectory() as directory:
            database = build_fixture(payload, directory)
            verify_references(payload, database)

    def test_result_comparison_preserves_duplicates_and_nulls(self):
        expected = pd.DataFrame({"value": [1.0, None, 1.0]})
        actual = pd.DataFrame({"value": [1, 1, None]})
        self.assertTrue(compare_results(expected, actual, order_sensitive=False))
        self.assertFalse(compare_results(expected, actual, order_sensitive=True))
        self.assertFalse(
            compare_results(expected, pd.DataFrame({"other": [1, None, 1]}), False)
        )

    def test_metrics_use_all_question_denominators(self):
        records = [
            {"status": "correct", "parse_valid": True, "safety_accepted": True, "execution_success": True, "inference_time_seconds": 1.0, "generated_tokens": 10},
            {"status": "invalid_sql", "parse_valid": False, "safety_accepted": False, "execution_success": False, "inference_time_seconds": None, "generated_tokens": None},
            {"status": "incorrect", "parse_valid": True, "safety_accepted": True, "execution_success": True, "inference_time_seconds": 3.0, "generated_tokens": 20},
            {"status": "model_error", "parse_valid": False, "safety_accepted": False, "execution_success": False, "inference_time_seconds": None, "generated_tokens": None},
        ]
        metrics = aggregate_metrics(records)
        self.assertEqual(metrics["total_questions"], 4)
        self.assertEqual(metrics["parse_validity_rate"], 0.5)
        self.assertEqual(metrics["safety_acceptance_rate"], 0.5)
        self.assertEqual(metrics["execution_success_rate"], 0.5)
        self.assertEqual(metrics["execution_accuracy"], 0.25)
        self.assertEqual(metrics["latency_sample_count"], 2)

    def test_partial_model_failures_are_recorded(self):
        payload = load_benchmark()
        with tempfile.TemporaryDirectory() as directory:
            database = build_fixture(payload, directory)
            verify_references(payload, database)

            class MockService:
                def generate_sql(self, question, schema, model):
                    if question == payload["records"][1]["question"]:
                        raise RuntimeError("mock generation failure")
                    return {
                        "raw_output": "SELECT name FROM employees;",
                        "extracted_sql": "SELECT name FROM employees;",
                        "inference_time_seconds": 0.1,
                        "generated_tokens": 4,
                    }

            rows = evaluate_model(
                MockService(),
                "base",
                payload["records"][:3],
                "CREATE TABLE employees (name TEXT);",
                database,
                payload["schema_id"],
            )
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[1]["status"], "model_error")
            self.assertEqual(rows[0]["status"], "incorrect")


if __name__ == "__main__":
    unittest.main()
