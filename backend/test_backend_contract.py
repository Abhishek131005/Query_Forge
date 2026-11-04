import unittest
from unittest.mock import patch

from backend import main
from backend.model_service import SQLModelService


class FakeService:
    def generate_sql(self, question, schema, model_type):
        return {
            "sql": "SELECT * FROM employees;",
            "raw_output": "```sql\nSELECT * FROM employees;\n```",
            "cleaned_output": "SELECT * FROM employees;",
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

    def test_extract_sql_uses_last_valid_statement_after_prompt_echo(self):
        raw_output = (
            'Schema: CREATE TABLE employees (name TEXT);\n'
            'SQL: SELECT name FROM employees;\n'
            'Question: List names\n'
            'SELECT name FROM queryforge_sample_employees;'
        )
        self.assertEqual(
            SQLModelService.extract_sql(raw_output),
            'SELECT name FROM queryforge_sample_employees;',
        )

    def test_schema_table_names_are_preserved_for_uploaded_tables(self):
        schema = '''CREATE TABLE "queryforge_sample_employees" (
            "name" TEXT,
            "department" TEXT,
            "salary" INTEGER
        );'''
        self.assertEqual(
            SQLModelService.schema_table_names(schema),
            ["queryforge_sample_employees"],
        )
        repaired = SQLModelService.repair_table_names(
            "SELECT name, AVG(salary) FROM employees GROUP BY department",
            schema,
        )
        self.assertIn("queryforge_sample_employees", repaired)
        self.assertNotIn("FROM employees", repaired)

    def test_average_by_department_uses_grouping_column(self):
        schema = '''CREATE TABLE "queryforge_sample_employees" (
            "name" TEXT,
            "department" TEXT,
            "salary" INTEGER
        );'''
        repaired = SQLModelService.repair_common_intent(
            "Find average salary of each department",
            "SELECT name, AVG(salary) FROM queryforge_sample_employees GROUP BY department",
            schema,
        )
        self.assertEqual(
            repaired,
            "SELECT department, AVG(salary) FROM queryforge_sample_employees GROUP BY department",
        )

    def test_table_repair_maps_known_suffixes_in_join_schema(self):
        schema = '''CREATE TABLE "queryforge_sample_employees" (
            "name" TEXT,
            "department" TEXT,
            "salary" INTEGER
        );
        CREATE TABLE "queryforge_sample_departments" (
            "department" TEXT,
            "manager" TEXT
        );'''
        repaired = SQLModelService.repair_table_names(
            "SELECT d.manager, AVG(e.salary) FROM employees AS e "
            "JOIN departments AS d ON e.department = d.department "
            "GROUP BY d.manager",
            schema,
        )
        self.assertIn("queryforge_sample_employees", repaired)
        self.assertIn("queryforge_sample_departments", repaired)
        self.assertNotIn("FROM employees", repaired)
        self.assertNotIn("JOIN departments", repaired)

    def test_average_salary_with_manager_uses_uploaded_join_tables(self):
        schema = '''CREATE TABLE "queryforge_sample_employees" (
            "name" TEXT,
            "department" TEXT,
            "salary" INTEGER
        );
        CREATE TABLE "queryforge_sample_departments" (
            "department" TEXT,
            "manager" TEXT
        );'''
        repaired = SQLModelService.repair_common_intent(
            "What is the average salary for each department, including the department manager?",
            "SELECT department, COUNT(name) FROM employees GROUP BY department, manager",
            schema,
        )
        self.assertEqual(
            repaired,
            "SELECT d.department, d.manager, AVG(e.salary) AS average_salary "
            "FROM queryforge_sample_employees AS e "
            "JOIN queryforge_sample_departments AS d "
            "ON e.department = d.department "
            "GROUP BY d.department, d.manager",
        )

    def test_fallback_sql_handles_base_model_without_select_output(self):
        schema = '''CREATE TABLE "queryforge_sample_employees" (
            "name" TEXT,
            "department" TEXT,
            "salary" INTEGER
        );
        CREATE TABLE "queryforge_sample_departments" (
            "department" TEXT,
            "manager" TEXT
        );'''
        self.assertEqual(
            SQLModelService.fallback_sql(
                "What is the average salary for each department, including the department manager?",
                schema,
            ),
            "SELECT d.department, d.manager, AVG(e.salary) AS average_salary "
            "FROM queryforge_sample_employees AS e "
            "JOIN queryforge_sample_departments AS d "
            "ON e.department = d.department "
            "GROUP BY d.department, d.manager",
        )

    def test_highest_paid_manager_query_uses_uploaded_join_tables(self):
        schema = '''CREATE TABLE "queryforge_sample_employees" (
            "name" TEXT,
            "department" TEXT,
            "salary" INTEGER
        );
        CREATE TABLE "queryforge_sample_departments" (
            "department" TEXT,
            "manager" TEXT
        );'''
        repaired = SQLModelService.repair_common_intent(
            "List employees earning more than 100000 with their department manager, "
            "ordered by salary from highest to lowest.",
            "SELECT e.name",
            schema,
        )
        self.assertEqual(
            repaired,
            "SELECT e.name, e.salary, e.department, d.manager "
            "FROM queryforge_sample_employees AS e "
            "JOIN queryforge_sample_departments AS d "
            "ON e.department = d.department "
            "WHERE e.salary > 100000 "
            "ORDER BY e.salary DESC",
        )

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
                self.assertEqual(payload["cleaned_output"], "SELECT * FROM employees;")
                self.assertEqual(payload["model"], model)
                self.assertGreater(payload["generated_tokens"], 0)
                self.assertGreaterEqual(payload["inference_time_seconds"], 0)
                self.assertGreaterEqual(payload["queue_wait_seconds"], 0)
                self.assertEqual(payload["generation_config"]["max_new_tokens"], 160)
                self.assertFalse(payload["postprocessed"])


if __name__ == "__main__":
    unittest.main()
