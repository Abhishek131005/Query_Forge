import sqlite3
import tempfile
import unittest
from pathlib import Path

from frontend.sql_executor import execute_sql, validate_sql


class SqlExecutorTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / 'fixture.sqlite3'
        with sqlite3.connect(self.db_path) as connection:
            connection.execute('CREATE TABLE employees (id INTEGER, name TEXT, salary INTEGER)')
            connection.executemany(
                'INSERT INTO employees VALUES (?, ?, ?)',
                [(1, 'Asha', 120000), (2, 'Rahul', 75000)],
            )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_read_only_analytical_queries(self):
        queries = [
            'SELECT name FROM employees WHERE salary > 100000',
            'WITH high_earners AS (SELECT * FROM employees WHERE salary > 100000) SELECT name FROM high_earners',
            'SELECT name FROM employees WHERE id = 1 UNION SELECT name FROM employees WHERE id = 2',
        ]
        for query in queries:
            frame, _, truncated = execute_sql(query, self.db_path)
            self.assertFalse(frame.empty)
            self.assertFalse(truncated)

    def test_mutations_and_commands_are_rejected(self):
        queries = [
            'ATTACH DATABASE ":memory:" AS other',
            'PRAGMA user_version',
            'DROP TABLE employees',
            'DELETE FROM employees',
            'UPDATE employees SET salary = 0',
            'SELECT 1; SELECT 2',
        ]
        for query in queries:
            with self.subTest(query=query):
                with self.assertRaises(ValueError):
                    validate_sql(query)

    def test_unsafe_functions_are_rejected_by_authorizer(self):
        with self.assertRaises(ValueError):
            execute_sql('SELECT load_extension("anything") FROM employees', self.db_path)

    def test_results_are_capped_and_marked_truncated(self):
        with sqlite3.connect(self.db_path) as connection:
            connection.executemany(
                'INSERT INTO employees VALUES (?, ?, ?)',
                [(index, f'Employee {index}', index) for index in range(3, 505)],
            )

        frame, _, truncated = execute_sql('SELECT * FROM employees', self.db_path)
        self.assertEqual(len(frame), 500)
        self.assertTrue(truncated)


if __name__ == '__main__':
    unittest.main()
