import json
import tempfile
import unittest
from pathlib import Path

from evaluation.dashboard import filter_rows, list_run_files, load_run


class DashboardTests(unittest.TestCase):
    def test_missing_and_latest_saved_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            path.write_text(
                json.dumps({"metadata": {"run_id": "r1"}, "models": {}}),
                encoding="utf-8",
            )
            self.assertEqual(list_run_files(directory), [path])
            self.assertEqual(load_run(path)["metadata"]["run_id"], "r1")

    def test_filters_are_applied_to_saved_records(self):
        rows = [
            {"difficulty": "easy", "tags": "filter,sorting", "status": "correct"},
            {"difficulty": "hard", "tags": "join", "status": "incorrect"},
        ]
        self.assertEqual(len(filter_rows(rows, difficulty="easy")), 1)
        self.assertEqual(len(filter_rows(rows, tag="join")), 1)
        self.assertEqual(len(filter_rows(rows, status="correct")), 1)


if __name__ == "__main__":
    unittest.main()
