import io
import unittest

from frontend.database import clean_identifier, unique_names


class DatabaseTests(unittest.TestCase):
    def test_identifiers_are_safe_and_unique(self):
        self.assertEqual(clean_identifier("Sales Data!.csv"), "sales_data_csv")
        self.assertEqual(unique_names(["Name", "Name", "salary"]), ["name", "name_2", "salary"])


if __name__ == "__main__":
    unittest.main()
