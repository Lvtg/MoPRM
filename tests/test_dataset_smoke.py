import tempfile
import unittest
from pathlib import Path

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moprm.datasets.public_sources import PublicSubsetConfig, make_problem
from moprm.io import load_jsonl, write_jsonl


class DatasetSmokeTest(unittest.TestCase):
    def test_jsonl_roundtrip(self) -> None:
        record = make_problem(
            problem_id="example_0001",
            domain="math",
            problem="1 + 1",
            answer="2",
            source="unit-test",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "records.jsonl"
            write_jsonl(path, [record])
            loaded = load_jsonl(path)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].problem_id, "example_0001")

    def test_public_subset_config_defaults_are_small(self) -> None:
        config = PublicSubsetConfig()
        self.assertLessEqual(config.math500_limit + config.gsm8k_limit, 200)


if __name__ == "__main__":
    unittest.main()

