import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moprm.evaluate import default_baselines
from moprm.io import load_jsonl


class EvaluateTest(unittest.TestCase):
    def test_smoke_baselines(self) -> None:
        records = load_jsonl(Path("examples/smoke_moprm.jsonl"))
        results = {result.method: result for result in default_baselines(records)}
        self.assertEqual(results["oracle_gate"].accuracy, 1.0)
        self.assertGreaterEqual(results["domain_rule_gate"].accuracy, 0.75)
        self.assertLess(results["single:math_prm"].accuracy, results["oracle_gate"].accuracy)


if __name__ == "__main__":
    unittest.main()
