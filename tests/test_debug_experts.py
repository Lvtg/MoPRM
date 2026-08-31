import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moprm.candidates.debug_generator import generate_debug_candidates
from moprm.scoring.debug_experts import score_debug_record
from moprm.schema import ProblemRecord


class DebugExpertTest(unittest.TestCase):
    def test_score_debug_record(self) -> None:
        record = ProblemRecord.from_dict(
            {
                "problem_id": "p1",
                "domain": "math",
                "problem": "1+1",
                "answer": "2",
                "candidates": [],
            }
        )
        generated = generate_debug_candidates(record, n=4)
        scored = score_debug_record(generated)
        self.assertEqual(scored.expert_names(), ["general_judge", "math_prm", "reflective_judge"])
        self.assertIn("expert_scoring", scored.metadata)


if __name__ == "__main__":
    unittest.main()

