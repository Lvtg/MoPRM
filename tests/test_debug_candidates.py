import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moprm.candidates.debug_generator import generate_debug_candidates
from moprm.labeling import label_candidate_correctness
from moprm.schema import ProblemRecord


class DebugCandidateTest(unittest.TestCase):
    def test_generate_debug_candidates(self) -> None:
        record = ProblemRecord.from_dict(
            {
                "problem_id": "p1",
                "domain": "logic",
                "problem": "Choose A.",
                "answer": "A",
                "candidates": [],
            }
        )
        generated = generate_debug_candidates(record, n=4)
        self.assertEqual(len(generated.candidates), 4)
        self.assertTrue(generated.candidates[0].metadata["uses_gold_answer"])
        labeled = label_candidate_correctness(generated.to_dict())
        self.assertTrue(labeled["candidates"][0]["is_correct"])
        self.assertFalse(labeled["candidates"][2]["is_correct"])


if __name__ == "__main__":
    unittest.main()

