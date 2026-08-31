import unittest

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moprm.labeling import label_candidate_correctness


class LabelingTest(unittest.TestCase):
    def test_label_candidate_correctness(self) -> None:
        record = {
            "problem_id": "logic_example",
            "domain": "logic",
            "problem": "Choose C.",
            "answer": "C",
            "candidates": [
                {"candidate_id": "c0", "solution": "Final answer: C"},
                {"candidate_id": "c1", "solution": "Final answer: A"},
            ],
        }
        labeled = label_candidate_correctness(record)
        self.assertTrue(labeled["candidates"][0]["is_correct"])
        self.assertFalse(labeled["candidates"][1]["is_correct"])

    def test_does_not_overwrite_existing_label_by_default(self) -> None:
        record = {
            "problem_id": "math_example",
            "domain": "math",
            "problem": "1+1",
            "answer": "2",
            "candidates": [{"candidate_id": "c0", "final_answer": "2", "is_correct": False}],
        }
        labeled = label_candidate_correctness(record)
        self.assertFalse(labeled["candidates"][0]["is_correct"])
        overwritten = label_candidate_correctness(record, overwrite=True)
        self.assertTrue(overwritten["candidates"][0]["is_correct"])


if __name__ == "__main__":
    unittest.main()

