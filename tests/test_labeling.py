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

    def test_overwrite_refreshes_final_answer_extraction(self) -> None:
        record = {
            "problem_id": "logic_example",
            "domain": "logic",
            "problem": "Choose E.",
            "answer": "E",
            "candidates": [
                {
                    "candidate_id": "c0",
                    "solution": (
                        "The answer is (E) because of the ordering.\n\n"
                        "Final answer: E"
                    ),
                    "final_answer": "(E) because of the ordering.",
                    "is_correct": False,
                }
            ],
        }
        labeled = label_candidate_correctness(record, overwrite=True)
        self.assertEqual(labeled["candidates"][0]["final_answer"], "E")
        self.assertTrue(labeled["candidates"][0]["is_correct"])

    def test_missing_explicit_final_answer_is_not_correct(self) -> None:
        record = {
            "problem_id": "logic_example",
            "domain": "logic",
            "problem": "Choose C.",
            "answer": "C",
            "candidates": [
                {
                    "candidate_id": "c0",
                    "solution": "I reason toward C but stop before giving the required final line.",
                }
            ],
        }
        labeled = label_candidate_correctness(record, overwrite=True)
        self.assertFalse(labeled["candidates"][0]["is_correct"])
        self.assertFalse(
            labeled["candidates"][0]["metadata"]["has_explicit_final_answer"]
        )


if __name__ == "__main__":
    unittest.main()
