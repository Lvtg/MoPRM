import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moprm.expert_rewrite import parse_mapping, rewrite_record_experts  # noqa: E402
from moprm.schema import ProblemRecord  # noqa: E402


class ExpertRewriteTest(unittest.TestCase):
    def test_parse_mapping(self) -> None:
        self.assertEqual(
            parse_mapping(["general_judge=openai_general_judge"]),
            {"general_judge": "openai_general_judge"},
        )
        with self.assertRaises(ValueError):
            parse_mapping(["broken"])

    def test_rewrite_record_experts(self) -> None:
        record = ProblemRecord.from_dict(
            {
                "problem_id": "p1",
                "domain": "math",
                "problem": "1+1",
                "answer": "2",
                "candidates": [
                    {
                        "candidate_id": "c1",
                        "solution": "Final answer: 2",
                        "expert_scores": {
                            "math_prm": 0.9,
                            "general_judge": 0.8,
                            "open_math_prm": 0.7,
                        },
                        "normalized_scores": {
                            "math_prm": 1.0,
                            "general_judge": 0.5,
                            "open_math_prm": 0.0,
                        },
                    }
                ],
            }
        )
        rewritten = rewrite_record_experts(
            record,
            rename={"general_judge": "openai_general_judge"},
            drop={"math_prm"},
        )
        candidate = rewritten.candidates[0]
        self.assertNotIn("math_prm", candidate.expert_scores)
        self.assertEqual(candidate.expert_scores["openai_general_judge"], 0.8)
        self.assertEqual(candidate.normalized_scores["open_math_prm"], 0.0)
        self.assertEqual(rewritten.metadata["expert_pool_rewrite"]["drop"], ["math_prm"])


if __name__ == "__main__":
    unittest.main()
