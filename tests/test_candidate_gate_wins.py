import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_candidate_gate_wins import compare_results, collect_cases  # noqa: E402
from moprm.evaluate import EvaluationResult  # noqa: E402
from moprm.schema import ProblemRecord  # noqa: E402


def make_record(problem_id: str, *, domain: str, correct: str) -> ProblemRecord:
    return ProblemRecord.from_dict(
        {
            "problem_id": problem_id,
            "domain": domain,
            "problem": "Choose the best candidate.",
            "answer": "A",
            "candidates": [
                {
                    "candidate_id": f"{problem_id}_a",
                    "solution": "candidate a",
                    "is_correct": correct == "a",
                    "expert_scores": {"expert": 1.0},
                },
                {
                    "candidate_id": f"{problem_id}_b",
                    "solution": "candidate b",
                    "is_correct": correct == "b",
                    "expert_scores": {"expert": 0.0},
                },
            ],
        }
    )


class CandidateGateWinsTest(unittest.TestCase):
    def test_compare_results_counts_wins_and_losses(self) -> None:
        records = [
            make_record("win", domain="math", correct="a"),
            make_record("loss", domain="math", correct="b"),
            make_record("both_correct", domain="logic", correct="a"),
            make_record("both_wrong", domain="logic", correct="a"),
        ]
        primary = EvaluationResult(
            method="candidate_gate:test",
            total=4,
            correct=2,
            accuracy=0.5,
            selections={
                "win": "win_a",
                "loss": "loss_a",
                "both_correct": "both_correct_a",
                "both_wrong": "both_wrong_b",
            },
        )
        baseline = EvaluationResult(
            method="baseline",
            total=4,
            correct=2,
            accuracy=0.5,
            selections={
                "win": "win_b",
                "loss": "loss_b",
                "both_correct": "both_correct_a",
                "both_wrong": "both_wrong_b",
            },
        )

        comparison = compare_results(
            records,
            primary=primary,
            baseline=baseline,
            group="all",
        )
        math_comparison = compare_results(
            records,
            primary=primary,
            baseline=baseline,
            group="math",
        )
        wins = collect_cases(
            records,
            primary=primary,
            baseline=baseline,
            normalization="rank",
            case_type="wins",
            limit=2,
        )

        self.assertEqual(comparison.wins, 1)
        self.assertEqual(comparison.losses, 1)
        self.assertEqual(comparison.both_correct, 1)
        self.assertEqual(comparison.both_wrong, 1)
        self.assertEqual(comparison.same_selection, 2)
        self.assertEqual(comparison.different_selection, 2)
        self.assertEqual(math_comparison.wins, 1)
        self.assertEqual(math_comparison.losses, 1)
        self.assertEqual([case["problem_id"] for case in wins], ["win"])


if __name__ == "__main__":
    unittest.main()
