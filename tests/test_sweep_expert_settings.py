import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from moprm.schema import ProblemRecord  # noqa: E402
from sweep_expert_settings import (  # noqa: E402
    candidate_upper_bound,
    filter_mixed_records,
    simplex_weight_grid,
)


def make_record(problem_id: str, correctness: list[bool]) -> ProblemRecord:
    return ProblemRecord.from_dict(
        {
            "problem_id": problem_id,
            "domain": "math",
            "problem": "Compute a small value.",
            "answer": "1",
            "candidates": [
                {
                    "candidate_id": f"{problem_id}_{index}",
                    "solution": "Final answer: 1",
                    "is_correct": is_correct,
                    "expert_scores": {"a": float(index), "b": float(len(correctness) - index)},
                }
                for index, is_correct in enumerate(correctness)
            ],
        }
    )


class SweepExpertSettingsTest(unittest.TestCase):
    def test_filter_mixed_records_keeps_only_partially_solved_sets(self) -> None:
        records = [
            make_record("all_wrong", [False, False]),
            make_record("mixed", [True, False]),
            make_record("all_correct", [True, True]),
        ]

        mixed = filter_mixed_records(records)

        self.assertEqual([record.problem_id for record in mixed], ["mixed"])
        self.assertEqual(candidate_upper_bound(records), (2, 3))
        self.assertEqual(candidate_upper_bound(mixed), (1, 1))

    def test_simplex_weight_grid_evenly_covers_two_experts(self) -> None:
        weights = list(simplex_weight_grid(["a", "b"], step=0.5))

        self.assertEqual(
            weights,
            [
                {"a": 0.0, "b": 1.0},
                {"a": 0.5, "b": 0.5},
                {"a": 1.0, "b": 0.0},
            ],
        )

    def test_simplex_weight_grid_rejects_uneven_step(self) -> None:
        with self.assertRaises(ValueError):
            list(simplex_weight_grid(["a", "b"], step=0.3))


if __name__ == "__main__":
    unittest.main()
