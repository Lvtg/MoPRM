import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moprm.candidate_gate import (  # noqa: E402
    add_candidate_gate_scores_to_record,
    candidate_feature_matrix,
    evaluate_candidate_gate_records,
    fit_candidate_feature_config,
    train_candidate_gate,
)
from moprm.schema import ProblemRecord  # noqa: E402


def make_record(problem_id: str, *, correct_index: int) -> ProblemRecord:
    return ProblemRecord.from_dict(
        {
            "problem_id": problem_id,
            "domain": "math",
            "problem": "Choose the best candidate.",
            "answer": "2",
            "candidates": [
                {
                    "candidate_id": f"{problem_id}_c0",
                    "solution": "candidate zero",
                    "is_correct": correct_index == 0,
                    "expert_scores": {
                        "a_expert": 0.9 if correct_index == 0 else 0.1,
                        "b_expert": 0.8 if correct_index == 0 else 0.2,
                    },
                },
                {
                    "candidate_id": f"{problem_id}_c1",
                    "solution": "candidate one",
                    "is_correct": correct_index == 1,
                    "expert_scores": {
                        "a_expert": 0.9 if correct_index == 1 else 0.1,
                        "b_expert": 0.8 if correct_index == 1 else 0.2,
                    },
                },
            ],
        }
    )


class CandidateGateTest(unittest.TestCase):
    def test_candidate_feature_matrix_has_one_row_per_labeled_candidate(self) -> None:
        records = [make_record("p0", correct_index=0), make_record("p1", correct_index=1)]
        config = fit_candidate_feature_config(["a_expert", "b_expert"])

        features, keys, targets = candidate_feature_matrix(records, config)

        self.assertEqual(features.shape, (4, config.dim))
        self.assertEqual(len(keys), 4)
        self.assertEqual(targets.tolist(), [1.0, 0.0, 0.0, 1.0])

    def test_candidate_gate_learns_to_select_correct_candidate(self) -> None:
        records = [
            make_record(f"p{index}", correct_index=index % 2)
            for index in range(12)
        ]

        model = train_candidate_gate(
            records,
            ["a_expert", "b_expert"],
            epochs=200,
            lr=0.08,
            l2=0.001,
            seed=3,
        )
        scores = model.predict_candidate_scores([records[0]])[0]
        scored = add_candidate_gate_scores_to_record(
            records[0],
            gate_name="candidate_gate_v2_cv",
            scores=scores,
        )
        result = evaluate_candidate_gate_records(
            [scored],
            gate_name="candidate_gate_v2_cv",
        )

        self.assertEqual(result.correct, 1)
        self.assertEqual(result.total, 1)


if __name__ == "__main__":
    unittest.main()
