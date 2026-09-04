import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moprm.schema import ProblemRecord  # noqa: E402
from moprm.trained_gate import (  # noqa: E402
    expert_success_targets,
    fit_feature_config,
    stratified_folds,
    train_linear_gate,
    transform_records,
)


def make_two_candidate_record(
    problem_id: str,
    *,
    domain: str,
    problem: str,
    first_correct: bool,
    math_prefers_first: bool,
    logic_prefers_first: bool,
) -> ProblemRecord:
    math_scores = (0.9, 0.1) if math_prefers_first else (0.1, 0.9)
    logic_scores = (0.9, 0.1) if logic_prefers_first else (0.1, 0.9)
    return ProblemRecord.from_dict(
        {
            "problem_id": problem_id,
            "domain": domain,
            "problem": problem,
            "answer": "A",
            "metadata": {
                "source": f"synthetic/{domain}",
                "task_name": domain,
            },
            "candidates": [
                {
                    "candidate_id": f"{problem_id}_0",
                    "solution": "First candidate.",
                    "is_correct": first_correct,
                    "expert_scores": {
                        "math_expert": math_scores[0],
                        "logic_expert": logic_scores[0],
                    },
                },
                {
                    "candidate_id": f"{problem_id}_1",
                    "solution": "Second candidate.",
                    "is_correct": not first_correct,
                    "expert_scores": {
                        "math_expert": math_scores[1],
                        "logic_expert": logic_scores[1],
                    },
                },
            ],
        }
    )


class TrainedGateTest(unittest.TestCase):
    def test_feature_config_and_transform_are_stable(self) -> None:
        records = [
            make_two_candidate_record(
                "math_0",
                domain="math",
                problem="Compute 2 + 2.",
                first_correct=True,
                math_prefers_first=True,
                logic_prefers_first=False,
            ),
            make_two_candidate_record(
                "logic_0",
                domain="logic",
                problem="If A is before B, which item is first?",
                first_correct=True,
                math_prefers_first=False,
                logic_prefers_first=True,
            ),
        ]

        config = fit_feature_config(records, hash_dim=16)
        first = transform_records(records, config)
        second = transform_records(records, config)

        self.assertEqual(first.shape, (2, config.dim))
        self.assertEqual(config.hash_dim, 16)
        self.assertTrue((first == second).all())

    def test_expert_success_targets_follow_single_expert_selection(self) -> None:
        record = make_two_candidate_record(
            "mixed_0",
            domain="math",
            problem="Compute 1 + 1.",
            first_correct=True,
            math_prefers_first=True,
            logic_prefers_first=False,
        )

        targets = expert_success_targets(
            [record],
            ["math_expert", "logic_expert"],
            normalization="rank",
        )

        self.assertEqual(targets.tolist(), [[1.0, 0.0]])

    def test_stratified_folds_cover_each_index_once(self) -> None:
        records = [
            make_two_candidate_record(
                f"math_{index}",
                domain="math",
                problem=f"Compute {index} + 1.",
                first_correct=True,
                math_prefers_first=True,
                logic_prefers_first=False,
            )
            for index in range(6)
        ]

        folds = stratified_folds(records, folds=3, seed=7)
        flattened = sorted(index for fold in folds for index in fold)

        self.assertEqual(flattened, list(range(6)))
        self.assertTrue(all(fold for fold in folds))

    def test_linear_gate_learns_domain_routing_signal(self) -> None:
        records: list[ProblemRecord] = []
        for index in range(6):
            records.append(
                make_two_candidate_record(
                    f"math_{index}",
                    domain="math",
                    problem=f"Compute {index} + {index}.",
                    first_correct=True,
                    math_prefers_first=True,
                    logic_prefers_first=False,
                )
            )
            records.append(
                make_two_candidate_record(
                    f"logic_{index}",
                    domain="logic",
                    problem=f"If object {index} is left of B, infer the order.",
                    first_correct=True,
                    math_prefers_first=False,
                    logic_prefers_first=True,
                )
            )

        model = train_linear_gate(
            records,
            ["math_expert", "logic_expert"],
            hash_dim=16,
            epochs=200,
            lr=0.08,
            l2=0.001,
            seed=11,
        )
        math_weights, logic_weights = model.predict_weight_dicts([records[0], records[1]])

        self.assertAlmostEqual(sum(math_weights.values()), 1.0)
        self.assertAlmostEqual(sum(logic_weights.values()), 1.0)
        self.assertGreater(math_weights["math_expert"], math_weights["logic_expert"])
        self.assertGreater(logic_weights["logic_expert"], logic_weights["math_expert"])


if __name__ == "__main__":
    unittest.main()
