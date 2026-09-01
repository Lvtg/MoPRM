import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moprm.scoring.skywork_math_prm import (  # noqa: E402
    OPEN_MATH_PRM_EXPERT,
    aggregate_step_rewards,
    prepare_skywork_input,
    score_record_with_skywork_math_prm,
    split_solution_steps,
)
from moprm.schema import ProblemRecord  # noqa: E402


class FakeTokenizer:
    bos_token = "<bos>"
    pad_token_id = 0

    def encode(self, text):
        return [ord(char) % 251 + 1 for char in text]


class FakeScorer:
    class Config:
        expert_name = OPEN_MATH_PRM_EXPERT

    config = Config()

    def score_text(self, problem, solution):
        return 0.75, {"provider": "fake", "num_reward_points": 2}


class SkyworkMathPRMTest(unittest.TestCase):
    def test_split_solution_steps_prefers_non_empty_lines(self) -> None:
        steps = split_solution_steps("Step 1\n\nStep 2\nFinal answer: 3")
        self.assertEqual(steps, ["Step 1", "Step 2", "Final answer: 3"])

    def test_aggregate_step_rewards(self) -> None:
        self.assertAlmostEqual(aggregate_step_rewards([0.2, 0.8], "mean"), 0.5)
        self.assertAlmostEqual(aggregate_step_rewards([0.2, 0.8], "min"), 0.2)
        self.assertAlmostEqual(aggregate_step_rewards([0.2, 0.8], "last"), 0.8)

    def test_prepare_skywork_input_marks_step_tokens(self) -> None:
        input_ids, steps, reward_flags = prepare_skywork_input(
            "Problem",
            "First\nSecond",
            FakeTokenizer(),
            step_token="\n",
        )
        self.assertEqual(len(input_ids), len(reward_flags))
        self.assertEqual(sum(reward_flags), 2)
        self.assertEqual(len(steps), 2)

    def test_score_record_attaches_open_math_prm(self) -> None:
        record = ProblemRecord.from_dict(
            {
                "problem_id": "p1",
                "domain": "math",
                "problem": "1+1",
                "answer": "2",
                "candidates": [
                    {
                        "candidate_id": "c1",
                        "solution": "1+1=2\nFinal answer: 2",
                        "final_answer": "2",
                        "is_correct": True,
                    }
                ],
            }
        )
        scored = score_record_with_skywork_math_prm(record, FakeScorer())
        candidate = scored.candidates[0]
        self.assertEqual(candidate.expert_scores[OPEN_MATH_PRM_EXPERT], 0.75)
        self.assertIn(OPEN_MATH_PRM_EXPERT, scored.metadata["open_source_experts"])


if __name__ == "__main__":
    unittest.main()
