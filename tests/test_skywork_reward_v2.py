import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moprm.scoring.skywork_reward_v2 import (  # noqa: E402
    OPEN_REASONING_RM_EXPERT,
    format_reward_conversation,
    score_record_with_skywork_reward_v2,
)
from moprm.schema import ProblemRecord  # noqa: E402


class FakeTokenizer:
    bos_token = "<bos>"

    def apply_chat_template(self, conversation, tokenize=False):
        assert tokenize is False
        return self.bos_token + "\n".join(
            f"{item['role']}: {item['content']}" for item in conversation
        )


class FakeScorer:
    class Config:
        expert_name = OPEN_REASONING_RM_EXPERT

    config = Config()

    def score_text(self, problem, solution):
        return 12.5, {
            "provider": "fake",
            "score_type": "sequence_reward_logit",
            "uses_gold_answer": False,
        }


class SkyworkRewardV2Test(unittest.TestCase):
    def test_format_reward_conversation_uses_problem_and_solution_only(self) -> None:
        formatted = format_reward_conversation(
            "Problem text",
            "Candidate solution",
            FakeTokenizer(),
        )
        self.assertFalse(formatted.startswith("<bos>"))
        self.assertIn("Problem text", formatted)
        self.assertIn("Candidate solution", formatted)

    def test_score_record_attaches_open_reasoning_rm(self) -> None:
        record = ProblemRecord.from_dict(
            {
                "problem_id": "p1",
                "domain": "logic",
                "problem": "A is before B. Which is first?",
                "answer": "A",
                "candidates": [
                    {
                        "candidate_id": "c1",
                        "solution": "A is first. Final answer: A",
                        "final_answer": "A",
                        "is_correct": True,
                    }
                ],
            }
        )
        scored = score_record_with_skywork_reward_v2(record, FakeScorer())
        candidate = scored.candidates[0]
        self.assertEqual(candidate.expert_scores[OPEN_REASONING_RM_EXPERT], 12.5)
        self.assertIn(OPEN_REASONING_RM_EXPERT, scored.metadata["open_source_experts"])
        self.assertFalse(candidate.metadata[OPEN_REASONING_RM_EXPERT]["uses_gold_answer"])


if __name__ == "__main__":
    unittest.main()
