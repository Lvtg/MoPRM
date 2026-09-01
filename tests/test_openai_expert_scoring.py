import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moprm.scoring.openai_experts import (  # noqa: E402
    OPENAI_EXPERTS,
    OpenAIExpertScoringConfig,
    build_scoring_prompt,
    parse_score_json,
    score_record_with_openai,
)
from moprm.schema import ProblemRecord  # noqa: E402


class FakeScoringClient:
    def create_response(self, **kwargs):
        return {
            "id": "resp_score_test",
            "output_text": (
                '{"math_prm": 0.9, "logic_judge": 0.2, '
                '"general_judge": 0.8, "reflective_judge": 0.4}'
            ),
            "usage": {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
        }


class OpenAIExpertScoringTest(unittest.TestCase):
    def test_parse_score_json_clamps_and_accepts_nested_scores(self) -> None:
        scores = parse_score_json(
            '```json\n{"scores": {"math_prm": 95, "logic_judge": 0.2, '
            '"general_judge": 0.8, "reflective_judge": 0.4}}\n```'
        )
        self.assertEqual(set(scores), set(OPENAI_EXPERTS))
        self.assertEqual(scores["math_prm"], 0.95)

    def test_scoring_prompt_does_not_include_gold_answer(self) -> None:
        record = ProblemRecord.from_dict(
            {
                "problem_id": "p1",
                "domain": "math",
                "problem": "What is 20 + 22?",
                "answer": "SECRET_GOLD",
                "candidates": [
                    {
                        "candidate_id": "c1",
                        "solution": "20+22=42. Final answer: 42",
                        "final_answer": "42",
                    }
                ],
            }
        )
        prompt = build_scoring_prompt(record, record.candidates[0])
        self.assertIn("What is 20 + 22?", prompt)
        self.assertIn("Final answer: 42", prompt)
        self.assertIn("explicit final answer marker: yes", prompt)
        self.assertNotIn("SECRET_GOLD", prompt)

    def test_score_record_with_openai_attaches_expert_scores(self) -> None:
        record = ProblemRecord.from_dict(
            {
                "problem_id": "p1",
                "domain": "math",
                "problem": "What is 20 + 22?",
                "answer": "42",
                "candidates": [
                    {
                        "candidate_id": "c1",
                        "solution": "20+22=42. Final answer: 42",
                        "final_answer": "42",
                        "is_correct": True,
                    }
                ],
            }
        )
        scored = score_record_with_openai(
            record,
            FakeScoringClient(),
            OpenAIExpertScoringConfig(model="gpt-test"),
        )
        candidate = scored.candidates[0]
        self.assertEqual(set(candidate.expert_scores), set(OPENAI_EXPERTS))
        self.assertEqual(candidate.metadata["openai_expert_scoring"]["usage"]["total_tokens"], 20)
        self.assertFalse(scored.metadata["expert_scoring"]["uses_gold_answer"])


if __name__ == "__main__":
    unittest.main()
