import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moprm.evaluate import default_baselines  # noqa: E402
from moprm.routing.openai_gate import (  # noqa: E402
    OpenAIGateConfig,
    build_gate_prompt,
    parse_gate_weights,
    route_record_with_openai,
)
from moprm.schema import ProblemRecord  # noqa: E402


class FakeGateClient:
    def create_response(self, **kwargs):
        return {
            "id": "resp_gate_test",
            "output_text": (
                '{"weights": {"math_prm": 3, "logic_judge": 1, '
                '"general_judge": 1, "reflective_judge": 0}}'
            ),
            "usage": {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10},
        }


class OpenAIGateTest(unittest.TestCase):
    def test_parse_gate_weights_normalizes(self) -> None:
        weights = parse_gate_weights(
            '{"weights": {"math_prm": 3, "logic_judge": 1}}',
            ["math_prm", "logic_judge", "general_judge"],
        )
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertGreater(weights["math_prm"], weights["logic_judge"])
        self.assertEqual(weights["general_judge"], 0.0)

    def test_gate_prompt_does_not_include_gold_answer(self) -> None:
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
                        "expert_scores": {
                            "math_prm": 0.9,
                            "logic_judge": 0.4,
                            "general_judge": 0.8,
                        },
                    }
                ],
            }
        )
        prompt = build_gate_prompt(record, record.expert_names())
        self.assertIn("What is 20 + 22?", prompt)
        self.assertNotIn("SECRET_GOLD", prompt)
        self.assertNotIn("20+22=42", prompt)

    def test_route_record_attaches_metadata_gate_used_by_eval(self) -> None:
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
                        "expert_scores": {
                            "math_prm": 0.9,
                            "logic_judge": 0.3,
                            "general_judge": 0.8,
                            "reflective_judge": 0.4,
                        },
                    }
                ],
            }
        )
        routed = route_record_with_openai(
            record,
            FakeGateClient(),
            OpenAIGateConfig(model="gpt-test"),
        )
        self.assertIn("openai_llm_gate", routed.metadata["gate_weights"])
        self.assertEqual(
            routed.metadata["gate_metadata"]["openai_llm_gate"]["usage"]["total_tokens"],
            10,
        )
        methods = {result.method for result in default_baselines([routed])}
        self.assertIn("metadata_gate:openai_llm_gate", methods)


if __name__ == "__main__":
    unittest.main()
