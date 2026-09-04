import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from expand_math_aggregation_experts import expand_record  # noqa: E402
from moprm.schema import ProblemRecord  # noqa: E402


class ExpandMathAggregationExpertsTest(unittest.TestCase):
    def test_expand_record_adds_pseudo_experts_and_remaps_gate_weight(self) -> None:
        record = ProblemRecord.from_dict(
            {
                "problem_id": "p0",
                "domain": "math",
                "problem": "Compute 1 + 1.",
                "answer": "2",
                "metadata": {
                    "gate_weights": {
                        "openai_llm_gate": {
                            "open_math_prm": 0.6,
                            "open_reasoning_rm": 0.4,
                        }
                    }
                },
                "candidates": [
                    {
                        "candidate_id": "p0_c0",
                        "solution": "1 + 1 = 2",
                        "is_correct": True,
                        "expert_scores": {
                            "open_math_prm": 0.6,
                            "open_reasoning_rm": 0.1,
                        },
                        "metadata": {
                            "open_math_prm": {
                                "step_rewards": [0.2, 1.0],
                            }
                        },
                    }
                ],
            }
        )

        expanded = expand_record(record, aggregations=["mean", "min"])
        scores = expanded.candidates[0].expert_scores
        gate_weights = expanded.metadata["gate_weights"]["openai_llm_gate"]

        self.assertNotIn("open_math_prm", scores)
        self.assertEqual(scores["open_math_prm_mean"], 0.6)
        self.assertEqual(scores["open_math_prm_min"], 0.2)
        self.assertEqual(gate_weights["open_reasoning_rm"], 0.4)
        self.assertEqual(gate_weights["open_math_prm_mean"], 0.3)
        self.assertEqual(gate_weights["open_math_prm_min"], 0.3)


if __name__ == "__main__":
    unittest.main()
