import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moprm.scoring.skywork_math_prm import OPEN_MATH_PRM_EXPERT  # noqa: E402
from moprm.schema import ProblemRecord  # noqa: E402

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT / "scripts"))

from reaggregate_skywork_math_prm import reaggregate_record  # noqa: E402


class ReaggregateSkyworkMathPRMTest(unittest.TestCase):
    def test_reaggregate_record_updates_score_from_step_rewards(self) -> None:
        record = ProblemRecord.from_dict(
            {
                "problem_id": "p1",
                "domain": "math",
                "problem": "1+1",
                "answer": "2",
                "candidates": [
                    {
                        "candidate_id": "c1",
                        "solution": "1+1=2",
                        "expert_scores": {OPEN_MATH_PRM_EXPERT: 0.5},
                        "metadata": {
                            OPEN_MATH_PRM_EXPERT: {
                                "aggregation": "mean",
                                "step_rewards": [0.1, 0.9],
                            }
                        },
                    }
                ],
            }
        )
        updated = reaggregate_record(
            record,
            expert_name=OPEN_MATH_PRM_EXPERT,
            aggregation="min",
        )
        candidate = updated.candidates[0]
        self.assertAlmostEqual(candidate.expert_scores[OPEN_MATH_PRM_EXPERT], 0.1)
        self.assertEqual(candidate.metadata[OPEN_MATH_PRM_EXPERT]["aggregation"], "min")
        self.assertTrue(
            candidate.metadata[OPEN_MATH_PRM_EXPERT]["reaggregated_from_step_rewards"]
        )


if __name__ == "__main__":
    unittest.main()
