import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moprm.datasets.sampling import sample_records
from moprm.schema import ProblemRecord


def make_record(problem_id: str, domain: str) -> ProblemRecord:
    return ProblemRecord.from_dict(
        {
            "problem_id": problem_id,
            "domain": domain,
            "problem": problem_id,
            "answer": "A",
            "candidates": [],
        }
    )


class SamplingTest(unittest.TestCase):
    def test_per_domain_sample(self) -> None:
        records = [
            make_record("m1", "math"),
            make_record("m2", "math"),
            make_record("l1", "logic"),
            make_record("l2", "logic"),
        ]
        sampled = sample_records(records, per_domain=1, seed=1)
        self.assertEqual(len(sampled), 2)
        self.assertEqual({record.domain for record in sampled}, {"math", "logic"})


if __name__ == "__main__":
    unittest.main()

