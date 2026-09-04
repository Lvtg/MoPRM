import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moprm.datasets.sampling import (
    parse_source_quota,
    sample_records,
    sample_records_by_source_quotas,
)
from moprm.schema import ProblemRecord


def make_record(problem_id: str, domain: str, source: str = "") -> ProblemRecord:
    return ProblemRecord.from_dict(
        {
            "problem_id": problem_id,
            "domain": domain,
            "problem": problem_id,
            "answer": "A",
            "candidates": [],
            "metadata": {"source": source},
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

    def test_parse_source_quota(self) -> None:
        self.assertEqual(
            parse_source_quota("math|HuggingFaceH4/MATH-500=50"),
            ("math", "HuggingFaceH4/MATH-500", 50),
        )

    def test_source_quota_sample(self) -> None:
        records = [
            make_record("math500_1", "math", "MATH500"),
            make_record("math500_2", "math", "MATH500"),
            make_record("gsm8k_1", "math", "GSM8K"),
            make_record("logic_1", "logic", "BBH"),
            make_record("logic_2", "logic", "BBH"),
        ]
        sampled = sample_records_by_source_quotas(
            records,
            [
                ("math", "MATH500", 2),
                ("math", "GSM8K", 1),
                ("logic", "BBH", 1),
            ],
            seed=1,
        )
        self.assertEqual(len(sampled), 4)
        self.assertEqual(
            sum(record.metadata["source"] == "MATH500" for record in sampled),
            2,
        )

    def test_source_quota_sample_excludes_problem_ids(self) -> None:
        records = [
            make_record("math500_1", "math", "MATH500"),
            make_record("math500_2", "math", "MATH500"),
            make_record("math500_3", "math", "MATH500"),
            make_record("logic_1", "logic", "BBH"),
            make_record("logic_2", "logic", "BBH"),
        ]
        sampled = sample_records_by_source_quotas(
            records,
            [
                ("math", "MATH500", 2),
                ("logic", "BBH", 1),
            ],
            seed=1,
            exclude_ids={"math500_1", "logic_1"},
        )

        self.assertEqual(len(sampled), 3)
        self.assertNotIn("math500_1", {record.problem_id for record in sampled})
        self.assertNotIn("logic_1", {record.problem_id for record in sampled})

    def test_per_domain_sample_excludes_problem_ids(self) -> None:
        records = [
            make_record("m1", "math"),
            make_record("m2", "math"),
            make_record("l1", "logic"),
            make_record("l2", "logic"),
        ]
        sampled = sample_records(
            records,
            per_domain=1,
            seed=1,
            exclude_ids={"m1", "l1"},
        )

        self.assertEqual({record.problem_id for record in sampled}, {"m2", "l2"})

    def test_source_quota_raises_when_pool_too_small(self) -> None:
        records = [make_record("math500_1", "math", "MATH500")]
        with self.assertRaises(ValueError):
            sample_records_by_source_quotas(
                records,
                [("math", "MATH500", 2)],
                seed=1,
            )


if __name__ == "__main__":
    unittest.main()
