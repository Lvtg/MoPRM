from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moprm.io import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a MoPRM JSONL dataset.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()

    records = load_jsonl(Path(args.input))
    domains = Counter(record.domain for record in records)
    sources = Counter(str(record.metadata.get("source", "unknown")) for record in records)
    empty_problems = [record.problem_id for record in records if not record.problem.strip()]
    empty_answers = [record.problem_id for record in records if not record.answer.strip()]
    candidate_counts = Counter(len(record.candidates) for record in records)

    print(f"Loaded {len(records)} records from {args.input}")
    print(f"Domains: {dict(domains)}")
    print(f"Sources: {dict(sources)}")
    print(f"Candidate counts: {dict(candidate_counts)}")
    print(f"Empty problems: {len(empty_problems)}")
    print(f"Empty answers: {len(empty_answers)}")

    if empty_problems:
        print(f"First empty problem ids: {empty_problems[:5]}")
    if empty_answers:
        print(f"First empty answer ids: {empty_answers[:5]}")

    print("\nExamples:")
    for record in records[: args.examples]:
        problem_preview = record.problem.replace("\n", " ")[:160]
        print(f"- {record.problem_id} [{record.domain}] answer={record.answer!r}")
        print(f"  {problem_preview}")


if __name__ == "__main__":
    main()

