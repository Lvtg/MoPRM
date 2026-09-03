from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moprm.io import load_jsonl, write_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter records by how many generated candidates are correct."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-correct", type=int, default=1)
    parser.add_argument("--max-correct", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(
            f"{output} already exists. Pass --overwrite to replace it deliberately."
        )

    records = load_jsonl(Path(args.input))
    selected = []
    count_histogram: Counter[int] = Counter()
    source_counts: Counter[str] = Counter()
    for record in records:
        correct = sum(1 for candidate in record.candidates if candidate.is_correct)
        total = len(record.candidates)
        max_correct = total if args.max_correct is None else args.max_correct
        if args.min_correct <= correct <= max_correct:
            selected.append(record)
            count_histogram[correct] += 1
            source_counts[f"{record.domain}|{record.metadata.get('source', '')}"] += 1

    write_jsonl(output, [record.to_dict() for record in selected])
    print(f"Selected {len(selected)} / {len(records)} records")
    print(f"Correct-count histogram: {dict(sorted(count_histogram.items()))}")
    print(f"Sources: {dict(sorted(source_counts.items()))}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
