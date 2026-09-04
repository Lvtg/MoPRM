from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moprm.io import load_jsonl, write_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge MoPRM JSONL record files with optional problem-id checks."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input JSONL path. Repeat to merge multiple files in order.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--allow-duplicate-problem-ids",
        action="store_true",
        help="Allow duplicate problem_id values across input files.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(
            f"{output} already exists. Pass --overwrite to replace it deliberately."
        )

    records = []
    seen: set[str] = set()
    duplicates: set[str] = set()
    for input_text in args.input:
        path = Path(input_text)
        for record in load_jsonl(path):
            if record.problem_id in seen:
                duplicates.add(record.problem_id)
            seen.add(record.problem_id)
            records.append(record)

    if duplicates and not args.allow_duplicate_problem_ids:
        preview = ", ".join(sorted(duplicates)[:10])
        raise SystemExit(
            "Duplicate problem_id values found while merging: "
            f"{preview}. Pass --allow-duplicate-problem-ids to override."
        )

    write_jsonl(output, [record.to_dict() for record in records])
    print(f"Merged {len(records)} records from {len(args.input)} files")
    print(f"Unique problem IDs: {len(seen)}")
    if duplicates:
        print(f"Duplicate problem IDs: {len(duplicates)}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
