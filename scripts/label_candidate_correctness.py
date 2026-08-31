from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moprm.labeling import label_candidate_correctness


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add candidate is_correct labels using gold answers for evaluation only."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    correct = 0
    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as target:
        for line_no, line in enumerate(source, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {input_path}:{line_no}") from exc
            labeled = label_candidate_correctness(record, overwrite=args.overwrite)
            for candidate in labeled.get("candidates", []):
                total += 1
                correct += 1 if candidate.get("is_correct") else 0
            target.write(json.dumps(labeled, ensure_ascii=False))
            target.write("\n")

    print(f"Labeled {total} candidates; {correct} are correct.")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

