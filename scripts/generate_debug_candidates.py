from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moprm.candidates.debug_generator import generate_debug_candidates
from moprm.io import load_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate gold-derived debug candidates for pipeline testing only."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    records = load_jsonl(Path(args.input))
    if args.limit is not None:
        records = records[: args.limit]
    generated = [
        generate_debug_candidates(record, n=args.num_candidates).to_dict()
        for record in records
    ]
    write_jsonl(Path(args.output), generated)
    print(
        f"Generated {len(generated)} records with {args.num_candidates} debug candidates each."
    )
    print("WARNING: debug candidates use gold answers and are not valid experiment results.")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

