from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moprm.io import load_jsonl, write_jsonl
from moprm.scoring.debug_experts import DEBUG_EXPERTS, score_debug_record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add synthetic debug expert scores for pipeline testing only."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records = load_jsonl(Path(args.input))
    scored = [score_debug_record(record).to_dict() for record in records]
    write_jsonl(Path(args.output), scored)
    print(f"Scored {len(scored)} records with debug experts: {', '.join(DEBUG_EXPERTS)}")
    print("WARNING: debug scores are synthetic and are not valid experiment results.")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

