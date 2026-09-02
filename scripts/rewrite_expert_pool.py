from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moprm.expert_rewrite import parse_mapping, rewrite_record_experts  # noqa: E402
from moprm.io import load_jsonl, write_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rename/drop expert score dimensions to build a clean comparison pool."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--rename",
        action="append",
        default=[],
        help="Expert rename mapping in old=new format. Can be passed multiple times.",
    )
    parser.add_argument(
        "--drop",
        action="append",
        default=[],
        help="Expert score key to remove. Can be passed multiple times.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(
            f"{output} already exists. Pass --overwrite to replace it deliberately."
        )

    rename = parse_mapping(args.rename)
    drop = set(args.drop)
    records = load_jsonl(Path(args.input))
    rewritten = [
        rewrite_record_experts(record, rename=rename, drop=drop).to_dict()
        for record in records
    ]
    write_jsonl(output, rewritten)

    expert_names = sorted(
        {
            expert
            for record in rewritten
            for candidate in record.get("candidates", [])
            for expert in candidate.get("expert_scores", {})
        }
    )
    print(f"Rewrote {len(rewritten)} records.")
    print(f"Experts: {', '.join(expert_names)}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
