from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moprm.datasets.sampling import sample_records
from moprm.io import load_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample a reproducible MoPRM dataset split.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--per-domain", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    if args.limit is None and args.per_domain is None:
        raise ValueError("Specify --limit or --per-domain")

    records = load_jsonl(Path(args.input))
    sampled = sample_records(
        records,
        limit=args.limit,
        per_domain=args.per_domain,
        seed=args.seed,
    )
    write_jsonl(Path(args.output), [record.to_dict() for record in sampled])
    counts = Counter(record.domain for record in sampled)
    print(f"Sampled {len(sampled)} records from {args.input}")
    print(f"Domains: {dict(counts)}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

