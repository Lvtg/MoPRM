from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moprm.datasets.sampling import parse_source_quota, sample_records, sample_records_by_source_quotas
from moprm.io import load_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample a reproducible MoPRM dataset split.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--per-domain", type=int, default=None)
    parser.add_argument(
        "--source-quota",
        action="append",
        default=[],
        help=(
            "Sample an exact source quota in the form domain|metadata.source=count. "
            "Can be repeated, e.g. --source-quota math|HuggingFaceH4/MATH-500=50."
        ),
    )
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    if args.limit is None and args.per_domain is None and not args.source_quota:
        raise ValueError("Specify --limit, --per-domain, or --source-quota")
    if args.source_quota and args.per_domain is not None:
        raise ValueError("Use either --source-quota or --per-domain, not both")

    records = load_jsonl(Path(args.input))
    if args.source_quota:
        quotas = [parse_source_quota(item) for item in args.source_quota]
        sampled = sample_records_by_source_quotas(records, quotas, seed=args.seed)
        if args.limit is not None:
            sampled = sampled[: args.limit]
    else:
        sampled = sample_records(
            records,
            limit=args.limit,
            per_domain=args.per_domain,
            seed=args.seed,
        )
    write_jsonl(Path(args.output), [record.to_dict() for record in sampled])
    counts = Counter(record.domain for record in sampled)
    source_counts = Counter(
        f"{record.domain}|{record.metadata.get('source', '')}" for record in sampled
    )
    print(f"Sampled {len(sampled)} records from {args.input}")
    print(f"Domains: {dict(counts)}")
    print(f"Sources: {dict(source_counts)}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
