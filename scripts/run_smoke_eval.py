from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moprm.evaluate import default_baselines
from moprm.io import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small MoPRM BoN smoke evaluation.")
    parser.add_argument("--input", default="examples/smoke_moprm.jsonl")
    parser.add_argument("--normalization", default="rank", choices=["rank", "minmax", "zscore"])
    parser.add_argument("--by-domain", action="store_true")
    parser.add_argument(
        "--by-source",
        action="store_true",
        help="Also report groups by domain plus metadata.source.",
    )
    args = parser.parse_args()

    records = load_jsonl(Path(args.input))
    groups = {"overall": records}
    if args.by_domain:
        for domain in sorted({record.domain for record in records}):
            groups[domain] = [record for record in records if record.domain == domain]
    if args.by_source:
        for key in sorted(
            {
                f"{record.domain}|{record.metadata.get('source', '')}"
                for record in records
            }
        ):
            domain, source = key.split("|", 1)
            groups[key] = [
                record
                for record in records
                if record.domain == domain and str(record.metadata.get("source", "")) == source
            ]

    print(f"Loaded {len(records)} problems from {args.input}")
    for group_name, group_records in groups.items():
        results = sorted(
            default_baselines(group_records, normalization=args.normalization),
            key=lambda item: item.method,
        )
        method_width = max(24, *(len(result.method) for result in results))
        print(f"\n[{group_name}]")
        print(f"{'method':<{method_width}} {'correct':>8} {'total':>8} {'accuracy':>10}")
        print("-" * (method_width + 30))
        for result in results:
            print(
                f"{result.method:<{method_width}} "
                f"{result.correct:>8} {result.total:>8} {result.accuracy:>10.3f}"
            )


if __name__ == "__main__":
    main()
