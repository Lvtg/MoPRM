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
    args = parser.parse_args()

    records = load_jsonl(Path(args.input))
    groups = {"overall": records}
    if args.by_domain:
        for domain in sorted({record.domain for record in records}):
            groups[domain] = [record for record in records if record.domain == domain]

    print(f"Loaded {len(records)} problems from {args.input}")
    for group_name, group_records in groups.items():
        results = sorted(
            default_baselines(group_records, normalization=args.normalization),
            key=lambda item: item.method,
        )
        print(f"\n[{group_name}]")
        print(f"{'method':<24} {'correct':>8} {'total':>8} {'accuracy':>10}")
        print("-" * 54)
        for result in results:
            print(f"{result.method:<24} {result.correct:>8} {result.total:>8} {result.accuracy:>10.3f}")


if __name__ == "__main__":
    main()
