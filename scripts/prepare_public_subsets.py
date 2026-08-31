from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moprm.datasets.public_sources import PublicSubsetConfig, prepare_public_subsets
from moprm.io import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare public MoPRM eval subsets.")
    parser.add_argument("--output-dir", default="data/cache/public_subsets")
    parser.add_argument("--math500-limit", type=int, default=80)
    parser.add_argument("--gsm8k-limit", type=int, default=80)
    parser.add_argument("--bbh-limit-per-task", type=int, default=60)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    config = PublicSubsetConfig(
        math500_limit=args.math500_limit,
        gsm8k_limit=args.gsm8k_limit,
        bbh_limit_per_task=args.bbh_limit_per_task,
        seed=args.seed,
    )
    outputs = prepare_public_subsets(Path(args.output_dir), config)

    print("Prepared public subsets:")
    for name, path in outputs.items():
        records = load_jsonl(path)
        domain_counts: dict[str, int] = {}
        for record in records:
            domain_counts[record.domain] = domain_counts.get(record.domain, 0) + 1
        print(f"- {name}: {path} ({len(records)} records, domains={domain_counts})")


if __name__ == "__main__":
    main()

