from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moprm.io import load_jsonl, write_jsonl  # noqa: E402
from moprm.scoring.skywork_math_prm import (  # noqa: E402
    OPEN_MATH_PRM_EXPERT,
    SKYWORK_MATH_PRM_MODEL,
    SkyworkMathPRMConfig,
    SkyworkMathPRMScorer,
    score_record_with_skywork_math_prm,
    split_solution_steps,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score candidates with Skywork's non-OpenAI open-source math PRM. "
            "The model checkpoint is downloaded to the ignored models/hf_cache path."
        )
    )
    parser.add_argument("--input", default="data/cache/openai_pilot10_n4_labeled.jsonl")
    parser.add_argument("--output", default="data/scored/skywork_math_prm_pilot.jsonl")
    parser.add_argument("--model", default=SKYWORK_MATH_PRM_MODEL)
    parser.add_argument("--expert-name", default=OPEN_MATH_PRM_EXPERT)
    parser.add_argument("--cache-dir", default="models/hf_cache")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--aggregation", default="mean", choices=["mean", "min", "last", "geomean"])
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--domains",
        default="math",
        help="Comma-separated domains to score, or 'all'. Default: math.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect records and step splitting without loading/downloading the model.",
    )
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not args.overwrite and not args.dry_run:
        raise SystemExit(
            f"{output} already exists. Pass --overwrite to replace it deliberately."
        )

    records = load_jsonl(Path(args.input))
    if args.limit is not None:
        records = records[: args.limit]

    domains = None if args.domains == "all" else set(args.domains.split(","))
    score_records = [
        record for record in records if domains is None or record.domain in domains
    ]
    total_candidates = sum(len(record.candidates) for record in score_records)
    print(
        "Skywork math PRM scoring plan: "
        f"records={len(score_records)}/{len(records)}, candidates={total_candidates}, "
        f"model={args.model}, expert={args.expert_name}, device={args.device}"
    )

    if args.dry_run:
        for record in score_records[:3]:
            print(f"[dry-run] {record.problem_id} ({record.domain})")
            if record.candidates:
                steps = split_solution_steps(record.candidates[0].solution, args.max_steps)
                print(f"  first candidate steps={len(steps)}")
                for step in steps[:3]:
                    print(f"  - {step[:120]}")
        return

    config = SkyworkMathPRMConfig(
        model_name=args.model,
        expert_name=args.expert_name,
        cache_dir=args.cache_dir,
        device=args.device,
        max_length=args.max_length,
        max_steps=args.max_steps,
        aggregation=args.aggregation,
    )
    scorer = SkyworkMathPRMScorer(config)

    scored_by_id = {}
    for index, record in enumerate(score_records, start=1):
        print(f"[{index}/{len(score_records)}] {record.problem_id} ({record.domain})")
        scored_by_id[record.problem_id] = score_record_with_skywork_math_prm(
            record,
            scorer,
        )
        output_records = [
            scored_by_id.get(record.problem_id, record).to_dict() for record in records
        ]
        write_jsonl(output, output_records)

    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
