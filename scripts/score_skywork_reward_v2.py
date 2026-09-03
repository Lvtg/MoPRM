from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moprm.io import load_jsonl, write_jsonl  # noqa: E402
from moprm.scoring.skywork_reward_v2 import (  # noqa: E402
    OPEN_REASONING_RM_EXPERT,
    SKYWORK_REWARD_V2_QWEN3_17B_MODEL,
    SkyworkRewardV2Config,
    SkyworkRewardV2Scorer,
    format_reward_conversation,
    score_record_with_skywork_reward_v2,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score candidates with Skywork Reward-V2 as a non-OpenAI "
            "response-level reasoning reward model."
        )
    )
    parser.add_argument("--input", default="data/scored/openai_dev40_n4_with_skywork_math.jsonl")
    parser.add_argument("--output", default="data/scored/openai_dev40_n4_with_open_experts.jsonl")
    parser.add_argument("--model", default=SKYWORK_REWARD_V2_QWEN3_17B_MODEL)
    parser.add_argument("--expert-name", default=OPEN_REASONING_RM_EXPERT)
    parser.add_argument("--cache-dir", default="models/hf_cache")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help=(
            "Torch dtype for model loading. Default auto uses bfloat16 on CUDA "
            "and float32 on CPU."
        ),
    )
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--domains",
        default="all",
        help="Comma-separated domains to score, or 'all'. Default: all.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect records and formatted prompt shape without loading the model.",
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
        "Skywork Reward-V2 scoring plan: "
        f"records={len(score_records)}/{len(records)}, candidates={total_candidates}, "
        f"model={args.model}, expert={args.expert_name}, device={args.device}, "
        f"dtype={args.dtype}"
    )

    if args.dry_run:
        class PreviewTokenizer:
            bos_token = "<bos>"

            def apply_chat_template(self, conversation, tokenize=False):
                assert tokenize is False
                return "\n".join(
                    f"{item['role']}: {item['content']}" for item in conversation
                )

        tokenizer = PreviewTokenizer()
        for record in score_records[:3]:
            print(f"[dry-run] {record.problem_id} ({record.domain})")
            if record.candidates:
                formatted = format_reward_conversation(
                    record.problem,
                    record.candidates[0].solution,
                    tokenizer,
                )
                print(f"  formatted_chars={len(formatted)}")
                print(f"  preview={formatted[:160].replace(chr(10), ' ')}")
        return

    config = SkyworkRewardV2Config(
        model_name=args.model,
        expert_name=args.expert_name,
        cache_dir=args.cache_dir,
        device=args.device,
        dtype=args.dtype,
        max_length=args.max_length,
    )
    scorer = SkyworkRewardV2Scorer(config)

    scored_by_id = {}
    for index, record in enumerate(score_records, start=1):
        print(f"[{index}/{len(score_records)}] {record.problem_id} ({record.domain})")
        scored_by_id[record.problem_id] = score_record_with_skywork_reward_v2(
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
