from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moprm.io import load_jsonl, write_jsonl  # noqa: E402
from moprm.openai_responses import OpenAIResponsesClient, resolve_secret  # noqa: E402
from moprm.scoring.openai_experts import (  # noqa: E402
    OPENAI_EXPERTS,
    OpenAIExpertScoringConfig,
    score_record_with_openai,
)


def _usage_from_records(records: list[dict]) -> dict[str, int]:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for record in records:
        for candidate in record.get("candidates", []):
            usage = (
                candidate.get("metadata", {})
                .get("openai_expert_scoring", {})
                .get("usage", {})
            )
            if not isinstance(usage, dict):
                continue
            for key in totals:
                value = usage.get(key)
                if isinstance(value, int):
                    totals[key] += value
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score generated candidates with lightweight OpenAI PRM-style experts. "
            "The scorer never receives gold answers."
        )
    )
    parser.add_argument("--input", default="data/cache/openai_dev_smoke_labeled.jsonl")
    parser.add_argument("--output", default="data/scored/openai_dev_smoke_scored.jsonl")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of problem records to process concurrently. Default: 1.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(
            f"{output} already exists. Pass --overwrite to replace it deliberately."
        )

    api_key = resolve_secret("OPENAI_API_KEY", args.env_file)
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY was not found in the process environment or the .env file."
        )

    records = load_jsonl(Path(args.input))
    if args.limit is not None:
        records = records[: args.limit]

    total_candidates = sum(len(record.candidates) for record in records)
    print(
        "Starting OpenAI expert scoring: "
        f"records={len(records)}, candidates={total_candidates}, "
        f"api_calls={total_candidates}, model={args.model}, "
        f"experts={','.join(OPENAI_EXPERTS)}"
    )

    client = OpenAIResponsesClient(
        api_key,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    config = OpenAIExpertScoringConfig(
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
    )

    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")

    scored_by_index: dict[int, dict] = {}
    if args.concurrency == 1:
        for index, record in enumerate(records, start=1):
            print(f"[{index}/{len(records)}] {record.problem_id} ({record.domain})")
            scored_record = score_record_with_openai(record, client, config)
            scored_by_index[index] = scored_record.to_dict()
            write_jsonl(output, [scored_by_index[i] for i in sorted(scored_by_index)])
    else:
        print(f"Using concurrency={args.concurrency}")

        def worker(index_and_record):
            index, record = index_and_record
            scored_record = score_record_with_openai(record, client, config)
            return index, record.problem_id, record.domain, scored_record.to_dict()

        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [
                executor.submit(worker, item)
                for item in enumerate(records, start=1)
            ]
            for future in as_completed(futures):
                index, problem_id, domain, scored_record = future.result()
                scored_by_index[index] = scored_record
                print(
                    f"[done {len(scored_by_index)}/{len(records)}] "
                    f"{problem_id} ({domain})"
                )
                write_jsonl(output, [scored_by_index[i] for i in sorted(scored_by_index)])

    scored = [scored_by_index[i] for i in sorted(scored_by_index)]
    usage = _usage_from_records(scored)
    print(f"Wrote {output}")
    if usage["total_tokens"]:
        print(
            "Token usage reported by API: "
            f"input={usage['input_tokens']}, output={usage['output_tokens']}, "
            f"total={usage['total_tokens']}"
        )


if __name__ == "__main__":
    main()
