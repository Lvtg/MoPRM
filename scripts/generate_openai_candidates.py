from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moprm.candidates.openai_generator import (  # noqa: E402
    OpenAICandidateConfig,
    generate_openai_candidates,
)
from moprm.io import load_jsonl, write_jsonl  # noqa: E402
from moprm.openai_responses import OpenAIResponsesClient, resolve_secret  # noqa: E402


def _usage_from_records(records: list[dict]) -> dict[str, int]:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for record in records:
        for candidate in record.get("candidates", []):
            usage = candidate.get("metadata", {}).get("usage", {})
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
            "Generate real OpenAI model candidates for MoPRM experiments. "
            "The default run is intentionally tiny to avoid accidental spend."
        )
    )
    parser.add_argument("--input", default="data/splits/dev_40.jsonl")
    parser.add_argument("--output", default="data/candidates/openai_dev_smoke.jsonl")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--num-candidates", type=int, default=2)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of problem records to process concurrently. Default: 1.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file.",
    )
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

    total_calls = len(records) * args.num_candidates
    print(
        "Starting OpenAI candidate generation: "
        f"records={len(records)}, candidates_per_record={args.num_candidates}, "
        f"total_api_calls={total_calls}, model={args.model}"
    )

    client = OpenAIResponsesClient(
        api_key,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    config = OpenAICandidateConfig(
        model=args.model,
        num_candidates=args.num_candidates,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
    )

    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")

    generated_by_index: dict[int, dict] = {}
    if args.concurrency == 1:
        for index, record in enumerate(records, start=1):
            print(f"[{index}/{len(records)}] {record.problem_id} ({record.domain})")
            generated_record = generate_openai_candidates(record, client, config)
            generated_by_index[index] = generated_record.to_dict()
            write_jsonl(output, [generated_by_index[i] for i in sorted(generated_by_index)])
    else:
        print(f"Using concurrency={args.concurrency}")

        def worker(index_and_record):
            index, record = index_and_record
            generated_record = generate_openai_candidates(record, client, config)
            return index, record.problem_id, record.domain, generated_record.to_dict()

        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [
                executor.submit(worker, item)
                for item in enumerate(records, start=1)
            ]
            for future in as_completed(futures):
                index, problem_id, domain, generated_record = future.result()
                generated_by_index[index] = generated_record
                print(
                    f"[done {len(generated_by_index)}/{len(records)}] "
                    f"{problem_id} ({domain})"
                )
                write_jsonl(output, [generated_by_index[i] for i in sorted(generated_by_index)])

    generated = [generated_by_index[i] for i in sorted(generated_by_index)]
    usage = _usage_from_records(generated)
    print(f"Wrote {output}")
    if usage["total_tokens"]:
        print(
            "Token usage reported by API: "
            f"input={usage['input_tokens']}, output={usage['output_tokens']}, "
            f"total={usage['total_tokens']}"
        )


if __name__ == "__main__":
    main()
