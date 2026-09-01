from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moprm.io import load_jsonl, write_jsonl  # noqa: E402
from moprm.openai_responses import OpenAIResponsesClient, resolve_secret  # noqa: E402
from moprm.routing.openai_gate import (  # noqa: E402
    DEFAULT_GATE_NAME,
    OpenAIGateConfig,
    route_record_with_openai,
)


def _usage_from_records(records: list[dict], gate_name: str) -> dict[str, int]:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for record in records:
        usage = (
            record.get("metadata", {})
            .get("gate_metadata", {})
            .get(gate_name, {})
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
            "Attach an OpenAI LLM question-level gate to scored MoPRM records. "
            "The gate sees no candidates and no gold answers."
        )
    )
    parser.add_argument("--input", default="data/scored/openai_pilot10_n4_scored.jsonl")
    parser.add_argument("--output", default="data/scored/openai_pilot10_n4_routed.jsonl")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--gate-name", default=DEFAULT_GATE_NAME)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=2)
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

    print(
        "Starting OpenAI gate routing: "
        f"records={len(records)}, api_calls={len(records)}, model={args.model}, "
        f"gate_name={args.gate_name}"
    )

    client = OpenAIResponsesClient(
        api_key,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    config = OpenAIGateConfig(
        model=args.model,
        gate_name=args.gate_name,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
    )

    routed: list[dict] = []
    for index, record in enumerate(records, start=1):
        print(f"[{index}/{len(records)}] {record.problem_id} ({record.domain})")
        routed_record = route_record_with_openai(record, client, config)
        routed.append(routed_record.to_dict())
        write_jsonl(output, routed)

    usage = _usage_from_records(routed, args.gate_name)
    print(f"Wrote {output}")
    if usage["total_tokens"]:
        print(
            "Token usage reported by API: "
            f"input={usage['input_tokens']}, output={usage['output_tokens']}, "
            f"total={usage['total_tokens']}"
        )


if __name__ == "__main__":
    main()
