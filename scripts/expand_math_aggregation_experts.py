from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moprm.io import load_jsonl, write_jsonl  # noqa: E402
from moprm.schema import Candidate, ProblemRecord  # noqa: E402
from moprm.scoring.skywork_math_prm import (  # noqa: E402
    OPEN_MATH_PRM_EXPERT,
    aggregate_step_rewards,
)


DEFAULT_AGGREGATIONS = ("mean", "min", "last", "geomean")


def parse_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def aggregation_expert_name(source_expert: str, aggregation: str) -> str:
    return f"{source_expert}_{aggregation}"


def expand_record(
    record: ProblemRecord,
    *,
    source_expert: str = OPEN_MATH_PRM_EXPERT,
    aggregations: tuple[str, ...] | list[str] = DEFAULT_AGGREGATIONS,
    drop_source_expert: bool = True,
) -> ProblemRecord:
    aggregation_tuple = tuple(aggregations)
    pseudo_experts = [
        aggregation_expert_name(source_expert, aggregation)
        for aggregation in aggregation_tuple
    ]
    candidates: list[Candidate] = []
    for candidate in record.candidates:
        candidate_metadata = dict(candidate.metadata)
        source_metadata = dict(candidate_metadata.get(source_expert, {}))
        step_rewards = source_metadata.get("step_rewards")
        expert_scores = dict(candidate.expert_scores)
        if drop_source_expert:
            expert_scores.pop(source_expert, None)

        if isinstance(step_rewards, list):
            numeric_rewards = [float(reward) for reward in step_rewards]
            for aggregation, pseudo_expert in zip(aggregation_tuple, pseudo_experts, strict=True):
                score = aggregate_step_rewards(numeric_rewards, method=aggregation)
                expert_scores[pseudo_expert] = score
                candidate_metadata[pseudo_expert] = {
                    "source_expert": source_expert,
                    "aggregation": aggregation,
                    "expanded_from_step_rewards": True,
                    "step_rewards": numeric_rewards,
                }

        candidates.append(
            Candidate(
                candidate_id=candidate.candidate_id,
                solution=candidate.solution,
                final_answer=candidate.final_answer,
                is_correct=candidate.is_correct,
                steps=candidate.steps,
                expert_scores=expert_scores,
                normalized_scores=candidate.normalized_scores,
                metadata=candidate_metadata,
            )
        )

    metadata = dict(record.metadata)
    metadata["math_aggregation_pseudo_experts"] = {
        "source_expert": source_expert,
        "experts": pseudo_experts,
        "drop_source_expert": drop_source_expert,
    }
    metadata = _remap_gate_weights_for_pseudo_experts(
        metadata,
        source_expert=source_expert,
        pseudo_experts=pseudo_experts,
        drop_source_expert=drop_source_expert,
    )

    return ProblemRecord(
        problem_id=record.problem_id,
        domain=record.domain,
        problem=record.problem,
        answer=record.answer,
        candidates=candidates,
        metadata=metadata,
    )


def _remap_gate_weights_for_pseudo_experts(
    metadata: dict,
    *,
    source_expert: str,
    pseudo_experts: list[str],
    drop_source_expert: bool,
) -> dict:
    gate_weights = metadata.get("gate_weights", {})
    if not isinstance(gate_weights, dict) or not pseudo_experts:
        return metadata
    remapped_gate_weights = {}
    for gate_name, weights in gate_weights.items():
        if not isinstance(weights, dict):
            remapped_gate_weights[gate_name] = weights
            continue
        updated = dict(weights)
        source_weight = float(updated.get(source_expert, 0.0))
        if drop_source_expert:
            updated.pop(source_expert, None)
        if source_weight > 0:
            split_weight = source_weight / len(pseudo_experts)
            for pseudo_expert in pseudo_experts:
                updated[pseudo_expert] = updated.get(pseudo_expert, 0.0) + split_weight
        remapped_gate_weights[gate_name] = updated
    metadata = dict(metadata)
    metadata["gate_weights"] = remapped_gate_weights
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Expand cached open_math_prm step rewards into aggregation-specific "
            "pseudo-experts such as open_math_prm_min."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-expert", default=OPEN_MATH_PRM_EXPERT)
    parser.add_argument(
        "--aggregations",
        default=",".join(DEFAULT_AGGREGATIONS),
        help="Comma-separated aggregation methods to expose as pseudo-experts.",
    )
    parser.add_argument(
        "--keep-source-expert",
        action="store_true",
        help="Keep the original source expert in addition to aggregation pseudo-experts.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(
            f"{output} already exists. Pass --overwrite to replace it deliberately."
        )

    records = load_jsonl(Path(args.input))
    aggregations = parse_csv(args.aggregations)
    expanded = [
        expand_record(
            record,
            source_expert=args.source_expert,
            aggregations=aggregations,
            drop_source_expert=not args.keep_source_expert,
        )
        for record in records
    ]
    write_jsonl(output, (record.to_dict() for record in expanded))
    print(
        f"Expanded {args.source_expert} into "
        f"{', '.join(aggregation_expert_name(args.source_expert, item) for item in aggregations)}"
    )
    print(f"Input records: {len(records)}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
