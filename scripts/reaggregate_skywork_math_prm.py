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


def reaggregate_record(
    record: ProblemRecord,
    *,
    expert_name: str,
    aggregation: str,
) -> ProblemRecord:
    candidates: list[Candidate] = []
    for candidate in record.candidates:
        candidate_metadata = dict(candidate.metadata)
        expert_metadata = dict(candidate_metadata.get(expert_name, {}))
        step_rewards = expert_metadata.get("step_rewards")
        if not isinstance(step_rewards, list):
            candidates.append(candidate)
            continue
        score = aggregate_step_rewards(
            [float(reward) for reward in step_rewards],
            method=aggregation,
        )
        expert_scores = dict(candidate.expert_scores)
        expert_scores[expert_name] = score
        expert_metadata["aggregation"] = aggregation
        expert_metadata["reaggregated_from_step_rewards"] = True
        candidate_metadata[expert_name] = expert_metadata
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

    return ProblemRecord(
        problem_id=record.problem_id,
        domain=record.domain,
        problem=record.problem,
        answer=record.answer,
        candidates=candidates,
        metadata=record.metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute Skywork math PRM scores from cached step rewards."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expert-name", default=OPEN_MATH_PRM_EXPERT)
    parser.add_argument(
        "--aggregation",
        required=True,
        choices=["mean", "min", "last", "geomean"],
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(
            f"{output} already exists. Pass --overwrite to replace it deliberately."
        )

    records = load_jsonl(Path(args.input))
    reaggregated = [
        reaggregate_record(
            record,
            expert_name=args.expert_name,
            aggregation=args.aggregation,
        )
        for record in records
    ]
    write_jsonl(output, [record.to_dict() for record in reaggregated])
    print(
        f"Reaggregated {len(reaggregated)} records for {args.expert_name} "
        f"using aggregation={args.aggregation}"
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
