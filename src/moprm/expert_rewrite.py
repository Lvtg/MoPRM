from __future__ import annotations

from dataclasses import replace

from moprm.schema import Candidate, ProblemRecord


def parse_mapping(items: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected mapping in old=new format, got: {item}")
        old, new = item.split("=", 1)
        old = old.strip()
        new = new.strip()
        if not old or not new:
            raise ValueError(f"Invalid empty mapping side: {item}")
        mapping[old] = new
    return mapping


def rewrite_score_dict(
    scores: dict[str, float],
    *,
    rename: dict[str, str],
    drop: set[str],
) -> dict[str, float]:
    rewritten: dict[str, float] = {}
    for expert, score in scores.items():
        if expert in drop:
            continue
        target = rename.get(expert, expert)
        rewritten[target] = score
    return rewritten


def rewrite_candidate(
    candidate: Candidate,
    *,
    rename: dict[str, str],
    drop: set[str],
) -> Candidate:
    return Candidate(
        candidate_id=candidate.candidate_id,
        solution=candidate.solution,
        final_answer=candidate.final_answer,
        is_correct=candidate.is_correct,
        steps=candidate.steps,
        expert_scores=rewrite_score_dict(candidate.expert_scores, rename=rename, drop=drop),
        normalized_scores=rewrite_score_dict(
            candidate.normalized_scores,
            rename=rename,
            drop=drop,
        ),
        metadata=candidate.metadata,
    )


def rewrite_record_experts(
    record: ProblemRecord,
    *,
    rename: dict[str, str],
    drop: set[str],
) -> ProblemRecord:
    metadata = dict(record.metadata)
    metadata["expert_pool_rewrite"] = {
        "rename": rename,
        "drop": sorted(drop),
    }
    return replace(
        record,
        candidates=[
            rewrite_candidate(candidate, rename=rename, drop=drop)
            for candidate in record.candidates
        ],
        metadata=metadata,
    )
