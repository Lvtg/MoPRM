from __future__ import annotations

from moprm.schema import Candidate, ProblemRecord


DEBUG_EXPERTS = ["math_prm", "general_judge", "reflective_judge"]


def score_debug_candidate(record: ProblemRecord, candidate: Candidate) -> dict[str, float]:
    provider = candidate.metadata.get("provider", "")
    style = candidate.metadata.get("style", "")
    is_gold_derived = provider == "debug_gold_answer"
    is_reflective = style == "reflective"

    if record.domain == "math":
        math_score = 0.85 if is_gold_derived and not is_reflective else 0.38
        general_score = 0.70 if is_gold_derived else 0.42
    else:
        math_score = 0.62 if not is_gold_derived else 0.35
        general_score = 0.88 if is_gold_derived else 0.28

    reflective_score = 0.92 if is_reflective else (0.58 if is_gold_derived else 0.25)
    return {
        "math_prm": math_score,
        "general_judge": general_score,
        "reflective_judge": reflective_score,
    }


def score_debug_record(record: ProblemRecord) -> ProblemRecord:
    candidates = [
        Candidate(
            candidate_id=candidate.candidate_id,
            solution=candidate.solution,
            final_answer=candidate.final_answer,
            is_correct=candidate.is_correct,
            steps=candidate.steps,
            expert_scores=score_debug_candidate(record, candidate),
            normalized_scores=candidate.normalized_scores,
            metadata=candidate.metadata,
        )
        for candidate in record.candidates
    ]
    metadata = dict(record.metadata)
    metadata["expert_scoring"] = "debug_synthetic_not_for_main_results"
    return ProblemRecord(
        problem_id=record.problem_id,
        domain=record.domain,
        problem=record.problem,
        answer=record.answer,
        candidates=candidates,
        metadata=metadata,
    )

