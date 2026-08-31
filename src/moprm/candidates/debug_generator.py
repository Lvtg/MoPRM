from __future__ import annotations

import re

from moprm.schema import Candidate, ProblemRecord


def make_wrong_answer(gold: str, index: int, domain: str) -> str:
    value = str(gold).strip()
    if domain == "logic":
        choices = ["A", "B", "C", "D", "E"]
        current = value.upper() if value.upper() in choices else "A"
        offset = (choices.index(current) + index) % len(choices)
        if choices[offset] == current:
            offset = (offset + 1) % len(choices)
        return choices[offset]

    numeric = re.fullmatch(r"[-+]?\d+", value.replace(",", ""))
    if numeric:
        return str(int(value.replace(",", "")) + index)

    frac = re.fullmatch(r"\\frac\{([-+]?\d+)\}\{(\d+)\}", value)
    if frac:
        return f"\\frac{{{int(frac.group(1)) + index}}}{{{frac.group(2)}}}"

    return f"{value}_incorrect_{index}"


def generate_debug_candidates(record: ProblemRecord, n: int = 4) -> ProblemRecord:
    if n < 2:
        raise ValueError("debug candidate generation needs n >= 2")

    candidates = [
        Candidate(
            candidate_id=f"{record.problem_id}_c000",
            solution=(
                "This is a pipeline-debug candidate derived from the gold answer. "
                f"Final answer: {record.answer}"
            ),
            final_answer=record.answer,
            metadata={"provider": "debug_gold_answer", "uses_gold_answer": True},
        )
    ]
    if n >= 3:
        candidates.append(
            Candidate(
                candidate_id=f"{record.problem_id}_c001",
                solution=(
                    "I first make an incorrect attempt, then explicitly correct it. "
                    f"After checking the reasoning, final answer: {record.answer}"
                ),
                final_answer=record.answer,
                metadata={
                    "provider": "debug_gold_answer",
                    "uses_gold_answer": True,
                    "style": "reflective",
                },
            )
        )

    start = len(candidates)
    for idx in range(start, n):
        wrong = make_wrong_answer(record.answer, idx, record.domain)
        candidates.append(
            Candidate(
                candidate_id=f"{record.problem_id}_c{idx:03d}",
                solution=(
                    "This is a pipeline-debug incorrect candidate. "
                    f"Final answer: {wrong}"
                ),
                final_answer=wrong,
                metadata={"provider": "debug_wrong_answer", "uses_gold_answer": True},
            )
        )

    metadata = dict(record.metadata)
    metadata["candidate_generation"] = "debug_uses_gold_answer_not_for_main_results"
    return ProblemRecord(
        problem_id=record.problem_id,
        domain=record.domain,
        problem=record.problem,
        answer=record.answer,
        candidates=candidates,
        metadata=metadata,
    )

