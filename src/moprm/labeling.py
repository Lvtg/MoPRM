from __future__ import annotations

from copy import deepcopy
from typing import Any

from moprm.answer_checking import (
    check_answer,
    extract_final_answer,
    has_explicit_final_answer,
)


def label_candidate_correctness(record: dict[str, Any], overwrite: bool = False) -> dict[str, Any]:
    labeled = deepcopy(record)
    domain = labeled.get("domain")
    gold = str(labeled["answer"])
    for candidate in labeled.get("candidates", []):
        solution = str(candidate.get("solution", ""))
        explicit = True
        if solution:
            explicit = has_explicit_final_answer(solution)
            metadata = dict(candidate.get("metadata", {}))
            metadata["has_explicit_final_answer"] = explicit
            candidate["metadata"] = metadata
        if solution and (overwrite or not candidate.get("final_answer")):
            candidate["final_answer"] = extract_final_answer(solution)
        if candidate.get("is_correct") is not None and not overwrite:
            continue
        if not explicit:
            candidate["is_correct"] = False
            continue
        prediction = candidate.get("final_answer") or candidate.get("solution", "")
        candidate["is_correct"] = check_answer(str(prediction), gold, domain=domain)
    return labeled
