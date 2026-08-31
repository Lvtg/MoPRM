from __future__ import annotations

from copy import deepcopy
from typing import Any

from moprm.answer_checking import check_answer


def label_candidate_correctness(record: dict[str, Any], overwrite: bool = False) -> dict[str, Any]:
    labeled = deepcopy(record)
    domain = labeled.get("domain")
    gold = str(labeled["answer"])
    for candidate in labeled.get("candidates", []):
        if candidate.get("is_correct") is not None and not overwrite:
            continue
        prediction = candidate.get("final_answer") or candidate.get("solution", "")
        candidate["is_correct"] = check_answer(str(prediction), gold, domain=domain)
    return labeled

