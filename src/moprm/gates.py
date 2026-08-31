from __future__ import annotations

from moprm.schema import ProblemRecord


def uniform_gate(expert_names: list[str]) -> dict[str, float]:
    if not expert_names:
        return {}
    weight = 1.0 / len(expert_names)
    return {expert: weight for expert in expert_names}


def domain_rule_gate(record: ProblemRecord, expert_names: list[str]) -> dict[str, float]:
    if not expert_names:
        return {}

    preferred_by_domain = {
        "math": ["math_prm", "reflective_judge", "general_judge"],
        "logic": ["general_judge", "logic_judge", "reflective_judge"],
        "code": ["code_verifier", "general_judge"],
    }
    preferred = preferred_by_domain.get(record.domain, [])
    for expert in preferred:
        if expert in expert_names:
            return {name: 1.0 if name == expert else 0.0 for name in expert_names}
    return uniform_gate(expert_names)


def oracle_gate(record: ProblemRecord, expert_names: list[str]) -> dict[str, float]:
    successful: list[str] = []
    for expert in expert_names:
        scored = [
            (candidate.normalized_scores.get(expert, candidate.expert_scores.get(expert)), candidate)
            for candidate in record.candidates
        ]
        scored = [(score, candidate) for score, candidate in scored if score is not None]
        if not scored:
            continue
        selected = max(scored, key=lambda item: (item[0], item[1].candidate_id))[1]
        if selected.is_correct:
            successful.append(expert)

    if not successful:
        return uniform_gate(expert_names)
    weight = 1.0 / len(successful)
    return {expert: weight if expert in successful else 0.0 for expert in expert_names}
