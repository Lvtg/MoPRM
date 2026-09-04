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
        "math": [
            "open_math_prm_min",
            "open_math_prm_last",
            "open_math_prm_geomean",
            "open_math_prm_mean",
            "open_math_prm",
            "openai_math_rubric",
            "math_prm",
            "openai_reflective_judge",
            "reflective_judge",
            "openai_general_judge",
            "general_judge",
        ],
        "logic": [
            "open_reasoning_rm",
            "open_logic_prm",
            "openai_logic_rubric",
            "logic_judge",
            "openai_general_judge",
            "general_judge",
            "openai_reflective_judge",
            "reflective_judge",
        ],
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


def metadata_gate(gate_name: str):
    def gate(record: ProblemRecord, expert_names: list[str]) -> dict[str, float]:
        gate_weights = record.metadata.get("gate_weights", {})
        if not isinstance(gate_weights, dict):
            return uniform_gate(expert_names)
        weights = gate_weights.get(gate_name, {})
        if not isinstance(weights, dict):
            return uniform_gate(expert_names)

        cleaned = {
            expert: max(0.0, float(weights.get(expert, 0.0)))
            for expert in expert_names
        }
        total = sum(cleaned.values())
        if total <= 0:
            return uniform_gate(expert_names)
        return {expert: weight / total for expert, weight in cleaned.items()}

    return gate
