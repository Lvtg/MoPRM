from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from moprm.gates import domain_rule_gate, oracle_gate, uniform_gate
from moprm.normalization import normalize_expert_scores
from moprm.schema import Candidate, ProblemRecord


GateFn = Callable[[ProblemRecord, list[str]], dict[str, float]]


@dataclass(frozen=True)
class EvaluationResult:
    method: str
    total: int
    correct: int
    accuracy: float
    selections: dict[str, str]


def attach_normalized_scores(record: ProblemRecord, method: str = "rank") -> ProblemRecord:
    expert_names = record.expert_names()
    raw_scores = {
        candidate.candidate_id: candidate.expert_scores for candidate in record.candidates
    }
    normalized = normalize_expert_scores(raw_scores, expert_names, method=method)
    candidates = [
        Candidate(
            candidate_id=candidate.candidate_id,
            solution=candidate.solution,
            final_answer=candidate.final_answer,
            is_correct=candidate.is_correct,
            steps=candidate.steps,
            expert_scores=candidate.expert_scores,
            normalized_scores=normalized.get(candidate.candidate_id, {}),
        )
        for candidate in record.candidates
    ]
    return ProblemRecord(
        problem_id=record.problem_id,
        domain=record.domain,
        problem=record.problem,
        answer=record.answer,
        candidates=candidates,
        metadata=record.metadata,
    )


def score_candidate(candidate: Candidate, weights: dict[str, float], use_normalized: bool = True) -> float:
    total = 0.0
    used_weight = 0.0
    source = candidate.normalized_scores if use_normalized else candidate.expert_scores
    for expert, weight in weights.items():
        if expert not in source:
            continue
        total += weight * source[expert]
        used_weight += weight
    if used_weight == 0:
        return float("-inf")
    return total / used_weight


def select_candidate(
    record: ProblemRecord,
    weights: dict[str, float],
    use_normalized: bool = True,
) -> Candidate:
    if not record.candidates:
        raise ValueError(f"Problem {record.problem_id} has no candidates")
    return max(
        record.candidates,
        key=lambda candidate: (
            score_candidate(candidate, weights, use_normalized=use_normalized),
            candidate.candidate_id,
        ),
    )


def single_expert_gate(expert_name: str) -> GateFn:
    def gate(_: ProblemRecord, expert_names: list[str]) -> dict[str, float]:
        return {name: 1.0 if name == expert_name else 0.0 for name in expert_names}

    return gate


def evaluate_records(
    records: list[ProblemRecord],
    method: str,
    gate: GateFn,
    normalization: str = "rank",
    use_normalized: bool = True,
) -> EvaluationResult:
    total = 0
    correct = 0
    selections: dict[str, str] = {}
    for record in records:
        normalized_record = attach_normalized_scores(record, method=normalization)
        expert_names = normalized_record.expert_names()
        weights = gate(normalized_record, expert_names)
        selected = select_candidate(normalized_record, weights, use_normalized=use_normalized)
        total += 1
        correct += 1 if selected.is_correct else 0
        selections[normalized_record.problem_id] = selected.candidate_id
    accuracy = correct / total if total else 0.0
    return EvaluationResult(method=method, total=total, correct=correct, accuracy=accuracy, selections=selections)


def default_baselines(records: list[ProblemRecord], normalization: str = "rank") -> list[EvaluationResult]:
    expert_names = sorted({expert for record in records for expert in record.expert_names()})
    results = [
        evaluate_records(records, "uniform_ensemble", lambda _r, names: uniform_gate(names), normalization),
        evaluate_records(records, "domain_rule_gate", domain_rule_gate, normalization),
        evaluate_records(records, "oracle_gate", oracle_gate, normalization),
    ]
    for expert in expert_names:
        results.append(evaluate_records(records, f"single:{expert}", single_expert_gate(expert), normalization))
    return results

