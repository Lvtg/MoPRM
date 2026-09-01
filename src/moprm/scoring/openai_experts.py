from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Protocol

from moprm.answer_checking import has_explicit_final_answer
from moprm.openai_responses import extract_output_text, extract_usage
from moprm.schema import Candidate, ProblemRecord


OPENAI_EXPERTS = [
    "math_prm",
    "logic_judge",
    "general_judge",
    "reflective_judge",
]


class ResponsesClient(Protocol):
    def create_response(
        self,
        *,
        model: str,
        instructions: str,
        input_text: str,
        max_output_tokens: int,
        temperature: float | None = None,
    ) -> dict:
        ...


@dataclass(frozen=True)
class OpenAIExpertScoringConfig:
    model: str
    max_output_tokens: int = 256
    temperature: float | None = 0.0


SCORING_INSTRUCTIONS = """You score candidate reasoning traces for a PRM-style Best-of-N experiment.
You are not given the reference answer. Do not infer that any hidden reference answer exists.
Evaluate only from the problem statement and the candidate solution.

Return JSON only, with these four numeric scores in [0, 1]:
{
  "math_prm": <math-specialist confidence; for non-math tasks use about 0.50 unless quantitative reasoning is central>,
  "logic_judge": <logic-specialist confidence; for non-logic tasks use about 0.50 unless formal deduction/order reasoning is central>,
  "general_judge": <overall task-solving quality and final-answer reliability>,
  "reflective_judge": <self-checking/error-recovery quality; correct but non-reflective solutions should not automatically get 1.0>
}

If the candidate has no explicit final answer marker, penalize final-answer
reliability heavily. A long or truncated solution without a clear final answer
should usually have general_judge <= 0.40.
"""


def build_scoring_prompt(record: ProblemRecord, candidate: Candidate) -> str:
    explicit_final = has_explicit_final_answer(candidate.solution)
    return "\n".join(
        [
            f"Domain: {record.domain}",
            "",
            "Problem:",
            record.problem,
            "",
            "Candidate solution:",
            candidate.solution,
            "",
            f"Candidate final answer: {candidate.final_answer}",
            f"Candidate has explicit final answer marker: {'yes' if explicit_final else 'no'}",
            "",
            "Important:",
            "- Do not use or ask for a reference/gold answer.",
            "- Scores should reflect whether this candidate should be selected over alternatives.",
            "- Return only the JSON object.",
        ]
    )


def _first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found in scorer output: {text[:200]!r}")

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise ValueError(f"Unclosed JSON object in scorer output: {text[:200]!r}")


def _coerce_score(value: object) -> float:
    score = float(value)
    if 1.0 < score <= 100.0:
        score = score / 100.0
    return max(0.0, min(1.0, score))


def parse_score_json(text: str, expected_experts: list[str] | None = None) -> dict[str, float]:
    expected = expected_experts or OPENAI_EXPERTS
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        data = json.loads(_first_json_object(stripped))

    if isinstance(data, dict) and isinstance(data.get("scores"), dict):
        data = data["scores"]
    if not isinstance(data, dict):
        raise ValueError(f"Scorer output JSON is not an object: {text[:200]!r}")

    scores: dict[str, float] = {}
    missing: list[str] = []
    for expert in expected:
        if expert not in data:
            missing.append(expert)
            continue
        scores[expert] = _coerce_score(data[expert])

    if missing:
        raise ValueError(f"Scorer output missing score keys: {', '.join(missing)}")
    return scores


def score_candidate_with_openai(
    record: ProblemRecord,
    candidate: Candidate,
    client: ResponsesClient,
    config: OpenAIExpertScoringConfig,
) -> tuple[dict[str, float], dict]:
    payload = client.create_response(
        model=config.model,
        instructions=SCORING_INSTRUCTIONS,
        input_text=build_scoring_prompt(record, candidate),
        max_output_tokens=config.max_output_tokens,
        temperature=config.temperature,
    )
    text = extract_output_text(payload)
    scores = parse_score_json(text)
    metadata = {
        "provider": "openai_responses",
        "model": config.model,
        "temperature": config.temperature,
        "max_output_tokens": config.max_output_tokens,
    }
    response_id = payload.get("id")
    if isinstance(response_id, str):
        metadata["response_id"] = response_id
    usage = extract_usage(payload)
    if usage:
        metadata["usage"] = usage
    return scores, metadata


def score_record_with_openai(
    record: ProblemRecord,
    client: ResponsesClient,
    config: OpenAIExpertScoringConfig,
) -> ProblemRecord:
    candidates: list[Candidate] = []
    for candidate in record.candidates:
        scores, scoring_metadata = score_candidate_with_openai(
            record, candidate, client, config
        )
        expert_scores = dict(candidate.expert_scores)
        expert_scores.update(scores)
        metadata = dict(candidate.metadata)
        metadata["openai_expert_scoring"] = scoring_metadata
        candidates.append(
            Candidate(
                candidate_id=candidate.candidate_id,
                solution=candidate.solution,
                final_answer=candidate.final_answer,
                is_correct=candidate.is_correct,
                steps=candidate.steps,
                expert_scores=expert_scores,
                normalized_scores=candidate.normalized_scores,
                metadata=metadata,
            )
        )

    metadata = dict(record.metadata)
    metadata["expert_scoring"] = {
        "provider": "openai_responses",
        "model": config.model,
        "experts": OPENAI_EXPERTS,
        "uses_gold_answer": False,
    }
    return ProblemRecord(
        problem_id=record.problem_id,
        domain=record.domain,
        problem=record.problem,
        answer=record.answer,
        candidates=candidates,
        metadata=metadata,
    )
