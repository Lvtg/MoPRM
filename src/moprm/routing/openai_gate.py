from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Protocol

from moprm.openai_responses import extract_output_text, extract_usage
from moprm.schema import ProblemRecord


DEFAULT_GATE_NAME = "openai_llm_gate"


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
class OpenAIGateConfig:
    model: str
    gate_name: str = DEFAULT_GATE_NAME
    max_output_tokens: int = 256
    temperature: float | None = 0.0


EXPERT_DESCRIPTIONS = {
    "open_math_prm": "Non-OpenAI open-source math PRM for mathematical process correctness.",
    "open_logic_prm": "Non-OpenAI open-source logic/reasoning PRM or reward model.",
    "math_prm": "Best for arithmetic, symbolic math, equations, and quantitative reasoning.",
    "logic_judge": "Best for formal deduction, ordering, relations, and multiple-choice logic.",
    "openai_general_judge": "OpenAI-based broad reasoning-quality judge across domains.",
    "openai_reflective_judge": "OpenAI-based judge for self-checking and error recovery.",
    "general_judge": "Broad reasoning-quality judge across domains.",
    "reflective_judge": "Best when self-checking, error recovery, or robust verification is important.",
}


ROUTING_INSTRUCTIONS = """You are the router/gate for a Mixture-of-PRMs system.
Given a problem and available expert scorers, assign nonnegative routing weights.
This is question-level routing: you do not see candidate solutions and you are not given a gold answer.

Return JSON only:
{
  "weights": {
    "<expert_name>": <nonnegative number>
  }
}
"""


def build_gate_prompt(record: ProblemRecord, expert_names: list[str]) -> str:
    expert_lines = [
        f"- {expert}: {EXPERT_DESCRIPTIONS.get(expert, 'Available PRM expert.')}"
        for expert in expert_names
    ]
    return "\n".join(
        [
            f"Domain hint: {record.domain}",
            "",
            "Problem:",
            record.problem,
            "",
            "Available experts:",
            *expert_lines,
            "",
            "Routing objective:",
            "- Put most weight on experts likely to rank candidate solutions correctly.",
            "- Use a mixture when the problem benefits from multiple signals.",
            "- Do not use or ask for a reference/gold answer.",
            "- Return only JSON.",
        ]
    )


def _first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found in gate output: {text[:200]!r}")

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

    raise ValueError(f"Unclosed JSON object in gate output: {text[:200]!r}")


def parse_gate_weights(text: str, expert_names: list[str]) -> dict[str, float]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        data = json.loads(_first_json_object(stripped))

    if isinstance(data, dict) and isinstance(data.get("weights"), dict):
        data = data["weights"]
    if not isinstance(data, dict):
        raise ValueError(f"Gate output JSON is not an object: {text[:200]!r}")

    weights: dict[str, float] = {}
    for expert in expert_names:
        try:
            weights[expert] = max(0.0, float(data.get(expert, 0.0)))
        except (TypeError, ValueError):
            weights[expert] = 0.0

    total = sum(weights.values())
    if total <= 0:
        uniform = 1.0 / len(expert_names) if expert_names else 0.0
        return {expert: uniform for expert in expert_names}
    return {expert: weight / total for expert, weight in weights.items()}


def route_record_with_openai(
    record: ProblemRecord,
    client: ResponsesClient,
    config: OpenAIGateConfig,
) -> ProblemRecord:
    expert_names = record.expert_names()
    payload = client.create_response(
        model=config.model,
        instructions=ROUTING_INSTRUCTIONS,
        input_text=build_gate_prompt(record, expert_names),
        max_output_tokens=config.max_output_tokens,
        temperature=config.temperature,
    )
    text = extract_output_text(payload)
    weights = parse_gate_weights(text, expert_names)

    metadata = dict(record.metadata)
    gate_weights = dict(metadata.get("gate_weights", {}))
    gate_weights[config.gate_name] = weights
    metadata["gate_weights"] = gate_weights

    gate_metadata = dict(metadata.get("gate_metadata", {}))
    scorer_metadata = {
        "provider": "openai_responses",
        "model": config.model,
        "temperature": config.temperature,
        "max_output_tokens": config.max_output_tokens,
        "uses_gold_answer": False,
    }
    response_id = payload.get("id")
    if isinstance(response_id, str):
        scorer_metadata["response_id"] = response_id
    usage = extract_usage(payload)
    if usage:
        scorer_metadata["usage"] = usage
    gate_metadata[config.gate_name] = scorer_metadata
    metadata["gate_metadata"] = gate_metadata

    return ProblemRecord(
        problem_id=record.problem_id,
        domain=record.domain,
        problem=record.problem,
        answer=record.answer,
        candidates=record.candidates,
        metadata=metadata,
    )
