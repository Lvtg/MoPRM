from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from moprm.answer_checking import extract_final_answer, has_explicit_final_answer
from moprm.openai_responses import extract_output_text, extract_usage
from moprm.schema import Candidate, ProblemRecord


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
class OpenAICandidateConfig:
    model: str
    num_candidates: int = 2
    max_output_tokens: int = 512
    temperature: float | None = 0.7


GENERATION_INSTRUCTIONS = """You generate candidate solutions for PRM/BoN experiments.
Do not mention hidden labels or gold answers. Solve the problem independently.
Use concise step-by-step reasoning.
End with exactly one final line in this format:
Final answer: <answer>
"""


def build_generation_prompt(record: ProblemRecord, candidate_index: int) -> str:
    domain_hint = {
        "math": "This is a math reasoning problem. The final answer should be compact.",
        "logic": (
            "This is a logic reasoning problem. If options are present, the final "
            "answer should be the selected option letter."
        ),
    }.get(record.domain, "This is a reasoning problem.")
    style = [
        "Use a direct concise solution path.",
        "Use a slightly more detailed solution path with a quick sanity check.",
        "Use an alternative reasoning path if one is natural.",
        "Use a fast scratchpad-style solution path while still ending cleanly.",
    ][candidate_index % 4]

    return "\n".join(
        [
            f"Domain: {record.domain}",
            domain_hint,
            f"Candidate index: {candidate_index}",
            f"Candidate style: {style}",
            "",
            "Problem:",
            record.problem,
            "",
            "Requirements:",
            "- Solve without using any reference answer.",
            "- Keep the solution under 8 sentences when possible.",
            "- Keep the solution concise but show enough reasoning for a reward model to judge it.",
            "- The last line must be exactly: Final answer: <answer>",
        ]
    )


def generate_candidate(
    record: ProblemRecord,
    client: ResponsesClient,
    config: OpenAICandidateConfig,
    candidate_index: int,
) -> Candidate:
    payload = client.create_response(
        model=config.model,
        instructions=GENERATION_INSTRUCTIONS,
        input_text=build_generation_prompt(record, candidate_index),
        max_output_tokens=config.max_output_tokens,
        temperature=config.temperature,
    )
    solution = extract_output_text(payload)
    usage = extract_usage(payload)
    response_id = payload.get("id")

    metadata = {
        "provider": "openai_responses",
        "model": config.model,
        "temperature": config.temperature,
        "max_output_tokens": config.max_output_tokens,
        "generation_index": candidate_index,
        "uses_gold_answer": False,
    }
    if isinstance(response_id, str):
        metadata["response_id"] = response_id
    if usage:
        metadata["usage"] = usage

    return Candidate(
        candidate_id=f"{record.problem_id}_openai_{candidate_index:03d}",
        solution=solution,
        final_answer=extract_final_answer(solution),
        metadata={**metadata, "has_explicit_final_answer": has_explicit_final_answer(solution)},
    )


def generate_openai_candidates(
    record: ProblemRecord,
    client: ResponsesClient,
    config: OpenAICandidateConfig,
) -> ProblemRecord:
    if config.num_candidates < 1:
        raise ValueError("num_candidates must be >= 1")

    candidates = [
        generate_candidate(record, client, config, index)
        for index in range(config.num_candidates)
    ]
    metadata = dict(record.metadata)
    metadata["candidate_generation"] = {
        "provider": "openai_responses",
        "model": config.model,
        "num_candidates": config.num_candidates,
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
