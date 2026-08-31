from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    solution: str
    final_answer: str | None = None
    is_correct: bool | None = None
    steps: list[str] = field(default_factory=list)
    expert_scores: dict[str, float] = field(default_factory=dict)
    normalized_scores: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Candidate":
        return cls(
            candidate_id=str(data["candidate_id"]),
            solution=str(data.get("solution", "")),
            final_answer=data.get("final_answer"),
            is_correct=data.get("is_correct"),
            steps=list(data.get("steps", [])),
            expert_scores={k: float(v) for k, v in data.get("expert_scores", {}).items()},
            normalized_scores={
                k: float(v) for k, v in data.get("normalized_scores", {}).items()
            },
        )


@dataclass(frozen=True)
class ProblemRecord:
    problem_id: str
    domain: str
    problem: str
    answer: str
    candidates: list[Candidate]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProblemRecord":
        return cls(
            problem_id=str(data["problem_id"]),
            domain=str(data.get("domain", "unknown")),
            problem=str(data["problem"]),
            answer=str(data["answer"]),
            candidates=[Candidate.from_dict(item) for item in data.get("candidates", [])],
            metadata=dict(data.get("metadata", {})),
        )

    def expert_names(self) -> list[str]:
        names: set[str] = set()
        for candidate in self.candidates:
            names.update(candidate.expert_scores)
            names.update(candidate.normalized_scores)
        return sorted(names)

