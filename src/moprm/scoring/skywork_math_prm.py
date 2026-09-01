from __future__ import annotations

from dataclasses import dataclass
from statistics import geometric_mean
from typing import Any

from moprm.schema import Candidate, ProblemRecord


SKYWORK_MATH_PRM_MODEL = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
OPEN_MATH_PRM_EXPERT = "open_math_prm"


def split_solution_steps(solution: str, max_steps: int | None = None) -> list[str]:
    """Split a generated solution into coarse PRM steps.

    Skywork's official example uses newline as the step separator. Our generated
    candidates are not guaranteed to be carefully step-delimited, so we use
    non-empty lines first and fall back to sentence-like chunks.
    """

    lines = [line.strip() for line in solution.replace("\r\n", "\n").split("\n")]
    steps = [line for line in lines if line]
    if len(steps) <= 1:
        rough = solution.replace("\n", " ").split(". ")
        steps = [chunk.strip() for chunk in rough if chunk.strip()]
    if not steps:
        steps = [solution.strip()] if solution.strip() else []
    if max_steps is not None and max_steps > 0:
        steps = steps[:max_steps]
    return steps


def aggregate_step_rewards(step_rewards: list[float], method: str = "mean") -> float:
    if not step_rewards:
        return 0.0
    if method == "mean":
        return sum(step_rewards) / len(step_rewards)
    if method == "min":
        return min(step_rewards)
    if method == "last":
        return step_rewards[-1]
    if method == "geomean":
        clipped = [max(1e-6, min(1.0, reward)) for reward in step_rewards]
        return float(geometric_mean(clipped))
    raise ValueError(f"Unknown aggregation method: {method}")


def prepare_skywork_input(
    problem: str,
    response: str,
    tokenizer: Any,
    *,
    step_token: str = "\n",
) -> tuple[list[int], list[str], list[int]]:
    """Prepare PRM input following Skywork's public inference example."""

    bos = tokenizer.bos_token or ""
    prompt_ids = tokenizer.encode(bos + problem + "\n")
    reward_flags = [0] * len(prompt_ids)
    response_ids: list[int] = []
    steps: list[str] = []
    step_token_id = tokenizer.encode(step_token)[-1]

    for step in response.split(step_token):
        if step != "":
            step_ids = tokenizer.encode(step)
        else:
            step_ids = []
        step_ids += [step_token_id]
        response_ids.extend(step_ids)
        reward_flags.extend([0] * (len(step_ids) - 1) + [1])
        steps.append(step + step_token)

    return prompt_ids + response_ids, steps, reward_flags


@dataclass(frozen=True)
class SkyworkMathPRMConfig:
    model_name: str = SKYWORK_MATH_PRM_MODEL
    expert_name: str = OPEN_MATH_PRM_EXPERT
    cache_dir: str = "models/hf_cache"
    device: str = "auto"
    max_length: int = 4096
    max_steps: int | None = 32
    step_token: str = "\n"
    aggregation: str = "mean"


class SkyworkMathPRMScorer:
    def __init__(self, config: SkyworkMathPRMConfig) -> None:
        self.config = config
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._device = None

    def load(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Skywork math PRM scoring requires torch and transformers. "
                "Install them in the local environment before running this scorer."
            ) from exc

        cuda_available = torch.cuda.is_available()
        if self.config.device == "auto":
            device = "cuda" if cuda_available else "cpu"
        else:
            device = self.config.device

        if device == "cuda" and not cuda_available:
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

        torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            cache_dir=self.config.cache_dir,
        )
        self._model = AutoModel.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            cache_dir=self.config.cache_dir,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        ).eval()
        self._model.to(device)
        self._torch = torch
        self._device = device

    def score_text(self, problem: str, solution: str) -> tuple[float, dict[str, Any]]:
        self.load()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None
        assert self._device is not None

        steps = split_solution_steps(solution, max_steps=self.config.max_steps)
        response = self.config.step_token.join(steps)
        input_ids, prepared_steps, reward_flags = prepare_skywork_input(
            problem,
            response,
            self._tokenizer,
            step_token=self.config.step_token,
        )
        if len(input_ids) > self.config.max_length:
            input_ids = input_ids[-self.config.max_length :]
            reward_flags = reward_flags[-self.config.max_length :]

        torch = self._torch
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self._device)
        attention_mask = torch.ones_like(input_tensor, device=self._device)
        reward_flag_tensor = torch.tensor(
            [reward_flags],
            dtype=torch.long,
            device=self._device,
        )

        with torch.no_grad():
            base_outputs = self._model.model(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                return_dict=True,
            )
            hidden_states = base_outputs.last_hidden_state
            logits = self._model.v_head(hidden_states).squeeze(-1)
            rewards = torch.sigmoid(logits)

        reward_indices = torch.nonzero(reward_flag_tensor[0] == 1).view(-1)
        step_rewards = [
            float(rewards[0, index].detach().cpu().item()) for index in reward_indices
        ]
        score = aggregate_step_rewards(step_rewards, method=self.config.aggregation)
        metadata = {
            "provider": "skywork_hf",
            "model": self.config.model_name,
            "device": self._device,
            "aggregation": self.config.aggregation,
            "num_steps": len(prepared_steps),
            "num_reward_points": len(step_rewards),
            "step_rewards": step_rewards,
        }
        return score, metadata


def score_record_with_skywork_math_prm(
    record: ProblemRecord,
    scorer: SkyworkMathPRMScorer,
) -> ProblemRecord:
    candidates: list[Candidate] = []
    for candidate in record.candidates:
        score, scoring_metadata = scorer.score_text(record.problem, candidate.solution)
        expert_scores = dict(candidate.expert_scores)
        expert_scores[scorer.config.expert_name] = score
        metadata = dict(candidate.metadata)
        metadata[scorer.config.expert_name] = scoring_metadata
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
    open_experts = list(metadata.get("open_source_experts", []))
    if scorer.config.expert_name not in open_experts:
        open_experts.append(scorer.config.expert_name)
    metadata["open_source_experts"] = open_experts

    return ProblemRecord(
        problem_id=record.problem_id,
        domain=record.domain,
        problem=record.problem,
        answer=record.answer,
        candidates=candidates,
        metadata=metadata,
    )
