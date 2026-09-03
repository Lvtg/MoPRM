from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from moprm.schema import Candidate, ProblemRecord


SKYWORK_REWARD_V2_QWEN3_17B_MODEL = "Skywork/Skywork-Reward-V2-Qwen3-1.7B"
OPEN_REASONING_RM_EXPERT = "open_reasoning_rm"


def format_reward_conversation(problem: str, solution: str, tokenizer: Any) -> str:
    """Format a problem-solution pair as the Skywork Reward-V2 chat input.

    The Skywork Reward-V2 model card recommends using the chat template without
    a system prompt. The gold/reference answer is deliberately not included.
    """

    conversation = [
        {"role": "user", "content": problem},
        {"role": "assistant", "content": solution},
    ]
    formatted = tokenizer.apply_chat_template(conversation, tokenize=False)
    bos_token = getattr(tokenizer, "bos_token", None)
    if bos_token is not None and formatted.startswith(bos_token):
        formatted = formatted[len(bos_token) :]
    return formatted


@dataclass(frozen=True)
class SkyworkRewardV2Config:
    model_name: str = SKYWORK_REWARD_V2_QWEN3_17B_MODEL
    expert_name: str = OPEN_REASONING_RM_EXPERT
    cache_dir: str = "models/hf_cache"
    device: str = "auto"
    dtype: str = "auto"
    max_length: int = 4096


class SkyworkRewardV2Scorer:
    def __init__(self, config: SkyworkRewardV2Config) -> None:
        self.config = config
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._device = None
        self._resolved_dtype = None

    def load(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Skywork Reward-V2 scoring requires torch and transformers. "
                "Install them in the local CUDA environment before running this scorer."
            ) from exc

        cuda_available = torch.cuda.is_available()
        if self.config.device == "auto":
            device = "cuda" if cuda_available else "cpu"
        else:
            device = self.config.device

        if device == "cuda" and not cuda_available:
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

        if self.config.dtype == "auto":
            torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        elif self.config.dtype == "float32":
            torch_dtype = torch.float32
        elif self.config.dtype == "float16":
            torch_dtype = torch.float16
        elif self.config.dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        else:
            raise ValueError(f"Unsupported dtype: {self.config.dtype}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir,
            dtype=torch_dtype,
            low_cpu_mem_usage=True,
            num_labels=1,
        ).eval()
        self._model.to(device)
        self._torch = torch
        self._device = device
        self._resolved_dtype = str(torch_dtype).replace("torch.", "")

    def score_text(self, problem: str, solution: str) -> tuple[float, dict[str, Any]]:
        self.load()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None
        assert self._device is not None

        formatted = format_reward_conversation(problem, solution, self._tokenizer)
        inputs = self._tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        ).to(self._device)

        with self._torch.no_grad():
            outputs = self._model(**inputs)
            score = float(outputs.logits[0][0].detach().cpu().item())

        if not math.isfinite(score):
            raise ValueError(
                "Skywork Reward-V2 produced a non-finite reward score. "
                "Try --dtype float32 or --device cpu for a numerically stable run."
            )

        metadata = {
            "provider": "skywork_hf",
            "model": self.config.model_name,
            "device": self._device,
            "dtype": self.config.dtype,
            "resolved_dtype": self._resolved_dtype,
            "max_length": self.config.max_length,
            "score_type": "sequence_reward_logit",
            "uses_gold_answer": False,
        }
        return score, metadata


def score_record_with_skywork_reward_v2(
    record: ProblemRecord,
    scorer: SkyworkRewardV2Scorer,
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
