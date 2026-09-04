from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

import numpy as np

from moprm.evaluate import EvaluationResult, attach_normalized_scores
from moprm.schema import Candidate, ProblemRecord
from moprm.trained_gate import Standardizer, sigmoid


CANDIDATE_GLOBAL_FEATURES = (
    "solution_char_count_log",
    "solution_word_count_log",
    "solution_line_count_log",
    "step_count_log",
    "has_final_answer",
    "has_boxed_answer",
    "mean_norm_score",
    "std_norm_score",
    "max_norm_score",
    "min_norm_score",
    "range_norm_score",
    "top_expert_count",
    "top_expert_frac",
    "open_source_norm_mean",
    "openai_norm_mean",
)
CANDIDATE_EXPERT_FEATURES = (
    "raw_score",
    "norm_score",
    "raw_minus_mean",
    "norm_minus_mean",
    "raw_z",
    "norm_z",
    "is_expert_top",
    "norm_gap_to_expert_top",
    "raw_gap_to_expert_top",
    "expert_norm_top_gap",
)
WORD_RE = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?")


@dataclass(frozen=True)
class CandidateFeatureConfig:
    expert_names: tuple[str, ...]
    normalization: str = "rank"
    global_features: tuple[str, ...] = CANDIDATE_GLOBAL_FEATURES
    expert_features: tuple[str, ...] = CANDIDATE_EXPERT_FEATURES

    @property
    def dim(self) -> int:
        return len(self.global_features) + len(self.expert_names) * len(self.expert_features)


@dataclass(frozen=True)
class CandidateGateModel:
    expert_names: tuple[str, ...]
    feature_config: CandidateFeatureConfig
    standardizer: Standardizer
    weights: np.ndarray
    bias: float

    def predict_candidate_scores(self, records: list[ProblemRecord]) -> list[dict[str, float]]:
        features, keys, _targets = candidate_feature_matrix(records, self.feature_config)
        if len(keys) == 0:
            return []
        probabilities = sigmoid(self.standardizer.transform(features) @ self.weights + self.bias)
        grouped: dict[str, dict[str, float]] = {}
        for (problem_id, candidate_id), probability in zip(keys, probabilities, strict=True):
            grouped.setdefault(problem_id, {})[candidate_id] = float(probability)
        return [grouped.get(record.problem_id, {}) for record in records]


def fit_candidate_feature_config(
    expert_names: tuple[str, ...] | list[str],
    *,
    normalization: str = "rank",
) -> CandidateFeatureConfig:
    expert_tuple = tuple(expert_names)
    if not expert_tuple:
        raise ValueError("expert_names must not be empty")
    return CandidateFeatureConfig(expert_names=expert_tuple, normalization=normalization)


def candidate_feature_matrix(
    records: list[ProblemRecord],
    config: CandidateFeatureConfig,
) -> tuple[np.ndarray, list[tuple[str, str]], np.ndarray]:
    rows: list[np.ndarray] = []
    keys: list[tuple[str, str]] = []
    targets: list[float] = []
    for record in records:
        normalized = attach_normalized_scores(record, method=config.normalization)
        record_rows = _candidate_feature_rows(normalized, config)
        for candidate, row in zip(normalized.candidates, record_rows, strict=True):
            if candidate.is_correct is None:
                continue
            rows.append(row)
            keys.append((normalized.problem_id, candidate.candidate_id))
            targets.append(1.0 if candidate.is_correct else 0.0)
    if not rows:
        return (
            np.zeros((0, config.dim), dtype=np.float64),
            [],
            np.zeros((0,), dtype=np.float64),
        )
    return (
        np.vstack(rows).astype(np.float64),
        keys,
        np.array(targets, dtype=np.float64),
    )


def train_candidate_gate(
    records: list[ProblemRecord],
    expert_names: tuple[str, ...] | list[str],
    *,
    normalization: str = "rank",
    epochs: int = 800,
    lr: float = 0.05,
    l2: float = 0.01,
    seed: int = 13,
) -> CandidateGateModel:
    if not records:
        raise ValueError("Cannot train a candidate gate on an empty record set")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    expert_tuple = tuple(expert_names)
    config = fit_candidate_feature_config(expert_tuple, normalization=normalization)
    features, _keys, targets = candidate_feature_matrix(records, config)
    if len(targets) == 0:
        raise ValueError("No candidate correctness labels available for training")

    standardizer = Standardizer.fit(features)
    features = standardizer.transform(features)
    rng = np.random.default_rng(seed)
    weights = rng.normal(loc=0.0, scale=0.01, size=(features.shape[1],))
    base_rate = float(np.clip(targets.mean(), 1e-3, 1.0 - 1e-3))
    bias = math.log(base_rate / (1.0 - base_rate))

    adam_m_w = np.zeros_like(weights)
    adam_v_w = np.zeros_like(weights)
    adam_m_b = 0.0
    adam_v_b = 0.0
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    n = float(len(targets))

    for step in range(1, epochs + 1):
        logits = features @ weights + bias
        probabilities = sigmoid(logits)
        grad_logits = (probabilities - targets) / n
        grad_w = features.T @ grad_logits + l2 * weights
        grad_b = float(grad_logits.sum())

        adam_m_w = beta1 * adam_m_w + (1.0 - beta1) * grad_w
        adam_v_w = beta2 * adam_v_w + (1.0 - beta2) * np.square(grad_w)
        adam_m_b = beta1 * adam_m_b + (1.0 - beta1) * grad_b
        adam_v_b = beta2 * adam_v_b + (1.0 - beta2) * grad_b * grad_b

        corrected_m_w = adam_m_w / (1.0 - beta1**step)
        corrected_v_w = adam_v_w / (1.0 - beta2**step)
        corrected_m_b = adam_m_b / (1.0 - beta1**step)
        corrected_v_b = adam_v_b / (1.0 - beta2**step)

        weights -= lr * corrected_m_w / (np.sqrt(corrected_v_w) + epsilon)
        bias -= lr * corrected_m_b / (math.sqrt(corrected_v_b) + epsilon)

    return CandidateGateModel(
        expert_names=expert_tuple,
        feature_config=config,
        standardizer=standardizer,
        weights=weights,
        bias=float(bias),
    )


def add_candidate_gate_scores_to_record(
    record: ProblemRecord,
    *,
    gate_name: str,
    scores: dict[str, float],
    gate_metadata: dict[str, Any] | None = None,
) -> ProblemRecord:
    metadata = dict(record.metadata)
    candidate_gate_scores = metadata.get("candidate_gate_scores", {})
    if not isinstance(candidate_gate_scores, dict):
        candidate_gate_scores = {}
    candidate_gate_scores = dict(candidate_gate_scores)
    candidate_gate_scores[gate_name] = {key: float(value) for key, value in scores.items()}
    metadata["candidate_gate_scores"] = candidate_gate_scores

    if gate_metadata is not None:
        existing_metadata = metadata.get("candidate_gate_metadata", {})
        if not isinstance(existing_metadata, dict):
            existing_metadata = {}
        existing_metadata = dict(existing_metadata)
        existing_metadata[gate_name] = dict(gate_metadata)
        metadata["candidate_gate_metadata"] = existing_metadata

    return ProblemRecord(
        problem_id=record.problem_id,
        domain=record.domain,
        problem=record.problem,
        answer=record.answer,
        candidates=record.candidates,
        metadata=metadata,
    )


def evaluate_candidate_gate_records(
    records: list[ProblemRecord],
    *,
    gate_name: str,
) -> EvaluationResult:
    total = 0
    correct = 0
    selections: dict[str, str] = {}
    for record in records:
        candidate_gate_scores = record.metadata.get("candidate_gate_scores", {})
        if not isinstance(candidate_gate_scores, dict):
            continue
        scores = candidate_gate_scores.get(gate_name, {})
        if not isinstance(scores, dict):
            continue
        scored_candidates = [
            (float(scores.get(candidate.candidate_id, float("-inf"))), candidate.candidate_id, candidate)
            for candidate in record.candidates
        ]
        if not scored_candidates:
            continue
        selected = max(scored_candidates, key=lambda item: (item[0], item[1]))[2]
        total += 1
        correct += 1 if selected.is_correct else 0
        selections[record.problem_id] = selected.candidate_id
    return EvaluationResult(
        method=f"candidate_gate:{gate_name}",
        total=total,
        correct=correct,
        accuracy=correct / total if total else 0.0,
        selections=selections,
    )


def _candidate_feature_rows(
    record: ProblemRecord,
    config: CandidateFeatureConfig,
) -> list[np.ndarray]:
    expert_names = config.expert_names
    raw_matrix = _score_matrix(record, expert_names, source="raw")
    norm_matrix = _score_matrix(record, expert_names, source="normalized")
    raw_mean = raw_matrix.mean(axis=0)
    norm_mean = norm_matrix.mean(axis=0)
    raw_std = np.where(raw_matrix.std(axis=0) < 1e-8, 1.0, raw_matrix.std(axis=0))
    norm_std = np.where(norm_matrix.std(axis=0) < 1e-8, 1.0, norm_matrix.std(axis=0))
    raw_top = raw_matrix.max(axis=0)
    norm_top = norm_matrix.max(axis=0)
    norm_sorted = np.sort(norm_matrix, axis=0)
    expert_top_gap = norm_sorted[-1, :] - (norm_sorted[-2, :] if len(record.candidates) > 1 else norm_sorted[-1, :])
    is_top = np.isclose(norm_matrix, norm_top.reshape(1, -1), atol=1e-12)

    rows: list[np.ndarray] = []
    for candidate_index, candidate in enumerate(record.candidates):
        norm_values = norm_matrix[candidate_index, :]
        raw_values = raw_matrix[candidate_index, :]
        top_count = float(is_top[candidate_index, :].sum())
        global_values = _candidate_global_features(
            candidate,
            norm_values=norm_values,
            top_count=top_count,
            expert_names=expert_names,
        )
        values: list[float] = [global_values[name] for name in config.global_features]
        for expert_index, _expert in enumerate(expert_names):
            feature_values = {
                "raw_score": float(raw_values[expert_index]),
                "norm_score": float(norm_values[expert_index]),
                "raw_minus_mean": float(raw_values[expert_index] - raw_mean[expert_index]),
                "norm_minus_mean": float(norm_values[expert_index] - norm_mean[expert_index]),
                "raw_z": float((raw_values[expert_index] - raw_mean[expert_index]) / raw_std[expert_index]),
                "norm_z": float((norm_values[expert_index] - norm_mean[expert_index]) / norm_std[expert_index]),
                "is_expert_top": 1.0 if is_top[candidate_index, expert_index] else 0.0,
                "norm_gap_to_expert_top": float(norm_top[expert_index] - norm_values[expert_index]),
                "raw_gap_to_expert_top": float(raw_top[expert_index] - raw_values[expert_index]),
                "expert_norm_top_gap": float(expert_top_gap[expert_index]),
            }
            values.extend(feature_values[name] for name in config.expert_features)
        rows.append(np.array(values, dtype=np.float64))
    return rows


def _candidate_global_features(
    candidate: Candidate,
    *,
    norm_values: np.ndarray,
    top_count: float,
    expert_names: tuple[str, ...],
) -> dict[str, float]:
    solution = candidate.solution
    open_source_indices = [
        index for index, expert in enumerate(expert_names)
        if not expert.startswith("openai_")
    ]
    openai_indices = [
        index for index, expert in enumerate(expert_names)
        if expert.startswith("openai_")
    ]
    return {
        "solution_char_count_log": math.log1p(len(solution)),
        "solution_word_count_log": math.log1p(len(WORD_RE.findall(solution))),
        "solution_line_count_log": math.log1p(solution.count("\n") + 1),
        "step_count_log": math.log1p(len(candidate.steps)),
        "has_final_answer": 1.0 if candidate.final_answer else 0.0,
        "has_boxed_answer": 1.0 if "\\boxed" in solution or "boxed" in solution.lower() else 0.0,
        "mean_norm_score": float(norm_values.mean()) if len(norm_values) else 0.0,
        "std_norm_score": float(norm_values.std()) if len(norm_values) else 0.0,
        "max_norm_score": float(norm_values.max()) if len(norm_values) else 0.0,
        "min_norm_score": float(norm_values.min()) if len(norm_values) else 0.0,
        "range_norm_score": float(norm_values.max() - norm_values.min()) if len(norm_values) else 0.0,
        "top_expert_count": top_count,
        "top_expert_frac": top_count / max(1, len(expert_names)),
        "open_source_norm_mean": (
            float(norm_values[open_source_indices].mean()) if open_source_indices else 0.0
        ),
        "openai_norm_mean": (
            float(norm_values[openai_indices].mean()) if openai_indices else 0.0
        ),
    }


def _score_matrix(
    record: ProblemRecord,
    expert_names: tuple[str, ...],
    *,
    source: str,
) -> np.ndarray:
    matrix = np.zeros((len(record.candidates), len(expert_names)), dtype=np.float64)
    for row_index, candidate in enumerate(record.candidates):
        for expert_index, expert in enumerate(expert_names):
            if source == "raw":
                score = candidate.expert_scores.get(expert)
            elif source == "normalized":
                score = candidate.normalized_scores.get(expert)
            else:
                raise ValueError(f"Unknown score source: {source}")
            matrix[row_index, expert_index] = 0.0 if score is None else float(score)
    return matrix
