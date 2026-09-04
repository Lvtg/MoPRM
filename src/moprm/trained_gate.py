from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
import math
import re
from typing import Any

import numpy as np

from moprm.evaluate import attach_normalized_scores, select_candidate
from moprm.schema import ProblemRecord


CATEGORICAL_FIELDS = ("domain", "source", "subject", "level", "task_name")
NUMERIC_FEATURES = (
    "char_count_log",
    "word_count_log",
    "line_count_log",
    "digit_count_log",
    "number_count_log",
    "math_symbol_count_log",
    "latex_command_count_log",
    "option_marker_count_log",
    "digit_ratio",
    "has_latex",
    "has_multiple_choice",
)
TOKEN_RE = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?")
LATEX_COMMAND_RE = re.compile(r"\\[A-Za-z]+")
OPTION_MARKER_RE = re.compile(r"(?:^|\n)\s*(?:\([A-H]\)|[A-H][\).\:])\s+", re.IGNORECASE)
MATH_SYMBOLS = set("+-=*/^_<>≤≥≠≈∑∏√∞∫πθλ{}[]()")


@dataclass(frozen=True)
class FeatureConfig:
    """Deterministic question-level feature schema for Gate-v1."""

    hash_dim: int
    categorical_values: dict[str, tuple[str, ...]]
    numeric_features: tuple[str, ...] = NUMERIC_FEATURES

    @property
    def dim(self) -> int:
        return (
            sum(len(values) for values in self.categorical_values.values())
            + len(self.numeric_features)
            + self.hash_dim
        )


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> "Standardizer":
        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        return cls(mean=mean, scale=scale)

    def transform(self, features: np.ndarray) -> np.ndarray:
        return (features - self.mean) / self.scale


@dataclass(frozen=True)
class LinearGateModel:
    """A lightweight multi-label logistic gate.

    The model predicts whether each expert would select a correct candidate for
    a problem. The predicted success probabilities are normalized into mixture
    weights.
    """

    expert_names: tuple[str, ...]
    feature_config: FeatureConfig
    standardizer: Standardizer
    weights: np.ndarray
    bias: np.ndarray

    def predict_logits(self, records: list[ProblemRecord]) -> np.ndarray:
        features = transform_records(records, self.feature_config)
        features = self.standardizer.transform(features)
        return features @ self.weights + self.bias

    def predict_probabilities(self, records: list[ProblemRecord]) -> np.ndarray:
        return sigmoid(self.predict_logits(records))

    def predict_weight_dicts(
        self,
        records: list[ProblemRecord],
        *,
        min_weight: float = 0.0,
        weight_power: float = 1.0,
    ) -> list[dict[str, float]]:
        if weight_power <= 0:
            raise ValueError("weight_power must be positive")
        probabilities = self.predict_probabilities(records)
        rows: list[dict[str, float]] = []
        for row in probabilities:
            clipped = np.maximum(row.astype(float), min_weight)
            clipped = np.power(clipped, weight_power)
            total = float(clipped.sum())
            if total <= 0:
                uniform = 1.0 / len(self.expert_names) if self.expert_names else 0.0
                rows.append({expert: uniform for expert in self.expert_names})
                continue
            rows.append(
                {
                    expert: float(weight / total)
                    for expert, weight in zip(self.expert_names, clipped, strict=True)
                }
            )
        return rows


def fit_feature_config(records: list[ProblemRecord], *, hash_dim: int = 256) -> FeatureConfig:
    if hash_dim <= 0:
        raise ValueError("hash_dim must be positive")
    categorical_values: dict[str, tuple[str, ...]] = {}
    for field in CATEGORICAL_FIELDS:
        values = sorted(
            {
                value
                for record in records
                if (value := _categorical_value(record, field)) != ""
            }
        )
        categorical_values[field] = tuple(values)
    return FeatureConfig(hash_dim=hash_dim, categorical_values=categorical_values)


def transform_records(records: list[ProblemRecord], config: FeatureConfig) -> np.ndarray:
    features = np.zeros((len(records), config.dim), dtype=np.float64)
    categorical_offsets: dict[str, tuple[int, dict[str, int]]] = {}
    offset = 0
    for field, values in config.categorical_values.items():
        categorical_offsets[field] = (offset, {value: index for index, value in enumerate(values)})
        offset += len(values)
    numeric_offset = offset
    hash_offset = numeric_offset + len(config.numeric_features)

    for row_index, record in enumerate(records):
        for field, (field_offset, value_to_index) in categorical_offsets.items():
            value = _categorical_value(record, field)
            if value in value_to_index:
                features[row_index, field_offset + value_to_index[value]] = 1.0

        numeric = _numeric_features(record.problem)
        for feature_index, name in enumerate(config.numeric_features):
            features[row_index, numeric_offset + feature_index] = numeric[name]

        hashed = _hashed_text_features(record.problem, config.hash_dim)
        features[row_index, hash_offset : hash_offset + config.hash_dim] = hashed
    return features


def expert_success_targets(
    records: list[ProblemRecord],
    expert_names: tuple[str, ...] | list[str],
    *,
    normalization: str = "rank",
) -> np.ndarray:
    expert_tuple = tuple(expert_names)
    targets = np.zeros((len(records), len(expert_tuple)), dtype=np.float64)
    for row_index, record in enumerate(records):
        normalized = attach_normalized_scores(record, method=normalization)
        for expert_index, expert in enumerate(expert_tuple):
            if not any(
                expert in candidate.expert_scores or expert in candidate.normalized_scores
                for candidate in normalized.candidates
            ):
                continue
            selected = select_candidate(normalized, {expert: 1.0})
            targets[row_index, expert_index] = 1.0 if selected.is_correct else 0.0
    return targets


def train_linear_gate(
    records: list[ProblemRecord],
    expert_names: tuple[str, ...] | list[str],
    *,
    normalization: str = "rank",
    hash_dim: int = 256,
    epochs: int = 800,
    lr: float = 0.05,
    l2: float = 0.01,
    seed: int = 13,
) -> LinearGateModel:
    if not records:
        raise ValueError("Cannot train Gate-v1 on an empty record set")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    expert_tuple = tuple(expert_names)
    if not expert_tuple:
        raise ValueError("expert_names must not be empty")

    config = fit_feature_config(records, hash_dim=hash_dim)
    features = transform_records(records, config)
    standardizer = Standardizer.fit(features)
    features = standardizer.transform(features)
    targets = expert_success_targets(records, expert_tuple, normalization=normalization)

    rng = np.random.default_rng(seed)
    weights = rng.normal(loc=0.0, scale=0.01, size=(features.shape[1], len(expert_tuple)))
    base_rates = np.clip(targets.mean(axis=0), 1e-3, 1.0 - 1e-3)
    bias = np.log(base_rates / (1.0 - base_rates))

    adam_m_w = np.zeros_like(weights)
    adam_v_w = np.zeros_like(weights)
    adam_m_b = np.zeros_like(bias)
    adam_v_b = np.zeros_like(bias)
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    n = float(len(records))

    for step in range(1, epochs + 1):
        logits = features @ weights + bias
        probabilities = sigmoid(logits)
        grad_logits = (probabilities - targets) / n
        grad_w = features.T @ grad_logits + l2 * weights
        grad_b = grad_logits.sum(axis=0)

        adam_m_w = beta1 * adam_m_w + (1.0 - beta1) * grad_w
        adam_v_w = beta2 * adam_v_w + (1.0 - beta2) * np.square(grad_w)
        adam_m_b = beta1 * adam_m_b + (1.0 - beta1) * grad_b
        adam_v_b = beta2 * adam_v_b + (1.0 - beta2) * np.square(grad_b)

        corrected_m_w = adam_m_w / (1.0 - beta1**step)
        corrected_v_w = adam_v_w / (1.0 - beta2**step)
        corrected_m_b = adam_m_b / (1.0 - beta1**step)
        corrected_v_b = adam_v_b / (1.0 - beta2**step)

        weights -= lr * corrected_m_w / (np.sqrt(corrected_v_w) + epsilon)
        bias -= lr * corrected_m_b / (np.sqrt(corrected_v_b) + epsilon)

    return LinearGateModel(
        expert_names=expert_tuple,
        feature_config=config,
        standardizer=standardizer,
        weights=weights,
        bias=bias,
    )


def stratified_folds(
    records: list[ProblemRecord],
    *,
    folds: int = 5,
    seed: int = 13,
) -> list[list[int]]:
    if folds <= 1:
        raise ValueError("folds must be greater than 1")
    if folds > len(records):
        raise ValueError("folds cannot exceed number of records")

    grouped: dict[tuple[str, str], list[int]] = {}
    for index, record in enumerate(records):
        key = (record.domain, str(record.metadata.get("source", "")))
        grouped.setdefault(key, []).append(index)

    rng = np.random.default_rng(seed)
    result: list[list[int]] = [[] for _ in range(folds)]
    for key in sorted(grouped):
        indices = grouped[key]
        rng.shuffle(indices)
        for offset, index in enumerate(indices):
            result[offset % folds].append(index)

    for fold in result:
        fold.sort()
    return result


def add_gate_weights_to_record(
    record: ProblemRecord,
    *,
    gate_name: str,
    weights: dict[str, float],
    gate_metadata: dict[str, Any] | None = None,
) -> ProblemRecord:
    metadata = dict(record.metadata)
    gate_weights = metadata.get("gate_weights", {})
    if not isinstance(gate_weights, dict):
        gate_weights = {}
    gate_weights = dict(gate_weights)
    gate_weights[gate_name] = {expert: float(weight) for expert, weight in weights.items()}
    metadata["gate_weights"] = gate_weights

    if gate_metadata is not None:
        existing_metadata = metadata.get("gate_metadata", {})
        if not isinstance(existing_metadata, dict):
            existing_metadata = {}
        existing_metadata = dict(existing_metadata)
        existing_metadata[gate_name] = dict(gate_metadata)
        metadata["gate_metadata"] = existing_metadata

    return ProblemRecord(
        problem_id=record.problem_id,
        domain=record.domain,
        problem=record.problem,
        answer=record.answer,
        candidates=record.candidates,
        metadata=metadata,
    )


def sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def _categorical_value(record: ProblemRecord, field: str) -> str:
    if field == "domain":
        value = record.domain
    else:
        value = record.metadata.get(field, "")
    if value is None:
        return ""
    return str(value)


def _numeric_features(text: str) -> dict[str, float]:
    words = TOKEN_RE.findall(text)
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    digits = sum(char.isdigit() for char in text)
    math_symbols = sum(char in MATH_SYMBOLS for char in text)
    latex_commands = LATEX_COMMAND_RE.findall(text)
    option_markers = OPTION_MARKER_RE.findall(text)
    char_count = len(text)
    return {
        "char_count_log": math.log1p(char_count),
        "word_count_log": math.log1p(len(words)),
        "line_count_log": math.log1p(text.count("\n") + 1),
        "digit_count_log": math.log1p(digits),
        "number_count_log": math.log1p(len(numbers)),
        "math_symbol_count_log": math.log1p(math_symbols),
        "latex_command_count_log": math.log1p(len(latex_commands)),
        "option_marker_count_log": math.log1p(len(option_markers)),
        "digit_ratio": digits / max(1, char_count),
        "has_latex": 1.0 if "\\" in text or "$" in text else 0.0,
        "has_multiple_choice": 1.0 if option_markers else 0.0,
    }


def _hashed_text_features(text: str, hash_dim: int) -> np.ndarray:
    buckets = np.zeros(hash_dim, dtype=np.float64)
    tokens = [token.lower() for token in TOKEN_RE.findall(text)]
    grams = tokens + [f"{left}_{right}" for left, right in zip(tokens, tokens[1:])]
    if not grams:
        return buckets

    for gram in grams:
        digest = blake2b(gram.encode("utf-8"), digest_size=8, person=b"moprmv1").digest()
        value = int.from_bytes(digest, byteorder="big", signed=False)
        index = value % hash_dim
        sign = 1.0 if (value >> 63) == 0 else -1.0
        buckets[index] += sign
    return buckets / math.sqrt(len(grams))
