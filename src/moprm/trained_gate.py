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
SCORE_SHAPE_FEATURES = (
    "score_coverage",
    "raw_mean",
    "raw_std",
    "raw_range",
    "raw_top_gap",
    "raw_top_minus_mean",
    "norm_mean",
    "norm_std",
    "norm_range",
    "norm_top",
    "norm_top_gap",
    "norm_top_minus_mean",
    "norm_softmax_confidence",
    "top_agreement_frac",
    "top_consensus_frac",
    "top_is_uniform_top",
)
GLOBAL_SCORE_SHAPE_FEATURES = (
    "candidate_count_log",
    "distinct_top_frac",
    "max_top_consensus_frac",
    "avg_pairwise_top_agreement",
    "avg_pairwise_norm_corr",
    "avg_expert_norm_std",
    "avg_expert_norm_top_gap",
)
PAIRWISE_SCORE_SHAPE_FEATURES = (
    "pair_top_same",
    "pair_norm_corr",
)
TOKEN_RE = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?")
LATEX_COMMAND_RE = re.compile(r"\\[A-Za-z]+")
OPTION_MARKER_RE = re.compile(r"(?:^|\n)\s*(?:\([A-H]\)|[A-H][\).\:])\s+", re.IGNORECASE)
MATH_SYMBOLS = set("+-=*/^_<>≤≥≠≈∑∏√∞∫πθλ{}[]()")


@dataclass(frozen=True)
class FeatureConfig:
    """Deterministic feature schema for lightweight trained gates.

    Gate-v1 uses only question and metadata features. Gate-v2 sets
    ``include_score_features=True`` and additionally consumes non-leaking
    candidate-pool score-shape features.
    """

    hash_dim: int
    categorical_values: dict[str, tuple[str, ...]]
    numeric_features: tuple[str, ...] = NUMERIC_FEATURES
    expert_names: tuple[str, ...] = ()
    include_score_features: bool = False
    normalization: str = "rank"
    score_features: tuple[str, ...] = SCORE_SHAPE_FEATURES
    global_score_features: tuple[str, ...] = GLOBAL_SCORE_SHAPE_FEATURES
    pairwise_score_features: tuple[str, ...] = PAIRWISE_SCORE_SHAPE_FEATURES

    @property
    def dim(self) -> int:
        return (
            sum(len(values) for values in self.categorical_values.values())
            + len(self.numeric_features)
            + self.hash_dim
            + self.score_shape_dim
        )

    @property
    def score_shape_dim(self) -> int:
        if not self.include_score_features:
            return 0
        pair_count = len(self.expert_names) * max(0, len(self.expert_names) - 1) // 2
        return (
            len(self.global_score_features)
            + len(self.expert_names) * len(self.score_features)
            + pair_count * len(self.pairwise_score_features)
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


def fit_feature_config(
    records: list[ProblemRecord],
    *,
    hash_dim: int = 256,
    expert_names: tuple[str, ...] | list[str] = (),
    include_score_features: bool = False,
    normalization: str = "rank",
) -> FeatureConfig:
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
    expert_tuple = tuple(expert_names)
    if include_score_features and not expert_tuple:
        expert_tuple = tuple(sorted({expert for record in records for expert in record.expert_names()}))
    return FeatureConfig(
        hash_dim=hash_dim,
        categorical_values=categorical_values,
        expert_names=expert_tuple,
        include_score_features=include_score_features,
        normalization=normalization,
    )


def transform_records(records: list[ProblemRecord], config: FeatureConfig) -> np.ndarray:
    features = np.zeros((len(records), config.dim), dtype=np.float64)
    categorical_offsets: dict[str, tuple[int, dict[str, int]]] = {}
    offset = 0
    for field, values in config.categorical_values.items():
        categorical_offsets[field] = (offset, {value: index for index, value in enumerate(values)})
        offset += len(values)
    numeric_offset = offset
    hash_offset = numeric_offset + len(config.numeric_features)
    score_shape_offset = hash_offset + config.hash_dim

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
        if config.include_score_features:
            score_features = _score_shape_feature_vector(record, config)
            features[
                row_index,
                score_shape_offset : score_shape_offset + config.score_shape_dim,
            ] = score_features
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
    include_score_features: bool = False,
    seed: int = 13,
) -> LinearGateModel:
    if not records:
        raise ValueError("Cannot train a gate on an empty record set")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    expert_tuple = tuple(expert_names)
    if not expert_tuple:
        raise ValueError("expert_names must not be empty")

    config = fit_feature_config(
        records,
        hash_dim=hash_dim,
        expert_names=expert_tuple,
        include_score_features=include_score_features,
        normalization=normalization,
    )
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


def _score_shape_feature_vector(record: ProblemRecord, config: FeatureConfig) -> np.ndarray:
    normalized = attach_normalized_scores(record, method=config.normalization)
    candidate_count = len(normalized.candidates)
    expert_names = config.expert_names
    top_by_expert = {
        expert: _top_candidate_id(normalized, expert)
        for expert in expert_names
    }
    present_top_ids = [top_id for top_id in top_by_expert.values() if top_id is not None]
    top_counts = {
        top_id: present_top_ids.count(top_id)
        for top_id in set(present_top_ids)
    }
    uniform_top = _weighted_top_candidate_id(
        normalized,
        {expert: 1.0 for expert in expert_names},
    )

    per_expert_values: dict[str, dict[str, float]] = {}
    for expert in expert_names:
        per_expert_values[expert] = _single_expert_score_shape_features(
            normalized,
            expert,
            top_by_expert=top_by_expert,
            top_counts=top_counts,
            uniform_top=uniform_top,
        )

    pairwise_values = _pairwise_score_shape_features(normalized, expert_names, top_by_expert)
    global_values = _global_score_shape_features(
        candidate_count=candidate_count,
        expert_names=expert_names,
        top_by_expert=top_by_expert,
        top_counts=top_counts,
        per_expert_values=per_expert_values,
        pairwise_values=pairwise_values,
    )

    values: list[float] = []
    values.extend(global_values[name] for name in config.global_score_features)
    for expert in expert_names:
        values.extend(per_expert_values[expert][name] for name in config.score_features)
    for left_index, left in enumerate(expert_names):
        for right in expert_names[left_index + 1 :]:
            pair_key = (left, right)
            values.extend(pairwise_values[pair_key][name] for name in config.pairwise_score_features)
    return np.array(values, dtype=np.float64)


def _single_expert_score_shape_features(
    record: ProblemRecord,
    expert: str,
    *,
    top_by_expert: dict[str, str | None],
    top_counts: dict[str, int],
    uniform_top: str | None,
) -> dict[str, float]:
    raw_scores = _expert_score_array(record, expert, source="raw")
    norm_scores = _expert_score_array(record, expert, source="normalized")
    candidate_count = max(1, len(record.candidates))
    coverage = len(norm_scores) / candidate_count

    raw_stats = _score_stats(raw_scores)
    norm_stats = _score_stats(norm_scores)
    top_id = top_by_expert.get(expert)
    other_top_ids = [
        other_top_id
        for other_expert, other_top_id in top_by_expert.items()
        if other_expert != expert and other_top_id is not None
    ]
    top_agreement = (
        sum(1 for other_top_id in other_top_ids if other_top_id == top_id) / len(other_top_ids)
        if top_id is not None and other_top_ids
        else 0.0
    )
    top_consensus = (
        top_counts.get(top_id, 0) / max(1, len([item for item in top_by_expert.values() if item is not None]))
        if top_id is not None
        else 0.0
    )

    return {
        "score_coverage": coverage,
        "raw_mean": raw_stats["mean"],
        "raw_std": raw_stats["std"],
        "raw_range": raw_stats["range"],
        "raw_top_gap": raw_stats["top_gap"],
        "raw_top_minus_mean": raw_stats["top_minus_mean"],
        "norm_mean": norm_stats["mean"],
        "norm_std": norm_stats["std"],
        "norm_range": norm_stats["range"],
        "norm_top": norm_stats["top"],
        "norm_top_gap": norm_stats["top_gap"],
        "norm_top_minus_mean": norm_stats["top_minus_mean"],
        "norm_softmax_confidence": _softmax_confidence(norm_scores),
        "top_agreement_frac": top_agreement,
        "top_consensus_frac": top_consensus,
        "top_is_uniform_top": 1.0 if top_id is not None and top_id == uniform_top else 0.0,
    }


def _global_score_shape_features(
    *,
    candidate_count: int,
    expert_names: tuple[str, ...],
    top_by_expert: dict[str, str | None],
    top_counts: dict[str, int],
    per_expert_values: dict[str, dict[str, float]],
    pairwise_values: dict[tuple[str, str], dict[str, float]],
) -> dict[str, float]:
    present_top_ids = [top_id for top_id in top_by_expert.values() if top_id is not None]
    pair_count = len(pairwise_values)
    avg_pairwise_top_agreement = (
        sum(values["pair_top_same"] for values in pairwise_values.values()) / pair_count
        if pair_count
        else 0.0
    )
    avg_pairwise_norm_corr = (
        sum(values["pair_norm_corr"] for values in pairwise_values.values()) / pair_count
        if pair_count
        else 0.0
    )
    avg_norm_std = (
        sum(per_expert_values[expert]["norm_std"] for expert in expert_names) / len(expert_names)
        if expert_names
        else 0.0
    )
    avg_norm_top_gap = (
        sum(per_expert_values[expert]["norm_top_gap"] for expert in expert_names) / len(expert_names)
        if expert_names
        else 0.0
    )
    return {
        "candidate_count_log": math.log1p(candidate_count),
        "distinct_top_frac": len(set(present_top_ids)) / max(1, len(present_top_ids)),
        "max_top_consensus_frac": max(top_counts.values(), default=0) / max(1, len(present_top_ids)),
        "avg_pairwise_top_agreement": avg_pairwise_top_agreement,
        "avg_pairwise_norm_corr": avg_pairwise_norm_corr,
        "avg_expert_norm_std": avg_norm_std,
        "avg_expert_norm_top_gap": avg_norm_top_gap,
    }


def _pairwise_score_shape_features(
    record: ProblemRecord,
    expert_names: tuple[str, ...],
    top_by_expert: dict[str, str | None],
) -> dict[tuple[str, str], dict[str, float]]:
    values: dict[tuple[str, str], dict[str, float]] = {}
    for left_index, left in enumerate(expert_names):
        for right in expert_names[left_index + 1 :]:
            left_top = top_by_expert.get(left)
            right_top = top_by_expert.get(right)
            values[(left, right)] = {
                "pair_top_same": 1.0 if left_top is not None and left_top == right_top else 0.0,
                "pair_norm_corr": _score_correlation(record, left, right, source="normalized"),
            }
    return values


def _score_stats(scores: np.ndarray) -> dict[str, float]:
    if len(scores) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "range": 0.0,
            "top": 0.0,
            "top_gap": 0.0,
            "top_minus_mean": 0.0,
        }
    sorted_scores = np.sort(scores)
    top = float(sorted_scores[-1])
    second = float(sorted_scores[-2]) if len(sorted_scores) > 1 else top
    mean = float(scores.mean())
    return {
        "mean": mean,
        "std": float(scores.std()),
        "range": float(scores.max() - scores.min()),
        "top": top,
        "top_gap": top - second,
        "top_minus_mean": top - mean,
    }


def _softmax_confidence(scores: np.ndarray) -> float:
    if len(scores) <= 1:
        return 0.0
    centered = scores - scores.max()
    exp_scores = np.exp(centered)
    probs = exp_scores / max(float(exp_scores.sum()), 1e-12)
    entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
    max_entropy = math.log(len(scores))
    return 1.0 - entropy / max_entropy if max_entropy > 0 else 0.0


def _expert_score_array(record: ProblemRecord, expert: str, *, source: str) -> np.ndarray:
    scores: list[float] = []
    for candidate in record.candidates:
        if source == "raw":
            score = candidate.expert_scores.get(expert)
        elif source == "normalized":
            score = candidate.normalized_scores.get(expert)
        else:
            raise ValueError(f"Unknown score source: {source}")
        if score is not None:
            scores.append(float(score))
    return np.array(scores, dtype=np.float64)


def _top_candidate_id(record: ProblemRecord, expert: str) -> str | None:
    scored = [
        (candidate.normalized_scores.get(expert), candidate.candidate_id)
        for candidate in record.candidates
        if candidate.normalized_scores.get(expert) is not None
    ]
    if not scored:
        return None
    return max(scored, key=lambda item: (float(item[0]), item[1]))[1]


def _weighted_top_candidate_id(record: ProblemRecord, weights: dict[str, float]) -> str | None:
    if not record.candidates:
        return None
    selected = select_candidate(record, weights)
    return selected.candidate_id


def _score_correlation(
    record: ProblemRecord,
    left: str,
    right: str,
    *,
    source: str,
) -> float:
    left_values: list[float] = []
    right_values: list[float] = []
    for candidate in record.candidates:
        if source == "normalized":
            left_score = candidate.normalized_scores.get(left)
            right_score = candidate.normalized_scores.get(right)
        elif source == "raw":
            left_score = candidate.expert_scores.get(left)
            right_score = candidate.expert_scores.get(right)
        else:
            raise ValueError(f"Unknown score source: {source}")
        if left_score is None or right_score is None:
            continue
        left_values.append(float(left_score))
        right_values.append(float(right_score))
    if len(left_values) < 2:
        return 0.0
    left_array = np.array(left_values, dtype=np.float64)
    right_array = np.array(right_values, dtype=np.float64)
    left_std = float(left_array.std())
    right_std = float(right_array.std())
    if left_std < 1e-8 or right_std < 1e-8:
        return 0.0
    return float(np.corrcoef(left_array, right_array)[0, 1])


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
