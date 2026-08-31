from __future__ import annotations

from collections import defaultdict
from statistics import mean, pstdev


def rank_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    if len(scores) == 1:
        key = next(iter(scores))
        return {key: 1.0}

    grouped: dict[float, list[str]] = defaultdict(list)
    for item_id, score in scores.items():
        grouped[score].append(item_id)

    sorted_scores = sorted(grouped)
    ranks: dict[str, float] = {}
    cursor = 0
    max_rank = len(scores) - 1
    for score in sorted_scores:
        item_ids = grouped[score]
        avg_rank = cursor + (len(item_ids) - 1) / 2
        normalized = avg_rank / max_rank
        for item_id in item_ids:
            ranks[item_id] = normalized
        cursor += len(item_ids)
    return ranks


def minmax_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo = min(scores.values())
    hi = max(scores.values())
    if hi == lo:
        return {key: 0.5 for key in scores}
    return {key: (value - lo) / (hi - lo) for key, value in scores.items()}


def zscore_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    sigma = pstdev(values)
    if sigma == 0:
        return {key: 0.0 for key in scores}
    mu = mean(values)
    return {key: (value - mu) / sigma for key, value in scores.items()}


def normalize_expert_scores(
    candidate_scores: dict[str, dict[str, float]],
    expert_names: list[str],
    method: str = "rank",
) -> dict[str, dict[str, float]]:
    normalizer = {
        "rank": rank_normalize,
        "minmax": minmax_normalize,
        "zscore": zscore_normalize,
    }.get(method)
    if normalizer is None:
        raise ValueError(f"Unknown normalization method: {method}")

    by_candidate = {candidate_id: {} for candidate_id in candidate_scores}
    for expert in expert_names:
        expert_scores = {
            candidate_id: scores[expert]
            for candidate_id, scores in candidate_scores.items()
            if expert in scores
        }
        normalized = normalizer(expert_scores)
        for candidate_id, score in normalized.items():
            by_candidate[candidate_id][expert] = score
    return by_candidate

