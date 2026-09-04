from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from moprm.evaluate import (  # noqa: E402
    attach_normalized_scores,
    default_baselines,
    select_candidate,
)
from moprm.io import load_jsonl  # noqa: E402
from moprm.schema import ProblemRecord  # noqa: E402
from reaggregate_skywork_math_prm import reaggregate_record  # noqa: E402


MAIN_METHODS = [
    "domain_rule_gate",
    "metadata_gate:openai_llm_gate",
    "uniform_ensemble",
    "oracle_gate",
]


def parse_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def filter_mixed_records(records: list[ProblemRecord]) -> list[ProblemRecord]:
    mixed = []
    for record in records:
        correct = sum(1 for candidate in record.candidates if candidate.is_correct)
        if 0 < correct < len(record.candidates):
            mixed.append(record)
    return mixed


def candidate_upper_bound(records: list[ProblemRecord]) -> tuple[int, int]:
    total = len(records)
    correct = sum(
        1
        for record in records
        if any(candidate.is_correct for candidate in record.candidates)
    )
    return correct, total


def simplex_weight_grid(experts: list[str], step: float = 0.1) -> Iterable[dict[str, float]]:
    if not experts:
        return
    units_float = 1.0 / step
    units = round(units_float)
    if abs(units - units_float) > 1e-9:
        raise ValueError("--grid-step must evenly divide 1.0")

    def compositions(length: int, remaining: int) -> Iterable[list[int]]:
        if length == 1:
            yield [remaining]
            return
        for value in range(remaining + 1):
            for rest in compositions(length - 1, remaining - value):
                yield [value, *rest]

    for values in compositions(len(experts), units):
        yield {
            expert: value / units
            for expert, value in zip(experts, values, strict=True)
        }


def evaluate_static_weights(
    records: list[ProblemRecord],
    weights: dict[str, float],
    *,
    normalization: str,
) -> tuple[int, int]:
    correct = 0
    total = 0
    for record in records:
        normalized = attach_normalized_scores(record, method=normalization)
        selected = select_candidate(normalized, weights)
        total += 1
        correct += 1 if selected.is_correct else 0
    return correct, total


def best_static_weights(
    records: list[ProblemRecord],
    experts: list[str],
    *,
    normalization: str,
    step: float,
) -> tuple[dict[str, float], int, int]:
    best_weights: dict[str, float] = {}
    best_correct = -1
    best_total = len(records)
    best_active = 10**9
    for weights in simplex_weight_grid(experts, step=step):
        correct, total = evaluate_static_weights(
            records,
            weights,
            normalization=normalization,
        )
        active = sum(1 for weight in weights.values() if weight > 0)
        if (correct, -active, str(sorted(weights.items()))) > (
            best_correct,
            -best_active,
            str(sorted(best_weights.items())),
        ):
            best_weights = weights
            best_correct = correct
            best_total = total
            best_active = active
    return best_weights, best_correct, best_total


def _accuracy(correct: int, total: int) -> float:
    return correct / total if total else 0.0


def _format_score(correct: int, total: int) -> str:
    return f"{correct}/{total}={_accuracy(correct, total):.3f}"


def _format_weights(weights: dict[str, float]) -> str:
    active = [
        f"{expert}:{weight:.1f}"
        for expert, weight in weights.items()
        if weight > 0
    ]
    return ",".join(active) if active else "none"


def evaluate_setting(
    records: list[ProblemRecord],
    *,
    group_name: str,
    aggregation: str,
    normalization: str,
    grid_step: float,
) -> dict[str, str]:
    results = {result.method: result for result in default_baselines(records, normalization)}
    singles = [
        result for method, result in results.items()
        if method.startswith("single:")
    ]
    best_single = max(
        singles,
        key=lambda result: (result.correct, result.method),
    )
    experts = sorted({expert for record in records for expert in record.expert_names()})
    weights, best_weight_correct, best_weight_total = best_static_weights(
        records,
        experts,
        normalization=normalization,
        step=grid_step,
    )
    candidate_correct, candidate_total = candidate_upper_bound(records)

    row = {
        "group": group_name,
        "aggregation": aggregation,
        "normalization": normalization,
        "candidate_upper": _format_score(candidate_correct, candidate_total),
        "best_single": f"{best_single.method.removeprefix('single:')} {_format_score(best_single.correct, best_single.total)}",
        "uniform": _format_score(results["uniform_ensemble"].correct, results["uniform_ensemble"].total),
        "domain": _format_score(results["domain_rule_gate"].correct, results["domain_rule_gate"].total),
        "llm_gate": "NA",
        "expert_oracle": _format_score(results["oracle_gate"].correct, results["oracle_gate"].total),
        "best_static": _format_score(best_weight_correct, best_weight_total),
        "best_static_weights": _format_weights(weights),
    }
    llm = results.get("metadata_gate:openai_llm_gate")
    if llm is not None:
        row["llm_gate"] = _format_score(llm.correct, llm.total)
    return row


def print_table(rows: list[dict[str, str]]) -> None:
    columns = [
        "group",
        "aggregation",
        "normalization",
        "candidate_upper",
        "best_single",
        "uniform",
        "domain",
        "llm_gate",
        "expert_oracle",
        "best_static",
        "best_static_weights",
    ]
    widths = {
        column: max(len(column), *(len(row[column]) for row in rows))
        for column in columns
    }
    print(" ".join(f"{column:<{widths[column]}}" for column in columns))
    print(" ".join("-" * widths[column] for column in columns))
    for row in rows:
        print(" ".join(f"{row[column]:<{widths[column]}}" for column in columns))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep open_math_prm aggregation, score normalization, and in-sample "
            "static weight calibration."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--aggregations",
        default="mean,min,last,geomean",
        help="Comma-separated open_math_prm aggregation methods.",
    )
    parser.add_argument(
        "--normalizations",
        default="rank,minmax,zscore",
        help="Comma-separated score normalization methods.",
    )
    parser.add_argument("--grid-step", type=float, default=0.1)
    args = parser.parse_args()

    base_records = load_jsonl(Path(args.input))
    rows: list[dict[str, str]] = []
    for aggregation in parse_csv(args.aggregations):
        records = [
            reaggregate_record(
                record,
                expert_name="open_math_prm",
                aggregation=aggregation,
            )
            for record in base_records
        ]
        groups = {
            "full": records,
            "mixed": filter_mixed_records(records),
        }
        for normalization in parse_csv(args.normalizations):
            for group_name, group_records in groups.items():
                if not group_records:
                    continue
                rows.append(
                    evaluate_setting(
                        group_records,
                        group_name=group_name,
                        aggregation=aggregation,
                        normalization=normalization,
                        grid_step=args.grid_step,
                    )
                )

    print(f"Loaded {len(base_records)} problems from {args.input}")
    print(f"Grid step: {args.grid_step}")
    print_table(rows)


if __name__ == "__main__":
    main()
