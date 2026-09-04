from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from moprm.evaluate import (  # noqa: E402
    EvaluationResult,
    attach_normalized_scores,
    default_baselines,
    evaluate_records,
    select_candidate,
)
from moprm.gates import metadata_gate  # noqa: E402
from moprm.io import load_jsonl, write_jsonl  # noqa: E402
from moprm.schema import ProblemRecord  # noqa: E402
from moprm.trained_gate import (  # noqa: E402
    add_gate_weights_to_record,
    stratified_folds,
    train_linear_gate,
)
from reaggregate_skywork_math_prm import reaggregate_record  # noqa: E402
from sweep_expert_settings import (  # noqa: E402
    best_static_weights,
    candidate_upper_bound,
    evaluate_static_weights,
    filter_mixed_records,
)


GateFn = Callable[[ProblemRecord, list[str]], dict[str, float]]


def parse_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def expert_names_for(records: list[ProblemRecord]) -> list[str]:
    return sorted({expert for record in records for expert in record.expert_names()})


def cross_validated_gate_records(
    records: list[ProblemRecord],
    *,
    gate_name: str,
    expert_names: list[str],
    normalization: str,
    folds: int,
    seed: int,
    hash_dim: int,
    epochs: int,
    lr: float,
    l2: float,
    weight_power: float,
    math_aggregation: str,
) -> list[ProblemRecord]:
    fold_indices = stratified_folds(records, folds=folds, seed=seed)
    oof_records: list[ProblemRecord | None] = [None] * len(records)
    for fold_no, test_indices in enumerate(fold_indices, start=1):
        test_index_set = set(test_indices)
        train_records = [
            record for index, record in enumerate(records)
            if index not in test_index_set
        ]
        test_records = [records[index] for index in test_indices]
        model = train_linear_gate(
            train_records,
            expert_names,
            normalization=normalization,
            hash_dim=hash_dim,
            epochs=epochs,
            lr=lr,
            l2=l2,
            seed=seed + fold_no,
        )
        predicted_weights = model.predict_weight_dicts(test_records, weight_power=weight_power)
        for index, record, weights in zip(test_indices, test_records, predicted_weights, strict=True):
            oof_records[index] = add_gate_weights_to_record(
                record,
                gate_name=gate_name,
                weights=weights,
                gate_metadata={
                    "type": "gate_v1_multilabel_logistic",
                    "fold": fold_no,
                    "folds": folds,
                    "train_size": len(train_records),
                    "normalization": normalization,
                    "math_aggregation": math_aggregation,
                    "hash_dim": hash_dim,
                    "epochs": epochs,
                    "lr": lr,
                    "l2": l2,
                    "weight_power": weight_power,
                    "seed": seed,
                },
            )
    return [record for record in oof_records if record is not None]


def cross_validated_static_records(
    records: list[ProblemRecord],
    *,
    gate_name: str,
    expert_names: list[str],
    normalization: str,
    folds: int,
    seed: int,
    grid_step: float,
    math_aggregation: str,
) -> tuple[list[ProblemRecord], list[dict[str, object]]]:
    fold_indices = stratified_folds(records, folds=folds, seed=seed)
    oof_records: list[ProblemRecord | None] = [None] * len(records)
    fold_summaries: list[dict[str, object]] = []
    for fold_no, test_indices in enumerate(fold_indices, start=1):
        test_index_set = set(test_indices)
        train_records = [
            record for index, record in enumerate(records)
            if index not in test_index_set
        ]
        test_records = [records[index] for index in test_indices]
        weights, train_correct, train_total = best_static_weights(
            train_records,
            expert_names,
            normalization=normalization,
            step=grid_step,
        )
        test_correct, test_total = evaluate_static_weights(
            test_records,
            weights,
            normalization=normalization,
        )
        fold_summaries.append(
            {
                "fold": fold_no,
                "train": f"{train_correct}/{train_total}",
                "test": f"{test_correct}/{test_total}",
                "weights": weights,
            }
        )
        for index, record in zip(test_indices, test_records, strict=True):
            oof_records[index] = add_gate_weights_to_record(
                record,
                gate_name=gate_name,
                weights=weights,
                gate_metadata={
                    "type": "cv_static_calibration",
                    "fold": fold_no,
                    "folds": folds,
                    "train_size": len(train_records),
                    "normalization": normalization,
                    "math_aggregation": math_aggregation,
                    "grid_step": grid_step,
                    "seed": seed,
                },
            )
    return [record for record in oof_records if record is not None], fold_summaries


def filter_group(records: list[ProblemRecord], group: str) -> list[ProblemRecord]:
    if group == "all":
        return records
    return [record for record in records if record.domain == group]


def get_result(
    records: list[ProblemRecord],
    *,
    method: str,
    gate: GateFn,
    normalization: str,
) -> EvaluationResult:
    if not records:
        return EvaluationResult(method=method, total=0, correct=0, accuracy=0.0, selections={})
    return evaluate_records(records, method, gate, normalization=normalization)


def result_score(result: EvaluationResult) -> str:
    if result.total == 0:
        return "NA"
    return f"{result.correct}/{result.total}={result.accuracy:.3f}"


def candidate_score(records: list[ProblemRecord]) -> str:
    correct, total = candidate_upper_bound(records)
    if total == 0:
        return "NA"
    return f"{correct}/{total}={correct / total:.3f}"


def best_single_score(records: list[ProblemRecord], *, normalization: str) -> str:
    if not records:
        return "NA"
    singles = [
        result
        for result in default_baselines(records, normalization=normalization)
        if result.method.startswith("single:")
    ]
    if not singles:
        return "NA"
    best = max(singles, key=lambda result: (result.correct, result.method))
    return f"{best.method.removeprefix('single:')} {result_score(best)}"


def baseline_score(
    records: list[ProblemRecord],
    *,
    method: str,
    normalization: str,
) -> str:
    if not records:
        return "NA"
    results = {result.method: result for result in default_baselines(records, normalization=normalization)}
    result = results.get(method)
    return result_score(result) if result is not None else "NA"


def top_gate_weight_summary(
    records: list[ProblemRecord],
    *,
    gate_name: str,
) -> str:
    totals: dict[str, float] = {}
    count = 0
    for record in records:
        gate_weights = record.metadata.get("gate_weights", {})
        if not isinstance(gate_weights, dict):
            continue
        weights = gate_weights.get(gate_name, {})
        if not isinstance(weights, dict):
            continue
        count += 1
        for expert, weight in weights.items():
            totals[expert] = totals.get(expert, 0.0) + float(weight)
    if count == 0:
        return "NA"
    means = sorted(
        ((expert, weight / count) for expert, weight in totals.items()),
        key=lambda item: (-item[1], item[0]),
    )
    return ",".join(f"{expert}:{weight:.2f}" for expert, weight in means)


def print_table(rows: list[dict[str, str]]) -> None:
    columns = [
        "aggregation",
        "group",
        "candidate_upper",
        "gate_v1_cv",
        "cv_static",
        "best_single",
        "uniform",
        "domain",
        "llm_gate",
        "expert_oracle",
        "gate_weight_mean",
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
            "Train and evaluate Gate-v1: a lightweight question-level "
            "multi-label logistic router for MoPRM."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output-dir",
        default="data/scored/gate_v1",
        help="Directory for out-of-fold JSONL records with Gate-v1 weights.",
    )
    parser.add_argument("--gate-name", default="trained_gate_v1_cv")
    parser.add_argument("--static-gate-name", default="cv_static_calibrated")
    parser.add_argument(
        "--math-aggregations",
        default="mean,min,last,geomean",
        help="Comma-separated open_math_prm aggregation methods.",
    )
    parser.add_argument("--normalization", default="rank")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--hash-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument(
        "--weight-power",
        type=float,
        default=1.0,
        help="Sharpen predicted expert probabilities before normalizing them into weights.",
    )
    parser.add_argument("--grid-step", type=float, default=0.1)
    parser.add_argument(
        "--include-non-mixed",
        action="store_true",
        help="Train/evaluate on all records instead of filtering to mixed records.",
    )
    args = parser.parse_args()

    loaded_records = load_jsonl(Path(args.input))
    output_dir = Path(args.output_dir)
    rows: list[dict[str, str]] = []

    for aggregation in parse_csv(args.math_aggregations):
        aggregated_records = [
            reaggregate_record(record, expert_name="open_math_prm", aggregation=aggregation)
            for record in loaded_records
        ]
        records = aggregated_records if args.include_non_mixed else filter_mixed_records(aggregated_records)
        experts = expert_names_for(records)
        gate_records = cross_validated_gate_records(
            records,
            gate_name=args.gate_name,
            expert_names=experts,
            normalization=args.normalization,
            folds=args.folds,
            seed=args.seed,
            hash_dim=args.hash_dim,
            epochs=args.epochs,
            lr=args.lr,
            l2=args.l2,
            weight_power=args.weight_power,
            math_aggregation=aggregation,
        )
        static_records, fold_summaries = cross_validated_static_records(
            records,
            gate_name=args.static_gate_name,
            expert_names=experts,
            normalization=args.normalization,
            folds=args.folds,
            seed=args.seed,
            grid_step=args.grid_step,
            math_aggregation=aggregation,
        )

        output_path = output_dir / f"{args.gate_name}_{aggregation}_{args.normalization}.jsonl"
        write_jsonl(output_path, (record.to_dict() for record in gate_records))

        static_output_path = output_dir / f"{args.static_gate_name}_{aggregation}_{args.normalization}.jsonl"
        write_jsonl(static_output_path, (record.to_dict() for record in static_records))

        for group in ("all", "math", "logic"):
            base_group = filter_group(records, group)
            gate_group = filter_group(gate_records, group)
            static_group = filter_group(static_records, group)
            rows.append(
                {
                    "aggregation": aggregation,
                    "group": group,
                    "candidate_upper": candidate_score(base_group),
                    "gate_v1_cv": result_score(
                        get_result(
                            gate_group,
                            method=f"metadata_gate:{args.gate_name}",
                            gate=metadata_gate(args.gate_name),
                            normalization=args.normalization,
                        )
                    ),
                    "cv_static": result_score(
                        get_result(
                            static_group,
                            method=f"metadata_gate:{args.static_gate_name}",
                            gate=metadata_gate(args.static_gate_name),
                            normalization=args.normalization,
                        )
                    ),
                    "best_single": best_single_score(base_group, normalization=args.normalization),
                    "uniform": baseline_score(
                        base_group,
                        method="uniform_ensemble",
                        normalization=args.normalization,
                    ),
                    "domain": baseline_score(
                        base_group,
                        method="domain_rule_gate",
                        normalization=args.normalization,
                    ),
                    "llm_gate": baseline_score(
                        base_group,
                        method="metadata_gate:openai_llm_gate",
                        normalization=args.normalization,
                    ),
                    "expert_oracle": baseline_score(
                        base_group,
                        method="oracle_gate",
                        normalization=args.normalization,
                    ),
                    "gate_weight_mean": top_gate_weight_summary(gate_group, gate_name=args.gate_name),
                }
            )

        print(f"Saved Gate-v1 OOF records: {output_path}")
        print(f"Saved CV-static OOF records: {static_output_path}")
        print(f"Static CV fold summaries for {aggregation}: {fold_summaries}")

    print(f"Loaded {len(loaded_records)} records from {args.input}")
    print(f"Evaluated {'all records' if args.include_non_mixed else 'mixed records only'}")
    print(
        "Gate-v1 config: "
        f"normalization={args.normalization}, folds={args.folds}, "
        f"hash_dim={args.hash_dim}, epochs={args.epochs}, lr={args.lr}, l2={args.l2}"
        f", weight_power={args.weight_power}"
    )
    print_table(rows)


if __name__ == "__main__":
    main()
