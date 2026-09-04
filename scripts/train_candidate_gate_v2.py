from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from moprm.candidate_gate import (  # noqa: E402
    add_candidate_gate_scores_to_record,
    evaluate_candidate_gate_records,
    train_candidate_gate,
)
from moprm.io import load_jsonl, write_jsonl  # noqa: E402
from moprm.schema import ProblemRecord  # noqa: E402
from moprm.trained_gate import stratified_folds  # noqa: E402
from train_gate_v1 import (  # noqa: E402
    baseline_score,
    best_single_score,
    candidate_score,
    cross_validated_static_records,
    expert_names_for,
    filter_group,
    maybe_reaggregate_records,
    parse_csv,
    result_score,
)
from sweep_expert_settings import filter_mixed_records  # noqa: E402


def cross_validated_candidate_gate_records(
    records: list[ProblemRecord],
    *,
    gate_name: str,
    expert_names: list[str],
    normalization: str,
    folds: int,
    seed: int,
    epochs: int,
    lr: float,
    l2: float,
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
        model = train_candidate_gate(
            train_records,
            expert_names,
            normalization=normalization,
            epochs=epochs,
            lr=lr,
            l2=l2,
            seed=seed + fold_no,
        )
        predicted_scores = model.predict_candidate_scores(test_records)
        for index, record, scores in zip(test_indices, test_records, predicted_scores, strict=True):
            oof_records[index] = add_candidate_gate_scores_to_record(
                record,
                gate_name=gate_name,
                scores=scores,
                gate_metadata={
                    "type": "gate_v2_candidate_aware_logistic_selector",
                    "fold": fold_no,
                    "folds": folds,
                    "train_size": len(train_records),
                    "normalization": normalization,
                    "math_aggregation": math_aggregation,
                    "epochs": epochs,
                    "lr": lr,
                    "l2": l2,
                    "seed": seed,
                    "expert_names": expert_names,
                },
            )
    return [record for record in oof_records if record is not None]


def print_table(rows: list[dict[str, str]]) -> None:
    columns = [
        "aggregation",
        "group",
        "candidate_upper",
        "candidate_gate_cv",
        "cv_static",
        "best_single",
        "uniform",
        "domain",
        "llm_gate",
        "expert_oracle",
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
            "Train Gate-v2 as a candidate-aware learned selector over expert "
            "score/rank features."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="data/scored/candidate_gate_v2")
    parser.add_argument("--gate-name", default="candidate_gate_v2_cv")
    parser.add_argument("--static-gate-name", default="cv_static_calibrated")
    parser.add_argument(
        "--math-aggregations",
        default="mean,min,last,geomean",
        help="Comma-separated open_math_prm aggregation methods; use none for pre-expanded pseudo-experts.",
    )
    parser.add_argument("--normalization", default="rank")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--grid-step", type=float, default=0.1)
    parser.add_argument("--skip-static", action="store_true")
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
        aggregated_records = maybe_reaggregate_records(loaded_records, aggregation=aggregation)
        records = aggregated_records if args.include_non_mixed else filter_mixed_records(aggregated_records)
        experts = expert_names_for(records)
        candidate_gate_records = cross_validated_candidate_gate_records(
            records,
            gate_name=args.gate_name,
            expert_names=experts,
            normalization=args.normalization,
            folds=args.folds,
            seed=args.seed,
            epochs=args.epochs,
            lr=args.lr,
            l2=args.l2,
            math_aggregation=aggregation,
        )
        if args.skip_static:
            static_records = []
        else:
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
            print(f"Static CV fold summaries for {aggregation}: {fold_summaries}")

        output_path = output_dir / f"{args.gate_name}_{aggregation}_{args.normalization}.jsonl"
        write_jsonl(output_path, (record.to_dict() for record in candidate_gate_records))
        print(f"Saved candidate-gate OOF records: {output_path}")
        if not args.skip_static:
            static_output_path = output_dir / f"{args.static_gate_name}_{aggregation}_{args.normalization}.jsonl"
            write_jsonl(static_output_path, (record.to_dict() for record in static_records))
            print(f"Saved CV-static OOF records: {static_output_path}")

        for group in ("all", "math", "logic"):
            base_group = filter_group(records, group)
            candidate_gate_group = filter_group(candidate_gate_records, group)
            static_group = filter_group(static_records, group)
            cv_static_score = "SKIP"
            if not args.skip_static:
                cv_static_score = baseline_score(
                    static_group,
                    method=f"metadata_gate:{args.static_gate_name}",
                    normalization=args.normalization,
                )
            rows.append(
                {
                    "aggregation": aggregation,
                    "group": group,
                    "candidate_upper": candidate_score(base_group),
                    "candidate_gate_cv": result_score(
                        evaluate_candidate_gate_records(
                            candidate_gate_group,
                            gate_name=args.gate_name,
                        )
                    ),
                    "cv_static": cv_static_score,
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
                }
            )

    print(f"Loaded {len(loaded_records)} records from {args.input}")
    print(f"Evaluated {'all records' if args.include_non_mixed else 'mixed records only'}")
    print(
        "Candidate Gate-v2 config: "
        f"normalization={args.normalization}, folds={args.folds}, "
        f"epochs={args.epochs}, lr={args.lr}, l2={args.l2}"
    )
    print_table(rows)


if __name__ == "__main__":
    main()
