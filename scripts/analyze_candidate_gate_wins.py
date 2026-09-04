from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moprm.candidate_gate import evaluate_candidate_gate_records  # noqa: E402
from moprm.evaluate import (  # noqa: E402
    EvaluationResult,
    attach_normalized_scores,
    default_baselines,
    select_candidate,
    single_expert_gate,
)
from moprm.gates import metadata_gate  # noqa: E402
from moprm.io import load_jsonl  # noqa: E402
from moprm.schema import Candidate, ProblemRecord  # noqa: E402


@dataclass(frozen=True)
class Comparison:
    baseline: str
    group: str
    total: int
    primary_correct: int
    baseline_correct: int
    both_correct: int
    both_wrong: int
    wins: int
    losses: int
    same_selection: int
    different_selection: int

    @property
    def net(self) -> int:
        return self.wins - self.losses


def filter_group(records: list[ProblemRecord], group: str) -> list[ProblemRecord]:
    if group == "all":
        return records
    if group.startswith("source:"):
        source = group.removeprefix("source:")
        return [record for record in records if str(record.metadata.get("source", "")) == source]
    return [record for record in records if record.domain == group]


def candidate_truths(record: ProblemRecord) -> str:
    return "".join("C" if candidate.is_correct else "W" for candidate in record.candidates)


def candidate_by_id(record: ProblemRecord, candidate_id: str) -> Candidate | None:
    for candidate in record.candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    return None


def result_map(records: list[ProblemRecord], result: EvaluationResult) -> dict[str, bool]:
    values: dict[str, bool] = {}
    by_problem = {record.problem_id: record for record in records}
    for problem_id, candidate_id in result.selections.items():
        candidate = candidate_by_id(by_problem[problem_id], candidate_id)
        values[problem_id] = bool(candidate and candidate.is_correct)
    return values


def expert_top_choices(record: ProblemRecord, *, normalization: str) -> dict[str, str]:
    normalized = attach_normalized_scores(record, method=normalization)
    experts = normalized.expert_names()
    return {
        expert: select_candidate(
            normalized,
            single_expert_gate(expert)(normalized, experts),
        ).candidate_id
        for expert in experts
    }


def load_static_result(
    static_input: Path | None,
    *,
    method: str,
    normalization: str,
) -> EvaluationResult | None:
    if static_input is None:
        return None
    static_records = load_jsonl(static_input)
    return evaluate_metadata_gate(static_records, method=method, normalization=normalization)


def evaluate_metadata_gate(
    records: list[ProblemRecord],
    *,
    method: str,
    normalization: str,
) -> EvaluationResult:
    prefix = "metadata_gate:"
    if not method.startswith(prefix):
        raise ValueError(f"Expected metadata gate method, got {method!r}")
    gate_name = method.removeprefix(prefix)
    from moprm.evaluate import evaluate_records  # noqa: PLC0415

    return evaluate_records(
        records,
        method,
        metadata_gate(gate_name),
        normalization=normalization,
    )


def build_results(
    records: list[ProblemRecord],
    *,
    primary_gate_name: str,
    normalization: str,
    static_input: Path | None,
) -> dict[str, EvaluationResult]:
    results = {
        result.method: result
        for result in default_baselines(records, normalization=normalization)
    }
    primary = evaluate_candidate_gate_records(records, gate_name=primary_gate_name)
    results[primary.method] = primary

    singles = [result for result in results.values() if result.method.startswith("single:")]
    if singles:
        best_single = max(singles, key=lambda item: (item.correct, item.method))
        results["best_single"] = EvaluationResult(
            method=f"best_single:{best_single.method.removeprefix('single:')}",
            total=best_single.total,
            correct=best_single.correct,
            accuracy=best_single.accuracy,
            selections=best_single.selections,
        )

    static = load_static_result(
        static_input,
        method="metadata_gate:cv_static_calibrated",
        normalization=normalization,
    )
    if static is not None:
        results["cv_static_calibrated"] = static
    return results


def compare_results(
    records: list[ProblemRecord],
    *,
    primary: EvaluationResult,
    baseline: EvaluationResult,
    group: str,
) -> Comparison:
    group_records = filter_group(records, group)
    group_ids = {record.problem_id for record in group_records}
    primary_correct_by_id = result_map(records, primary)
    baseline_correct_by_id = result_map(records, baseline)
    wins = losses = both_correct = both_wrong = same_selection = different_selection = 0
    for problem_id in sorted(group_ids):
        primary_selected = primary.selections.get(problem_id)
        baseline_selected = baseline.selections.get(problem_id)
        if primary_selected is None or baseline_selected is None:
            continue
        primary_correct = primary_correct_by_id[problem_id]
        baseline_correct = baseline_correct_by_id[problem_id]
        if primary_selected == baseline_selected:
            same_selection += 1
        else:
            different_selection += 1
        if primary_correct and baseline_correct:
            both_correct += 1
        elif not primary_correct and not baseline_correct:
            both_wrong += 1
        elif primary_correct and not baseline_correct:
            wins += 1
        else:
            losses += 1
    total = both_correct + both_wrong + wins + losses
    return Comparison(
        baseline=baseline.method,
        group=group,
        total=total,
        primary_correct=both_correct + wins,
        baseline_correct=both_correct + losses,
        both_correct=both_correct,
        both_wrong=both_wrong,
        wins=wins,
        losses=losses,
        same_selection=same_selection,
        different_selection=different_selection,
    )


def collect_cases(
    records: list[ProblemRecord],
    *,
    primary: EvaluationResult,
    baseline: EvaluationResult,
    normalization: str,
    case_type: str,
    limit: int,
) -> list[dict[str, object]]:
    by_problem = {record.problem_id: record for record in records}
    primary_correct_by_id = result_map(records, primary)
    baseline_correct_by_id = result_map(records, baseline)
    cases: list[dict[str, object]] = []
    for problem_id in sorted(primary.selections):
        if problem_id not in baseline.selections:
            continue
        primary_correct = primary_correct_by_id[problem_id]
        baseline_correct = baseline_correct_by_id[problem_id]
        if case_type == "wins" and not (primary_correct and not baseline_correct):
            continue
        if case_type == "losses" and not (not primary_correct and baseline_correct):
            continue
        if case_type == "selection_diff" and primary.selections[problem_id] == baseline.selections[problem_id]:
            continue
        record = by_problem[problem_id]
        primary_selected = primary.selections[problem_id]
        baseline_selected = baseline.selections[problem_id]
        cases.append(
            {
                "problem_id": problem_id,
                "domain": record.domain,
                "source": record.metadata.get("source", ""),
                "primary_selected": primary_selected,
                "primary_correct": primary_correct,
                "baseline_selected": baseline_selected,
                "baseline_correct": baseline_correct,
                "truths": candidate_truths(record),
                "expert_top_choices": expert_top_choices(record, normalization=normalization),
                "problem_preview": record.problem.replace("\n", " ")[:180],
            }
        )
        if len(cases) >= limit:
            break
    return cases


def print_comparison_table(comparisons: list[Comparison]) -> None:
    columns = [
        "baseline",
        "group",
        "primary",
        "baseline_acc",
        "wins",
        "losses",
        "net",
        "same_sel",
        "diff_sel",
        "both_wrong",
    ]
    rows = [
        {
            "baseline": comparison.baseline,
            "group": comparison.group,
            "primary": f"{comparison.primary_correct}/{comparison.total}",
            "baseline_acc": f"{comparison.baseline_correct}/{comparison.total}",
            "wins": str(comparison.wins),
            "losses": str(comparison.losses),
            "net": f"{comparison.net:+d}",
            "same_sel": str(comparison.same_selection),
            "diff_sel": str(comparison.different_selection),
            "both_wrong": str(comparison.both_wrong),
        }
        for comparison in comparisons
    ]
    widths = {
        column: max(len(column), *(len(row[column]) for row in rows))
        for column in columns
    }
    print(" ".join(f"{column:<{widths[column]}}" for column in columns))
    print(" ".join("-" * widths[column] for column in columns))
    for row in rows:
        print(" ".join(f"{row[column]:<{widths[column]}}" for column in columns))


def print_cases(
    cases: list[dict[str, object]],
    *,
    title: str,
) -> None:
    print(f"\n[{title}]")
    if not cases:
        print("none")
        return
    for case in cases:
        print(
            f"{case['problem_id']} ({case['domain']} | {case['source']}) "
            f"primary={case['primary_selected']}:{case['primary_correct']} "
            f"baseline={case['baseline_selected']}:{case['baseline_correct']} "
            f"truths={case['truths']}"
        )
        top_choices = case["expert_top_choices"]
        if isinstance(top_choices, dict):
            top_text = ", ".join(f"{expert}->{candidate}" for expert, candidate in sorted(top_choices.items()))
            print(f"  expert_tops: {top_text}")
        print(f"  preview: {case['problem_preview']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Candidate Gate-v2 selections against baseline methods."
    )
    parser.add_argument("--input", required=True, help="Candidate Gate-v2 OOF JSONL.")
    parser.add_argument(
        "--static-input",
        help="Optional CV-static OOF JSONL generated with the same aggregation/normalization.",
    )
    parser.add_argument("--primary-gate-name", default="candidate_gate_v2_cv")
    parser.add_argument("--normalization", default="rank")
    parser.add_argument(
        "--baselines",
        default=(
            "best_single,"
            "cv_static_calibrated,"
            "uniform_ensemble,"
            "metadata_gate:openai_llm_gate,"
            "domain_rule_gate,"
            "oracle_gate"
        ),
    )
    parser.add_argument("--case-baseline", default="best_single")
    parser.add_argument("--max-cases", type=int, default=8)
    args = parser.parse_args()

    records = load_jsonl(Path(args.input))
    results = build_results(
        records,
        primary_gate_name=args.primary_gate_name,
        normalization=args.normalization,
        static_input=Path(args.static_input) if args.static_input else None,
    )
    primary_method = f"candidate_gate:{args.primary_gate_name}"
    primary = results[primary_method]
    baselines = [item.strip() for item in args.baselines.split(",") if item.strip()]

    print(f"Loaded {len(records)} records from {args.input}")
    print(f"Primary: {primary.method} {primary.correct}/{primary.total}={primary.accuracy:.3f}")
    print("\n[method summary]")
    for method in [primary_method, *baselines]:
        result = results.get(method)
        if result is None:
            print(f"{method:<36} MISSING")
            continue
        print(f"{result.method:<36} {result.correct:>3}/{result.total:<3} {result.accuracy:.3f}")

    source_counts = Counter(str(record.metadata.get("source", "")) for record in records)
    groups = ["all", "math", "logic", *[f"source:{source}" for source in sorted(source_counts)]]
    comparisons: list[Comparison] = []
    for baseline_name in baselines:
        baseline = results.get(baseline_name)
        if baseline is None:
            continue
        for group in groups:
            comparisons.append(
                compare_results(
                    records,
                    primary=primary,
                    baseline=baseline,
                    group=group,
                )
            )

    print("\n[win/loss table]")
    print_comparison_table(comparisons)

    case_baseline = results.get(args.case_baseline)
    if case_baseline is not None:
        print_cases(
            collect_cases(
                records,
                primary=primary,
                baseline=case_baseline,
                normalization=args.normalization,
                case_type="wins",
                limit=args.max_cases,
            ),
            title=f"wins versus {case_baseline.method}",
        )
        print_cases(
            collect_cases(
                records,
                primary=primary,
                baseline=case_baseline,
                normalization=args.normalization,
                case_type="losses",
                limit=args.max_cases,
            ),
            title=f"losses versus {case_baseline.method}",
        )


if __name__ == "__main__":
    main()
