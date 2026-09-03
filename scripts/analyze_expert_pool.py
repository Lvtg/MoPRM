from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moprm.evaluate import (  # noqa: E402
    attach_normalized_scores,
    default_baselines,
    select_candidate,
    single_expert_gate,
)
from moprm.io import load_jsonl  # noqa: E402


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _avg(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _candidate_truths(record) -> str:
    return "".join("C" if candidate.is_correct else "W" for candidate in record.candidates)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze expert complementarity, gate weights, and oracle gaps."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--normalization", default="rank", choices=["rank", "minmax", "zscore"])
    parser.add_argument("--metadata-gate", default="openai_llm_gate")
    parser.add_argument("--max-cases", type=int, default=8)
    args = parser.parse_args()

    records = load_jsonl(Path(args.input))
    normalized_records = [attach_normalized_scores(record, method=args.normalization) for record in records]
    experts = sorted({expert for record in normalized_records for expert in record.expert_names()})

    print(f"Loaded {len(records)} problems from {args.input}")
    print(f"Experts: {', '.join(experts)}")

    print("\n[correct candidate availability]")
    by_domain_counts: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for record in records:
        by_domain_counts[record.domain].append(
            (
                sum(1 for candidate in record.candidates if candidate.is_correct),
                len(record.candidates),
            )
        )
    for domain in ["overall", *sorted(by_domain_counts)]:
        count_pairs = (
            [item for values in by_domain_counts.values() for item in values]
            if domain == "overall"
            else by_domain_counts[domain]
        )
        if not count_pairs:
            continue
        counts = [correct for correct, _total in count_pairs]
        print(
            f"{domain:<8} avg_correct_candidates={_fmt(_avg(counts))} "
            f"all_wrong={sum(1 for count in counts if count == 0)} "
            f"all_correct={sum(1 for correct, total in count_pairs if correct == total)}"
        )

    print("\n[top-choice complementarity]")
    expert_correct = {expert: 0 for expert in experts}
    all_same = 0
    any_disagreement = 0
    disagreement_cases: list[tuple[str, str, dict[str, str], str]] = []
    for record in normalized_records:
        choices: dict[str, str] = {}
        for expert in experts:
            selected = select_candidate(record, single_expert_gate(expert)(record, experts))
            choices[expert] = selected.candidate_id
            expert_correct[expert] += 1 if selected.is_correct else 0
        if len(set(choices.values())) == 1:
            all_same += 1
        else:
            any_disagreement += 1
            if len(disagreement_cases) < args.max_cases:
                disagreement_cases.append(
                    (record.problem_id, record.domain, choices, _candidate_truths(record))
                )
    print(f"all_experts_same_top_choice={all_same}")
    print(f"any_expert_disagreement={any_disagreement}")
    for expert, correct in expert_correct.items():
        print(f"single:{expert:<24} {correct:>3}/{len(normalized_records)}")

    print("\n[gate weights]")
    gate_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        gate_weights = record.metadata.get("gate_weights", {})
        if not isinstance(gate_weights, dict):
            continue
        weights = gate_weights.get(args.metadata_gate)
        if not isinstance(weights, dict):
            continue
        for expert in experts:
            gate_values["overall"][expert].append(float(weights.get(expert, 0.0)))
            gate_values[record.domain][expert].append(float(weights.get(expert, 0.0)))
    if not gate_values:
        print(f"No metadata gate named {args.metadata_gate!r} found.")
    else:
        for domain in ["overall", *sorted(k for k in gate_values if k != "overall")]:
            print(f"{domain}:")
            for expert in experts:
                print(f"  {expert:<24} {_fmt(_avg(gate_values[domain][expert]))}")

    print("\n[raw score separation]")
    score_groups: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for record in records:
        for candidate in record.candidates:
            label = "correct" if candidate.is_correct else "wrong"
            for expert, score in candidate.expert_scores.items():
                if expert not in experts:
                    continue
                score_groups["overall"][expert][label].append(score)
                score_groups[record.domain][expert][label].append(score)
    for domain in ["overall", *sorted(k for k in score_groups if k != "overall")]:
        print(f"{domain}:")
        for expert in experts:
            correct_scores = score_groups[domain][expert]["correct"]
            wrong_scores = score_groups[domain][expert]["wrong"]
            if not correct_scores and not wrong_scores:
                continue
            delta = _avg(correct_scores) - _avg(wrong_scores)
            print(
                f"  {expert:<24} "
                f"correct={_fmt(_avg(correct_scores))} "
                f"wrong={_fmt(_avg(wrong_scores))} "
                f"delta={_fmt(delta)}"
            )

    print("\n[oracle gaps]")
    results = {result.method: result for result in default_baselines(records, args.normalization)}
    oracle = results.get("oracle_gate")
    if oracle is None:
        print("No oracle result.")
        return
    methods = [
        "domain_rule_gate",
        f"metadata_gate:{args.metadata_gate}",
        "uniform_ensemble",
    ]
    for method in methods:
        result = results.get(method)
        if result is None:
            continue
        selection_diff_cases = []
        accuracy_gap_cases = []
        for record in records:
            problem_id = record.problem_id
            selected_id = result.selections.get(problem_id, "")
            oracle_id = oracle.selections.get(problem_id, "")
            if oracle_id == selected_id:
                continue
            by_id = {candidate.candidate_id: candidate for candidate in record.candidates}
            selected_candidate = by_id.get(selected_id)
            oracle_candidate = by_id.get(oracle_id)
            case = (
                problem_id,
                record.domain,
                selected_id,
                oracle_id,
                _candidate_truths(record),
            )
            selection_diff_cases.append(case)
            if (
                selected_candidate is not None
                and oracle_candidate is not None
                and not selected_candidate.is_correct
                and oracle_candidate.is_correct
            ):
                accuracy_gap_cases.append(case)
        print(
            f"{method:<32} accuracy={_fmt(result.accuracy)} "
            f"selection_diff={len(selection_diff_cases)} "
            f"accuracy_gap={len(accuracy_gap_cases)}"
        )
        for problem_id, domain, selected, oracle_selected, truths in accuracy_gap_cases[
            : args.max_cases
        ]:
            print(
                f"  {problem_id} ({domain}) selected={selected} "
                f"oracle={oracle_selected} truths={truths}"
            )


if __name__ == "__main__":
    main()
