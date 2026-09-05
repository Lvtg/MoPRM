# Three-Expert No-RM Ablation after Label Cleanup

Date: 2026-09-05

This note re-runs the three-expert ablation after the conservative answer-label
cleanup. It should be read together with:

```text
notes/label_cleanup_v2_results.md
notes/three_expert_no_reasoning_rm_ablation.md
```

The earlier three-expert note used the pre-cleanup labels. This clean-label
rerun supersedes the old headline numbers for final reporting.

## Motivation

The original three-expert no-RM ablation removed the strongest expert,
`open_reasoning_rm`, to test whether the MoPRM result was mostly carried by a
single very strong reward model.

Under the old labels, removing `open_reasoning_rm` still produced a large
Candidate Gate-v2 gain:

```text
pre-clean-label no-RM result:
Candidate Gate-v2:       74 / 84 = 0.881
best no-RM single:       66 / 84 = 0.786
net headline gain:       +8 problems
```

However, the label-cleaning audit found that many candidate answers previously
marked wrong were actually answer-equivalent to the gold answer, especially
because of LaTeX wrappers and simple unit suffixes. Therefore, the no-RM
ablation needed to be re-run under the cleaned labels.

The updated question is:

> After removing the strongest RM and cleaning answer labels, do the remaining
> three experts still show useful routing signal?

## Setup

Clean-label input:

```text
data/scored/openai_hard_mix_scout320_n8_mixed_pool_routed_clean_labels.jsonl
```

Temporary no-RM pool:

```bash
python scripts/rewrite_expert_pool.py \
  --input data/scored/openai_hard_mix_scout320_n8_mixed_pool_routed_clean_labels.jsonl \
  --output data/scored/tmp_no_open_reasoning_rm_clean_labels_mean.jsonl \
  --drop open_reasoning_rm \
  --overwrite
```

Remaining experts:

```text
open_math_prm
openai_general_judge
openai_reflective_judge
```

Unless otherwise stated:

```text
normalization: rank
folds: 5
seed: 41
epochs: 800
lr: 0.05
l2: 0.02
```

Two evaluation views are reported:

```text
1. clean mixed 67:
   Only records still mixed after label cleanup.

2. original 84 with clean labels:
   The original mixed pool, including 17 records that became all-correct after
   label cleanup.
```

This distinction matters because the label cleanup shrinks the hard mixed
subset from 84 to 67 examples.

## Clean-label candidate availability

After label cleanup:

```text
overall  avg_correct_candidates=5.655 all_wrong=0 all_correct=17 mixed=67 candidate_upper=84/84
logic    avg_correct_candidates=5.217 all_wrong=0 all_correct=0  mixed=23 candidate_upper=23/23
math     avg_correct_candidates=5.820 all_wrong=0 all_correct=17 mixed=44 candidate_upper=61/61
```

The original 84 examples are no longer all hard mixed cases. Seventeen math
examples become all-correct under the cleaned labels.

## Three-expert baselines under cleaned labels

With `open_reasoning_rm` removed:

```text
overall, original 84:
single:open_math_prm              58 / 84 = 0.690
single:openai_general_judge       79 / 84 = 0.940
single:openai_reflective_judge    79 / 84 = 0.940
uniform_ensemble                  75 / 84 = 0.893
domain_rule_gate                  67 / 84 = 0.798
OpenAI LLM gate                   72 / 84 = 0.857
expert top-choice oracle          81 / 84 = 0.964
```

By domain:

```text
math, original 61:
single:open_math_prm              45 / 61 = 0.738
single:openai_general_judge       57 / 61 = 0.934
single:openai_reflective_judge    57 / 61 = 0.934
uniform_ensemble                  53 / 61 = 0.869
domain_rule_gate                  45 / 61 = 0.738
OpenAI LLM gate                   50 / 61 = 0.820
expert top-choice oracle          59 / 61 = 0.967

logic, original 23:
single:open_math_prm              13 / 23 = 0.565
single:openai_general_judge       22 / 23 = 0.957
single:openai_reflective_judge    22 / 23 = 0.957
uniform_ensemble                  22 / 23 = 0.957
domain_rule_gate                  22 / 23 = 0.957
OpenAI LLM gate                   22 / 23 = 0.957
expert top-choice oracle          22 / 23 = 0.957
```

Compared with the pre-clean-label ablation, the two OpenAI judge-style experts
become much stronger. This is the most important change:

```text
pre-clean-label:
openai_general_judge       63 / 84 = 0.750
openai_reflective_judge    66 / 84 = 0.786

clean-label:
openai_general_judge       79 / 84 = 0.940
openai_reflective_judge    79 / 84 = 0.940
```

So a substantial part of the older no-RM routing gain came from label noise.

## Complementarity

Top-choice success patterns for the three experts under cleaned labels:

```text
expert order:
open_math_prm, openai_general_judge, openai_reflective_judge

111 count = 55
011 count = 22
000 count = 2
101 count = 1
100 count = 1
110 count = 1
001 count = 1
010 count = 1

all experts succeed: 55 / 84
no expert succeeds:   2 / 84

unique success:
open_math_prm             1
openai_general_judge      1
openai_reflective_judge   1
```

The three-expert pool still has disagreements, but much less
accuracy-relevant complementarity than before label cleanup. The old labels had
10 cases where no expert's top choice was correct; under cleaned labels this
drops to 2.

## Score separation

Raw score separation is measured as average expert score on correct candidates
minus average expert score on wrong candidates:

```text
overall:
open_math_prm             -0.040
openai_general_judge      +0.341
openai_reflective_judge   +0.404

math:
open_math_prm             -0.039
openai_general_judge      +0.332
openai_reflective_judge   +0.404

logic:
open_math_prm             -0.018
openai_general_judge      +0.347
openai_reflective_judge   +0.387
```

This explains why the no-RM pool is dominated by the OpenAI judge-style experts
after label cleanup. `open_math_prm` is still useful in individual cases, but
its average scalar score is not well separated on this candidate pool.

## Static aggregation sweep

The non-cross-validated static sweep is only a diagnostic. It shows the best
static weights when tuned on the same evaluation records.

Under mean/rank:

```text
original 84:
best static = 79 / 84 = 0.940
weights     = openai_general_judge: 1.0

clean mixed 67:
best static = 62 / 67 = 0.925
weights     = openai_general_judge: 1.0
```

Under last/rank, static mixing can use a small amount of `open_math_prm`:

```text
original 84:
best static = 81 / 84 = 0.964
weights     = open_math_prm: 0.2, openai_general_judge: 0.8

clean mixed 67:
best static = 64 / 67 = 0.955
weights     = open_math_prm: 0.2, openai_general_judge: 0.8
```

This suggests that the `open_math_prm` aggregation choice matters. The
candidate-aware model should therefore report aggregation sensitivity instead
of treating mean aggregation as intrinsically best.

## Candidate Gate-v2 without `open_reasoning_rm`

### Clean mixed 67 only

Command:

```bash
python scripts/train_candidate_gate_v2.py \
  --input data/scored/tmp_no_open_reasoning_rm_clean_labels_mean.jsonl \
  --output-dir data/scored/tmp_candidate_gate_v2_no_rm_clean_labels_mixed67_l2_02 \
  --math-aggregations mean,min,last,geomean \
  --normalization rank \
  --folds 5 \
  --seed 41 \
  --epochs 800 \
  --lr 0.05 \
  --l2 0.02 \
  --grid-step 0.1
```

Best tested aggregation on clean mixed 67:

```text
open_math_prm aggregation: last

Candidate Gate-v2:       63 / 67 = 0.940
best single expert:      62 / 67 = 0.925  # openai_reflective_judge
CV-static calibration:   61 / 67 = 0.910
uniform ensemble:        62 / 67 = 0.925
OpenAI LLM gate:         52 / 67 = 0.776
domain-rule gate:        49 / 67 = 0.731
expert top-choice oracle:66 / 67 = 0.985
```

By domain:

```text
math:  40 / 44 = 0.909
logic: 23 / 23 = 1.000
```

Aggregation sweep:

```text
mean:    62 / 67 = 0.925
min:     61 / 67 = 0.910
last:    63 / 67 = 0.940
geomean: 62 / 67 = 0.925
```

Win/loss against the best no-RM single expert:

```text
overall: 3 wins, 2 losses, net +1
math:    2 wins, 2 losses, net +0
logic:   1 win,  0 losses, net +1
```

Win/loss against no-RM CV-static calibration:

```text
overall: 4 wins, 2 losses, net +2
math:    3 wins, 2 losses, net +1
logic:   1 win,  0 losses, net +1
```

### All original 84 records with clean labels

Command:

```bash
python scripts/train_candidate_gate_v2.py \
  --input data/scored/tmp_no_open_reasoning_rm_clean_labels_mean.jsonl \
  --output-dir data/scored/tmp_candidate_gate_v2_no_rm_clean_labels_all84_l2_02 \
  --math-aggregations mean,min,last,geomean \
  --normalization rank \
  --folds 5 \
  --seed 41 \
  --epochs 800 \
  --lr 0.05 \
  --l2 0.02 \
  --grid-step 0.1 \
  --include-non-mixed
```

Best tested aggregations on all 84:

```text
open_math_prm aggregation: mean or geomean

Candidate Gate-v2:       81 / 84 = 0.964
best single expert:      79 / 84 = 0.940  # openai_reflective_judge
CV-static calibration:   77 / 84 = 0.917
uniform ensemble:        75 / 84 = 0.893
OpenAI LLM gate:         72 / 84 = 0.857
domain-rule gate:        67 / 84 = 0.798
expert top-choice oracle:81 / 84 = 0.964
```

By domain:

```text
math:  58 / 61 = 0.951
logic: 23 / 23 = 1.000
```

Aggregation sweep:

```text
mean:    81 / 84 = 0.964
min:     78 / 84 = 0.929
last:    80 / 84 = 0.952
geomean: 81 / 84 = 0.964
```

Win/loss against the best no-RM single expert:

```text
overall: 3 wins, 1 loss, net +2
math:    2 wins, 1 loss, net +1
logic:   1 win,  0 losses, net +1
```

Win/loss against no-RM CV-static calibration:

```text
overall: 5 wins, 1 loss, net +4
math:    4 wins, 1 loss, net +3
logic:   1 win,  0 losses, net +1
```

## Representative cases

Candidate Gate-v2 wins over the best no-RM single expert on:

```text
bbh_logical_deduction_seven_objects_0077
truths: CCWWWWWW
primary:  candidate 001, correct
baseline: candidate 006, wrong

math500_0018
truths: CWWWWWWW
primary:  candidate 000, correct
baseline: candidate 006, wrong

math500_0323
truths: CCCWWWCC
primary:  candidate 000, correct
baseline: candidate 005, wrong
```

The main repeated loss is:

```text
math500_0358
truths: WCWCCCCC
primary:  candidate 002, wrong
baseline: candidate 006, correct
```

On the clean mixed 67 view, `math500_0377` is another loss against the best
single expert.

## Interpretation

The clean-label result changes the three-expert story:

```text
pre-clean-label no-RM:
Candidate Gate-v2: 74 / 84 = 0.881
best single:       66 / 84 = 0.786
gain:              +8 problems

clean-label no-RM, clean mixed 67:
Candidate Gate-v2: 63 / 67 = 0.940
best single:       62 / 67 = 0.925
gain:              +1 problem

clean-label no-RM, all original 84:
Candidate Gate-v2: 81 / 84 = 0.964
best single:       79 / 84 = 0.940
gain:              +2 problems
```

The old no-RM result was directionally useful but overstated the size of the
routing gain because the labels were noisy. After label cleanup, the remaining
three-expert pool is already very strong, especially because the two OpenAI
judge-style experts reach 79 / 84.

The strongest defensible claim is:

> Removing `open_reasoning_rm` does not collapse the system. Candidate-aware
> routing still provides a small positive gain under cleaned labels, but the
> effect is much smaller than the pre-clean-label ablation suggested.

For the final report, this should be presented as a robustness ablation, not as
the main result. The main result should remain the heterogeneous four-expert
pool with clean-label evaluation, while this no-RM analysis addresses the
concern that the method only works because `open_reasoning_rm` is strong.

## Suggested reporting language

Use cautious wording:

```text
To test whether MoPRM was dominated by the strongest reward model, we removed
`open_reasoning_rm` and re-evaluated the remaining three experts after label
cleanup. The candidate-aware gate still improved over the best remaining
single expert, but only modestly: +1 problem on the clean mixed subset and +2
problems on the original 84-example pool. This indicates that the routing
mechanism does not collapse without the strongest RM, while also showing that
part of the earlier, larger no-RM gain came from answer-checking noise.
```
