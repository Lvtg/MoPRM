# Three-Expert No-RM Ablation

Date: 2026-09-04

This note records a small but important ablation: remove the strongest expert,
`open_reasoning_rm`, and evaluate whether the remaining three experts still
contain useful mixture/routing signal.

## Motivation

In the four-expert MoPRM pool, `open_reasoning_rm` is the strongest single
expert on `hard_mix_scout_320_n8_mixed`:

```text
single:open_reasoning_rm = 67 / 84 = 0.798
```

This raises a reasonable concern: maybe the MoPRM result is mostly carried by
one very strong reward model, rather than by meaningful expert complementarity.
To test this, we remove `open_reasoning_rm` and keep only:

```text
open_math_prm
openai_general_judge
openai_reflective_judge
```

The question is not whether this three-expert pool is the final system. The
question is diagnostic:

> If the strongest RM is removed, do the remaining experts still have enough
> complementary signal for mixture/routing to matter?

## Setup

Input:

```text
data/scored/openai_hard_mix_scout320_n8_mixed_pool_routed.jsonl
```

Temporary no-RM pool:

```bash
python scripts/rewrite_expert_pool.py \
  --input data/scored/openai_hard_mix_scout320_n8_mixed_pool_routed.jsonl \
  --output data/scored/tmp_no_open_reasoning_rm_mean.jsonl \
  --drop open_reasoning_rm \
  --overwrite
```

The reported split is the 84-problem mixed subset:

```text
overall mixed: 84
math mixed:    61 MATH500
logic mixed:   23 BBH logical_deduction_seven_objects
```

Unless otherwise noted:

```text
open_math_prm aggregation: mean
normalization: rank
```

## Three-Expert Baselines

With `open_reasoning_rm` removed:

```text
overall:
single:open_math_prm              46 / 84 = 0.548
single:openai_general_judge       63 / 84 = 0.750
single:openai_reflective_judge    66 / 84 = 0.786
uniform_ensemble                  62 / 84 = 0.738
domain_rule_gate                  55 / 84 = 0.655
OpenAI LLM gate                   59 / 84 = 0.702
expert top-choice oracle          73 / 84 = 0.869
```

By domain:

```text
math mixed:
single:open_math_prm              33 / 61 = 0.541
single:openai_general_judge       41 / 61 = 0.672
single:openai_reflective_judge    44 / 61 = 0.721
uniform_ensemble                  40 / 61 = 0.656
domain_rule_gate                  33 / 61 = 0.541
OpenAI LLM gate                   37 / 61 = 0.607
expert top-choice oracle          51 / 61 = 0.836

logic mixed:
single:open_math_prm              13 / 23 = 0.565
single:openai_general_judge       22 / 23 = 0.957
single:openai_reflective_judge    22 / 23 = 0.957
uniform_ensemble                  22 / 23 = 0.957
domain_rule_gate                  22 / 23 = 0.957
OpenAI LLM gate                   22 / 23 = 0.957
expert top-choice oracle          22 / 23 = 0.957
```

Interpretation:

- `open_reasoning_rm` is clearly strong, but removing it does not destroy the
  pool.
- `openai_reflective_judge` becomes the best single expert.
- The three-expert oracle remains high at `73 / 84`, showing that the remaining
  experts still disagree in accuracy-relevant ways.
- Logic is still mostly solved by the two OpenAI judges, so the interesting
  signal remains concentrated in MATH500 mixed examples.

## Complementarity

Top-choice success patterns for the three experts:

```text
expert order:
open_math_prm, openai_general_judge, openai_reflective_judge

111 count = 38
011 count = 21
000 count = 10
001 count = 5
100 count = 4
101 count = 2
010 count = 2
110 count = 2

all experts succeed: 38 / 84
no expert succeeds:  10 / 84

unique success:
open_math_prm             4
openai_general_judge      2
openai_reflective_judge   5
```

This is useful for the paper story. `open_math_prm` is weak as a standalone
selector, but it is not useless: it uniquely rescues 4 problems that neither
OpenAI judge gets right as its own top choice.

## Score Separation

Raw score separation, measured as average score on correct candidates minus
average score on wrong candidates:

```text
overall:
open_math_prm             +0.006
openai_general_judge      +0.208
openai_reflective_judge   +0.241

math:
open_math_prm             -0.001
openai_general_judge      +0.179
openai_reflective_judge   +0.211

logic:
open_math_prm             -0.018
openai_general_judge      +0.347
openai_reflective_judge   +0.387
```

This explains why static routing prefers the reflective judge: it has the
cleanest raw separation among the remaining experts.

## Static Aggregation Sweep

When `open_reasoning_rm` is removed, simple static mixtures mostly collapse to
the best OpenAI judge:

```text
mean + rank:
best static = 66 / 84 = 0.786
weights     = openai_reflective_judge: 1.0

min + rank:
best static = 66 / 84 = 0.786
weights     = openai_reflective_judge: 1.0

last + minmax:
best static = 67 / 84 = 0.798
weights     = open_math_prm: 0.2,
              openai_general_judge: 0.7,
              openai_reflective_judge: 0.1

last + zscore:
best static = 67 / 84 = 0.798
weights     = open_math_prm: 0.2,
              openai_general_judge: 0.8
```

So the three-expert pool does contain some complementary signal, but naive
static mixing does not reliably exploit it.

## Candidate Gate-v2 Without `open_reasoning_rm`

Command:

```bash
python scripts/train_candidate_gate_v2.py \
  --input data/scored/tmp_no_open_reasoning_rm_mean.jsonl \
  --output-dir data/scored/tmp_candidate_gate_v2_no_rm_l2_02 \
  --math-aggregations mean,min,last,geomean \
  --normalization rank \
  --folds 5 \
  --seed 41 \
  --epochs 800 \
  --lr 0.05 \
  --l2 0.02 \
  --grid-step 0.1
```

Candidate Gate-v2 result:

```text
mean aggregation:
overall Candidate Gate-v2   74 / 84 = 0.881
math Candidate Gate-v2      52 / 61 = 0.852
logic Candidate Gate-v2     22 / 23 = 0.957

geomean aggregation:
overall Candidate Gate-v2   71 / 84 = 0.845

last aggregation:
overall Candidate Gate-v2   70 / 84 = 0.833

min aggregation:
overall Candidate Gate-v2   66 / 84 = 0.786
```

For the main no-RM setting, `open_math_prm` mean aggregation is best.

Comparison under mean aggregation:

```text
Candidate Gate-v2           74 / 84 = 0.881
best single expert          66 / 84 = 0.786  # openai_reflective_judge
CV-static calibration        64 / 84 = 0.762
uniform ensemble             62 / 84 = 0.738
OpenAI LLM gate              59 / 84 = 0.702
domain-rule gate             55 / 84 = 0.655
expert top-choice oracle     73 / 84 = 0.869
```

Important caveat: the `oracle_gate` in the evaluator is an expert top-choice
oracle. It only chooses among each expert's own top candidate. Candidate Gate-v2
is a candidate-level learned selector, so it can choose a candidate that is not
the top choice of any single expert. Therefore `74 / 84` being above the
expert top-choice oracle `73 / 84` is possible and does not imply leakage.

The candidate upper bound remains:

```text
candidate upper bound = 84 / 84 = 1.000
```

## Win/Loss Summary

Candidate Gate-v2 without `open_reasoning_rm` versus the no-RM best single
expert, `openai_reflective_judge`:

```text
overall: 12 wins, 4 losses, net +8
math:    12 wins, 4 losses, net +8
logic:    0 wins, 0 losses, net +0
```

Against no-RM CV-static calibration:

```text
overall: 13 wins, 3 losses, net +10
math:    13 wins, 3 losses, net +10
logic:    0 wins, 0 losses, net +0
```

Against other baselines:

```text
uniform ensemble:      16 wins, 4 losses, net +12
OpenAI LLM gate:       16 wins, 1 loss,  net +15
domain-rule gate:      20 wins, 1 loss,  net +19
expert oracle:          6 wins, 5 losses, net +1
```

The gains are almost entirely from MATH500 mixed examples.

## Takeaway

This ablation answers the original concern in a nuanced way.

`open_reasoning_rm` is indeed very strong. In the full four-expert pool, it is
the best single expert and can dominate simple expert-level comparisons.

However, the MoPRM result is not only a story of one strong RM carrying the
system. After removing `open_reasoning_rm`, the remaining three experts still
show substantial complementarity:

```text
three-expert top-choice oracle: 73 / 84
Candidate Gate-v2 no-RM:       74 / 84
best no-RM single expert:      66 / 84
```

This supports a stronger interpretation:

> Candidate-aware routing can exploit score-pattern information even when the
> strongest standalone RM is removed.

The main remaining caution is different:

> In the no-RM pool, two of the three experts are OpenAI judges. So this
> ablation reduces concern about `open_reasoning_rm` dominance, but it increases
> the need to be careful about how much of the no-RM signal comes from OpenAI
> judge-style scoring.

For the final report, this should be presented as an ablation, not as the main
method. The main method should remain the heterogeneous four-expert pool, while
this no-RM experiment shows that the positive V2 behavior does not vanish when
the strongest expert is removed.
