# Conservative Label Cleanup and Candidate Gate-v2 Re-evaluation

Date: 2026-09-05

## What changed

The answer checker was updated with two conservative normalization fixes:

```text
1. Remove LaTeX inline/display math wrappers:
   \( ... \), \[ ... \]

2. For numeric answers only, ignore simple unit suffixes:
   36 seconds == 36
```

The checker still keeps percentage semantics:

```text
36% != 36
```

The motivation came from the Candidate Gate-v2 loss audit. Many reported math
losses were answer-equivalent to the gold answer but were marked wrong because
the selected candidate used inline LaTeX wrappers or a simple unit suffix.

## Commands

Relabel the scored expert pool without recomputing expert scores:

```bash
python scripts/label_candidate_correctness.py \
  --input data/scored/openai_hard_mix_scout320_n8_mixed_pool_routed.jsonl \
  --output data/scored/openai_hard_mix_scout320_n8_mixed_pool_routed_clean_labels.jsonl \
  --overwrite
```

Relabel the previous Candidate Gate-v2 OOF output for direct apples-to-apples
evaluation of the old V2 selections:

```bash
python scripts/label_candidate_correctness.py \
  --input data/scored/candidate_gate_v2_l2_02/candidate_gate_v2_cv_mean_rank.jsonl \
  --output data/scored/candidate_gate_v2_l2_02/candidate_gate_v2_cv_mean_rank_clean_labels_eval.jsonl \
  --overwrite
```

Retrain Candidate Gate-v2 on the cleaned labels, using the same main
configuration:

```bash
python scripts/train_candidate_gate_v2.py \
  --input data/scored/openai_hard_mix_scout320_n8_mixed_pool_routed_clean_labels.jsonl \
  --output-dir data/scored/candidate_gate_v2_clean_labels_l2_02 \
  --math-aggregations mean,min,last,geomean \
  --normalization rank \
  --folds 5 \
  --seed 41 \
  --epochs 800 \
  --lr 0.05 \
  --l2 0.02 \
  --grid-step 0.1
```

Also retrain/evaluate on all 84 originally mixed records, keeping the 17 newly
all-correct records instead of filtering them out:

```bash
python scripts/train_candidate_gate_v2.py \
  --input data/scored/openai_hard_mix_scout320_n8_mixed_pool_routed_clean_labels.jsonl \
  --output-dir data/scored/candidate_gate_v2_clean_labels_all84_l2_02 \
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

## Label-change summary

Input records:

```text
84 problems
672 generated candidates
```

Candidate-level labels:

```text
old correct candidates: 371 / 672
new correct candidates: 475 / 672
changed records:        23 / 84
```

Per-problem correct-count histogram:

```text
old:
1 correct candidate: 10
2 correct candidates: 12
3 correct candidates: 11
4 correct candidates: 6
5 correct candidates: 9
6 correct candidates: 17
7 correct candidates: 19

new:
1 correct candidate: 4
2 correct candidates: 5
3 correct candidates: 7
4 correct candidates: 5
5 correct candidates: 10
6 correct candidates: 18
7 correct candidates: 18
8 correct candidates: 17
```

The original 84 examples were all mixed under the old labels. After label
cleanup:

```text
clean mixed records:     67
newly all-correct records: 17
all-wrong records:        0
```

This confirms that the previous mixed subset contained a substantial number of
formatting-induced pseudo-errors.

## Old Candidate Gate-v2 selections under cleaned labels

Using the old out-of-fold Candidate Gate-v2 scores but cleaned labels:

```text
all original 84 records:
Candidate Gate-v2:       83 / 84 = 0.988
best single expert:      81 / 84 = 0.964  # open_reasoning_rm
uniform ensemble:        80 / 84 = 0.952
domain-rule gate:        68 / 84 = 0.810
OpenAI LLM gate:         73 / 84 = 0.869
expert oracle:           82 / 84 = 0.976

clean mixed 67 records:
Candidate Gate-v2:       66 / 67 = 0.985
best single expert:      64 / 67 = 0.955  # open_reasoning_rm
uniform ensemble:        63 / 67 = 0.940
domain-rule gate:        51 / 67 = 0.761
OpenAI LLM gate:         56 / 67 = 0.836
expert oracle:           65 / 67 = 0.970
```

The only remaining wrong old-V2 selection is:

```text
bbh_logical_deduction_seven_objects_0077

gold: C
selected final answer: (D) Ana finished last
truths: CCWWWWWW

expert top choices:
open_math_prm             -> 000, correct
open_reasoning_rm         -> 001, correct
openai_general_judge      -> 006, wrong
openai_reflective_judge   -> 006, wrong
```

This is a genuine semantic logic failure caused by two OpenAI judge-style
experts agreeing on a wrong candidate.

## Retrained Candidate Gate-v2 under cleaned labels

### Clean mixed 67 only

When the newly all-correct examples are filtered out, the clean mixed subset has
67 records:

```text
best setting among tested aggregations: mean + rank + l2=0.02

Candidate Gate-v2:       64 / 67 = 0.955
CV static calibration:   64 / 67 = 0.955
best single expert:      64 / 67 = 0.955  # open_reasoning_rm
uniform ensemble:        63 / 67 = 0.940
domain-rule gate:        51 / 67 = 0.761
OpenAI LLM gate:         56 / 67 = 0.836
expert oracle:           65 / 67 = 0.970
```

By domain:

```text
math:  42 / 44 = 0.955
logic: 22 / 23 = 0.957
```

On this cleaned mixed subset, the learned gate no longer clearly beats the best
single/static baselines. This is expected: after label cleanup the subset is
smaller and much easier, leaving less room for routing gains.

### All original 84 records

Keeping all 84 originally mixed records after label cleanup:

```text
best settings among tested aggregations: mean/geomean + rank + l2=0.02

Candidate Gate-v2:       82 / 84 = 0.976
CV static calibration:   81 / 84 = 0.964
best single expert:      81 / 84 = 0.964  # open_reasoning_rm
uniform ensemble:        80 / 84 = 0.952
domain-rule gate:        68 / 84 = 0.810
OpenAI LLM gate:         73 / 84 = 0.869
expert oracle:           82 / 84 = 0.976
```

By domain:

```text
math:  59 / 61 = 0.967
logic: 23 / 23 = 1.000
```

Win/loss against best single expert:

```text
overall: 2 wins, 1 loss, net +1
math:    2 wins, 1 loss, net +1
logic:   0 wins, 0 losses
```

Retrained all-84 Candidate Gate-v2 wrong cases:

```text
math500_0030
gold: 52_8
selected final answer: 52

math500_0473
gold: 7
selected final answer: :
```

`math500_0030` is a borderline base-notation case: the selected solution
explains `52_8`, but its final answer line is only `52`. It may be acceptable
in context, but the current conservative checker does not read problem context
or infer omitted base notation.

`math500_0473` is a genuine extraction/completion failure: the selected
candidate's final answer is incomplete.

## Interpretation

The cleaned-label result changes the story:

```text
Before cleanup:
Candidate Gate-v2: 72 / 84 = 0.857

Old V2 selections under cleaned labels:
Candidate Gate-v2: 83 / 84 = 0.988

Retrained V2 on cleaned labels, all original 84 records:
Candidate Gate-v2: 82 / 84 = 0.976

Retrained V2 on cleaned mixed 67:
Candidate Gate-v2: 64 / 67 = 0.955
```

Therefore the earlier V2 errors were mostly not model failures. They were
mostly automatic answer-checking artifacts.

For the project narrative:

```text
1. Candidate Gate-v2 is robust under cleaned labels and remains at or above all
   practical baselines.
2. The remaining margin over best single/static baselines is smaller after
   cleanup because the task becomes easier and 17 cases become all-correct.
3. The most valuable next step is not Gate-v3 training, but larger/fresher data
   with cleaner labels.
4. If a Gate-v3 is included, it should be framed as future work or a small
   diagnostic using frozen problem embeddings/tags, not as the main result.
```
