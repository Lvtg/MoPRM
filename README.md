# MoPRM

MoPRM is a course-project experiment on **router-guided mixtures of heterogeneous process reward models** for reasoning tasks.

The project studies whether a lightweight gate can choose or weight multiple independently sourced PRM/verifier experts for different problems and candidate solutions, improving Best-of-N selection compared with any single expert, a naive ensemble, or an OpenAI-only multi-rubric judge baseline.

## Project Goal

The revised goal is to build a **heterogeneous MoPRM** system:

- at least **two non-OpenAI PRM/reward experts** in the main expert pool;
- one math-specialized open-source PRM;
- one logic/reasoning-specialized open-source PRM or reward model;
- two OpenAI-based judges kept as general and reflective baselines/supporting experts;
- a question-level gate first, followed by a trained gate once heterogeneous scores exist.

The current OpenAI scorer is intentionally treated as an **OpenAI multi-rubric judge baseline**, not as the final MoPRM expert pool.

## Current Status

As of 2026-09-04, the project has:

- integrated the main heterogeneous expert pool;
- run `hard_dev_100, N=8` as the first harder full split;
- expanded to `hard_mix_scout_320_n8`, producing 84 mixed candidate sets;
- implemented and evaluated Gate-v1, the first trained question-level router;
- implemented Candidate-aware Gate-v2, which is the current best trained method.

Main expert pool:

| Expert name | Source | Score type |
|---|---|---|
| `open_math_prm` | `Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B` | step-level math PRM; `min` step aggregation works best so far |
| `open_reasoning_rm` | `Skywork/Skywork-Reward-V2-Qwen3-1.7B` | response-level reward logit |
| `openai_general_judge` | OpenAI Responses API | general reliability rubric |
| `openai_reflective_judge` | OpenAI Responses API | self-checking / error-recovery rubric |

Latest result with the default mean aggregation for `open_math_prm`:

```text
split: hard_dev_100
candidates per problem: N=8
main pool: open_math_prm + open_reasoning_rm + openai_general_judge + openai_reflective_judge

overall:
domain_rule_gate              67 / 100 = 0.670
openai_llm_gate               69 / 100 = 0.690
uniform_ensemble              70 / 100 = 0.700
best single expert            71 / 100 = 0.710  # open_reasoning_rm
oracle_gate                   71 / 100 = 0.710
```

After reaggregating the cached Skywork math PRM step rewards with `min` instead
of `mean`, the main table improves:

```text
min aggregation + rank normalization:
domain_rule_gate              71 / 100 = 0.710
openai_llm_gate               70 / 100 = 0.700
uniform_ensemble              71 / 100 = 0.710
best single expert            71 / 100 = 0.710  # open_reasoning_rm
oracle_gate                   72 / 100 = 0.720

min aggregation + minmax normalization:
domain_rule_gate              71 / 100 = 0.710
openai_llm_gate               71 / 100 = 0.710
uniform_ensemble              70 / 100 = 0.700
best single expert            71 / 100 = 0.710  # open_reasoning_rm
oracle_gate                   72 / 100 = 0.720
```

Interpretation at this stage: the harder `N=8` run was much more useful than
`dev_40`, but `hard_dev_100_n8` still had limited routing headroom. This led to
the later mixed-rich scout splits below.

`hard_dev_100_n8` diagnostic finding:

```text
mixed problems:                  18 / 100
candidate upper bound:           77 / 100 full, 18 / 18 mixed
best single expert:              71 / 100 full, 12 / 18 mixed  # open_reasoning_rm
best calibrated static mixture:  72 / 100 full, 13 / 18 mixed
best static weights:             open_math_prm + openai_reflective_judge
```

The oracle gap was real but small. This diagnostic showed that the bottleneck
was not merely the gate, but candidate-set composition and expert calibration.

## Current Scope

Recommended 15-day scope:

- domains: math reasoning and logic reasoning, with code reasoning as an optional extension;
- experts: at least two non-OpenAI PRM/reward experts plus two OpenAI-based judge experts;
- gate: question-level trained router first, candidate-aware router only if time allows;
- main metric: `PRM@8`, with `PRM@16` as an extension;
- baselines: OpenAI multi-rubric judge, best single expert, uniform ensemble, domain-rule gate, LLM gate, trained gate, oracle gate.

Target expert pool:

| Expert name | Source type | Intended role |
|---|---|---|
| `open_math_prm` | non-OpenAI open-source PRM | math process correctness |
| `open_reasoning_rm` | non-OpenAI open-source RM | response-level logic and general reasoning quality |
| `openai_general_judge` | OpenAI judge | broad candidate reliability |
| `openai_reflective_judge` | OpenAI judge | self-checking and error-recovery quality |

## Main Plan

The current project goal is in [notes/project_goal.md](notes/project_goal.md).

The current experiment plan is in [notes/moprm_15_day_experiment_plan.md](notes/moprm_15_day_experiment_plan.md).

## Latest Experiment: `hard_dev_100_n8`

Completed run:

```text
split: hard_dev_100
total problems: 100
candidate count: N=8
math/logic mix: 60 math + 40 logic
math source mix: 50 MATH500 + 10 GSM8K
logic source mix: 20 BBH seven-object + 10 five-object + 10 three-object
```

Why this instead of another balanced `dev_80`:

- `dev_40, N=4` already has very little oracle gap, so simply doubling size is
  less useful than making the split harder.
- MATH500 should replace most GSM8K because GSM8K is too easy for the current
  generator/judges.
- `N=8` creates more candidate diversity and gives the gate more chances to
  differ from uniform or single-expert selection.
- `100 x 8 = 800` candidates is still manageable for the two local models and
  the OpenAI scoring budget.

Observed OpenAI API usage:

```text
candidate generation: 800 API calls, 408,180 tokens
OpenAI expert scoring: 800 API calls, 603,493 tokens
LLM gate routing: 100 API calls, 38,368 tokens
```

The two open-source experts run locally from `models/hf_cache` and should not
consume API budget.

Create the split:

```bash
python scripts/sample_dataset.py \
  --input data/cache/public_subsets/math_logic_combined.jsonl \
  --output data/splits/hard_dev_100.jsonl \
  --source-quota "math|HuggingFaceH4/MATH-500=50" \
  --source-quota "math|openai/gsm8k=10" \
  --source-quota "logic|BIG-Bench-Hard/logical_deduction_seven_objects=20" \
  --source-quota "logic|BIG-Bench-Hard/logical_deduction_five_objects=10" \
  --source-quota "logic|BIG-Bench-Hard/logical_deduction_three_objects=10" \
  --seed 23
```

Then run the candidate/scoring pipeline:

```bash
python scripts/generate_openai_candidates.py --input data/splits/hard_dev_100.jsonl --output data/candidates/openai_hard_dev100_n8.jsonl --limit 100 --num-candidates 8 --max-output-tokens 512 --temperature 1.0 --concurrency 4 --overwrite
python scripts/label_candidate_correctness.py --input data/candidates/openai_hard_dev100_n8.jsonl --output data/cache/openai_hard_dev100_n8_labeled.jsonl --overwrite
python scripts/score_openai_experts.py --input data/cache/openai_hard_dev100_n8_labeled.jsonl --output data/scored/openai_hard_dev100_n8_scored.jsonl --concurrency 4 --overwrite
```

Local open-source expert scoring:

```powershell
$env:HF_HUB_OFFLINE='1'
.\.venv_cuda\Scripts\python.exe scripts\score_skywork_math_prm.py --input data\scored\openai_hard_dev100_n8_scored.jsonl --output data\scored\openai_hard_dev100_n8_with_skywork_math.jsonl --domains all --device cuda --dtype float32 --overwrite
.\.venv_cuda\Scripts\python.exe scripts\score_skywork_reward_v2.py --input data\scored\openai_hard_dev100_n8_with_skywork_math.jsonl --output data\scored\openai_hard_dev100_n8_with_open_experts.jsonl --domains all --device cuda --dtype auto --overwrite
```

Build and evaluate the main heterogeneous pool:

```bash
python scripts/rewrite_expert_pool.py --input data/scored/openai_hard_dev100_n8_with_open_experts.jsonl --output data/scored/openai_hard_dev100_n8_two_open_expert_pool.jsonl --drop math_prm --drop logic_judge --rename general_judge=openai_general_judge --rename reflective_judge=openai_reflective_judge --overwrite
python scripts/route_openai_gate.py --input data/scored/openai_hard_dev100_n8_two_open_expert_pool.jsonl --output data/scored/openai_hard_dev100_n8_two_open_expert_pool_routed.jsonl --overwrite
python scripts/run_smoke_eval.py --input data/scored/openai_hard_dev100_n8_two_open_expert_pool_routed.jsonl --by-domain --by-source
python scripts/analyze_expert_pool.py --input data/scored/openai_hard_dev100_n8_two_open_expert_pool_routed.jsonl
```

Optional but currently recommended: reaggregate the cached `open_math_prm`
step rewards with `min` aggregation and evaluate again:

```bash
python scripts/reaggregate_skywork_math_prm.py --input data/scored/openai_hard_dev100_n8_two_open_expert_pool_routed.jsonl --output data/scored/openai_hard_dev100_n8_two_open_expert_pool_routed_math_min.jsonl --aggregation min --overwrite
python scripts/run_smoke_eval.py --input data/scored/openai_hard_dev100_n8_two_open_expert_pool_routed_math_min.jsonl --by-domain --by-source
python scripts/run_smoke_eval.py --input data/scored/openai_hard_dev100_n8_two_open_expert_pool_routed_math_min.jsonl --by-domain --by-source --normalization minmax
python scripts/analyze_expert_pool.py --input data/scored/openai_hard_dev100_n8_two_open_expert_pool_routed_math_min.jsonl
```

## Latest Scout: `hard_mix_scout_320_n8`

Completed on 2026-09-04 by appending a non-overlapping 160-problem batch to
`hard_mix_scout_160_n8`. Both batches use eight distinct candidate-generation
styles.

```text
split: 240 MATH500 + 80 BBH logical_deduction_seven_objects
candidates: 320 x 8 = 2560
candidate generation tokens: 1,459,205
OpenAI expert scoring tokens for mixed subset only: 618,571
LLM gate tokens for mixed subset: 32,194
```

Candidate correctness:

```text
correct candidates: 1379 / 2560 = 0.539
all-wrong problems: 110 / 320
all-correct problems: 126 / 320
mixed problems: 84 / 320
candidate upper bound: 210 / 320

mixed sources:
- MATH500: 61
- BBH seven-object: 23
```

Mixed-subset result with the main heterogeneous expert pool:

```text
experts:
- open_math_prm
- open_reasoning_rm
- openai_general_judge
- openai_reflective_judge

best single expert:        67 / 84 = 0.798  # open_reasoning_rm
best uniform ensemble:     66 / 84 = 0.786  # open_math_prm geomean + rank
best static mixture:       70 / 84 = 0.833  # open_math_prm min + rank
oracle gate:               76 / 84 = 0.905
```

Interpretation: this scout split is much more selection-informative than
`hard_dev_100_n8`. The mixed count now exceeds the target for a lightweight
trained-gate experiment, and the oracle gap is large enough to make routing
meaningful.

## Gate-v1: Lightweight Trained Router

Implemented on 2026-09-04:

| Component | Choice |
|---|---|
| Gate model | multi-label logistic regression |
| Features | question text hashes + lightweight metadata/text statistics |
| Labels | whether each expert's own top-ranked candidate is correct |
| Split | 5-fold source/domain-stratified cross-validation |
| Evaluation | out-of-fold `metadata_gate:trained_gate_v1_cv` weights |

Gate-v1 deliberately avoids large-model training. It learns a question-level
distribution over the four heterogeneous experts and writes out-of-fold weights
back into `record.metadata.gate_weights.trained_gate_v1_cv`, so the existing
MoPRM evaluator can score it exactly like the LLM gate.

Current Gate-v1 command:

```bash
python scripts/train_gate_v1.py \
  --input data/scored/openai_hard_mix_scout320_n8_mixed_pool_routed.jsonl \
  --output-dir data/scored/gate_v1_p4 \
  --math-aggregations mean,min,last,geomean \
  --normalization rank \
  --folds 5 \
  --seed 41 \
  --hash-dim 256 \
  --epochs 800 \
  --lr 0.05 \
  --l2 0.01 \
  --weight-power 4 \
  --grid-step 0.1
```

Main mixed-only CV result:

```text
best Gate-v1 setting: mean aggregation + rank normalization + weight_power=4

Gate-v1 CV:                 67 / 84 = 0.798
CV static calibration:      67 / 84 = 0.798
best single expert:         67 / 84 = 0.798  # open_reasoning_rm
uniform ensemble:           64 / 84 = 0.762
domain-rule gate:           56 / 84 = 0.667
OpenAI LLM gate:            60 / 84 = 0.714
expert oracle:              76 / 84 = 0.905
```

Best calibration diagnostic remains:

```text
open_math_prm min aggregation + rank normalization + CV static calibration
CV static calibration:      69 / 84 = 0.821
Gate-v1 CV:                 63 / 84 = 0.750
```

Interpretation: Gate-v1 is now a working trained-router baseline and improves
over uniform/domain/LLM routing in its best setting, but it does not yet beat
the strongest cross-validated static calibration. This is a useful boundary:
the next research step should make the gate less purely question-level, or give
it better labels/features, rather than only claiming that a learned gate already
wins.

## Gate-v2: Candidate-Aware Learned Selector

Implemented on 2026-09-04:

| Component | Choice |
|---|---|
| Gate model | candidate-level logistic selector |
| Features | candidate text statistics + per-expert raw/rank scores + expert top-choice indicators + score margins |
| Train labels | candidate final-answer correctness, available only in training folds |
| Split | 5-fold source/domain-stratified split by problem, not by candidate |
| Evaluation | out-of-fold candidate scores; select highest-scoring candidate per problem |

This version is more candidate-aware than Gate-v1. Gate-v1 predicts one expert
weight vector per problem; Candidate Gate-v2 predicts a learned score for each
candidate using the full expert-score pattern available at selection time. It
does not see the gold answer at inference.

Current main command:

```bash
python scripts/train_candidate_gate_v2.py \
  --input data/scored/openai_hard_mix_scout320_n8_mixed_pool_routed.jsonl \
  --output-dir data/scored/candidate_gate_v2_l2_02 \
  --math-aggregations mean,min,last,geomean \
  --normalization rank \
  --folds 5 \
  --seed 41 \
  --epochs 800 \
  --lr 0.05 \
  --l2 0.02 \
  --grid-step 0.1
```

Main result on `hard_mix_scout_320_n8_mixed`:

```text
best setting: open_math_prm mean aggregation + rank normalization + l2=0.02

Candidate Gate-v2:         72 / 84 = 0.857
CV static calibration:     67 / 84 = 0.798
best single expert:        67 / 84 = 0.798  # open_reasoning_rm
uniform ensemble:          64 / 84 = 0.762
domain-rule gate:          56 / 84 = 0.667
OpenAI LLM gate:           60 / 84 = 0.714
expert oracle:             76 / 84 = 0.905

math mixed:
Candidate Gate-v2:         50 / 61 = 0.820
best single expert:        44 / 61 = 0.721
expert oracle:             53 / 61 = 0.869

logic mixed:
Candidate Gate-v2:         22 / 23 = 0.957
open_reasoning_rm/oracle:  23 / 23 = 1.000
```

Aggregation result for Candidate Gate-v2:

```text
mean:     72 / 84 = 0.857  # best for learned candidate selector
last:     71 / 84 = 0.845
geomean:  71 / 84 = 0.845
min:      69 / 84 = 0.821
```

Normalization check for the best mean aggregation:

```text
rank:    72 / 84 = 0.857
minmax:  68 / 84 = 0.810
zscore:  68 / 84 = 0.810
```

Aggregation-as-pseudo-experts was also tested by expanding `open_math_prm` into
`open_math_prm_mean`, `open_math_prm_min`, `open_math_prm_last`, and
`open_math_prm_geomean`.

```text
pseudo-expert oracle:      78 / 84 = 0.929
Candidate Gate-v2:         68 / 84 = 0.810
```

Interpretation: aggregation variants do add oracle headroom, but exposing all
four as separate experts makes the current 84-problem training set too small for
a stable learned gate. For the current main table, use `open_math_prm` mean
aggregation with Candidate Gate-v2.

### Clean-label Check

Added on 2026-09-05 after the V2 loss-case audit.

The answer checker now handles conservative formatting artifacts:

- LaTeX inline/display wrappers such as `\(...\)` and `\[...\]`;
- simple numeric unit suffixes such as `36 seconds` vs `36`;
- percentage semantics are preserved, so `36%` is not equal to `36`.

Relabeling the 84-problem mixed pool changed 23 records:

```text
old correct candidate labels: 371 / 672
new correct candidate labels: 475 / 672

old mixed records: 84
new mixed records: 67
new all-correct records: 17
```

Old Candidate Gate-v2 selections under cleaned labels:

```text
all original 84 records:
Candidate Gate-v2:       83 / 84 = 0.988
best single expert:      81 / 84 = 0.964
uniform ensemble:        80 / 84 = 0.952
OpenAI LLM gate:         73 / 84 = 0.869
expert oracle:           82 / 84 = 0.976
```

Retrained Candidate Gate-v2 under cleaned labels:

```text
all original 84 records:
Candidate Gate-v2:       82 / 84 = 0.976
CV static calibration:   81 / 84 = 0.964
best single expert:      81 / 84 = 0.964
expert oracle:           82 / 84 = 0.976

clean mixed 67 records:
Candidate Gate-v2:       64 / 67 = 0.955
CV static calibration:   64 / 67 = 0.955
best single expert:      64 / 67 = 0.955
expert oracle:           65 / 67 = 0.970
```

Interpretation: the earlier `72 / 84` result was strongly underestimated by
answer-checking noise. After cleanup, the remaining evidence favors keeping
Candidate Gate-v2 as the main method and expanding to a larger clean-label split
rather than training a more expressive Gate-v3 immediately.

### Gate-v2 Win/Loss Analysis

Command:

```bash
python scripts/analyze_candidate_gate_wins.py \
  --input data/scored/candidate_gate_v2_l2_02/candidate_gate_v2_cv_mean_rank.jsonl \
  --static-input data/scored/candidate_gate_v2_l2_02/cv_static_calibrated_min_rank.jsonl \
  --normalization rank \
  --case-baseline cv_static_calibrated
```

Selection-level summary:

```text
Candidate Gate-v2 vs best single open_reasoning_rm:
overall: 9 wins, 4 losses, net +5
math:    9 wins, 3 losses, net +6
logic:   0 wins, 1 loss,  net -1

Candidate Gate-v2 vs strongest CV-static calibration:
overall: 8 wins, 5 losses, net +3
math:    8 wins, 4 losses, net +4
logic:   0 wins, 1 loss,  net -1

Candidate Gate-v2 vs uniform ensemble:
overall: 12 wins, 4 losses, net +8

Candidate Gate-v2 vs OpenAI LLM gate:
overall: 16 wins, 4 losses, net +12

Candidate Gate-v2 vs domain-rule gate:
overall: 21 wins, 5 losses, net +16
```

Interpretation: the V2 gain is almost entirely from MATH500 mixed examples. It
rescues many math problems where `open_reasoning_rm` or static calibration pick
a confidently wrong candidate. The main cost is one BBH logic case where
`open_reasoning_rm` was correct but V2 trusted OpenAI judge-style signals too
much.

### Three-Expert No-RM Ablation

Because `open_reasoning_rm` is the strongest single expert in the four-expert
pool, we ran a diagnostic ablation that removes it and keeps only
`open_math_prm`, `openai_general_judge`, and `openai_reflective_judge`.

Detailed note:
[notes/three_expert_no_reasoning_rm_ablation.md](notes/three_expert_no_reasoning_rm_ablation.md)

Key result:

```text
without open_reasoning_rm, mean aggregation + rank normalization:

Candidate Gate-v2:          74 / 84 = 0.881
best single expert:         66 / 84 = 0.786  # openai_reflective_judge
CV-static calibration:      64 / 84 = 0.762
uniform ensemble:           62 / 84 = 0.738
expert top-choice oracle:   73 / 84 = 0.869
```

Interpretation: `open_reasoning_rm` is indeed very strong, but the V2 gain does
not vanish when it is removed. The remaining three experts still contain
accuracy-relevant complementarity, especially on MATH500 mixed examples. This
should be reported as an ablation rather than the main method, since the no-RM
pool relies more heavily on the two OpenAI judge experts.

## Next Step

Current decision after Gate-v2:

1. Use Candidate Gate-v2 as the current main trained MoPRM result.
2. Report Gate-v1 as the question-level trained baseline.
3. Use `open_math_prm` mean aggregation for Candidate Gate-v2, while noting that
   static calibration prefers `min`.
4. Keep aggregation-as-pseudo-experts as an oracle/headroom diagnostic, not the
   main method yet.
5. Next: run robustness on a larger mixed split or build a cleaner held-out
   split so the `72/84` result is not overfit to one scout sample.

## Local Smoke Test

Run the toy evaluation without downloading models or datasets:

```bash
python -m unittest discover -s tests
python scripts/run_smoke_eval.py
```

To label candidate correctness for evaluation-only use:

```bash
python scripts/label_candidate_correctness.py --input examples/smoke_moprm.jsonl --output data/cache/smoke_labeled.jsonl
```

Prepare the first public math+logic subsets:

```bash
python scripts/prepare_public_subsets.py --math500-limit 80 --gsm8k-limit 80 --bbh-limit-per-task 60
python scripts/inspect_dataset.py --input data/cache/public_subsets/math_logic_combined.jsonl
python scripts/sample_dataset.py --input data/cache/public_subsets/math_logic_combined.jsonl --output data/splits/dev_40.jsonl --per-domain 20 --seed 13
```

Prepared public data is written under `data/cache/` and is intentionally ignored by git.

The sampler also supports exact source quotas, which is useful for creating
harder splits with more MATH500 and fewer GSM8K examples.

Generate gold-derived debug candidates for end-to-end pipeline testing only:

```bash
python scripts/generate_debug_candidates.py --input data/splits/dev_40.jsonl --output data/candidates/debug_math_logic.jsonl --num-candidates 4
python scripts/label_candidate_correctness.py --input data/candidates/debug_math_logic.jsonl --output data/cache/debug_math_logic_labeled.jsonl --overwrite
python scripts/score_debug_experts.py --input data/cache/debug_math_logic_labeled.jsonl --output data/scored/debug_math_logic_scored.jsonl
python scripts/run_smoke_eval.py --input data/scored/debug_math_logic_scored.jsonl
```

Generate real candidates with the OpenAI Responses API. The default command is a tiny
smoke run: two problems, two candidates per problem.

```bash
python scripts/generate_openai_candidates.py --input data/splits/dev_40.jsonl --output data/candidates/openai_dev_smoke.jsonl --limit 2 --num-candidates 2
python scripts/label_candidate_correctness.py --input data/candidates/openai_dev_smoke.jsonl --output data/cache/openai_dev_smoke_labeled.jsonl --overwrite
```

The script reads `OPENAI_API_KEY` from the process environment first, then from a
local `.env` file. Do not commit `.env` or generated candidate files.
For long runs, pass `--resume` to continue from a partially written output file
instead of regenerating completed problems.

Score generated candidates with lightweight OpenAI PRM-style experts. The scorer
does not receive gold answers; it only sees the problem and candidate solution.

```bash
python scripts/score_openai_experts.py --input data/cache/openai_dev_smoke_labeled.jsonl --output data/scored/openai_dev_smoke_scored.jsonl --overwrite
python scripts/run_smoke_eval.py --input data/scored/openai_dev_smoke_scored.jsonl --by-domain
```

Attach an LLM-based question-level gate and evaluate it alongside baselines:

```bash
python scripts/route_openai_gate.py --input data/scored/openai_dev_smoke_scored.jsonl --output data/scored/openai_dev_smoke_routed.jsonl --overwrite
python scripts/run_smoke_eval.py --input data/scored/openai_dev_smoke_routed.jsonl --by-domain
```

Inspect expert complementarity, LLM-gate weights, raw-score separation, and
oracle gaps:

```bash
python scripts/analyze_expert_pool.py --input data/scored/openai_dev40_n4_hetero_math_pool_routed.jsonl
```

Score math candidates with the first non-OpenAI open-source PRM adapter:

```bash
python scripts/score_skywork_math_prm.py --input data/cache/openai_pilot10_n4_labeled.jsonl --output data/scored/skywork_math_prm_pilot.jsonl --domains math --limit 10 --device auto --overwrite
```

The first selected math PRM is `Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B`; selection notes are in [notes/open_source_prm_selection.md](notes/open_source_prm_selection.md).

Score candidates with the second non-OpenAI expert, a response-level reasoning
reward model:

```powershell
$env:HF_HUB_OFFLINE='1'
.\.venv_cuda\Scripts\python.exe scripts/score_skywork_reward_v2.py --input data/scored/openai_dev40_n4_with_skywork_math.jsonl --output data/scored/openai_dev40_n4_with_open_experts.jsonl --domains all --device cuda --dtype auto --overwrite
```

The selected reasoning RM is `Skywork/Skywork-Reward-V2-Qwen3-1.7B`. It writes
scores as `expert_scores.open_reasoning_rm`.

Build a transitional heterogeneous expert pool after adding `open_math_prm`:

```bash
python scripts/rewrite_expert_pool.py --input data/scored/openai_pilot10_n4_with_skywork_math.jsonl --output data/scored/openai_pilot10_n4_hetero_math_pool.jsonl --drop math_prm --rename logic_judge=openai_logic_rubric --rename general_judge=openai_general_judge --rename reflective_judge=openai_reflective_judge --overwrite
```

Build the current main heterogeneous expert pool after both open-source experts
are scored:

```bash
python scripts/rewrite_expert_pool.py --input data/scored/openai_dev40_n4_with_open_experts.jsonl --output data/scored/openai_dev40_n4_two_open_expert_pool.jsonl --drop math_prm --drop logic_judge --rename general_judge=openai_general_judge --rename reflective_judge=openai_reflective_judge --overwrite
```

For local PRM inference, install optional dependencies into the ignored `.venv`:

```bash
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[local-prm]"
```

Dry-run the adapter before downloading model weights:

```bash
python scripts/score_skywork_math_prm.py --input data/cache/openai_pilot10_n4_labeled.jsonl --domains math --limit 3 --dry-run
```

If Hugging Face downloads are unstable, use the resumable helper:

```bash
.\.venv_cuda\Scripts\python.exe scripts/download_skywork_weight.py
```

## Relation to Earlier Work

Earlier notes in this repo focus on **Beyond the First Error: Process Reward Models for Reflective Mathematical Reasoning**. That paper remains useful as motivation: it shows that different reasoning styles need different process supervision signals. In MoPRM, reflective PRM scoring can be treated as one expert among several, instead of the whole project scope.

## Git Policy

Important source code, experiment configs, notes, and result summaries should be committed regularly. Large datasets, model weights, generated candidates, scored caches, rendered PDFs, and temporary images should stay out of git.
