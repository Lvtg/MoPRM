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

As of 2026-09-04, the main heterogeneous expert pool has been run at
`hard_dev_100, N=8` scale:

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

## Next Step

Current decision:

1. Use `hard_mix_scout_320_n8_mixed` as the first trained-gate dataset.
2. Keep `open_math_prm` step aggregation as an ablation. `min`, `last`, and
   `geomean` each win under different diagnostics, so aggregation should be a
   reported choice rather than a hidden constant.
3. Use rank normalization for the main table and minmax as a calibration
   sensitivity check.
4. Train a small question-level gate on the 84 mixed problems, with a
   source/domain-stratified split.
5. Report both full-scout candidate statistics and mixed-only PRM@8 results.

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
