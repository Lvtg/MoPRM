# Experiment Log

## 2026-08-31: Public Data and Debug Pipeline

Prepared first public subsets:

```text
MATH-500: 80 records
GSM8K: 80 records
BBH logical deduction: 180 records
Combined: 340 records
```

Created a balanced development split:

```text
data/splits/dev_40.jsonl
math: 20
logic: 20
```

Ran a debug-only end-to-end pipeline:

```text
generate debug candidates -> label correctness -> add synthetic debug expert scores -> evaluate BoN
```

Important caveat:

Debug candidates and debug scores use synthetic/gold-derived signals. They are valid only for pipeline testing and must not be reported as real experiment results.

Verification:

```text
python -m unittest discover -s tests
16 tests OK
```

## 2026-09-01: OpenAI Candidate Generation Scaffold

Added a real candidate-generation path using the OpenAI Responses API:

```text
scripts/generate_openai_candidates.py
src/moprm/openai_responses.py
src/moprm/candidates/openai_generator.py
```

Default run is intentionally tiny:

```text
limit: 2 problems
num_candidates: 2 per problem
max_output_tokens: 512
default model: gpt-4.1-mini
```

The generator does not include gold answers in prompts. Gold answers are used only
after generation by the evaluation-labeling script.

Added a first OpenAI expert scorer:

```text
scripts/score_openai_experts.py
src/moprm/scoring/openai_experts.py
```

It scores each candidate once and returns four expert dimensions:

```text
math_prm
logic_judge
general_judge
reflective_judge
```

The scorer prompt receives no gold answer. This keeps expert scoring separate
from evaluation labels.

Rubric note:

```text
Domain-specialist scores are intentionally not pure correctness labels.
For example, math_prm should not automatically score a non-math logic task as 1.0.
This helps the gate have meaningful expert-specialization signals.
```

Added a first question-level OpenAI router:

```text
scripts/route_openai_gate.py
src/moprm/routing/openai_gate.py
```

The router sees the problem and available expert names/descriptions, but no
candidate solutions and no gold answer. It writes normalized expert weights to
`record.metadata.gate_weights.openai_llm_gate`, which the evaluator now includes
as `metadata_gate:openai_llm_gate`.

## 2026-09-01: First OpenAI Pilot Run

Created a balanced pilot split:

```text
data/splits/pilot_10.jsonl
math: 5
logic: 5
```

Generated candidates:

```text
input: data/splits/pilot_10.jsonl
output: data/candidates/openai_pilot10_n4.jsonl
model: gpt-4.1-mini
temperature: 1.0
num_candidates: 4
problems: 10
candidates: 40
reported generation tokens: 16,960
```

Labeled candidate correctness for evaluation only:

```text
correct candidates: 32 / 40
math: 15 / 20
logic: 17 / 20
```

Scored candidates with final-answer-aware OpenAI expert rubric:

```text
output: data/scored/openai_pilot10_n4_scored.jsonl
reported scoring tokens: 25,994
explicit-final-answer missing candidates: 3
```

Attached question-level LLM gate:

```text
output: data/scored/openai_pilot10_n4_routed.jsonl
reported gate tokens: 3,636
```

Evaluation:

```text
domain_rule_gate:              10 / 10
metadata_gate:openai_llm_gate: 10 / 10
oracle_gate:                   10 / 10
single experts:                10 / 10 each
uniform_ensemble:              10 / 10
```

Interpretation:

This run validates the real OpenAI generation -> evaluation labeling -> expert
scoring -> LLM gate -> BoN evaluation pipeline. It is not yet evidence that
MoPRM improves over baselines, because this pilot is too small and the scorer is
strong enough that all methods select correct candidates.

## 2026-09-01: Goal Revision Toward Heterogeneous Experts

Decision after project discussion:

```text
The OpenAI multi-rubric scorer remains a baseline.
The final MoPRM expert pool should include at least two non-OpenAI PRM/reward experts.
```

Target expert pool:

```text
open_math_prm           non-OpenAI open-source math PRM
open_logic_prm          non-OpenAI open-source logic/reasoning RM or PRM
openai_general_judge    OpenAI general reliability judge
openai_reflective_judge OpenAI reflective/error-recovery judge
```

Rationale:

Using four rubric dimensions from the same OpenAI model is too homogeneous for
the main MoPRM claim. It is useful as a baseline and scaffolding, but the final
experiment should test routing over independently sourced experts.

Plan impact:

```text
Next immediate stage: integrate one open-source math PRM.
Second stage: integrate one non-OpenAI logic/reasoning reward model or PRM.
Only then train/evaluate the main gate against heterogeneous expert scores.
```

## 2026-09-01: Skywork Math PRM Adapter

Hardware check:

```text
GPU: NVIDIA GeForce RTX 5070 Laptop
VRAM: 8,151 MiB
RAM: about 31.4 GiB
Python: 3.10.10
```

Selected first open-source math PRM:

```text
Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B
```

Rationale:

```text
The model is a real non-OpenAI math PRM and its main weight file is about 3.09 GB,
making it much more suitable for the local 8GB GPU than 7B/8B PRMs.
```

Implementation added:

```text
src/moprm/scoring/skywork_math_prm.py
scripts/score_skywork_math_prm.py
tests/test_skywork_math_prm.py
notes/open_source_prm_selection.md
```

Dry-run result:

```text
python scripts/score_skywork_math_prm.py --input data/cache/openai_pilot10_n4_labeled.jsonl --domains math --limit 3 --dry-run
records: 1 math record from first 3 pilot records
candidates: 4
step splitting worked
```

Environment note:

```text
.venv installed CPU torch + transformers + accelerate.
CUDA torch 12.8 was available from the PyTorch wheel index, but the 2.75GB wheel
download was repeatedly interrupted and projected to take over an hour, so it was
paused.
```

Attempted one-record Skywork PRM smoke:

```text
command: .\.venv\Scripts\python.exe scripts\score_skywork_math_prm.py --input data\cache\openai_pilot10_n4_labeled.jsonl --output data\scored\skywork_math_prm_pilot1.jsonl --domains math --limit 1 --device auto --overwrite
result: small config/tokenizer/custom-code files downloaded successfully
issue: 3.09GB model weight download did not progress beyond small cache files
action: interrupted the run and kept the adapter ready for the next download attempt
```

## 2026-09-02: Skywork Math PRM CUDA Pilot

Environment update:

```text
.venv_cuda
torch: 2.7.1+cu128
CUDA available: yes
GPU: NVIDIA GeForce RTX 5070 Laptop GPU
```

Weight cache:

```text
model: Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B
weight: pytorch_model.bin
size: 3,087,498,236 bytes
cache: models/hf_cache/.../snapshots/.../pytorch_model.bin
SHA256/etag: verified externally before this run
```

CPU smoke output:

```text
data/scored/skywork_math_prm_pilot1.jsonl
record: gsm8k_0076
candidates: 4
open_math_prm scores: 0.555142, 0.577435, 0.493211, 0.446626
```

CUDA dtype finding:

```text
CUDA bf16 produced NaN step rewards for two candidates on gsm8k_0076.
CPU float32 and CUDA float32 were stable.
Adapter updated to support --dtype and to reject non-finite step rewards.
Default auto dtype is conservative float32 for this Skywork PRM integration.
```

Full pilot scoring:

```text
input: data/scored/openai_pilot10_n4_scored.jsonl
output: data/scored/openai_pilot10_n4_with_skywork_math.jsonl
device: cuda
dtype: float32
records: 10
candidates: 40
open_math_prm coverage: 40 / 40
non-finite scores: 0
```

Open math PRM score distribution:

```text
math candidates: 20, min=0.218508, max=0.746356, mean=0.552536
logic candidates: 20, min=0.486544, max=0.729964, mean=0.580342
```

Transitional heterogeneous pool:

```text
output: data/scored/openai_pilot10_n4_hetero_math_pool.jsonl
experts:
- open_math_prm
- openai_logic_rubric
- openai_general_judge
- openai_reflective_judge
rewrite:
- dropped OpenAI math_prm rubric
- renamed OpenAI logic/general/reflective rubrics for clarity
```

Evaluation on transitional heterogeneous pool:

```text
domain_rule_gate:               8 / 10 overall, 3 / 5 math
metadata_gate:openai_llm_gate:  8 / 10 overall, 3 / 5 math
uniform_ensemble:               8 / 10 overall, 3 / 5 math
single:open_math_prm:           8 / 10 overall, 3 / 5 math
oracle_gate:                   10 / 10 overall, 5 / 5 math
```

Interpretation:

The first non-OpenAI math PRM is integrated and runnable, but it is not
automatically better on the current generated candidate pool. The LLM/domain
gates over-trust the specialized math expert on math problems, so this pilot
motivates calibration and trained routing rather than naive domain routing.

## 2026-09-02: Skywork Weight Download Completed

Download status:

```text
model: Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B
revision: 98d69606595eedbdbbbf0a7d28efdcd462ba6a67
weight: pytorch_model.bin
size: 3,087,498,236 bytes
cache: models/hf_cache
```

The earlier Hugging Face CLI attempts created process-unique `.incomplete`
files, so interrupted partial downloads were not reused automatically. Added a
small resume helper that keeps a stable `.part` file, uses HTTP Range requests,
verifies the final SHA256 against the Hugging Face etag, and then links the
weight into the local snapshot cache.

Implementation update:

```text
scripts/download_skywork_weight.py
src/moprm/scoring/skywork_math_prm.py now passes use_cache=False for PRM scoring
```

Offline one-record smoke:

```text
command: HF_HUB_OFFLINE=1 python scripts/score_skywork_math_prm.py --input data/cache/openai_pilot10_n4_labeled.jsonl --output data/scored/skywork_math_prm_pilot1.jsonl --domains math --limit 1 --device cpu --overwrite
result: succeeded
record: gsm8k_0076
candidates scored: 4
open_math_prm scores: 0.555142, 0.577435, 0.493211, 0.446626
```

## 2026-09-02: dev_40 Heterogeneous Math-PRM Pilot

Candidate generation:

```text
input: data/splits/dev_40.jsonl
output: data/candidates/openai_dev40_n4.jsonl
model: gpt-4.1-mini
temperature: 1.0
num_candidates: 4
problems: 40
candidates: 160
```

Evaluation-only correctness labels:

```text
correct candidates: 129 / 160
avg correct candidates per problem: 3.225 / 4
all-candidate-wrong problems: 6 / 40, all in math
all-candidate-correct problems: 30 / 40
logic: avg 3.900 correct candidates/problem, 0 all-wrong
math: avg 2.550 correct candidates/problem, 6 all-wrong
```

OpenAI baseline expert scoring:

```text
input: data/cache/openai_dev40_n4_labeled.jsonl
output: data/scored/openai_dev40_n4_scored.jsonl
records: 40
candidates: 160
reported scoring tokens: 98,058
```

Skywork math PRM scoring:

```text
input: data/scored/openai_dev40_n4_scored.jsonl
output: data/scored/openai_dev40_n4_with_skywork_math.jsonl
device: cuda
dtype: float32
records: 40
candidates: 160
open_math_prm coverage: 160 / 160
```

Clean transitional expert pool:

```text
output: data/scored/openai_dev40_n4_hetero_math_pool.jsonl
experts:
- open_math_prm
- openai_logic_rubric
- openai_general_judge
- openai_reflective_judge
rewrite:
- dropped OpenAI math_prm rubric
- renamed OpenAI logic/general/reflective rubrics for clarity
```

LLM gate routing:

```text
output: data/scored/openai_dev40_n4_hetero_math_pool_routed.jsonl
records: 40
reported gate tokens: 15,242
```

Evaluation on routed clean pool:

```text
overall:
domain_rule_gate:               33 / 40 = 0.825
metadata_gate:openai_llm_gate:  33 / 40 = 0.825
uniform_ensemble:               33 / 40 = 0.825
single:open_math_prm:           33 / 40 = 0.825
single:openai_general_judge:    32 / 40 = 0.800
single:openai_logic_rubric:     32 / 40 = 0.800
single:openai_reflective_judge: 33 / 40 = 0.825
oracle_gate:                    34 / 40 = 0.850

logic:
all methods:                    20 / 20 = 1.000

math:
domain_rule_gate:               13 / 20 = 0.650
metadata_gate:openai_llm_gate:  13 / 20 = 0.650
uniform_ensemble:               13 / 20 = 0.650
single:open_math_prm:           13 / 20 = 0.650
single:openai_general_judge:    12 / 20 = 0.600
single:openai_logic_rubric:     12 / 20 = 0.600
single:openai_reflective_judge: 13 / 20 = 0.650
oracle_gate:                    14 / 20 = 0.700
```

Gate-weight diagnostics:

```text
average openai_llm_gate weights:
logic:
  open_math_prm            0.000
  openai_general_judge     0.185
  openai_logic_rubric      0.570
  openai_reflective_judge  0.245
math:
  open_math_prm            0.570
  openai_general_judge     0.005
  openai_logic_rubric      0.110
  openai_reflective_judge  0.315
```

Raw score separation:

```text
overall delta(correct - wrong):
open_math_prm:            +0.007
openai_general_judge:     +0.055
openai_logic_rubric:      +0.030
openai_reflective_judge:  +0.034

math delta(correct - wrong):
open_math_prm:            +0.020
openai_general_judge:     +0.057
openai_logic_rubric:      +0.031
openai_reflective_judge:  +0.039
```

Interpretation:

The dev_40 run confirms that the first non-OpenAI PRM is integrated in the real
pipeline, but the current benchmark slice is near its candidate-generation
ceiling. Six math problems have no correct candidate at all, so the maximum
selectable accuracy is 34/40. Current routing reaches 33/40 and the oracle only
recovers one additional problem. This means the next useful experiment should
increase the available complementarity rather than only tweak the gate:

```text
1. add a second non-OpenAI expert, preferably a logic/reasoning RM or PRM;
2. create a harder or larger split with more mixed correct/wrong candidates;
3. then train/calibrate a lightweight gate on held-out problems.
```

Implementation utility added:

```text
scripts/analyze_expert_pool.py
```

## 2026-09-03: Second Non-OpenAI Expert Integrated

Selected second open-source expert:

```text
model: Skywork/Skywork-Reward-V2-Qwen3-1.7B
revision: e51ea3e08fb81326c3b812a7ff0cb9cee83e59cc
weight: model.safetensors
size: 3,441,189,792 bytes
cache: models/hf_cache
expert name: open_reasoning_rm
score type: response-level sequence reward logit
```

Implementation added:

```text
src/moprm/scoring/skywork_reward_v2.py
scripts/score_skywork_reward_v2.py
tests/test_skywork_reward_v2.py
```

Smoke result:

```text
input: data/scored/openai_dev40_n4_with_skywork_math.jsonl
output: data/scored/skywork_reward_v2_smoke1.jsonl
device: cuda
dtype: auto, resolved to CUDA bfloat16
records: 1
candidates: 4
open_reasoning_rm scores: 6.21875, 7.34375, 7.5, 5.5
```

Full dev_40 scoring:

```text
input: data/scored/openai_dev40_n4_with_skywork_math.jsonl
output: data/scored/openai_dev40_n4_with_open_experts.jsonl
device: cuda
dtype: auto
records: 40
candidates: 160
open_reasoning_rm coverage: 160 / 160
```

Open reasoning RM score distribution:

```text
logic candidates: 80, min=-0.781250, max=11.750000, mean=7.087109
math candidates: 80, min=2.156250, max=12.500000, mean=9.074805
```

Current main heterogeneous pool:

```text
output: data/scored/openai_dev40_n4_two_open_expert_pool.jsonl
experts:
- open_math_prm
- open_reasoning_rm
- openai_general_judge
- openai_reflective_judge
rewrite:
- dropped OpenAI math_prm rubric
- dropped OpenAI logic_judge rubric
- renamed OpenAI general/reflective rubrics for clarity
```

LLM gate routing:

```text
output: data/scored/openai_dev40_n4_two_open_expert_pool_routed.jsonl
records: 40
reported gate tokens: 15,232
```

Evaluation on current main heterogeneous pool:

```text
overall:
domain_rule_gate:               33 / 40 = 0.825
metadata_gate:openai_llm_gate:  33 / 40 = 0.825
uniform_ensemble:               32 / 40 = 0.800
single:open_math_prm:           33 / 40 = 0.825
single:open_reasoning_rm:       32 / 40 = 0.800
single:openai_general_judge:    32 / 40 = 0.800
single:openai_reflective_judge: 33 / 40 = 0.825
oracle_gate:                    34 / 40 = 0.850

logic:
all methods:                    20 / 20 = 1.000

math:
domain_rule_gate:               13 / 20 = 0.650
metadata_gate:openai_llm_gate:  13 / 20 = 0.650
uniform_ensemble:               12 / 20 = 0.600
single:open_math_prm:           13 / 20 = 0.650
single:open_reasoning_rm:       12 / 20 = 0.600
single:openai_general_judge:    12 / 20 = 0.600
single:openai_reflective_judge: 13 / 20 = 0.650
oracle_gate:                    14 / 20 = 0.700
```

Gate-weight diagnostics:

```text
average openai_llm_gate weights:
logic:
  open_math_prm            0.000
  open_reasoning_rm        0.505
  openai_general_judge     0.300
  openai_reflective_judge  0.195
math:
  open_math_prm            0.615
  open_reasoning_rm        0.190
  openai_general_judge     0.085
  openai_reflective_judge  0.110
```

Raw score separation:

```text
overall delta(correct - wrong):
open_math_prm:            +0.007
open_reasoning_rm:        -0.233
openai_general_judge:     +0.055
openai_reflective_judge:  +0.034

logic delta(correct - wrong):
open_math_prm:            -0.092
open_reasoning_rm:        +7.165
openai_general_judge:     +0.450
openai_reflective_judge:  +0.383

math delta(correct - wrong):
open_math_prm:            +0.020
open_reasoning_rm:        +0.381
openai_general_judge:     +0.057
openai_reflective_judge:  +0.039
```

Interpretation:

The main MoPRM pool now satisfies the heterogeneity requirement: two non-OpenAI
experts plus two OpenAI supporting judges. On dev_40, the LLM gate routes in the
expected direction, assigning most logic weight to `open_reasoning_rm` and most
math weight to `open_math_prm`. The accuracy does not improve beyond 33/40
because this split still has a very small true oracle gap: six math problems have
no correct candidate, and only one additional problem can be rescued by expert
selection. The next experiment should therefore create a harder/larger candidate
set, such as N=8, before investing in a trained gate.

## 2026-09-03: Next Harder Experiment Plan

Decision:

```text
Next experiment: hard_dev_100_n8
```

Rationale:

```text
The previous dev_40, N=4 run had too little oracle headroom:
- current routing: 33 / 40
- oracle gate:     34 / 40

Therefore the next step should increase candidate diversity and task difficulty
before training the gate.
```

Chosen split:

```text
total: 100 problems
math: 60
logic: 40

math sources:
- HuggingFaceH4/MATH-500: 50
- openai/gsm8k: 10

logic sources:
- BIG-Bench-Hard/logical_deduction_seven_objects: 20
- BIG-Bench-Hard/logical_deduction_five_objects: 10
- BIG-Bench-Hard/logical_deduction_three_objects: 10

candidates per problem for next run: N=8
```

Implementation update:

```text
scripts/sample_dataset.py now supports repeated --source-quota arguments.
```

Verified split command:

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

Observed output:

```text
Domains: {'logic': 40, 'math': 60}
Sources:
- logic|BIG-Bench-Hard/logical_deduction_seven_objects: 20
- logic|BIG-Bench-Hard/logical_deduction_five_objects: 10
- logic|BIG-Bench-Hard/logical_deduction_three_objects: 10
- math|HuggingFaceH4/MATH-500: 50
- math|openai/gsm8k: 10
```

## 2026-09-03: hard_dev_100_n8 Main Run

Candidate generation:

```text
input: data/splits/hard_dev_100.jsonl
output: data/candidates/openai_hard_dev100_n8.jsonl
model: gpt-4.1-mini
temperature: 1.0
num_candidates: 8
problems: 100
candidates: 800
concurrency: 4
reported generation tokens: 408,180
```

Implementation note:

```text
scripts/generate_openai_candidates.py and scripts/score_openai_experts.py now
support --concurrency. The default remains 1, but hard_dev_100_n8 used
--concurrency 4 for API-bound stages.
```

Evaluation-only correctness labels:

```text
correct candidates: 534 / 800 = 0.667

overall:
avg correct candidates/problem: 5.34 / 8
all-wrong problems: 23 / 100
all-correct problems: 59 / 100
mixed problems: 18 / 100

logic:
correct candidates: 280 / 320 = 0.875
avg correct candidates/problem: 7.00 / 8
all-wrong problems: 3 / 40
all-correct problems: 31 / 40

math:
correct candidates: 254 / 480 = 0.529
avg correct candidates/problem: 4.23 / 8
all-wrong problems: 20 / 60
all-correct problems: 28 / 60
```

Source-level candidate correctness:

```text
MATH500:
problems: 50
correct candidates: 204 / 400 = 0.510
avg correct candidates/problem: 4.08 / 8
all-wrong: 18
all-correct: 22

GSM8K:
problems: 10
correct candidates: 50 / 80 = 0.625
avg correct candidates/problem: 5.00 / 8
all-wrong: 2
all-correct: 6

BBH seven-object logical deduction:
problems: 20
correct candidates: 120 / 160 = 0.750
avg correct candidates/problem: 6.00 / 8
all-wrong: 3
all-correct: 11

BBH five-object and three-object subsets:
all candidates correct in this run.
```

OpenAI baseline expert scoring:

```text
input: data/cache/openai_hard_dev100_n8_labeled.jsonl
output: data/scored/openai_hard_dev100_n8_scored.jsonl
records: 100
candidates: 800
concurrency: 4
reported scoring tokens: 603,493
```

Open-source expert scoring:

```text
open_math_prm:
input: data/scored/openai_hard_dev100_n8_scored.jsonl
output: data/scored/openai_hard_dev100_n8_with_skywork_math.jsonl
device: cuda
dtype: float32
aggregation: mean
coverage: 800 / 800

open_reasoning_rm:
input: data/scored/openai_hard_dev100_n8_with_skywork_math.jsonl
output: data/scored/openai_hard_dev100_n8_with_open_experts.jsonl
device: cuda
dtype: auto
coverage: 800 / 800
```

Main heterogeneous pool:

```text
output: data/scored/openai_hard_dev100_n8_two_open_expert_pool.jsonl
experts:
- open_math_prm
- open_reasoning_rm
- openai_general_judge
- openai_reflective_judge
```

LLM gate routing:

```text
output: data/scored/openai_hard_dev100_n8_two_open_expert_pool_routed.jsonl
records: 100
reported gate tokens: 38,368
```

Evaluation with mean step aggregation for `open_math_prm`:

```text
overall:
domain_rule_gate:               67 / 100 = 0.670
metadata_gate:openai_llm_gate:  69 / 100 = 0.690
uniform_ensemble:               70 / 100 = 0.700
single:open_math_prm:           63 / 100 = 0.630
single:open_reasoning_rm:       71 / 100 = 0.710
single:openai_general_judge:    68 / 100 = 0.680
single:openai_reflective_judge: 69 / 100 = 0.690
oracle_gate:                    71 / 100 = 0.710

logic:
domain_rule_gate:               37 / 40 = 0.925
metadata_gate:openai_llm_gate:  36 / 40 = 0.900
single:open_reasoning_rm:       37 / 40 = 0.925
oracle_gate:                    37 / 40 = 0.925

math:
domain_rule_gate:               30 / 60 = 0.500
metadata_gate:openai_llm_gate:  33 / 60 = 0.550
uniform_ensemble:               34 / 60 = 0.567
single:open_math_prm:           30 / 60 = 0.500
single:open_reasoning_rm:       34 / 60 = 0.567
oracle_gate:                    34 / 60 = 0.567
```

Aggregation calibration:

```text
mean aggregation:
single:open_math_prm: 63 / 100 overall, 30 / 60 math

min aggregation:
single:open_math_prm: 68 / 100 overall, 34 / 60 math
domain_rule_gate:     71 / 100 overall, 34 / 60 math
uniform_ensemble:     71 / 100 overall, 35 / 60 math
oracle_gate:          72 / 100 overall, 35 / 60 math

last aggregation:
single:open_math_prm: 66 / 100 overall, 31 / 60 math

geomean aggregation:
single:open_math_prm: 63 / 100 overall, 30 / 60 math
```

Evaluation after reaggregating routed pool with `open_math_prm` min aggregation:

```text
rank normalization:
overall:
domain_rule_gate:               71 / 100 = 0.710
metadata_gate:openai_llm_gate:  70 / 100 = 0.700
uniform_ensemble:               71 / 100 = 0.710
single:open_math_prm:           68 / 100 = 0.680
single:open_reasoning_rm:       71 / 100 = 0.710
oracle_gate:                    72 / 100 = 0.720

math:
domain_rule_gate:               34 / 60 = 0.567
metadata_gate:openai_llm_gate:  34 / 60 = 0.567
uniform_ensemble:               35 / 60 = 0.583
single:open_math_prm:           34 / 60 = 0.567
single:open_reasoning_rm:       34 / 60 = 0.567
oracle_gate:                    35 / 60 = 0.583
```

Normalization ablation after `open_math_prm` min aggregation:

```text
rank:
metadata_gate:openai_llm_gate:  70 / 100 overall, 34 / 60 math
uniform_ensemble:               71 / 100 overall, 35 / 60 math

minmax:
metadata_gate:openai_llm_gate:  71 / 100 overall, 35 / 60 math
uniform_ensemble:               70 / 100 overall, 34 / 60 math

zscore:
metadata_gate:openai_llm_gate:  70 / 100 overall, 34 / 60 math
uniform_ensemble:               70 / 100 overall, 34 / 60 math
```

Mixed-candidate analysis:

```text
definition: keep problems with 1..7 correct candidates
selected: 18 / 100 problems
sources:
- BIG-Bench-Hard/logical_deduction_seven_objects: 6
- HuggingFaceH4/MATH-500: 10
- openai/gsm8k: 2
correct-count histogram:
- 1 correct candidate: 5 problems
- 2 correct candidates: 2 problems
- 4 correct candidates: 7 problems
- 5 correct candidates: 1 problem
- 6 correct candidates: 1 problem
- 7 correct candidates: 2 problems
```

Mixed-candidate PRM@8 after `open_math_prm` min aggregation:

```text
overall:
domain_rule_gate:               12 / 18 = 0.667
metadata_gate:openai_llm_gate:  11 / 18 = 0.611
uniform_ensemble:               12 / 18 = 0.667
single:open_math_prm:            9 / 18 = 0.500
single:open_reasoning_rm:       12 / 18 = 0.667
oracle_gate:                    13 / 18 = 0.722

math:
domain_rule_gate:                6 / 12 = 0.500
metadata_gate:openai_llm_gate:   6 / 12 = 0.500
uniform_ensemble:                7 / 12 = 0.583
single:open_math_prm:            6 / 12 = 0.500
single:open_reasoning_rm:        6 / 12 = 0.500
oracle_gate:                     7 / 12 = 0.583
```

Diagnostics after min aggregation:

```text
all_experts_same_top_choice: 1 / 100
any_expert_disagreement: 99 / 100

average openai_llm_gate weights:
logic:
  open_math_prm            0.000
  open_reasoning_rm        0.500
  openai_general_judge     0.300
  openai_reflective_judge  0.200
math:
  open_math_prm            0.587
  open_reasoning_rm        0.208
  openai_general_judge     0.088
  openai_reflective_judge  0.117

accuracy gap versus oracle:
domain_rule_gate:              1 problem
metadata_gate:openai_llm_gate: 2 problems
uniform_ensemble:              1 problem
```

Interpretation:

`hard_dev_100_n8` is a better stress test than `dev_40_n4`, but it still has
limited useful routing headroom. The current pool's best single expert
(`open_reasoning_rm`) and uniform/domain-rule baselines are already close to the
expert-oracle gate. The main failure is no longer expert heterogeneity; it is
candidate-set composition and expert calibration. The Skywork math PRM should
not be reported with mean aggregation as the only result; cached step rewards
show that `min` aggregation is stronger on this run.

Next recommended work:

```text
1. Treat min aggregation as the current preferred open_math_prm setting.
2. Build a mixed-candidate analysis split from problems with 1..N-1 correct
   candidates, because all-correct/all-wrong problems cannot show routing gains.
3. Only train a gate after checking whether mixed problems have enough expert
   complementarity beyond open_reasoning_rm and uniform fusion.
4. Consider a more diverse/weaker candidate generator or generation prompt if
   mixed examples remain too rare.
```

## 2026-09-04: Mixed Subset and Calibration Sweep

Goal:

```text
Analyze whether the current hard_dev_100_n8 split has enough selection-relevant
expert complementarity to justify training a gate.
```

Code updates:

```text
scripts/analyze_expert_pool.py
- now reports candidate upper bound, mixed/all-correct/all-wrong counts, expert
  success patterns, and unique-success counts.

scripts/sweep_expert_settings.py
- sweeps open_math_prm step aggregation: mean, min, last, geomean;
- sweeps score normalization: rank, minmax, zscore;
- searches in-sample static expert weights as a calibration diagnostic.

src/moprm/candidates/openai_generator.py
- future N=8 candidate generation now uses eight distinct candidate styles
  instead of repeating four styles twice.
```

Mixed-candidate availability after `open_math_prm` min aggregation:

```text
overall  avg_correct_candidates=5.340 all_wrong=23 all_correct=59 mixed=18 candidate_upper=77/100
logic    avg_correct_candidates=7.000 all_wrong=3  all_correct=31 mixed=6  candidate_upper=37/40
math     avg_correct_candidates=4.233 all_wrong=20 all_correct=28 mixed=12 candidate_upper=40/60
```

Expert top-choice complementarity:

```text
full split:
single:open_math_prm             68 / 100
single:open_reasoning_rm         71 / 100
single:openai_general_judge      68 / 100
single:openai_reflective_judge   69 / 100

success patterns, expert order:
open_math_prm, open_reasoning_rm, openai_general_judge, openai_reflective_judge

1111 count=65
0000 count=28
0111 count=3
1000 count=1
1100 count=1
0100 count=1
1101 count=1

unique_success:open_math_prm            1
unique_success:open_reasoning_rm        1
unique_success:openai_general_judge     0
unique_success:openai_reflective_judge  0
```

Mixed-only top-choice complementarity:

```text
mixed problems: 18
single:open_math_prm              9 / 18
single:open_reasoning_rm         12 / 18
single:openai_general_judge       9 / 18
single:openai_reflective_judge   10 / 18
expert oracle                    13 / 18
unique successes total:           2 problems
```

Aggregation and calibration sweep:

```text
Best full-split setting:
open_math_prm aggregation: min
normalization: rank/minmax/zscore all reach the same oracle ceiling
best static mixture: 72 / 100

Best mixed-subset setting:
open_math_prm aggregation: min
best static mixture: 13 / 18

Best static weights:
rank:   open_math_prm 0.5 + openai_reflective_judge 0.5
minmax: open_math_prm 0.4 + openai_reflective_judge 0.6
zscore: open_math_prm 0.3 + openai_reflective_judge 0.7
```

Decision:

```text
Do not train the main gate on hard_dev_100_n8 yet.

Reason:
- only 18 / 100 problems are mixed;
- the current expert oracle is only 1 point above domain/uniform and 1 point
  above the best single expert after calibration;
- most examples are either all experts succeed or no expert succeeds, so a gate
  would mostly learn trivial source/domain behavior or tie-breaking.

Next:
- build a larger mixed-rich scout split;
- increase MATH500 share and remove/reduce easy GSM8K;
- use the new eight-style N=8 generator;
- only train a gate if the mixed subset reaches roughly 60+ problems and keeps
  a meaningful oracle gap over best-single/uniform.
```

Recommended next run:

```text
prepare_public_subsets:
  MATH500=500, GSM8K=40, BBH per task=100

hard_mix_scout_160:
  120 MATH500
   40 BBH logical_deduction_seven_objects
  N=8
  temperature around 1.05

success criterion:
  mixed problems >= about 60
  expert oracle clearly above best single and uniform routing
```

## 2026-09-04: hard_mix_scout_160_n8

Goal:

```text
Create a harder, mixed-candidate-rich scout split with N=8 and eight distinct
candidate-generation styles, then decide whether trained-gate work is justified.
```

Data preparation:

```text
Prepared public subsets:
- MATH500: 500
- GSM8K: 40
- BBH logical deduction: 300, 100 per task

Sampled split:
- 120 HuggingFaceH4/MATH-500
-  40 BIG-Bench-Hard/logical_deduction_seven_objects
```

Candidate generation:

```text
input: data/splits/hard_mix_scout_160.jsonl
output: data/candidates/openai_hard_mix_scout160_n8.jsonl
problems: 160
candidates per problem: 8
total candidates: 1280
temperature: 1.05
candidate styles: 8 distinct prompt styles
reported OpenAI tokens: 724,484
```

Correctness labeling:

```text
labeled output: data/cache/openai_hard_mix_scout160_n8_labeled.jsonl
correct candidates: 682 / 1280 = 0.533
all-wrong problems: 56 / 160
all-correct problems: 61 / 160
mixed problems: 43 / 160
candidate upper bound: 104 / 160

correct-count histogram:
0: 56
1: 4
2: 6
3: 5
4: 4
5: 6
6: 9
7: 9
8: 61

mixed sources:
- MATH500: 26
- BBH logical_deduction_seven_objects: 17
```

Decision after labeling:

```text
The split is harder than hard_dev_100_n8:
- candidate correctness drops from 0.667 to 0.533;
- mixed examples improve from 18 / 100 to 43 / 160.

The mixed count is still below the target of roughly 60+ for trained-gate
claims, but it is high enough to score the mixed subset and inspect expert
complementarity.
```

Mixed subset expert scoring:

```text
input: data/splits/hard_mix_scout160_n8_mixed.jsonl
problems: 43
candidates: 344

OpenAI expert scoring tokens: 332,906
LLM gate tokens: 17,245

open-source experts:
- open_math_prm, Skywork-o1-Open-PRM-Qwen-2.5-1.5B, local CUDA
- open_reasoning_rm, Skywork-Reward-V2-Qwen3-1.7B, local CUDA
```

Mixed subset result, `open_math_prm` mean aggregation:

```text
domain_rule_gate:               32 / 43 = 0.744
openai_llm_gate:                32 / 43 = 0.744
uniform_ensemble:               33 / 43 = 0.767
single:open_math_prm:           26 / 43 = 0.605
single:open_reasoning_rm:       35 / 43 = 0.814
single:openai_general_judge:    31 / 43 = 0.721
single:openai_reflective_judge: 33 / 43 = 0.767
oracle_gate:                    40 / 43 = 0.930
```

Mixed subset result, `open_math_prm` min aggregation:

```text
domain_rule_gate:               27 / 43 = 0.628
openai_llm_gate:                31 / 43 = 0.721
uniform_ensemble:               34 / 43 = 0.791
single:open_math_prm:           24 / 43 = 0.558
single:open_reasoning_rm:       35 / 43 = 0.814
single:openai_reflective_judge: 33 / 43 = 0.767
oracle_gate:                    40 / 43 = 0.930
```

Mixed subset result, `open_math_prm` last aggregation:

```text
rank normalization:
domain_rule_gate:               33 / 43 = 0.767
openai_llm_gate:                33 / 43 = 0.767
uniform_ensemble:               36 / 43 = 0.837
single:open_math_prm:           27 / 43 = 0.628
single:open_reasoning_rm:       35 / 43 = 0.814
single:openai_reflective_judge: 33 / 43 = 0.767
oracle_gate:                    41 / 43 = 0.953

math only:
uniform_ensemble:               19 / 26 = 0.731
single:open_reasoning_rm:       18 / 26 = 0.692
oracle_gate:                    24 / 26 = 0.923

logic only:
open_reasoning_rm/domain/uniform/oracle all reach 17 / 17.
```

Aggregation and calibration sweep:

```text
Best static mixture on mixed subset:
- mean/rank: 36 / 43, open_math_prm 0.2 + open_reasoning_rm 0.8
- min/rank:  37 / 43, open_math_prm 0.2 + open_reasoning_rm 0.8
- last/rank: 38 / 43, open_math_prm 0.4 + openai_general_judge 0.5 + openai_reflective_judge 0.1

Best oracle:
- last aggregation reaches 41 / 43.
```

Interpretation:

```text
This scout succeeded at creating useful routing signal, but not enough mixed
examples yet for a confident trained-gate result.

Important change from the previous hard_dev_100_n8 finding:
- min aggregation was strongest on hard_dev_100_n8;
- last aggregation is strongest on this mixed scout.

Therefore, open_math_prm step aggregation should remain an ablation or a
calibrated choice, not a fixed assumption.
```

Next plan:

```text
Do not train the final gate on only 43 mixed examples.

Recommended next step:
- expand the same recipe to about 240 raw problems, or add roughly 80 more raw
  problems, targeting 60+ mixed examples;
- score only the resulting mixed subset first;
- if the oracle gap remains large, train a question-level gate.

Current expected gain from expansion:
- observed mixed rate: 43 / 160 = 26.9%;
- 240 raw problems would likely give about 64 mixed problems under the same
  distribution.
```

## 2026-09-04: hard_mix_scout_320_n8

Goal:

```text
Append 160 non-overlapping raw problems using the same hard-mix recipe and
bring the mixed-candidate pool above the threshold for trained-gate work.
```

Code updates:

```text
scripts/sample_dataset.py
- added --exclude-input to sample a new split while excluding problem IDs from
  previous JSONL files.

src/moprm/datasets/sampling.py
- added exclude_ids support to exact source-quota sampling and per-domain
  sampling.

scripts/merge_jsonl_records.py
- added a small utility for merging MoPRM JSONL files with duplicate
  problem_id checks.
```

Append split:

```text
input pool: data/cache/public_subsets/math_logic_combined.jsonl
exclude: data/splits/hard_mix_scout_160.jsonl
output: data/splits/hard_mix_scout_160_append1.jsonl
quota:
- 120 HuggingFaceH4/MATH-500
-  40 BIG-Bench-Hard/logical_deduction_seven_objects
seed: 31

overlap with first hard_mix_scout_160 split: 0
combined raw problems: 320
```

Append candidate generation:

```text
input: data/splits/hard_mix_scout_160_append1.jsonl
output: data/candidates/openai_hard_mix_scout160_append1_n8.jsonl
problems: 160
candidates per problem: 8
total candidates: 1280
temperature: 1.05
candidate styles: 8 distinct prompt styles
reported OpenAI tokens: 734,721
```

Append correctness labeling:

```text
correct candidates: 697 / 1280 = 0.545
all-wrong problems: 54 / 160
all-correct problems: 65 / 160
mixed problems: 41 / 160
candidate upper bound: 106 / 160

correct-count histogram:
0: 54
1: 6
2: 6
3: 6
4: 2
5: 3
6: 8
7: 10
8: 65

mixed sources:
- MATH500: 35
- BBH logical_deduction_seven_objects: 6
```

Combined 320 correctness:

```text
raw split: data/splits/hard_mix_scout_320.jsonl
labeled:   data/cache/openai_hard_mix_scout320_n8_labeled.jsonl
mixed:     data/splits/hard_mix_scout320_n8_mixed.jsonl

raw problems: 320
math/logic: 240 MATH500 + 80 BBH seven-object
candidates: 2560
correct candidates: 1379 / 2560 = 0.539
all-wrong problems: 110 / 320
all-correct problems: 126 / 320
mixed problems: 84 / 320
candidate upper bound: 210 / 320

correct-count histogram:
0: 110
1: 10
2: 12
3: 11
4: 6
5: 9
6: 17
7: 19
8: 126

mixed sources:
- MATH500: 61
- BBH logical_deduction_seven_objects: 23
```

OpenAI/API cost for the combined 320 scout:

```text
candidate generation tokens:        1,459,205
mixed-only OpenAI expert scoring:     618,571
mixed-only LLM gate routing:            32,194
```

Combined mixed-subset result, mean aggregation:

```text
domain_rule_gate:               56 / 84 = 0.667
openai_llm_gate:                60 / 84 = 0.714
uniform_ensemble:               64 / 84 = 0.762
single:open_math_prm:           46 / 84 = 0.548
single:open_reasoning_rm:       67 / 84 = 0.798
single:openai_general_judge:    63 / 84 = 0.750
single:openai_reflective_judge: 66 / 84 = 0.786
oracle_gate:                    76 / 84 = 0.905
```

Combined mixed-subset result, min aggregation:

```text
domain_rule_gate:               50 / 84 = 0.595
openai_llm_gate:                58 / 84 = 0.690
uniform_ensemble:               63 / 84 = 0.750
single:open_math_prm:           45 / 84 = 0.536
single:open_reasoning_rm:       67 / 84 = 0.798
single:openai_general_judge:    63 / 84 = 0.750
single:openai_reflective_judge: 66 / 84 = 0.786
oracle_gate:                    74 / 84 = 0.881
```

Combined mixed-subset result, last aggregation:

```text
domain_rule_gate:               53 / 84 = 0.631
openai_llm_gate:                57 / 84 = 0.679
uniform_ensemble:               65 / 84 = 0.774
single:open_math_prm:           44 / 84 = 0.524
single:open_reasoning_rm:       67 / 84 = 0.798
single:openai_general_judge:    63 / 84 = 0.750
single:openai_reflective_judge: 66 / 84 = 0.786
oracle_gate:                    76 / 84 = 0.905

math only:
best single expert:             44 / 61 = 0.721
best uniform ensemble:          42 / 61 = 0.689
oracle_gate:                    53 / 61 = 0.869

logic only:
open_reasoning_rm/domain/uniform/oracle all reach 23 / 23.
```

Aggregation and calibration sweep on the combined mixed subset:

```text
best single expert:
open_reasoning_rm:              67 / 84 = 0.798

best uniform ensemble:
geomean + rank:                 66 / 84 = 0.786

best static calibrated mixture:
min + rank:                     70 / 84 = 0.833
weights: open_math_prm 0.2 + open_reasoning_rm 0.7 + openai_general_judge 0.1

best oracle:
mean/last/geomean:              76 / 84 = 0.905
```

Expert complementarity diagnostics:

```text
mean aggregation:
all_experts_success: 36 / 84
no_expert_success:    7 / 84
unique_success:
- open_math_prm:            2
- open_reasoning_rm:        3
- openai_general_judge:     0
- openai_reflective_judge:  3

min aggregation:
all_experts_success: 39 / 84
no_expert_success:    8 / 84
unique_success:
- open_math_prm:            1
- open_reasoning_rm:        4
- openai_general_judge:     1
- openai_reflective_judge:  3

last aggregation:
all_experts_success: 33 / 84
no_expert_success:    7 / 84
unique_success:
- open_math_prm:            2
- open_reasoning_rm:        2
- openai_general_judge:     0
- openai_reflective_judge:  3
```

Decision:

```text
The project now has enough mixed examples to train the first lightweight gate.

Why:
- mixed examples reached 84 / 320, above the 60+ target;
- oracle is 76 / 84, clearly above best single 67 / 84 and best uniform 66 / 84;
- best static calibration reaches 70 / 84, proving that nontrivial score fusion
  can already beat any single expert on this split.

Main caveat:
- logic mixed examples are basically solved by open_reasoning_rm;
- the real routing challenge is MATH500 mixed selection.

Next:
- build a train/dev split over hard_mix_scout320_n8_mixed;
- train a question-level gate;
- report held-out PRM@8 against best single, uniform, LLM gate, static
  calibration, and oracle.
```

## 2026-09-04: Gate-v1 Trained Question-Level Router

Goal:

```text
Build the first trainable MoPRM router on hard_mix_scout_320_n8_mixed and
evaluate it with out-of-fold predictions rather than in-sample routing.
```

Implementation added:

```text
src/moprm/trained_gate.py
- deterministic question-level feature extraction;
- source/domain/task metadata features;
- numeric text statistics;
- hashed unigram/bigram problem-text features;
- multi-label logistic regression trained with NumPy/Adam;
- source/domain-stratified K-fold splitting;
- utility to attach trained gate weights into record.metadata.gate_weights.

scripts/train_gate_v1.py
- trains Gate-v1 with 5-fold CV;
- writes out-of-fold metadata-gate records;
- compares Gate-v1 against best single, uniform, domain-rule, OpenAI LLM gate,
  expert oracle, and cross-validated static calibration.

tests/test_trained_gate.py
- validates feature determinism, expert-success labels, stratified folds, and
  basic learned routing behavior.
```

Training label:

```text
For each problem and expert, label the expert as positive if that expert's
own top-ranked candidate is correct under the chosen normalization.
```

This gives Gate-v1 a multi-label target because several experts can be
successful on the same problem.

Verification:

```text
python -m unittest discover -s tests
53 tests OK
```

Main command:

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

Gate sharpness sweep:

```text
weight_power=1:
- mean aggregation:    Gate-v1 65 / 84 = 0.774
- geomean aggregation: Gate-v1 65 / 84 = 0.774

weight_power=2:
- mean aggregation:    Gate-v1 66 / 84 = 0.786
- geomean aggregation: Gate-v1 66 / 84 = 0.786

weight_power=4:
- mean aggregation:    Gate-v1 67 / 84 = 0.798
- geomean aggregation: Gate-v1 66 / 84 = 0.786

weight_power=8:
- mean aggregation:    Gate-v1 65 / 84 = 0.774
- geomean aggregation: Gate-v1 64 / 84 = 0.762
```

Best Gate-v1 result:

```text
setting: mean aggregation + rank normalization + weight_power=4

overall mixed:
Gate-v1 CV:                 67 / 84 = 0.798
CV static calibration:      67 / 84 = 0.798
best single expert:         67 / 84 = 0.798  # open_reasoning_rm
uniform ensemble:           64 / 84 = 0.762
domain-rule gate:           56 / 84 = 0.667
OpenAI LLM gate:            60 / 84 = 0.714
expert oracle:              76 / 84 = 0.905

math mixed:
Gate-v1 CV:                 44 / 61 = 0.721
CV static calibration:      44 / 61 = 0.721
best single expert:         44 / 61 = 0.721  # openai_reflective_judge
uniform ensemble:           41 / 61 = 0.672
domain-rule gate:           33 / 61 = 0.541
OpenAI LLM gate:            37 / 61 = 0.607
expert oracle:              53 / 61 = 0.869

logic mixed:
Gate-v1 CV:                 23 / 23 = 1.000
open_reasoning_rm/domain/uniform/LLM/oracle all reach 23 / 23.
```

Strongest cross-validated static calibration:

```text
setting: open_math_prm min aggregation + rank normalization

overall mixed:
CV static calibration:      69 / 84 = 0.821
Gate-v1 CV:                 63 / 84 = 0.750
best single expert:         67 / 84 = 0.798
uniform ensemble:           63 / 84 = 0.750
expert oracle:              74 / 84 = 0.881

learned per-fold static weights usually concentrate on:
open_math_prm 0.2 + open_reasoning_rm 0.7/0.8 + sometimes openai_general_judge 0.1
```

Interpretation:

```text
Gate-v1 is a useful trained-router baseline, not yet the final winning method.

What works:
- it is fully reproducible and cheap;
- it beats uniform/domain/LLM routing in its best setting;
- it reaches the best single expert under mean aggregation;
- it gives us a clean paper-ready negative/diagnostic if needed.

What does not work yet:
- the question-only gate does not beat the strongest CV-static calibration;
- average weights remain close to a soft ensemble and only mildly favor
  open_reasoning_rm;
- logic mixed examples are already solved, so the real challenge is math mixed
  selection.
```

Decision:

```text
Keep Gate-v1 as the trained baseline.

Next improvement should be one of:
1. candidate-aware Gate-v2: add candidate/expert score-shape features, not only
   problem text;
2. math-only gate: train/evaluate on the 61 MATH500 mixed examples where the
   oracle gap is large;
3. aggregation-aware gate: treat open_math_prm mean/min/last/geomean as
   separate pseudo-experts or add aggregation choice as a predicted variable;
4. expand to hard_mix_scout_480_n8 if more training data is needed.
```

## 2026-09-04: Gate-v2 Candidate-Aware Experiments

Goal:

```text
Move beyond the question-only Gate-v1 by giving the trained method access to
candidate-pool information available at selection time.
```

Implementation added:

```text
src/moprm/trained_gate.py
- extended FeatureConfig with optional score-shape features for problem-level
  Gate-v2;
- score-shape features include per-expert score variance, top gaps, top-choice
  agreement, pairwise score correlations, and consensus statistics.

scripts/train_gate_v2.py
- thin wrapper around the trained-gate runner with score-shape features enabled.

scripts/expand_math_aggregation_experts.py
- expands cached open_math_prm step rewards into aggregation-specific
  pseudo-experts:
  open_math_prm_mean, open_math_prm_min, open_math_prm_last,
  open_math_prm_geomean.

src/moprm/candidate_gate.py
- implements Candidate Gate-v2 as a candidate-level logistic selector;
- features include candidate text statistics, per-expert raw/normalized scores,
  per-expert top-candidate indicators, z-scored margins, and top gaps.

scripts/train_candidate_gate_v2.py
- trains Candidate Gate-v2 with problem-level 5-fold CV;
- writes out-of-fold candidate scores into
  record.metadata.candidate_gate_scores.candidate_gate_v2_cv;
- compares against best single expert, uniform, domain-rule, OpenAI LLM gate,
  CV-static calibration, and expert oracle.

tests added:
- tests/test_candidate_gate.py
- tests/test_expand_math_aggregation_experts.py
```

Verification:

```text
python -m unittest discover -s tests
58 tests OK
```

### Problem-level score-shape Gate-v2

Command:

```bash
python scripts/train_gate_v2.py \
  --input data/scored/openai_hard_mix_scout320_n8_mixed_pool_routed.jsonl \
  --output-dir data/scored/gate_v2_p4 \
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

Result:

```text
Problem-level score-shape Gate-v2 did not materially improve over Gate-v1.

Best setting:
mean aggregation + rank normalization + weight_power=4

Gate-v2 score-shape:       67 / 84 = 0.798
Gate-v1 question-only:     67 / 84 = 0.798
best single expert:        67 / 84 = 0.798
uniform ensemble:          64 / 84 = 0.762
OpenAI LLM gate:           60 / 84 = 0.714
expert oracle:             76 / 84 = 0.905
```

Interpretation:

```text
Simply adding problem-level score-shape features is not enough. The gate still
produces soft, nearly static expert mixtures and does not exploit candidate-level
differences.
```

### Aggregation-as-pseudo-experts diagnostic

Command:

```bash
python scripts/expand_math_aggregation_experts.py \
  --input data/scored/openai_hard_mix_scout320_n8_mixed_pool_routed.jsonl \
  --output data/scored/openai_hard_mix_scout320_n8_mixed_pool_routed_math_agg_experts.jsonl \
  --aggregations mean,min,last,geomean \
  --overwrite
```

Result with problem-level score-shape gate:

```text
expert pool:
- open_math_prm_mean
- open_math_prm_min
- open_math_prm_last
- open_math_prm_geomean
- open_reasoning_rm
- openai_general_judge
- openai_reflective_judge

expert oracle:             78 / 84 = 0.929
CV-static calibration:     68 / 84 = 0.810
problem-level Gate-v2:     65 / 84 = 0.774
```

Candidate-level Gate-v2 on the same pseudo-expert pool:

```text
Candidate Gate-v2:         68 / 84 = 0.810
math mixed:                46 / 61 = 0.754
logic mixed:               22 / 23 = 0.957
```

Interpretation:

```text
Aggregation variants create real oracle headroom, but exposing all four math
aggregations as separate experts increases the expert space from 4 to 7. With
only 84 mixed problems, the trained gate becomes less stable. Keep this as a
diagnostic/future direction, not the current main method.
```

### Candidate Gate-v2

Main command:

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

Best result:

```text
setting: open_math_prm mean aggregation + rank normalization + l2=0.02

overall mixed:
Candidate Gate-v2:         72 / 84 = 0.857
CV static calibration:     67 / 84 = 0.798
best single expert:        67 / 84 = 0.798  # open_reasoning_rm
uniform ensemble:          64 / 84 = 0.762
domain-rule gate:          56 / 84 = 0.667
OpenAI LLM gate:           60 / 84 = 0.714
expert oracle:             76 / 84 = 0.905

math mixed:
Candidate Gate-v2:         50 / 61 = 0.820
CV static calibration:     44 / 61 = 0.721
best single expert:        44 / 61 = 0.721  # openai_reflective_judge
uniform ensemble:          41 / 61 = 0.672
domain-rule gate:          33 / 61 = 0.541
OpenAI LLM gate:           37 / 61 = 0.607
expert oracle:             53 / 61 = 0.869

logic mixed:
Candidate Gate-v2:         22 / 23 = 0.957
open_reasoning_rm/oracle:  23 / 23 = 1.000
```

Aggregation sweep for Candidate Gate-v2 with rank normalization and l2=0.02:

```text
mean:       72 / 84 = 0.857
last:       71 / 84 = 0.845
geomean:    71 / 84 = 0.845
min:        69 / 84 = 0.821
```

Regularization sanity sweep for mean aggregation:

```text
l2=0.001: 69 / 84 = 0.821
l2=0.005: 71 / 84 = 0.845
l2=0.020: 72 / 84 = 0.857
l2=0.050: 70 / 84 = 0.833
```

Normalization sanity sweep for mean aggregation and l2=0.02:

```text
rank:       72 / 84 = 0.857
minmax:     68 / 84 = 0.810
zscore:     68 / 84 = 0.810
```

Interpretation:

```text
Candidate Gate-v2 is now the strongest trained method. The improvement is
concentrated on MATH500 mixed examples: 50 / 61 versus 44 / 61 for the best
single expert and 41 / 61 for uniform ensemble.

This is stronger evidence for MoPRM than Gate-v1 because the model uses
candidate-level expert-score patterns rather than only coarse problem routing.
The remaining caveat is sample size: this should be validated on a fresh or
larger mixed split before making a very strong claim.
```

Decision:

```text
Use Candidate Gate-v2 as the current main method:
- open_math_prm aggregation: mean
- normalization: rank
- l2: 0.02
- split: 5-fold source/domain-stratified by problem

Keep these as ablations/diagnostics:
- Gate-v1 question-only baseline
- problem-level score-shape Gate-v2 negative result
- open_math_prm min aggregation for static calibration
- aggregation-as-pseudo-experts oracle/headroom diagnostic
```

## 2026-09-04: Candidate Gate-v2 Win/Loss Analysis

Goal:

```text
Inspect where Candidate Gate-v2 gains or loses relative to the strongest
baselines at selection level, not only aggregate accuracy.
```

Implementation added:

```text
scripts/analyze_candidate_gate_wins.py
- compares a candidate_gate output against default baselines, best single expert,
  optional CV-static metadata gate output, and expert oracle;
- reports wins/losses/both-correct/both-wrong/same-selection/different-selection
  by all/math/logic/source groups;
- prints representative win/loss cases with candidate correctness strings and
  each expert's top candidate.

tests/test_candidate_gate_wins.py
- verifies win/loss accounting on a synthetic example.
```

Verification:

```text
python -m unittest tests.test_candidate_gate_wins
1 test OK
```

Main command against the same-aggregation CV-static baseline:

```bash
python scripts/analyze_candidate_gate_wins.py \
  --input data/scored/candidate_gate_v2_l2_02/candidate_gate_v2_cv_mean_rank.jsonl \
  --static-input data/scored/candidate_gate_v2_l2_02/cv_static_calibrated_mean_rank.jsonl \
  --normalization rank \
  --case-baseline best_single \
  --max-cases 20
```

Method summary:

```text
Candidate Gate-v2:           72 / 84 = 0.857
best single open_reasoning:  67 / 84 = 0.798
CV-static, mean aggregation: 67 / 84 = 0.798
uniform ensemble:            64 / 84 = 0.762
OpenAI LLM gate:             60 / 84 = 0.714
domain-rule gate:            56 / 84 = 0.667
expert oracle:               76 / 84 = 0.905
```

Win/loss summary:

```text
Candidate Gate-v2 vs best single open_reasoning_rm:
overall: 9 wins, 4 losses, net +5
math:    9 wins, 3 losses, net +6
logic:   0 wins, 1 loss,  net -1

Candidate Gate-v2 vs same-aggregation CV-static:
overall: 9 wins, 4 losses, net +5
math:    9 wins, 3 losses, net +6
logic:   0 wins, 1 loss,  net -1

Candidate Gate-v2 vs uniform ensemble:
overall: 12 wins, 4 losses, net +8
math:    12 wins, 3 losses, net +9
logic:   0 wins, 1 loss,  net -1

Candidate Gate-v2 vs OpenAI LLM gate:
overall: 16 wins, 4 losses, net +12
math:    16 wins, 3 losses, net +13
logic:   0 wins, 1 loss,  net -1

Candidate Gate-v2 vs domain-rule gate:
overall: 21 wins, 5 losses, net +16
math:    21 wins, 4 losses, net +17
logic:   0 wins, 1 loss,  net -1
```

Direct comparison against the previously strongest CV-static setting:

```bash
python scripts/analyze_candidate_gate_wins.py \
  --input data/scored/candidate_gate_v2_l2_02/candidate_gate_v2_cv_mean_rank.jsonl \
  --static-input data/scored/candidate_gate_v2_l2_02/cv_static_calibrated_min_rank.jsonl \
  --normalization rank \
  --case-baseline cv_static_calibrated \
  --max-cases 20
```

Result:

```text
Candidate Gate-v2, mean aggregation:       72 / 84 = 0.857
strongest CV-static, min aggregation:      69 / 84 = 0.821

overall: 8 wins, 5 losses, net +3
math:    8 wins, 4 losses, net +4
logic:   0 wins, 1 loss,  net -1
```

Wins versus best single open_reasoning_rm:

```text
math500_0018  truths=CWWWWWWW
math500_0030  truths=WCCCCWCC
math500_0046  truths=CWWCCCCC
math500_0048  truths=CWWWWWCW
math500_0165  truths=WWCCWWWW
math500_0323  truths=WWWWWWWC
math500_0389  truths=WWWWWWCW
math500_0441  truths=WWWWWCWC
math500_0499  truths=CCWWWCCC
```

Losses versus best single open_reasoning_rm:

```text
bbh_logical_deduction_seven_objects_0077  truths=CCWWWWWW
math500_0034                              truths=WCWWCWWW
math500_0095                              truths=WCWWWCCW
math500_0364                              truths=WCWWWWCC
```

Wins versus strongest CV-static min aggregation:

```text
math500_0018  truths=CWWWWWWW
math500_0030  truths=WCCCCWCC
math500_0048  truths=CWWWWWCW
math500_0165  truths=WWCCWWWW
math500_0323  truths=WWWWWWWC
math500_0328  truths=WWWWCWWW
math500_0389  truths=WWWWWWCW
math500_0441  truths=WWWWWCWC
```

Losses versus strongest CV-static min aggregation:

```text
bbh_logical_deduction_seven_objects_0077  truths=CCWWWWWW
math500_0034                              truths=WCWWCWWW
math500_0095                              truths=WCWWWCCW
math500_0334                              truths=WCCCCCWW
math500_0364                              truths=WCWWWWCC
```

Expert-oracle comparison:

```text
Candidate Gate-v2: 72 / 84
expert oracle:     76 / 84

V2 beats expert oracle on 4 problems and loses to it on 8 problems.
Net gap: -4.
```

Important nuance:

```text
The oracle here is the existing expert-oracle gate, not a candidate oracle. It
can still select a wrong candidate when multiple oracle-successful experts are
combined and their scores conflict. Therefore Candidate Gate-v2 can beat this
expert oracle on a few cases, even though it remains below oracle overall.
```

Interpretation:

```text
The positive effect is strongly concentrated in MATH500 mixed cases. Candidate
Gate-v2 often recovers problems where open_reasoning_rm or static calibration
selects a wrong candidate with high reward-model confidence. The method's main
failure mode is over-trusting OpenAI judge-style score patterns in one BBH logic
case and still missing several math cases where the correct candidate is
available but expert scores are misleading.

This gives a clean error-analysis story:
- V2 is not simply copying the best expert, because it differs from
  open_reasoning_rm on 62 / 84 selections;
- the net gain over strong baselines comes from candidate-level score-pattern
  learning on math;
- the next robustness step should test whether this math gain persists on a
  fresh mixed split.
```

## 2026-09-05: Candidate Gate-v2 Loss Case Audit

Goal:

```text
Inspect Candidate Gate-v2's own wrong selections and check whether they indicate
a need for a more expressive Gate-v3.
```

Full note:

```text
notes/candidate_gate_v2_loss_cases.md
```

Summary:

```text
Candidate Gate-v2 reported accuracy: 72 / 84 = 0.857
Wrong selections under current labels: 12

wrong-case domain/source distribution:
math / HuggingFaceH4/MATH-500:                         11
logic / BIG-Bench-Hard logical_deduction_seven_objects: 1

correct-candidate-count histogram among wrong cases:
1 correct candidate: 3
2 correct candidates: 4
3 correct candidates: 3
4 correct candidates: 1
5 correct candidates: 1
```

Important label finding:

```text
11 / 12 wrong selections become answer-equivalent to the gold answer under a
simple loose diagnostic that removes inline LaTeX math delimiters such as
\(...\) and simple unit suffixes.

This loose diagnostic is not used in the official metric yet. It indicates that
the current loss set is dominated by answer-normalization artifacts.
```

Only clearly semantic loss in this audit:

```text
bbh_logical_deduction_seven_objects_0077

correct candidates: 000, 001
Candidate Gate-v2 selected: 006, wrong

expert top choices:
open_math_prm             -> 000, correct
open_reasoning_rm         -> 001, correct
openai_general_judge      -> 006, wrong
openai_reflective_judge   -> 006, wrong
```

Interpretation:

```text
Do not train Gate-v3 immediately. The dataset is small and the apparent loss
set is mostly label-normalization noise. The next best experimental step is to
improve/audit the answer checker, re-label the existing candidate files, and
then re-run the same Gate-v2 evaluation.

A problem-text-aware gate remains plausible, but should be treated as a
low-capacity extension: frozen problem embeddings or LLM-generated problem tags
plus the existing candidate/expert score features, not end-to-end text encoder
training on 84 mixed problems.
```

## 2026-09-05: Conservative Label Cleanup and Clean-label V2 Effect

Goal:

```text
Fix obvious automatic answer-checking artifacts found by the V2 loss audit and
measure Candidate Gate-v2 under cleaned labels.
```

Implementation:

```text
src/moprm/answer_checking.py
- remove LaTeX inline/display math wrappers \(...\) and \[...\];
- for numeric answers only, ignore simple unit suffixes such as seconds,
  degrees, units, gallons, etc.;
- preserve percentage semantics, so 36% is still not equal to 36.

tests/test_answer_checking.py
- add regression coverage for inline math wrappers, coordinate tuples, simple
  unit suffixes, and percentage non-equivalence.
```

Verification:

```text
python -m unittest discover -s tests
61 tests OK
```

Full note:

```text
notes/label_cleanup_v2_results.md
```

Label-change summary:

```text
records:              84
candidates:           672
old correct labels:   371 / 672
new correct labels:   475 / 672
changed records:      23 / 84

old mixed records:    84
new mixed records:    67
new all-correct:      17
new all-wrong:        0
```

Old V2 selections under cleaned labels:

```text
all original 84:
Candidate Gate-v2:       83 / 84 = 0.988
best single expert:      81 / 84 = 0.964  # open_reasoning_rm
uniform ensemble:        80 / 84 = 0.952
domain-rule gate:        68 / 84 = 0.810
OpenAI LLM gate:         73 / 84 = 0.869
expert oracle:           82 / 84 = 0.976

clean mixed 67:
Candidate Gate-v2:       66 / 67 = 0.985
best single expert:      64 / 67 = 0.955
uniform ensemble:        63 / 67 = 0.940
domain-rule gate:        51 / 67 = 0.761
OpenAI LLM gate:         56 / 67 = 0.836
expert oracle:           65 / 67 = 0.970
```

Retrained V2 under cleaned labels:

```text
clean mixed 67:
Candidate Gate-v2:       64 / 67 = 0.955
CV static calibration:   64 / 67 = 0.955
best single expert:      64 / 67 = 0.955
expert oracle:           65 / 67 = 0.970

all original 84:
Candidate Gate-v2:       82 / 84 = 0.976
CV static calibration:   81 / 84 = 0.964
best single expert:      81 / 84 = 0.964
uniform ensemble:        80 / 84 = 0.952
OpenAI LLM gate:         73 / 84 = 0.869
expert oracle:           82 / 84 = 0.976
```

Interpretation:

```text
The original 72 / 84 V2 result was strongly underestimated by answer-checking
noise. After conservative label cleanup, old V2 selections score 83 / 84, while
the retrained all-84 clean-label V2 scores 82 / 84.

The cleaned mixed subset shrinks from 84 to 67 because 17 examples become
all-correct. On this smaller/easier subset, retrained V2 ties best single and
CV-static baselines. Therefore there is no strong evidence that Gate-v3 training
is necessary right now.

Next recommendation: freeze Gate-v2 as the main method, use label cleanup as an
important error-analysis result, and prioritize a larger/fresher clean-label
split if more experimental work is needed.
```

## 2026-09-05: Two-Open-PRM Clean-label Ablation

Goal:

```text
Remove both OpenAI judge-style experts and test whether Candidate Gate-v2 still
works with only the two non-OpenAI open-source reward experts.
```

Motivation:

```text
The candidate answers are OpenAI-generated, and two experts in the main pool are
OpenAI judge-style scorers. This ablation tests whether the routing result
depends on those OpenAI judge experts.
```

Input:

```text
data/scored/openai_hard_mix_scout320_n8_mixed_pool_routed_clean_labels.jsonl
```

Temporary two-expert pool:

```bash
python scripts/rewrite_expert_pool.py \
  --input data/scored/openai_hard_mix_scout320_n8_mixed_pool_routed_clean_labels.jsonl \
  --output data/scored/tmp_open_prm_2expert_clean_labels_mean.jsonl \
  --drop openai_general_judge \
  --drop openai_reflective_judge \
  --overwrite
```

Retained experts:

```text
open_math_prm
open_reasoning_rm
```

Clean-label pool status:

```text
all original records:     84
candidate-mixed records:  67
all-correct records:      17
all-wrong records:         0

math records:             61
logic records:            23
math candidate-mixed:     44
logic candidate-mixed:    23
```

Two-expert baseline with open_math_prm mean aggregation and rank normalization:

```text
overall:
single:open_math_prm          58 / 84 = 0.690
single:open_reasoning_rm      81 / 84 = 0.964
uniform ensemble              75 / 84 = 0.893
domain-rule gate              68 / 84 = 0.810
OpenAI LLM gate               70 / 84 = 0.833
expert top-choice oracle      83 / 84 = 0.988

math:
single:open_math_prm          45 / 61 = 0.738
single:open_reasoning_rm      58 / 61 = 0.951
expert top-choice oracle      60 / 61 = 0.984

logic:
single:open_math_prm          13 / 23 = 0.565
single:open_reasoning_rm      23 / 23 = 1.000
expert top-choice oracle      23 / 23 = 1.000
```

Complementarity:

```text
success pattern legend: open_math_prm, open_reasoning_rm

11 count = 56
01 count = 25
10 count = 2
00 count = 1

unique success:
open_math_prm       2
open_reasoning_rm   25
```

Candidate Gate-v2, clean mixed 67:

```text
best tested aggregations: mean/last/geomean + rank

Candidate Gate-v2:       65 / 67 = 0.970
CV static calibration:   64 / 67 = 0.955
best single expert:      64 / 67 = 0.955  # open_reasoning_rm
uniform ensemble:        58 / 67 = 0.866
domain-rule gate:        51 / 67 = 0.761
OpenAI LLM gate:         53 / 67 = 0.791
expert oracle:           66 / 67 = 0.985

math:                    42 / 44 = 0.955
logic:                   23 / 23 = 1.000
```

Candidate Gate-v2, all original 84 under cleaned labels:

```text
best tested aggregations: mean/geomean + rank

Candidate Gate-v2:       83 / 84 = 0.988
CV static calibration:   81 / 84 = 0.964
best single expert:      81 / 84 = 0.964  # open_reasoning_rm
uniform ensemble:        75 / 84 = 0.893
domain-rule gate:        68 / 84 = 0.810
OpenAI LLM gate:         70 / 84 = 0.833
expert oracle:           83 / 84 = 0.988

math:                    60 / 61 = 0.984
logic:                   23 / 23 = 1.000
```

Win/loss against best single open_reasoning_rm:

```text
clean mixed 67:
overall: 1 win, 0 losses, net +1
math:    1 win, 0 losses, net +1
logic:   0 wins, 0 losses

all original 84:
overall: 2 wins, 0 losses, net +2
math:    2 wins, 0 losses, net +2
logic:   0 wins, 0 losses
```

Routing-relevant subset:

```text
records where at least one open PRM top choice is wrong: 28
all-84 trained Candidate Gate-v2 on this subset: 27 / 28 = 0.964

pattern 11: 56 / 56
pattern 01: 25 / 25
pattern 10:  2 / 2
pattern 00:  0 / 1
```

Only all-84 V2 failure:

```text
math500_0018
gold: 28
selected final answer: 152
candidate truths: CWWWWWWW
```

Interpretation:

```text
This is the strongest current robustness ablation against the OpenAI-judge
homogeneity concern. After removing both OpenAI judge experts, Candidate Gate-v2
still improves over the strongest open-source single expert and reaches the
two-expert oracle on the all-84 clean-label evaluation.

Caveat: candidate answers are still OpenAI-generated, and after label cleanup
the 84-problem subset is small and near ceiling. This result should be used as a
robustness ablation, not as a claim that the full pipeline is OpenAI-free.
```

Full note:

```text
notes/open_prm_2expert_clean_label_ablation.md
```

## 2026-09-05: Clean-label Three-Expert No-RM Ablation

Detailed note:

```text
notes/three_expert_no_reasoning_rm_clean_label_ablation.md
```

Purpose:

```text
Re-run the three-expert no-RM ablation after conservative label cleanup.
The earlier no-RM result was useful, but likely overstated routing gain because
many pre-clean-label errors were answer-checking artifacts.
```

Setup:

```text
input: data/scored/openai_hard_mix_scout320_n8_mixed_pool_routed_clean_labels.jsonl
removed expert: open_reasoning_rm
remaining experts:
- open_math_prm
- openai_general_judge
- openai_reflective_judge
```

Clean-label no-RM baselines on original 84:

```text
single:open_math_prm              58 / 84 = 0.690
single:openai_general_judge       79 / 84 = 0.940
single:openai_reflective_judge    79 / 84 = 0.940
uniform_ensemble                  75 / 84 = 0.893
domain_rule_gate                  67 / 84 = 0.798
OpenAI LLM gate                   72 / 84 = 0.857
expert top-choice oracle          81 / 84 = 0.964
```

Candidate Gate-v2 no-RM results:

```text
clean mixed 67, best aggregation = last + rank:
Candidate Gate-v2:       63 / 67 = 0.940
best single expert:      62 / 67 = 0.925
CV-static calibration:   61 / 67 = 0.910
expert oracle:           66 / 67 = 0.985

all original 84, best aggregation = mean/geomean + rank:
Candidate Gate-v2:       81 / 84 = 0.964
best single expert:      79 / 84 = 0.940
CV-static calibration:   77 / 84 = 0.917
expert oracle:           81 / 84 = 0.964
```

Interpretation:

```text
Removing open_reasoning_rm does not collapse the system: clean-label no-RM
Candidate Gate-v2 still improves by +1 problem on the clean mixed 67 subset
and +2 problems on the original 84-example clean-label pool.

However, the effect is much smaller than the pre-clean-label no-RM result
(74 / 84 vs 66 / 84). The final report should present the clean-label no-RM
analysis as a robustness ablation, while treating the older result as part of
the label-cleaning/error-analysis narrative.
```
