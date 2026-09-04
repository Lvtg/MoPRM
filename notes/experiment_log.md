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
