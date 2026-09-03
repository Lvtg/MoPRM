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
