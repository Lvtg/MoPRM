# Resource Requirements for MoPRM

This note answers what we need to download or prepare for the 15-day MoPRM project.

## Short Answer

Do not download large models blindly. The first step is to verify hardware and run tiny smoke tests, but the main experiment now requires at least two non-OpenAI PRM/reward experts.

Minimum resources:

- small benchmark subsets for math and logic;
- a candidate solution generator, either local or API-based;
- 4 scoring experts, including at least 2 non-OpenAI experts;
- a lightweight embedding model or feature extractor for the gate;
- exact-answer checking scripts.

The current OpenAI multi-rubric scorer is a baseline. It is not enough for the final heterogeneous MoPRM claim.

## Data

Recommended first-stage datasets:

| Domain | Dataset | Need to download? | Notes |
|---|---|---:|---|
| Math | MATH500 or MATH subset | Yes | Main math reasoning set. Use a small subset first. |
| Math | GSM8K | Optional | Easier math sanity check. |
| Logic | BBH logical deduction subset | Yes | Clean final-answer checking if multiple-choice. |
| Logic | LogiQA/ReClor-style data | Optional | Useful if we need more realistic logic tasks. |
| Code | HumanEval or MBPP | Optional | Only add if math+logic runs smoothly. |

The repo should store only processed metadata or tiny samples. Full raw datasets should live under ignored paths such as `data/raw/`.

## Models and Experts

Recommended final expert pool:

| Expert | Source type | First target | Notes |
|---|---|---|---|
| `open_math_prm` | non-OpenAI open-source PRM | Qwen2.5-Math-PRM, Skywork PRM, or RLHFlow/OpenR-style math PRM | Prioritize a smaller/runnable checkpoint first; step aggregation is required. |
| `open_logic_prm` | non-OpenAI open-source logic/reasoning RM or PRM | LogicReward-style model or open-source reasoning reward model | Logic PRMs may be less plug-and-play than math PRMs; document limitations if using a general reasoning RM. |
| `openai_general_judge` | OpenAI judge | existing scorer split into a general-only expert | Baseline/supporting expert. |
| `openai_reflective_judge` | OpenAI judge | existing scorer split into a reflective-only expert | Uses BFE-inspired self-checking/error-recovery rubric. |
| Non-leaking verifier | local scripts | optional auxiliary expert | Use only checks available at selection time, such as format, consistency, public code tests, or symbolic constraints. |

Gold-answer exact matching is used for evaluation and oracle-label construction only. It must not be treated as a normal expert for math or logic, because that would leak the benchmark answer into selection.

Optional later expert:

| Expert | Status | Notes |
|---|---|---|
| MC-style expert | Optional | Prefer an existing MC-style PRM over doing expensive rollouts. |

For the gate:

- start with a lightweight classifier over embeddings;
- use a local sentence embedding model if available;
- otherwise use simple lexical/domain features for the first baseline.

## What We Should Not Download First

Avoid downloading these until the data/scoring pipeline works and the first target checkpoint is chosen:

- multiple 7B PRMs at once;
- large training datasets;
- full model checkpoints for generation;
- full MC rollout data.

The project bottleneck is not only model size. It is having one clean shared candidate set, reliable answer checking, calibrated expert scores, and comparable baselines. However, after the OpenAI baseline, model heterogeneity becomes a core requirement.

## First Download Checklist

Day 1-2 only:

- one math dataset subset;
- one logic dataset subset;
- one small embedding model or a no-download feature baseline;
- metadata for candidate PRM experts.

Day 3-6:

- one runnable open-source math PRM;
- candidate solution data or a generation API path;
- OpenAI general/reflective judge baseline.

Current selected first math PRM:

```text
Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B
```

Local setup status:

```text
.venv has CPU torch + transformers + accelerate installed.
.venv_cuda has CUDA torch 2.7.1+cu128 and can see the RTX 5070 Laptop GPU.
Skywork 1.5B weights are cached locally under models/hf_cache.
```

Day 7-9:

- one runnable non-OpenAI logic/reasoning PRM or reward model;
- adapter that writes scores into the same JSONL schema;
- heterogeneous expert disagreement analysis.

After the first baseline table:

- train or fit the gate only after heterogeneous expert scores exist;
- expand from `N=8` to `N=16` only if scoring cost is under control.
