# Resource Requirements for MoPRM

This note answers what we need to download or prepare for the 15-day MoPRM project.

## Short Answer

Do not download large models immediately. The first two days should only verify availability and run tiny smoke tests.

Minimum resources:

- small benchmark subsets for math and logic;
- a candidate solution generator, either local or API-based;
- 2-3 scoring experts;
- a lightweight embedding model or feature extractor for the gate;
- exact-answer checking scripts.

Large PRM models are useful but not mandatory for the first prototype.

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

Recommended expert pool:

| Expert | First version | Notes |
|---|---|---|
| Math PRM | Try an open-source PRM if local hardware allows | Candidate options include Qwen2.5-Math-PRM, Math-Shepherd-style PRM, or Skywork PRM. |
| Reflective/general judge | LLM-as-judge | Can reuse rules inspired by Beyond the First Error. |
| Non-leaking verifier | Local scripts | Use only checks available at selection time, such as format, consistency, public code tests, or symbolic constraints. |

Gold-answer exact matching is used for evaluation and oracle-label construction only. It must not be treated as a normal expert for math or logic, because that would leak the benchmark answer into selection.
| MC-style expert | Optional | Prefer an existing MC-style PRM over doing expensive rollouts. |

For the gate:

- start with a lightweight classifier over embeddings;
- use a local sentence embedding model if available;
- otherwise use simple lexical/domain features for the first baseline.

## What We Should Not Download First

Avoid downloading these until the data/scoring pipeline works:

- multiple 7B PRMs;
- large training datasets;
- full model checkpoints for generation;
- full MC rollout data.

The project bottleneck is not model size at the beginning. It is having one clean shared candidate set, reliable answer checking, calibrated expert scores, and comparable baselines.

## First Download Checklist

Day 1-2 only:

- one math dataset subset;
- one logic dataset subset;
- one small embedding model or a no-download feature baseline;
- metadata for candidate PRM experts.

Day 3-6:

- one runnable math PRM, if feasible;
- candidate solution data or a generation API path;
- optional general LLM judge setup.

After the first baseline table:

- add one more expert only if it creates clear complementarity;
- expand from `N=8` to `N=16` only if scoring cost is under control.
