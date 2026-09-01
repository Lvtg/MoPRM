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
| `open_logic_prm` | non-OpenAI open-source logic/reasoning RM or PRM | logical consistency and reasoning validity |
| `openai_general_judge` | OpenAI judge | broad candidate reliability |
| `openai_reflective_judge` | OpenAI judge | self-checking and error-recovery quality |

## Main Plan

The current project goal is in [notes/project_goal.md](notes/project_goal.md).

The current experiment plan is in [notes/moprm_15_day_experiment_plan.md](notes/moprm_15_day_experiment_plan.md).

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

## Relation to Earlier Work

Earlier notes in this repo focus on **Beyond the First Error: Process Reward Models for Reflective Mathematical Reasoning**. That paper remains useful as motivation: it shows that different reasoning styles need different process supervision signals. In MoPRM, reflective PRM scoring can be treated as one expert among several, instead of the whole project scope.

## Git Policy

Important source code, experiment configs, notes, and result summaries should be committed regularly. Large datasets, model weights, generated candidates, scored caches, rendered PDFs, and temporary images should stay out of git.
