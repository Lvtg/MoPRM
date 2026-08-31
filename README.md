# MoPRM

MoPRM is a course-project experiment on **router-guided mixtures of process reward models** for reasoning tasks.

The project studies whether a lightweight gate can choose or weight multiple PRM/verifier experts for different problems and candidate solutions, improving Best-of-N selection compared with any single expert or a naive ensemble.

## Current Scope

Recommended 15-day scope:

- domains: math reasoning and logic reasoning, with code reasoning as an optional extension;
- experts: math PRM, reflective/general judge, rule-based verifier, and optionally an MC-style recoverability expert;
- gate: question-level trained router first, candidate-aware router only if time allows;
- main metric: `PRM@8`, with `PRM@16` as an extension;
- baselines: best single expert, uniform ensemble, domain-rule gate, LLM gate, trained gate, oracle gate.

## Main Plan

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

## Relation to Earlier Work

Earlier notes in this repo focus on **Beyond the First Error: Process Reward Models for Reflective Mathematical Reasoning**. That paper remains useful as motivation: it shows that different reasoning styles need different process supervision signals. In MoPRM, reflective PRM scoring can be treated as one expert among several, instead of the whole project scope.

## Git Policy

Important source code, experiment configs, notes, and result summaries should be committed regularly. Large datasets, model weights, generated candidates, scored caches, rendered PDFs, and temporary images should stay out of git.
