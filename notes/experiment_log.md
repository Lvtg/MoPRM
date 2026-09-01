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
