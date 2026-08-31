# MoPRM 15-Day Experiment Plan

## 0. Project Positioning

Working title:

**MoPRM: Router-Guided Mixture of Process Reward Models for Multi-Domain Reasoning**

Core question:

Can a lightweight gate dynamically route different reasoning problems or candidate solutions to suitable process reward model experts, improving Best-of-N selection over any single PRM or naive ensemble?

The project shifts from improving one PRM to coordinating multiple PRMs. This is a good course-project scope because we do not need to train large PRMs from scratch. The main trainable component is a small gate/router, while the expert pool can reuse existing PRMs, reward models, rule-based verifiers, or LLM judges.

Relation to our earlier discussions:

- **Beyond the First Error** remains useful as a motivation and expert source: it shows that different reasoning styles, especially reflective long-CoT, require different process supervision signals.
- **MC/recoverability** can become one expert or one signal, not the whole project.
- **Reflection-aware aggregation** can become a special aggregation baseline inside the math/reflective subset.
- The new main contribution is broader: **process reward routing** across domains and reasoning styles.

## 1. Method Overview

For each problem, generate or collect multiple candidate solutions. Each expert PRM/verifier scores every candidate. A gate predicts how much to trust each expert. The final score is a weighted aggregation.

Pipeline:

```text
problem q
  -> generate N candidate solutions {s_1, ..., s_N}
  -> split solutions into steps when needed
  -> expert PRMs/verifiers produce scores E_k(q, s_j)
  -> normalize expert scores within the same problem
  -> gate predicts weights w_k(q) or w_k(q, s_j)
  -> final score S(q, s_j) = sum_k w_k * norm(E_k(q, s_j))
  -> choose argmax_j S(q, s_j)
  -> evaluate final-answer correctness
```

Recommended first version:

- Use a **question-level gate** first: `w = Gate(q)`.
- Add a **candidate-aware gate** only if time allows: `w = Gate(q, s_j)`.
- Use top-k sparse routing if possible: activate only the top 1 or top 2 experts.

## 2. Expert Pool

The expert pool should be heterogeneous but manageable. The first 15-day version should target 3-4 experts.

Recommended experts:

| Expert | Role | Priority | Notes |
|---|---|---:|---|
| Math PRM | Strong on mathematical step correctness | High | Example: Qwen2.5-Math-PRM, Math-Shepherd-style PRM, Skywork PRM if runnable |
| Reflective PRM or BFE-style judge | Strong on self-correction / reflective traces | High | Can be an LLM judge using Error Propagation / Error Cessation rules if the original model is unavailable |
| General LLM judge | Broad but slower baseline expert | Medium | Useful for logic and mixed-domain reasoning |
| Non-leaking verifier | High precision checks that do not use hidden gold answers | Medium | Format, consistency, symbolic constraints, public code tests, or other checks available at selection time |

Gold-answer exact matching is for **evaluation and oracle-label construction only**. It should not be used as a normal scoring expert for math or logic, otherwise the experiment leaks the benchmark answer into the selector. Execution-based code verification is only a normal expert when the tests are genuinely available at selection time.
| MC-style/recoverability expert | Future solvability signal | Optional | Can use an existing MC-style PRM if runnable; avoid expensive rollouts in the first version |

Important calibration issue:

Different experts' scores are not directly comparable. Use at least one of:

- rank normalization within each problem's N candidates;
- z-score normalization within each problem;
- temperature scaling on a validation set;
- learned linear calibration before routing.

The safest first version is **rank normalization**, because it is robust when expert score scales differ.

## 3. Data Scope

Keep the first version to 2-3 domains. The goal is not to prove universal routing, but to show that expert specialization exists and a gate can exploit it.

Recommended domain set:

| Domain | Candidate dataset | Final-answer check | Priority |
|---|---|---|---:|
| Math reasoning | MATH500, GSM8K, AIME-style subset | exact answer / boxed answer parsing | High |
| Logical reasoning | BBH logical deduction, LogiQA/ReClor-style multiple choice | option match | High |
| Code reasoning | HumanEval/MBPP subset | unit tests | Optional |

Minimum viable scale:

```text
Math: 100-200 problems x 8 candidates
Logic: 100-200 problems x 8 candidates
Optional code: 50-100 problems x 5-8 candidates
```

Stronger version:

```text
Total: 500-800 problems
N: 8 or 16 candidates per problem
Experts: 3-4
```

Use `N=8` first. If the pipeline works, add `N=16` for the final table.

## 4. Gate Design

### 4.1 No-Training Gate Baselines

These are easy and useful:

- **Uniform ensemble**: all experts have equal weights.
- **Domain rule gate**: math problems go to math PRM, logic to logic/general judge, code to verifier.
- **LLM gate**: prompt an LLM to assign expert weights based on problem type and reasoning style.

LLM gate prompt idea:

```text
Given the problem and a list of reward model experts, assign weights to experts.
Experts:
1. Math PRM
2. Reflective reasoning PRM
3. General LLM judge
4. Non-leaking verifier

Return JSON weights that sum to 1.
```

### 4.2 Trained Gate

Recommended model:

- input embedding from `bge`, `e5`, `sentence-transformers`, or a small local encoder;
- classifier or MLP producing expert weights;
- start with question-only features.

Training label construction:

For each problem `q`:

1. Each expert scores all N candidates.
2. Each expert selects its top candidate.
3. Check whether that candidate's final answer is correct.
4. Experts that select a correct candidate are treated as positive experts.
5. If multiple experts succeed, use a soft multi-hot target.
6. If no expert succeeds, skip the sample for routing training or use uniform target.

Target example:

```text
Math PRM selected correct answer: yes
Reflective PRM selected correct answer: no
General judge selected correct answer: yes
Non-leaking verifier selected correct answer: no

target gate distribution = [0.5, 0.0, 0.5, 0.0]
```

Loss:

```text
L_gate = cross_entropy(target_distribution, predicted_weights)
```

Alternative training objective:

Train weights to maximize final candidate correctness directly using validation PRM@N, but this is harder and less stable. Use it only as an extension.

## 5. Evaluation

Primary metric:

- **BoN / PRM@N accuracy**: for each problem, select one candidate using the method; check final answer correctness.

Secondary metrics:

- per-domain PRM@N;
- routing accuracy against oracle expert labels;
- regret versus oracle gate;
- cost: average number of experts activated;
- calibration/error analysis;
- win/loss breakdown over best single expert.

Main comparison table:

| Method | Math PRM@8 | Logic PRM@8 | Code PRM@8 | Overall PRM@8 | Avg experts |
|---|---:|---:|---:|---:|---:|
| Best single expert | | | | | 1 |
| Uniform ensemble | | | | | all |
| Domain rule gate | | | | | 1 |
| LLM gate | | | | | 1-2 |
| Trained gate | | | | | 1-2 |
| Oracle gate | | | | | 1 |

Ablations:

| Ablation | Purpose |
|---|---|
| no score normalization | show calibration is necessary |
| rank norm vs z-score norm | compare calibration choices |
| top-1 vs top-2 routing | accuracy-cost tradeoff |
| question-only vs question+candidate gate | test whether candidate content helps |
| remove each expert | show expert complementarity |

Diagnostic analysis:

- where trained gate beats the best single expert;
- where uniform ensemble fails because a weak expert pollutes the score;
- examples where math PRM misroutes logic or reflective cases;
- cases where the gate is just learning domain labels versus finer reasoning style.

## 6. Minimal Implementation Plan

Suggested modules:

```text
data/
  raw/
  candidates/
  scored/
  splits/

src/
  datasets/
    load_math.py
    load_logic.py
    load_code.py
  generation/
    generate_candidates.py
  scoring/
    score_math_prm.py
    score_general_judge.py
    score_nonleaking_verifier.py
    normalize_scores.py
  routing/
    build_oracle_labels.py
    train_gate.py
    llm_gate.py
  evaluation/
    evaluate_bon.py
    analyze_routing.py
  utils/
    answer_checking.py

experiments/
  configs/
  results/
  tables/
```

Data record schema:

```json
{
  "problem_id": "math_0001",
  "domain": "math",
  "problem": "...",
  "answer": "...",
  "candidates": [
    {
      "candidate_id": "math_0001_c00",
      "solution": "...",
      "steps": ["...", "..."],
      "final_answer": "...",
      "is_correct": true,
      "expert_scores": {
        "math_prm": 0.82,
        "reflective_prm": 0.76,
        "general_judge": 0.69,
        "nonleaking_verifier": 0.74
      },
      "normalized_scores": {
        "math_prm": 0.91,
        "reflective_prm": 0.83,
        "general_judge": 0.75,
        "nonleaking_verifier": 0.74
      }
    }
  ]
}
```

## 7. 15-Day Schedule

### Day 1: Scope Lock and Repro Setup

Goals:

- finalize domains: math + logic, code optional;
- list candidate datasets and experts;
- create repo structure and experiment config template;
- check which PRM models can actually run locally or through available APIs.

Deliverables:

- dataset/model availability table;
- final scope decision;
- initial README for experiment pipeline.

### Day 2: Related Work and Baseline Definition

Goals:

- summarize adjacent work: reward routing, MoE reward models, PRM benchmarks;
- define exact baselines and metrics;
- decide what claims the project will and will not make.

Deliverables:

- `related_work_notes.md`;
- baseline checklist;
- risk table.

### Day 3: Data Loader and Answer Checking

Goals:

- implement math and logic dataset loaders;
- implement final answer extraction and correctness checking;
- manually inspect 20 examples per domain.

Deliverables:

- normalized problem files;
- answer checker unit tests or manual QA sheet.

### Day 4: Candidate Solution Collection

Goals:

- generate or load N candidate solutions per problem;
- start with `N=8`;
- store candidate solutions in a fixed JSONL schema.

Deliverables:

- first candidate solution set;
- basic statistics: average length, correct rate, domain distribution.

### Day 5: Expert 1-2 Integration

Goals:

- integrate the easiest two experts first;
- likely choices: math PRM + general judge;
- run scoring on a 30-problem smoke test.

Deliverables:

- scored smoke-test dataset;
- first PRM@8 numbers for individual experts.

### Day 6: Expert 3-4 Integration

Goals:

- add reflective/BFE-style judge and optional MC-style expert;
- define step splitting if needed;
- run all experts on the small subset.

Deliverables:

- full expert score schema;
- expert runtime and cost estimate.

### Day 7: Score Normalization and Single-Expert Baselines

Goals:

- implement rank normalization and z-score normalization;
- compute single-expert PRM@8;
- compute uniform ensemble baseline.

Deliverables:

- first baseline table;
- calibration sanity plots or tables.

### Day 8: Oracle Gate and Routing Labels

Goals:

- build oracle expert labels from final-answer correctness;
- compute oracle gate upper bound;
- inspect cases where experts disagree.

Deliverables:

- gate training dataset;
- oracle gate result;
- 10 qualitative disagreement examples.

### Day 9: LLM Gate Baseline

Goals:

- implement prompt-based LLM gate;
- compare JSON weights versus domain-rule routing;
- test top-1 and top-2 routing.

Deliverables:

- LLM gate result table;
- prompt and parsing script;
- failure examples.

### Day 10: Trained Gate

Goals:

- train question-only lightweight gate;
- tune on validation split;
- evaluate on held-out split.

Deliverables:

- trained gate checkpoint;
- main result table v1.

### Day 11: Main Experiment Run

Goals:

- run full selected dataset;
- compare all methods under the same candidate set;
- produce `PRM@8`, and `PRM@16` if possible.

Deliverables:

- main result table;
- per-domain result table;
- saved experiment logs.

### Day 12: Ablations

Goals:

- run no-normalization ablation;
- run top-1 vs top-2 expert activation;
- run remove-one-expert ablation.

Deliverables:

- ablation table;
- cost-performance table.

### Day 13: Robustness and Generalization

Goals:

- test on harder subset or different generator if available;
- check whether routing still helps when candidate quality distribution changes;
- optional: include reflective math subset inspired by Beyond the First Error.

Deliverables:

- robustness table;
- 5-10 representative success/failure cases.

### Day 14: Error Analysis and Figure Preparation

Goals:

- analyze failure modes;
- create routing confusion matrix;
- create expert complementarity plot;
- prepare clean tables for presentation.

Deliverables:

- figures and tables;
- error analysis notes;
- final experiment summary.

### Day 15: Freeze and Handoff Package

Goals:

- freeze code, config, and data splits;
- rerun key commands for reproducibility;
- write concise technical summary for later paper writing.

Deliverables:

- reproducible experiment package;
- final result tables;
- project report outline;
- list of claims supported by evidence.

## 8. Resource Estimate

Minimum version:

- 200-300 problems total;
- `N=8` candidates per problem;
- 2-3 experts;
- no PRM training, only gate training;
- can run in several days with API scoring or one local GPU.

Recommended version:

- 400-800 problems total;
- `N=8` first, `N=16` for final run if feasible;
- 3-4 experts;
- one lightweight trained gate;
- expected 1-2 days for data generation/scoring, depending on model/API speed.

Compute-sensitive part:

```text
num_problems x num_candidates x num_experts x scoring_cost
```

For example:

```text
500 problems x 8 candidates x 4 experts = 16,000 candidate-level scores
```

If experts score every step instead of every solution:

```text
500 problems x 8 candidates x 10 steps x 4 experts = 160,000 step-level scores
```

Therefore:

- candidate-level scoring should be the default first pass;
- step-level PRM scoring should be used for math/reflective subset or final stronger version.

## 9. Main Risks and Backups

| Risk | Why it matters | Backup |
|---|---|---|
| PRM models are hard to run locally | 7B models can be slow and large | use fewer experts, quantization, or LLM-as-judge experts |
| Experts are all math-heavy | cross-domain claim becomes weak | frame as reasoning-style routing first; add logic LLM judge |
| Gate only learns domain labels | contribution becomes shallow | add mixed/hard subsets and compare against domain-rule gate |
| Scores are not calibrated | ensemble may underperform | use rank normalization and validation calibration |
| Answer checking is noisy | PRM@N becomes unreliable | manually inspect sampled predictions; use multiple-choice or exact-answer tasks |
| No improvement over best expert | possible in small datasets | report oracle gap and expert complementarity; adjust expert pool and top-2 routing |

## 10. Recommended Final Scope

For a 15-day course project, the strongest manageable version is:

```text
MoPRM with:
- 2 domains: math + logic
- 3 experts: math PRM, reflective/general judge, and a non-leaking verifier if available
- question-level trained gate
- LLM gate baseline
- rank-normalized score fusion
- PRM@8 main metric
- oracle gate upper bound
```

Optional upgrades if early results look good:

- add code reasoning;
- add `N=16`;
- add candidate-aware gate;
- add a reflective math subset;
- add MC-style recoverability expert.

## 11. Paper/Presentation Claim Template

Safe claim:

> We propose MoPRM, a lightweight router-guided framework that combines multiple process reward experts for multi-domain reasoning verification. Instead of assuming one PRM is universally reliable, MoPRM learns when to trust each expert and improves Best-of-N selection under controlled compute.

Stronger claim, only if results support it:

> The gain is not merely from ensembling: the trained gate outperforms uniform averaging and domain-rule routing, suggesting that PRM specialization depends on finer reasoning patterns beyond coarse task domains.

Claim to avoid unless fully proven:

> MoPRM is the first mixture-of-PRMs method.

Existing reward routing and MoE reward-modeling work means we should position the contribution as process-level, multi-domain, and experimentally lightweight rather than claiming firstness.

## 12. References to Track

- Beyond the First Error: Process Reward Models for Reflective Mathematical Reasoning.
- Math-Shepherd: Verify and Reinforce LLMs Step-by-Step without Human Annotations.
- ProcessBench: Identifying Process Errors in Mathematical Reasoning.
- PRMBench: A Fine-Grained Process Reward Model Benchmark.
- RewardBench / RewardBench-style RM evaluation, if using general reward models.
- RouteLLM: Learning to Route LLMs with Preference Data.
- LASeR: reward model selection/routing with multi-armed bandits.
- DMoERM: Mixture-of-Experts Reward Modeling.
