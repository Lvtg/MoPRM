# Public Data Sources

Initial MoPRM experiments use public datasets where final answers can be checked automatically.

## Selected First-Stage Sources

| Domain | Source | Use |
|---|---|---|
| Math | `openai/gsm8k` | Easy math sanity check and answer-checker debugging |
| Math | `HuggingFaceH4/MATH-500` | Main math reasoning subset aligned with common PRM/BoN evaluation |
| Logic | `suzgunmirac/BIG-Bench-Hard` logical deduction tasks | Multi-choice logic reasoning with simple answer checking |

Source links:

- `openai/gsm8k`: https://huggingface.co/datasets/openai/gsm8k
- `HuggingFaceH4/MATH-500`: https://huggingface.co/datasets/HuggingFaceH4/MATH-500
- `BIG-Bench-Hard`: https://github.com/suzgunmirac/BIG-Bench-Hard

## Why These Sources

- They are public and widely used.
- The first versions are small enough to download quickly.
- Final-answer correctness is easy to evaluate automatically.
- They cover at least two domains, allowing us to test whether routing beats single experts and uniform ensembling.

## Download Policy

The scripts download prepared subsets into `data/cache/public_subsets/`. Those files are ignored by git. We commit only loaders, scripts, and metadata notes.

## Planned First Run

```text
MATH-500: 80 problems
GSM8K: 80 problems
BBH logical deduction: 60 examples each from 3/5/7-object tasks
Total: 340 problems before candidate generation
```

After the pipeline works, expand toward 500-800 problems and `N=8` candidates per problem.

## Current QA Notes

The first download produced:

```text
MATH-500: 80 records
GSM8K: 80 records
BBH logical deduction: 180 records
Combined: 340 records
Domains: math=160, logic=180
```

All records had non-empty problems and answers.

Answer-checking risk:

- BBH logical deduction should be reliable because labels are multiple-choice.
- GSM8K should be reliable after extracting the answer after `####`.
- MATH-500 includes LaTeX fractions, coordinates, text answers, and diagrams. Exact matching will need extra normalization and sampled manual QA before we trust large-scale `PRM@N` numbers.
