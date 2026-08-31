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

