# Open-Source PRM Selection

## Local Hardware Snapshot

Checked on 2026-09-01:

```text
GPU: NVIDIA GeForce RTX 5070 Laptop
VRAM: 8,151 MiB
System RAM: about 31.4 GiB
CPU threads: 24
Python: 3.10.10
```

The original local `.venv` had CPU PyTorch. A CUDA environment was later prepared
at:

```text
.venv_cuda
```

Current CUDA environment:

```text
torch: 2.7.1+cu128
CUDA available: yes
GPU: NVIDIA GeForce RTX 5070 Laptop GPU
```

## First Math PRM Choice

Selected first target:

```text
Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B
```

Reasons:

- It is a real non-OpenAI PRM/reward model, not another OpenAI rubric.
- It is much smaller than 7B PRMs and is the best fit for an 8GB laptop GPU.
- The model file is about 3.09 GB, which is acceptable for local experiments.
- The official model card provides a step-reward inference path.
- It is trained from Qwen2.5-Math-1.5B-Instruct, so it is naturally aligned with
  the project's math domain.

Alternatives:

| Model | Fit for this laptop | Notes |
|---|---|---|
| `Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B` | Best first choice | Smallest practical open PRM target. |
| `Qwen/Qwen2.5-Math-PRM-7B` | Later / risky | Stronger math PRM, but 7B is tight on 8GB VRAM. |
| `Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B` | Later / risky | Same family, larger and likely needs quantization/offload. |
| `RLHFlow/Llama3.1-8B-PRM-Deepseek-Data` | Later / risky | 8B model; useful comparison but not first integration. |

## Integration Plan

Adapter added:

```text
src/moprm/scoring/skywork_math_prm.py
scripts/score_skywork_math_prm.py
scripts/download_skywork_weight.py
```

The adapter writes scores as:

```text
expert_scores.open_math_prm
```

The score is the aggregate of per-step PRM rewards. The first aggregation method
is mean step reward, matching the common PRM Best-of-N setup.

## Immediate Experiment

Start with the math subset only:

```bash
python scripts/score_skywork_math_prm.py \
  --input data/cache/openai_pilot10_n4_labeled.jsonl \
  --output data/scored/skywork_math_prm_pilot.jsonl \
  --domains math \
  --limit 10 \
  --device auto \
  --overwrite
```

Use `--dtype float32` for the first CUDA runs. On this laptop, bf16 CUDA scoring
produced NaN step rewards for some candidates, while CPU float32 and CUDA
float32 were stable.
