# MoPRM 完整实验分析

日期：2026-09-05

依据范围：`README.md`、`notes/` 下当前实验报告、`data/` 下本地 JSONL 实验产物，以及当前评测脚本复算结果。`local_archive/legacy_paper_ppt_materials/` 中的 Beyond the First Error 材料主要作为背景动机，不作为本项目 MoPRM 的实测结果合并进主表。

## 1. 总体结论

当前项目已经完成一个可复现的异构 MoPRM 实验闭环：

```text
公开 math/logic 数据
-> OpenAI 生成 N 个候选解
-> evaluation-only final-answer 标签
-> 四专家打分
-> rank/minmax/zscore 归一化与路由
-> Gate-v1 / Candidate Gate-v2 交叉验证
-> clean-label 与专家移除消融
```

最稳妥的主结论是：

> Candidate-aware Gate-v2 能利用异构专家的候选级 score/rank/margin 模式，在当前 mixed-rich scout split 上优于 best single、uniform ensemble、domain-rule gate、OpenAI LLM gate 和 CV-static calibration。经过保守答案标签清理后，任务接近天花板，V2 的优势缩小，但仍保持在 practical baselines 之上或持平；在只保留两个非 OpenAI 开源专家的消融中，V2 仍能超过最强单专家并达到 two-expert top-choice oracle。

最重要的限定是：

```text
1. 主正结果仍来自一个 84 条 originally mixed 的 scout subset；
2. clean-label 后真正 mixed 的样本缩到 67 条；
3. 候选答案仍由 OpenAI 生成；
4. 自动 answer checking 是核心风险源，已发现并修复一批格式性误判；
5. 当前不应声称大规模泛化，也不应声称完整 pipeline 已经 OpenAI-free。
```

## 2. 数据与实验资产

当前本地公共数据缓存已经扩展到：

| 文件 | 记录数 | 领域 |
|---|---:|---|
| `data/cache/public_subsets/math500.jsonl` | 500 | math |
| `data/cache/public_subsets/gsm8k.jsonl` | 40 | math |
| `data/cache/public_subsets/bbh_logic.jsonl` | 300 | logic |
| `data/cache/public_subsets/math_logic_combined.jsonl` | 840 | math 540, logic 300 |

主要候选集与标签分布：

| 实验集 | problems | candidates | correct candidates | mixed | all-correct | all-wrong | 作用 |
|---|---:|---:|---:|---:|---:|---:|---|
| `pilot_10, N=4` | 10 | 40 | 32 | 3 | 7 | 0 | 验证 OpenAI 生成、打分、路由闭环 |
| `dev_40, N=4` | 40 | 160 | 129 | 4 | 30 | 6 | 验证首个异构池，但 oracle 空间太小 |
| `hard_dev_100, N=8` | 100 | 800 | 534 | 18 | 59 | 23 | 更难主运行，仍只有小 oracle gap |
| `hard_mix_scout_320, N=8` | 320 | 2560 | 1379 | 84 | 126 | 110 | mixed-rich scout，为训练 gate 提供数据 |
| `scout320 originally mixed, clean labels` | 84 | 672 | 475 | 67 | 17 | 0 | clean-label 后的主分析对象 |

API 使用量中，报告中明确记录的核心量级为：

| 阶段 | candidate generation | OpenAI expert scoring | LLM gate routing |
|---|---:|---:|---:|
| `pilot_10_n4` | 16,960 tokens | 25,994 tokens | 3,636 tokens |
| `dev_40_n4` | 67,805 tokens | 98,058 tokens | about 15,232 tokens |
| `hard_dev_100_n8` | 408,180 tokens | 603,493 tokens | 38,368 tokens |
| `hard_mix_scout_320_n8` | 1,459,205 tokens, full generation | 618,571 tokens, mixed subset scoring | 32,194 tokens, mixed subset gate |

两个 Skywork 开源专家均使用本地 `models/hf_cache`，不再消耗 API token。

## 3. 专家池与方法设置

主异构专家池：

| Expert | 来源 | 类型 | 当前角色 |
|---|---|---|---|
| `open_math_prm` | `Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B` | step-level math PRM | 数学过程正确性信号 |
| `open_reasoning_rm` | `Skywork/Skywork-Reward-V2-Qwen3-1.7B` | response-level reward logit | 逻辑/通用推理质量信号 |
| `openai_general_judge` | OpenAI Responses API | LLM judge | broad reliability baseline/supporting expert |
| `openai_reflective_judge` | OpenAI Responses API | LLM judge | self-checking / error-recovery rubric |

重要工程判断：

```text
open_math_prm:
  CUDA bf16 出现 NaN step rewards；
  CPU float32 和 CUDA float32 稳定；
  因此主运行使用 float32。

open_reasoning_rm:
  CUDA auto resolved to bfloat16；
  输出 raw sequence reward logit；
  通过 per-problem rank normalization 处理跨专家尺度差异。
```

主评测指标：

```text
PRM@N / BoN selection accuracy:
每道题从 N 个候选解中选一个，检查最终答案是否正确。
```

主要 baseline：

```text
best single expert
uniform ensemble
domain-rule gate
OpenAI LLM question-level gate
CV-static calibration
Gate-v1 question-level trained router
Candidate Gate-v2 candidate-aware selector
expert top-choice oracle
```

## 4. 实验阶段分析

### 4.1 Pilot：OpenAI-only 与首个开放 PRM 接入

`pilot_10_n4` 上所有 OpenAI-only 方法均为 `10/10`，这只证明 pipeline 能跑通，不证明 MoPRM 有收益。

加入 `open_math_prm` 后，`pilot_10_n4_hetero_math_pool` 变成：

```text
domain_rule_gate / LLM gate / uniform / open_math_prm: 8 / 10
oracle_gate:                                      10 / 10
OpenAI general/reflective judges:                 10 / 10
```

解释：首个非 OpenAI math PRM 已经可用，但直接按 domain 信任它会伤害结果。这很早就说明，项目不能停在“专家越专越好”的简单故事，而需要 normalization、calibration 和 trained gate。

### 4.2 `dev_40, N=4`：异构池成立，但评测空间不足

四专家池复算结果：

| Method | Overall | Math | Logic |
|---|---:|---:|---:|
| domain-rule gate | 33/40 | 13/20 | 20/20 |
| OpenAI LLM gate | 33/40 | 13/20 | 20/20 |
| uniform ensemble | 32/40 | 12/20 | 20/20 |
| best single | 33/40 | 13/20 | 20/20 |
| oracle gate | 34/40 | 14/20 | 20/20 |

候选分布为 `30/40 all-correct`、`6/40 all-wrong`、只有 `4/40 mixed`。也就是说，在大多数题上没有真正的 selection 问题：要么谁选都对，要么谁也救不了。这个实验适合验证工程链路，不适合训练或证明路由收益。

### 4.3 `hard_dev_100, N=8`：更难，但 routing headroom 仍小

mean aggregation 结果：

| Method | Accuracy |
|---|---:|
| best single `open_reasoning_rm` | 71/100 |
| uniform ensemble | 70/100 |
| OpenAI LLM gate | 69/100 |
| domain-rule gate | 67/100 |
| oracle gate | 71/100 |

将 `open_math_prm` 从 mean 改为 min aggregation 后：

| Method | Accuracy |
|---|---:|
| domain-rule gate | 71/100 |
| OpenAI LLM gate | 70/100 |
| uniform ensemble | 71/100 |
| best single `open_reasoning_rm` | 71/100 |
| oracle gate | 72/100 |

mixed-only 子集只有 `18/100`。在 `math_min` 下：

```text
domain_rule_gate: 12 / 18
OpenAI LLM gate:  11 / 18
uniform ensemble: 12 / 18
best single:      12 / 18
oracle_gate:      13 / 18
```

解释：这个 split 比 `dev_40` 更真实，但仍没有足够路由空间。瓶颈不是 gate 还不够强，而是候选集合和专家分歧没有制造出足够多可学习的 mixed cases。

### 4.4 `hard_mix_scout_320, N=8`：第一个真正 selection-informative 的 split

完整 scout：

```text
problems: 320
candidates: 2560
correct candidates: 1379 / 2560 = 0.539
all-wrong: 110 / 320
all-correct: 126 / 320
mixed: 84 / 320
mixed sources: 61 MATH500 + 23 BBH seven-object
```

pre-clean-label 四专家 mixed-only 结果：

| Method | Accuracy |
|---|---:|
| Candidate Gate-v2 | 72/84 = 0.857 |
| best CV-static calibration | 69/84 = 0.821 |
| best single `open_reasoning_rm` | 67/84 = 0.798 |
| uniform ensemble | 64/84 = 0.762 |
| OpenAI LLM gate | 60/84 = 0.714 |
| domain-rule gate | 56/84 = 0.667 |
| expert oracle | 76/84 = 0.905 |

按领域看：

```text
math mixed:
  Candidate Gate-v2: 50 / 61
  best single:       44 / 61
  uniform:           41 / 61
  oracle:            53 / 61

logic mixed:
  Candidate Gate-v2: 22 / 23
  open_reasoning_rm: 23 / 23
  oracle:            23 / 23
```

这说明 V2 的主要贡献不是粗粒度 math/logic routing，而是 MATH500 上的候选级 pattern learning。逻辑子集基本由 `open_reasoning_rm` 解决。

## 5. Gate 实验分析

### 5.1 Gate-v1：问题级 trained router

Gate-v1 使用 question text hash + 轻量元数据/文本统计，训练 multi-label logistic regression；目标是预测每个专家 top choice 是否正确。

最佳设置：

```text
open_math_prm aggregation: mean
normalization: rank
weight_power: 4
5-fold source/domain-stratified CV
```

结果：

| Method | Accuracy |
|---|---:|
| Gate-v1 CV | 67/84 = 0.798 |
| best single `open_reasoning_rm` | 67/84 = 0.798 |
| uniform ensemble | 64/84 = 0.762 |
| OpenAI LLM gate | 60/84 = 0.714 |
| domain-rule gate | 56/84 = 0.667 |
| expert oracle | 76/84 = 0.905 |

但是在 `open_math_prm min + rank` 下，CV-static calibration 达到 `69/84`，Gate-v1 不能超过它。因此 Gate-v1 应报告为“训练型 question-level baseline”，不是主方法。

### 5.2 Candidate Gate-v2：当前主方法

Candidate Gate-v2 是 candidate-level logistic selector，使用：

```text
candidate text statistics
per-expert raw scores
per-expert rank-normalized scores
expert top-choice indicators
score margins / gaps
open-source vs OpenAI score aggregates
```

它不看 gold answer；训练标签只在训练 folds 中使用 candidate final-answer correctness；评估为 problem-level 5-fold out-of-fold。

pre-clean 最佳：

```text
open_math_prm aggregation: mean
normalization: rank
l2: 0.02

Candidate Gate-v2:     72 / 84
best single:           67 / 84
best CV-static:        69 / 84
expert oracle:         76 / 84
```

aggregation sweep：

| open_math_prm aggregation | Candidate Gate-v2 |
|---|---:|
| mean | 72/84 |
| last | 71/84 |
| geomean | 71/84 |
| min | 69/84 |

normalization sweep：

| normalization | Candidate Gate-v2 |
|---|---:|
| rank | 72/84 |
| minmax | 68/84 |
| zscore | 68/84 |

regularization sanity：

| l2 | Candidate Gate-v2 |
|---:|---:|
| 0.001 | 69/84 |
| 0.005 | 71/84 |
| 0.020 | 72/84 |
| 0.050 | 70/84 |

结论：Candidate-level 信息确实比 question-level router 更有效；rank normalization 对当前异构专家尺度最稳。

## 6. Label Cleanup 后的重新解释

loss-case audit 发现：pre-clean Candidate Gate-v2 的 12 个 wrong selections 中，11 个是明显答案格式误判：

```text
LaTeX inline/display wrappers: \(...\), \[...\]
simple numeric unit suffix: 36 seconds vs 36
```

清理前后标签变化：

| 指标 | Before | After |
|---|---:|---:|
| correct candidate labels | 371/672 | 475/672 |
| mixed records | 84 | 67 |
| all-correct records | 0 | 17 |
| all-wrong records | 0 | 0 |
| changed records | - | 23/84 |

旧 V2 selections 在 clean labels 下：

```text
Candidate Gate-v2: 83 / 84
best single:       81 / 84
uniform:           80 / 84
OpenAI LLM gate:   73 / 84
domain-rule gate:  68 / 84
expert oracle:     82 / 84
```

重新训练 clean-label V2：

| Evaluation view | Candidate Gate-v2 | Best single | CV-static | Uniform | Oracle |
|---|---:|---:|---:|---:|---:|
| clean mixed 67 | 64/67 | 64/67 | 64/67 | 63/67 | 65/67 |
| all original 84 | 82/84 | 81/84 | 81/84 | 80/84 | 82/84 |

解释：

```text
1. pre-clean 的 72/84 被 answer-checking noise 严重低估；
2. clean-label 后任务接近天花板，优势自然缩小；
3. clean mixed 67 上 V2 与 best single/CV-static 持平；
4. all-84 clean-label view 上 V2 仍有 +1 problem 的净收益并达到 expert oracle；
5. 当前没有足够证据说明需要立即训练 Gate-v3。
```

## 7. 专家行为与互补性

### 7.1 `open_reasoning_rm` 是最强单专家

clean-label all-84 四专家池：

```text
open_reasoning_rm: 81 / 84
logic:             23 / 23
math:              58 / 61
```

它不仅适合 logic，在 MATH500 上也很强。这使得很多 baseline 已经接近天花板，是 V2 后续提升空间变小的核心原因之一。

### 7.2 `open_math_prm` 单独较弱，但提供少量关键互补信号

clean-label all-84：

```text
open_math_prm: 58 / 84
math:          45 / 61
logic:         13 / 23
```

两开源专家消融中的 top-choice pattern：

```text
open_math_prm, open_reasoning_rm

11: 56
01: 25
10:  2
00:  1
```

解释：`open_math_prm` 不是好的 standalone selector，但有两个只有它能 rescue 的 math cases。Candidate Gate-v2 的价值，正是从强 RM 主导的局面中识别这些少数切换点。

### 7.3 OpenAI judges 在 clean-label 后变强

pre-clean no-RM ablation 中：

```text
openai_general_judge:    63 / 84
openai_reflective_judge: 66 / 84
```

clean-label no-RM ablation 中：

```text
openai_general_judge:    79 / 84
openai_reflective_judge: 79 / 84
```

这说明旧标签把许多实际正确的数学答案误判为错，从而低估了 judge-style experts。旧 no-RM 的大幅 V2 增益应保留为历史诊断，不能作为最终 headline。

## 8. 消融实验结论

### 8.1 两开源专家 only：去掉所有 OpenAI judge experts

保留：

```text
open_math_prm
open_reasoning_rm
```

去掉：

```text
openai_general_judge
openai_reflective_judge
```

clean-label 结果：

| Evaluation view | Candidate Gate-v2 | Best single | CV-static | Uniform | Expert oracle |
|---|---:|---:|---:|---:|---:|
| clean mixed 67 | 65/67 | 64/67 | 64/67 | 58/67 | 66/67 |
| all original 84 | 83/84 | 81/84 | 81/84 | 75/84 | 83/84 |

这是当前最有力的鲁棒性消融：即使移除两个 OpenAI judge-style experts，V2 仍超过最强开源单专家，并在 all-84 clean-label evaluation 上达到 two-expert oracle。

但边界仍然存在：

```text
候选答案仍由 OpenAI 生成；
clean-label 后样本小且接近 ceiling；
因此这是 robustness ablation，不是 OpenAI-free pipeline claim。
```

### 8.2 No-RM：去掉最强 `open_reasoning_rm`

保留：

```text
open_math_prm
openai_general_judge
openai_reflective_judge
```

clean-label 结果：

| Evaluation view | Candidate Gate-v2 | Best single | CV-static | Uniform | Expert oracle |
|---|---:|---:|---:|---:|---:|
| clean mixed 67, best aggregation = last | 63/67 | 62/67 | 61/67 | 62/67 | 66/67 |
| all original 84, best aggregation = mean/geomean | 81/84 | 79/84 | 77/84 | 75/84 | 81/84 |

解释：去掉最强 RM 后系统没有崩；V2 仍有小幅正收益。但增益远小于 pre-clean 的 `74/84 vs 66/84`，说明旧 no-RM 结果被 label noise 放大。

## 9. 错误模式

pre-clean V2 loss audit：

```text
reported wrong selections: 12
math/MATH500: 11
logic/BBH seven-object: 1
11 / 12 在 loose diagnostic 下与 gold answer 等价
```

唯一明确语义错误：

```text
bbh_logical_deduction_seven_objects_0077
gold: C
V2 selected: candidate 006, wrong

expert top choices:
open_math_prm             -> candidate 000, correct
open_reasoning_rm         -> candidate 001, correct
openai_general_judge      -> candidate 006, wrong
openai_reflective_judge   -> candidate 006, wrong
```

这说明 V2 会在某些逻辑题中被两个 OpenAI judge-style experts 的一致错误信号带偏。

clean-label retrained all-84 V2 的 wrong cases：

```text
math500_0030:
  gold: 52_8
  selected final answer: 52
  边界情况。解题过程可能表达了 base-8，但 final answer line 丢了 base notation。

math500_0473:
  gold: 7
  selected final answer: :
  真正的 final answer extraction/completion failure。
```

two-open-PRM all-84 V2 的唯一 failure：

```text
math500_0018
gold: 28
selected final answer: 152

correct candidate:
  open_math_prm:     0.2281
  open_reasoning_rm: 5.21875

selected wrong candidate:
  open_math_prm:     0.2886
  open_reasoning_rm: 6.1875
```

这类问题不是 gate 容易修复的，因为两个保留专家都把 wrong candidate 排得更高。

## 10. 与 Beyond the First Error 背景的关系

旧归档材料说明了项目早期动机来自 Beyond the First Error：

```text
传统 PRM 的 first-error 标注不适合 long-CoT / reflective reasoning；
Error Propagation 和 Error Cessation 说明不同推理阶段需要不同过程信号；
PRM@N 与 step-level F1 是重要评价方式；
MC-style recoverability 与 local correctness 不是同一类信号。
```

但当前 MoPRM 项目已经转向“异构专家混合与路由”：

```text
旧论文是背景动机；
当前贡献不是训练新的 reflective PRM；
当前实验证据来自多个独立专家的 candidate selection 与 routing。
```

因此最终报告可以把 BFE 放在 related work / motivation 中，用来解释为什么单一 PRM 不一定适合所有 reasoning trajectory；但主结果应报告 MoPRM 的专家池、gate 和 ablation。

## 11. 支持的 claim 与不支持的 claim

当前支持的 claim：

```text
1. 项目实现了完整可复现的 MoPRM pipeline。
2. 主专家池满足异构性要求：两个非 OpenAI 开源专家 + 两个 OpenAI judge baselines/supporting experts。
3. 在 pre-clean mixed-rich scout 上，Candidate Gate-v2 明显优于强 baselines。
4. 标签清理显示旧错误大多来自 answer normalization artifacts，而不是 V2 选择失败。
5. 在 clean-label all-84 视角下，V2 仍略优于 best single/CV-static，并达到 expert oracle。
6. 两开源专家 only 的 clean-label 消融中，V2 仍超过最强开源单专家并达到 two-expert oracle。
7. V2 的收益主要来自 MATH500 mixed cases，而不是简单学习 math/logic domain 标签。
```

应避免或谨慎表达的 claim：

```text
1. 不应说四个 OpenAI rubric scores 本身构成最终 MoPRM。
2. 不应说 pipeline 完全不依赖 OpenAI，因为候选生成仍来自 OpenAI。
3. 不应把 pre-clean 72/84 当作最终能力上限或最终错误率。
4. 不应声称 Gate-v3 必要；当前证据更支持先扩 clean-label 数据。
5. 不应声称大规模泛化，当前主样本是 84 originally mixed / 67 clean mixed。
```

## 12. 最终报告建议

推荐主表顺序：

1. `dev_40_n4`：pipeline validation，强调 oracle gap 太小。
2. `hard_dev_100_n8`：更难评估，但 mixed 只有 18，说明需要 mixed-rich scout。
3. `hard_mix_scout_320_n8` pre-clean：Candidate Gate-v2 主效果。
4. clean-label reevaluation：说明 pre-clean 错误大部分是 answer checking noise。
5. two-open-PRM clean-label ablation：最重要鲁棒性消融。
6. no-RM clean-label ablation：说明去掉最强 RM 不会崩，但增益较小。

推荐 headline 数字：

```text
Pre-clean mixed-rich scout:
Candidate Gate-v2: 72 / 84
best single:       67 / 84
CV-static:         69 / 84
uniform:           64 / 84
LLM gate:          60 / 84
domain-rule:       56 / 84
oracle:            76 / 84

Clean-label four-expert all-84:
Candidate Gate-v2: 82 / 84
best single:       81 / 84
CV-static:         81 / 84
uniform:           80 / 84
oracle:            82 / 84

Clean-label four-expert mixed-67:
Candidate Gate-v2: 64 / 67
best single:       64 / 67
CV-static:         64 / 67
oracle:            65 / 67

Two-open-PRM clean-label all-84:
Candidate Gate-v2: 83 / 84
best open single:  81 / 84
CV-static:         81 / 84
oracle:            83 / 84
```

建议最终论述口径：

> The strongest evidence for MoPRM is not that every learned gate always beats every baseline after label cleanup. Rather, the evidence chain is that candidate-level expert-score patterns are useful on genuinely mixed candidate sets, label hygiene is essential for fair PRM@N evaluation, and the learned selector continues to extract the small remaining complementarity even when OpenAI judge experts are removed.

## 13. 下一步优先级

最高优先级：

```text
1. 构造 fresh/larger clean-label mixed split，例如 hard_mix_scout_480_n8 或 640_n8。
2. 在新 split 上先跑 answer checker QA，再跑专家和 V2，不要先上 Gate-v3。
3. 增加非 OpenAI candidate generation 条件，至少做一组 robustness split。
4. 对 MATH500 final-answer checker 做更系统的 manual audit，尤其 base notation、坐标、LaTeX、单位、表达式等。
```

中优先级：

```text
1. 继续报告 aggregation sensitivity：mean 对 V2 最好，min 对 static calibration 更好。
2. 做 cost-performance 表：single expert、top-2 routing、all experts。
3. 分析 V2 feature importance / learned weights，确认它不是简单复制 `open_reasoning_rm`。
4. 对 bbh_logical_deduction_seven_objects_0077 做定性案例展示。
```

低优先级或 future work：

```text
1. Gate-v3：只有在更大 clean split 上仍有明显 oracle gap 时再做。
2. 如果做 Gate-v3，应使用 frozen problem embeddings 或 LLM-generated problem tags，而不是在 67/84 个样本上训练高容量文本编码器。
3. aggregation-as-pseudo-experts 可以作为 oracle/headroom 诊断，不适合当前主方法，因为训练集太小会不稳定。
```

## 14. 一句话总结

当前 MoPRM 实验已经能支持一个谨慎但有价值的结论：异构专家的候选级分数模式确实包含可利用的互补信息，Candidate Gate-v2 是目前最好的路由/选择方法；但真正让结果可信的不是单个 72/84 数字，而是从 mixed-rich split、clean-label audit、win/loss、两开源专家消融到 no-RM 消融形成的一条完整证据链。
