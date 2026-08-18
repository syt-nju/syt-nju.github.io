---
title: "On-Policy Distillation 问题地图"
topic: on-policy-distillation
summary: "OPD 的核心问题是：在学生自己访问的状态上，用多密、多可靠的老师信号做后训练，而不是继续模仿老师轨迹或只拿稀疏对错奖励。"
lang: zh-CN
updated: 2026-08-17
order: 1
sources:
  - title: "On-Policy Distillation"
    url: "https://thinkingmachines.ai/blog/on-policy-distillation/"
  - title: "Training LLMs using Off-Policy vs On-Policy Distillation"
    url: "https://saraswatmks.github.io/2026/07/on-policy-distillation-thinking-machines.html"
  - title: "Distillation (Tinker Cookbook)"
    url: "https://tinker-docs.thinkingmachines.ai/cookbook/recipes/distillation/"
  - title: "Rethinking On-Policy Distillation of Large Language Models"
    url: "https://arxiv.org/abs/2604.13016"
  - title: "Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes"
    url: "https://arxiv.org/abs/2603.25562"
  - title: "OPD深度解析：从数学推导到DeepSeek V4、SWIFT与verl实践"
    url: "https://zhuanlan.zhihu.com/p/2033212181823608430"
  - title: "SFT, RL, and On-Policy Distillation Through a Distributional Lens"
    url: "https://nrehiew.github.io/blog/sft_rl_opd/"
raw:
  - raw/on-policy-distillation/2025-10-27-on-policy-distillation.md
  - raw/on-policy-distillation/2026-07-on-policy-distillation-floating-bytes.md
  - raw/on-policy-distillation/tinker-cookbook-distillation.md
  - raw/on-policy-distillation/2026-04-14-rethinking-on-policy-distillation.md
  - raw/on-policy-distillation/2026-03-26-revisiting-on-policy-distillation.md
  - raw/on-policy-distillation/zhihu-opd-deep-dive-v2.md
  - raw/on-policy-distillation/2026-05-10-sft-rl-opd-distributional-lens.md
---

## Overview

On-Policy Distillation（OPD）把后训练拆成彼此独立的选择：训练状态从谁来、每个 prefix 上比较什么、以及老师分数何时可学。学生在自己的 rollout 上学习，老师在这些前缀上给 dense 反馈。这同时针对 SFT 的 [exposure bias](/wiki/on-policy-distillation/sft-rl-opd/#exposure-bias) 和 RL 的 [稀疏 credit](/wiki/on-policy-distillation/sft-rl-opd/#sparse-credit)。OPD 不是单一算法。

本 topic 覆盖白盒、可取老师 logprob 的设定，以及同模型、参考答案作特权前缀的 [OPSD](/wiki/on-policy-distillation/when-opd-works/#opsd)。黑盒老师尚未摄入。DeepSeek V4 报告本身未摄入，相关主张来自[知乎对该报告的阅读](https://zhuanlan.zhihu.com/p/2033212181823608430)和 [nrehiew 对工业管线的转述](https://nrehiew.github.io/blog/sft_rl_opd/)。

## 问题地图

### 训练状态该跟谁对齐

Off-policy 蒸馏训练的是老师常去的前缀；推理时学生走自己的错路，错误会沿序列放大。OPD 改成学生采样、老师打分，让训练分布贴近推理分布。这条比较构成后训练的 [2×2](/wiki/on-policy-distillation/sft-rl-opd/#sampling-density)：SFT 是 off-policy + dense，RL 是 on-policy + sparse，OPD 是 on-policy + dense。on-policy 还有一层几何后果：更新被约束在当前策略附近，抗遗忘靠的是数据来源，不是显式 KL 惩罚。原文：[Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/)、[nrehiew](https://nrehiew.github.io/blog/sft_rl_opd/)。

### 每个 prefix 上比较什么

Dense 仍有粒度。[sampled-token](/wiki/on-policy-distillation/teacher-signal-granularity/#sampled-token) 只评价学生抽出的那一个 token；[top-k](/wiki/on-policy-distillation/teacher-signal-granularity/#top-k) 比较老师支持的一小撮候选；[full-vocab](/wiki/on-policy-distillation/teacher-signal-granularity/#full-vocab) 比较整个词表。KL 还可以选 [forward / reverse / JSD](/wiki/on-policy-distillation/teacher-signal-granularity/#kl-direction)，loss 可以是直接反传或 policy-gradient。[Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/) 的默认实现只是这个格子里的一格。

### 老师信号何时可靠

监督密，不等于老师在学生前缀上的分数可学。老师与学生的 [thinking pattern](/wiki/on-policy-distillation/when-opd-works/#thinking-pattern) 要对上，老师还要带来学生训练中没见过的[新能力](/wiki/on-policy-distillation/when-opd-works/#new-capability)；否则更强老师也可能蒸不动。原文：[Rethinking OPD](https://arxiv.org/abs/2604.13016)。

## 共同主张

- 后训练的关键差在[状态来源和监督密度](/wiki/on-policy-distillation/sft-rl-opd/#sampling-density)，不在「像不像 RL 的代码路径」。原文：[Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/)。
- [on-policy 数据](/wiki/on-policy-distillation/sft-rl-opd/#on-policy-data)是抗遗忘和 KL 预算的承重件，不是显式 KL 惩罚。老师分布和学生对齐的状态是两支旋钮。原文：[nrehiew](https://nrehiew.github.io/blog/sft_rl_opd/)。
- [sampled-token reverse KL](/wiki/on-policy-distillation/teacher-signal-granularity/#sampled-token) 是便宜的默认实现，可当作把 RL 的 KL regularizer 换成老师；它不是 OPD 的唯一定义。原文：[Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/)。
- [top-k](/wiki/on-policy-distillation/teacher-signal-granularity/#top-k) 是信息量和成本之间最常被采用的折中；[full-vocab](/wiki/on-policy-distillation/teacher-signal-granularity/#full-vocab) 最完整，但一般团队很难负担。原文：[Revisiting OPD](https://arxiv.org/abs/2603.25562)、[知乎整理](https://zhuanlan.zhihu.com/p/2033212181823608430)。
- 先用老师轨迹做 SFT、再 OPD，是工业上反复出现的[冷启动](/wiki/on-policy-distillation/when-opd-works/#cold-start)，不是可有可无的细节。原文：[Rethinking OPD](https://arxiv.org/abs/2604.13016)、[Tinker cookbook](https://tinker-docs.thinkingmachines.ai/cookbook/recipes/distillation/)。
- 老师分数更高既不充分也不必要；[pattern 匹配](/wiki/on-policy-distillation/when-opd-works/#thinking-pattern)和[可迁移的新能力](/wiki/on-policy-distillation/when-opd-works/#new-capability)更重要。原文：[Rethinking OPD](https://arxiv.org/abs/2604.13016)。在老师已经会做任务时，[SFT overtrain 再 OPD](/wiki/on-policy-distillation/sft-rl-opd/#sft-teacher-opd) 可以少继承 SFT 的遗忘。原文：[nrehiew](https://nrehiew.github.io/blog/sft_rl_opd/)。
- 长轨迹上应保持局部监督（discount 接近 0），但不要把局部比较收成单个 sampled token。原文：[Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/)、[Revisiting OPD](https://arxiv.org/abs/2603.25562)。
- OPD 擅长把老师已经发现的能力迁给学生，不擅长发现老师不会的新策略；它更像 post-training glue，不是 RL 的替代品。原文：[知乎整理](https://zhuanlan.zhihu.com/p/2033212181823608430)。

## 边界

当前证据主要来自数学推理、指令跟随恢复，以及最小代码编辑上的遗忘/泛化对照。多轮 tool-use 已有 [Tinker Harbor recipe](https://tinker-docs.thinkingmachines.ai/cookbook/recipes/distillation/)，但还没有与 math 同等强度的对照实验写入本 topic。失败模式页只讨论白盒 logprob 监督；outcome-gated 或 black-box OPD 留待后续来源。SWIFT / verl 的配置映射写在粒度页的操作要点，不单独建系统页。GLM 5 / DeepSeek V4 / MiMo-V2 Flash 的管线数字未经那些报告原文核验，只保留 nrehiew 的转述方向。

## See Also

- [SFT、RL 与 OPD](/wiki/on-policy-distillation/sft-rl-opd/)
- [sampled-token、top-k 与 full-vocab](/wiki/on-policy-distillation/teacher-signal-granularity/)
- [老师信号何时可靠](/wiki/on-policy-distillation/when-opd-works/)
