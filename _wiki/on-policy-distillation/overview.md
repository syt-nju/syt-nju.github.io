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

On-Policy Distillation（OPD）的出发点很简单：SFT 信号密，但训练在老师或数据的轨迹上；RL 训练在学生自己的轨迹上，但奖励通常太稀疏。OPD 试图把两者接起来：让学生自己 rollout，再让老师在学生已经走到的 prefix 上给 dense 反馈。这样它同时针对 SFT 的 [exposure bias](/wiki/on-policy-distillation/sft-rl-opd/#exposure-bias) 和 RL 的 [稀疏 credit](/wiki/on-policy-distillation/sft-rl-opd/#sparse-credit)。

本 topic 覆盖白盒、可取老师 logprob 的设定，以及同模型、参考答案作特权前缀的 [OPSD](/wiki/on-policy-distillation/when-opd-works/#opsd)。黑盒老师尚未摄入。DeepSeek V4 报告本身未摄入，相关主张来自[知乎对该报告的阅读](https://zhuanlan.zhihu.com/p/2033212181823608430)和 [nrehiew 对工业管线的转述](https://nrehiew.github.io/blog/sft_rl_opd/)。

## 主线

### 1. SFT 信号密，但状态错

普通蒸馏和 SFT 都在外部轨迹上训练：数据里每一步通常是正确前缀，学生推理时却会走到自己的错误前缀。前面一步偏掉，后面就进入训练没见过的状态，错误沿序列放大。这就是 [exposure bias](/wiki/on-policy-distillation/sft-rl-opd/#exposure-bias)。所以只把老师答案喂给学生，不等于学生会在自己的分布上走稳。

### 2. RL 状态对，但 credit 太疏

RL 让学生在自己的 rollout 上学习，因此状态分布是对的。问题是反馈往往只有最终对错，知道整条轨迹失败，不代表知道哪一步该改。这是 [稀疏 credit](/wiki/on-policy-distillation/sft-rl-opd/#sparse-credit)。Thinking Machines 的说法是：RL 每条 episode 只给 \(O(1)\) bit，而蒸馏可以在 token 级给 \(O(N)\) bit。原文：[Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/)。

### 3. OPD 的最小想法：学生走，老师评

OPD 把训练状态改成学生自己的 prefix，同时把反馈从 outcome reward 换成老师的 token 级分数。最小实现可以是 [sampled-token reverse KL](/wiki/on-policy-distillation/teacher-signal-granularity/#sampled-token)：只看学生这一步实际采出的 token，比较学生和老师给它的 logprob。这很便宜，也能复用 RL trainer，把 KL regularizer 的 reference 换成老师。原文：[Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/)。

### 4. 最小实现不是 OPD 的全部

一旦 prefix 已经来自学生，还要决定老师在这个 prefix 上给什么。[sampled-token](/wiki/on-policy-distillation/teacher-signal-granularity/#sampled-token) 只评价学生抽出的一个 token；[top-k](/wiki/on-policy-distillation/teacher-signal-granularity/#top-k) 比较老师最看好的候选集合；[full-vocab](/wiki/on-policy-distillation/teacher-signal-granularity/#full-vocab) 比较整个词表。KL 方向、是否直接反传、是否用 policy-gradient，都是独立旋钮。把 sampled-token 当成 OPD 的定义，会漏掉后续工作真正修的失败点。

### 5. Dense 监督也可能变成 dense 噪声

监督密，不代表老师信号一定可学。失败有两类：一类是比较粒度太窄，单 token 信号会被噪声、OOD prefix、tokenizer 或 special token 错配放大；另一类是老师和学生的 [thinking pattern](/wiki/on-policy-distillation/when-opd-works/#thinking-pattern) 不一致，或老师没有学生可迁移的[新能力](/wiki/on-policy-distillation/when-opd-works/#new-capability)。这时更强老师也可能蒸不动。原文：[Rethinking OPD](https://arxiv.org/abs/2604.13016)、[Revisiting OPD](https://arxiv.org/abs/2603.25562)。

## 设计轴

- **状态从谁来。** SFT 是 off-policy + dense，RL 是 on-policy + sparse，OPD 是 on-policy + dense。on-policy 数据也是抗遗忘和 KL 预算的关键承重件。证据见 [状态来源和监督密度](/wiki/on-policy-distillation/sft-rl-opd/#sampling-density) 与 [on-policy 数据为什么承重](/wiki/on-policy-distillation/sft-rl-opd/#on-policy-data)。
- **每个 prefix 上比较什么。** OPD 可以比较 sampled token、top-k 局部分布或 full vocab 分布。便宜程度、信息量和稳定性都不同。机制见 [每个 prefix 比较什么](/wiki/on-policy-distillation/teacher-signal-granularity/#granularity)。
- **老师信号何时可靠。** 老师要和学生共享足够的 pattern，还要提供可迁移的新能力；必要时先 off-policy cold start，再 OPD。条件见 [thinking-pattern consistency](/wiki/on-policy-distillation/when-opd-works/#thinking-pattern) 和 [冷启动和 prompt](/wiki/on-policy-distillation/when-opd-works/#cold-start)。

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
