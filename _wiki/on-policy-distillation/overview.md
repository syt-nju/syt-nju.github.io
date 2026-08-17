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
raw:
  - raw/on-policy-distillation/2025-10-27-on-policy-distillation.md
  - raw/on-policy-distillation/2026-07-on-policy-distillation-floating-bytes.md
  - raw/on-policy-distillation/tinker-cookbook-distillation.md
  - raw/on-policy-distillation/2026-04-14-rethinking-on-policy-distillation.md
  - raw/on-policy-distillation/2026-03-26-revisiting-on-policy-distillation.md
  - raw/on-policy-distillation/zhihu-opd-deep-dive-v2.md
---

## Overview

On-Policy Distillation（OPD）把后训练拆成彼此独立的选择：训练状态从谁来、每个 prefix 上比较什么、以及老师分数何时可学。学生在自己的 rollout 上学习，老师在这些前缀上给 dense 反馈。这同时针对 SFT 的 exposure bias 和 RL 的稀疏 credit。OPD 不是单一算法。

本 topic 目前只覆盖白盒、可取老师 logprob 的设定。黑盒老师、privileged self-distillation 尚未摄入。DeepSeek V4 报告本身未摄入，相关主张来自知乎对该报告的阅读。

## 问题地图

### 训练状态该跟谁对齐

Off-policy 蒸馏训练的是老师常去的前缀；推理时学生走自己的错路，错误会沿序列放大。OPD 改成学生采样、老师打分，让训练分布贴近推理分布。这条比较构成后训练的 2×2：SFT 是 off-policy + dense，RL 是 on-policy + sparse，OPD 是 on-policy + dense。详见 [SFT、RL 与 OPD](/wiki/on-policy-distillation/sft-rl-opd/)。

### 每个 prefix 上比较什么

Dense 仍有粒度。sampled-token 只评价学生抽出的那一个 token；top-k 比较老师支持的一小撮候选；full-vocab 比较整个词表。KL 还可以选 forward / reverse / JSD，loss 可以是直接反传或 policy-gradient。Thinking Machines 的默认实现只是这个格子里的一格。详见 [sampled-token、top-k 与 full-vocab](/wiki/on-policy-distillation/teacher-signal-granularity/)。

### 老师信号何时可靠

监督密，不等于老师在学生前缀上的分数可学。老师与学生的 thinking pattern 要对上，老师还要带来学生训练中没见过的能力；否则更强老师也可能蒸不动。详见 [老师信号何时可靠](/wiki/on-policy-distillation/when-opd-works/)。

## 共同主张

- 后训练的关键差在状态来源和监督密度，不在「像不像 RL 的代码路径」。
- sampled-token reverse KL 是便宜的默认实现，可当作把 RL 的 KL regularizer 换成老师；它不是 OPD 的唯一定义。
- top-k 是信息量和成本之间最常被采用的折中；full-vocab 最完整，但一般团队很难负担。
- 先用老师轨迹做 SFT、再 OPD，是工业上反复出现的冷启动，不是可有可无的细节。
- 老师分数更高既不充分也不必要；pattern 匹配和可迁移的新能力更重要。
- 长轨迹上应保持局部监督（discount 接近 0），但不要把局部比较收成单个 sampled token。
- OPD 擅长把老师已经发现的能力迁给学生，不擅长发现老师不会的新策略；它更像 post-training glue，不是 RL 的替代品。

## 边界

当前证据主要来自数学推理和指令跟随恢复。多轮 tool-use 已有 Tinker Harbor recipe，但还没有与 math 同等强度的对照实验写入本 topic。失败模式页只讨论白盒 logprob 监督；outcome-gated 或 black-box OPD 留待后续来源。SWIFT / verl 的配置映射写在粒度页的操作要点，不单独建系统页。

## See Also

- [SFT、RL 与 OPD](/wiki/on-policy-distillation/sft-rl-opd/)
- [sampled-token、top-k 与 full-vocab](/wiki/on-policy-distillation/teacher-signal-granularity/)
- [老师信号何时可靠](/wiki/on-policy-distillation/when-opd-works/)
