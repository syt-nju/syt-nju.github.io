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
raw:
  - raw/on-policy-distillation/2025-10-27-on-policy-distillation.md
  - raw/on-policy-distillation/2026-07-on-policy-distillation-floating-bytes.md
  - raw/on-policy-distillation/tinker-cookbook-distillation.md
  - raw/on-policy-distillation/2026-04-14-rethinking-on-policy-distillation.md
  - raw/on-policy-distillation/2026-03-26-revisiting-on-policy-distillation.md
---

## Overview

On-Policy Distillation（OPD）把后训练拆成两个彼此独立的选择：训练状态从谁来，以及每条轨迹给多少监督。学生在自己的 rollout 上学习，老师只在这些前缀上给 dense 反馈，通常是 reverse KL。这同时针对 SFT 的 exposure bias 和 RL 的稀疏 credit。

本 topic 目前只覆盖白盒、可取老师 logprob 的设定。黑盒老师、privileged self-distillation、跨 tokenizer 对齐尚未摄入。

## 问题地图

### 训练状态该跟谁对齐

Off-policy 蒸馏训练的是老师常去的前缀；推理时学生走自己的错路，错误会沿序列放大。OPD 改成学生采样、老师打分，让训练分布贴近推理分布。这条比较构成后训练的 2×2：SFT 是 off-policy + dense，RL 是 on-policy + sparse，OPD 是 on-policy + dense。详见 [SFT、RL 与 OPD](/wiki/on-policy-distillation/sft-rl-opd/)。

### 老师信号何时可靠

Dense 并不自动等于可学。老师与学生的 thinking pattern 要对上，老师还要带来学生训练中没见过的能力；否则更强老师也可能蒸不动。即便条件满足，sampled-token reverse KL 在长轨迹上仍会偏、噪、被分词干扰。稳定化是领域共同问题，不是某一篇的附录。详见 [老师信号何时可靠](/wiki/on-policy-distillation/when-opd-works/)。

## 共同主张

- 后训练的关键差在状态来源和监督密度，不在「像不像 RL 的代码路径」。
- 默认实现是学生 rollout 上的 token-level reverse KL，可当作把 RL 的 KL regularizer 换成老师。
- 先用老师轨迹做 SFT、再 OPD，是工业上反复出现的冷启动，不是可有可无的细节。
- 老师分数更高既不充分也不必要；pattern 匹配和可迁移的新能力更重要。
- 长轨迹上应保持局部监督（discount 接近 0），但不要把局部比较收成单个 sampled token。

## 边界

当前证据主要来自数学推理和指令跟随恢复。多轮 tool-use 已有 Tinker Harbor recipe，但还没有与 math 同等强度的对照实验写入本 topic。失败模式页只讨论白盒 logprob 监督；outcome-gated 或 black-box OPD 留待后续来源。

## See Also

- [SFT、RL 与 OPD](/wiki/on-policy-distillation/sft-rl-opd/)
- [老师信号何时可靠](/wiki/on-policy-distillation/when-opd-works/)
