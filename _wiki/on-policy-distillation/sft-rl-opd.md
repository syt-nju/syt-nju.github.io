---
title: "SFT、RL 与 OPD"
topic: on-policy-distillation
summary: "OPD 是学生自采样加上 dense 老师监督；它补的是 SFT 的状态错配和 RL 的稀疏奖励，默认用 token-level reverse KL 实现。"
lang: zh-CN
updated: 2026-08-17
order: 2
sources:
  - title: "On-Policy Distillation"
    url: "https://thinkingmachines.ai/blog/on-policy-distillation/"
  - title: "Training LLMs using Off-Policy vs On-Policy Distillation"
    url: "https://saraswatmks.github.io/2026/07/on-policy-distillation-thinking-machines.html"
  - title: "Distillation (Tinker Cookbook)"
    url: "https://tinker-docs.thinkingmachines.ai/cookbook/recipes/distillation/"
raw:
  - raw/on-policy-distillation/2025-10-27-on-policy-distillation.md
  - raw/on-policy-distillation/2026-07-on-policy-distillation-floating-bytes.md
  - raw/on-policy-distillation/tinker-cookbook-distillation.md
---

## Overview

按采样来源和监督密度，常见后训练可以画成一张表。Thinking Machines Lab 把 OPD 放在「on-policy + dense」这一格：学生生成轨迹，老师给每个 token 打分。

| 方法 | 采样 | 监督 |
| --- | --- | --- |
| Supervised finetuning | off-policy | dense |
| Reinforcement learning | on-policy | sparse |
| On-policy distillation | on-policy | dense |

## Off-policy 密，但对齐的是老师的世界

SFT / 普通蒸馏在老师轨迹上做 next-token 学习。老师可以提供完整思维链，也可以提供 full next-token 分布（logit distillation）。序列采样是对老师分布的无偏估计，目标与 logit 蒸馏一致。

问题是 compounding error：学生一旦走出老师从未访问的前缀，后续 token 都在训练分布之外。长序列上这个问题更重。学生还可能只学会老师的语气和自信，而不是事实正确性。

## On-policy 疏，但状态是对的

RL 在学生自己的 rollout 上学习，因此能直接惩罚自己会犯的错。代价是每条轨迹通常只有对错这一类稀疏信号。Thinking Machines 用信息论说法：RL 每 episode 教 \(O(1)\) bit，蒸馏教 \(O(N)\) bit，\(N\) 是 token 数。

## OPD：学生走，老师评

OPD 保留学生采样，把老师当成逐步评分器。直觉是棋手自己下棋，引擎给每步标好坏，而不是只看终局，也不是只看大师棋谱。

默认目标是 token-level reverse KL：在学生自己的前缀上，最小化 \(\log\pi_\theta-\log\pi_{\text{teacher}}\)。它 mode-seeking，期望在学生分布下计算，因此是 on-policy。Thinking Machines 取 discount 为 0，只优化当前 token；他们写明实践中 \(\gt 0\) 的 discount 没有带来收益。Tinker cookbook 同样表示提高 `kl_discount_factor` 一般无帮助。

实现上可以是 RL trainer 的一行改动：把 KL regularizer 的 reference 换成老师，把 per-token advantage 设为负的 reverse KL，再用 importance sampling 更新学生。老师只需一次 `compute_logprobs`，不必反传。轨迹由较小的学生生成。

Floating Bytes 把同一 loop 写成：无梯度采样 → 学生再前向拿带梯度 logprob → 老师前向打分 → reverse KL。采样不可微，所以学生需要第二次前向。

## 算力证据

Qwen3 技术报告 Table 21（Thinking Machines 转引）：off-policy 蒸馏 AIME’24 55.0%、GPQA-Diamond 55.6%；加 RL 后 67.6% / 61.3%，17,920 GPU hours；加 OPD 后 74.4% / 63.3%，1,800 GPU hours。

Thinking Machines 自己的复现：Qwen3-8B-Base 在 OpenThoughts-3 上 SFT 400K prompts 后 AIME’24 为 60%。继续 off-policy 外推到 70% 约需 2M prompts。从该 400K checkpoint 做 OPD，约 150 steps（约 77K prompts，每 prompt 4 samples）到 70%。相对 SFT-2M，他们报 9-30x 的 compute efficiency，取决于是否把老师采样成本算进 off-policy 基线。

同初始化、把 RL 训出的老师再蒸回 base 时，OPD 大约用 7-10x 更少的 gradient steps 追上老师，综合算力他们估计 50-100x：蒸馏可用更短 context，强 SFT 初始化下 batch 也可以更小。

Tinker cookbook 把学生/老师换成 Qwen3.5-9B-Base / Qwen3.5-9B 后：rank-128 LoRA SFT 3000 steps 约 65% AIME’24；再 OPD 200 steps、16k-token rollouts 约 76.7%。评测用 `temperature=1.0`、`top_p=1.0`、`max_tokens=64000`。

LoRA 在大规模 SFT 上落后更明显；Thinking Machines 写 rank = 32 时，SFT 后 LoRA 落后 full finetuning 13%，OPD 后只落后 6%。

## 行为恢复是同一机制的应用

域数据 mid-train 会打掉指令跟随。Qwen3-8B 的 IF-eval 从 85% 降到纯文档 mid-train 的 45%；70% 文档 + 30% chat 混训后是 79%。再用**旧版 Qwen3-8B** 当老师、在 Tulu3 prompt 上 OPD，IF-eval 回到 83%，internal QA 从 36% 升到 41%。Tinker 写 IF-eval 大约 100 steps 内恢复。

对模型自己 `temperature = 1.0` 的样本做 SFT，即使期望 KL 为 0，实用学习率仍会让 IF-eval 下降：有限 batch 使训练逐渐变成 off-policy。锁行为要用老师固定的 OPD，而不是 self-SFT。

这些数字说明 OPD 可以当行为恢复工具，但不单独构成一条与 2×2 平级的设计轴。何时蒸得动，见 [老师信号何时可靠](/wiki/on-policy-distillation/when-opd-works/)。

## See Also

- [On-Policy Distillation 问题地图](/wiki/on-policy-distillation/overview/)
- [老师信号何时可靠](/wiki/on-policy-distillation/when-opd-works/)
