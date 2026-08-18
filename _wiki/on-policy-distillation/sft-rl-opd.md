---
title: "SFT、RL 与 OPD"
topic: on-policy-distillation
summary: "OPD 是学生自采样加上 dense 老师监督；它补的是 SFT 的状态错配和 RL 的稀疏奖励。承重的是 on-policy 数据，不是显式 KL 惩罚。"
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
  - title: "Rethinking On-Policy Distillation of Large Language Models"
    url: "https://arxiv.org/abs/2604.13016"
  - title: "SFT, RL, and On-Policy Distillation Through a Distributional Lens"
    url: "https://nrehiew.github.io/blog/sft_rl_opd/"
raw:
  - raw/on-policy-distillation/2025-10-27-on-policy-distillation.md
  - raw/on-policy-distillation/2026-07-on-policy-distillation-floating-bytes.md
  - raw/on-policy-distillation/tinker-cookbook-distillation.md
  - raw/on-policy-distillation/2026-04-14-rethinking-on-policy-distillation.md
  - raw/on-policy-distillation/2026-05-10-sft-rl-opd-distributional-lens.md
---

## Overview {#sampling-density}

本页只负责 OPD 主线里的**状态来源**问题：SFT 为什么会错在 off-policy 轨迹，RL 为什么赢在 on-policy 状态，OPD 为什么把 on-policy 数据和 dense 老师监督接起来。粒度选择放在 [每个 prefix 比较什么](/wiki/on-policy-distillation/teacher-signal-granularity/#granularity)，老师是否值得听放在 [thinking-pattern consistency](/wiki/on-policy-distillation/when-opd-works/#thinking-pattern)。

按采样来源和监督密度，常见后训练可以画成一张表。[Thinking Machines Lab](https://thinkingmachines.ai/blog/on-policy-distillation/) 把 OPD 放在「on-policy + dense」这一格：学生生成轨迹，老师给每个 token 打分。

| 方法 | 采样 | 监督 |
| --- | --- | --- |
| Supervised finetuning | off-policy | dense |
| Reinforcement learning | on-policy | sparse |
| On-policy distillation | on-policy | dense |

## Exposure bias {#exposure-bias}

SFT / 普通蒸馏在老师或数据的轨迹上做 next-token 学习。老师可以提供完整思维链，也可以提供 full next-token 分布（logit distillation）。训练时每一步的前缀几乎都是「正确的那条路」。

推理时前缀换成模型自己采出来的 token。前面一旦写错，后面就进入训练时没见过的状态，下一步更容易再错，错误沿序列放大。这就是 exposure bias，也叫 compounding error。长序列更重。学生还可能只学会老师的语气和自信，而不是事实正确性。

术语来自 Bengio 等人 2015 的 scheduled sampling；[Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/) 和 [Floating Bytes](https://saraswatmks.github.io/2026/07/on-policy-distillation-thinking-machines.html) 用它指 off-policy 蒸馏的结构问题，不只是数据不够。[Rethinking OPD](https://arxiv.org/abs/2604.13016) 同样把 train–inference 分布错配写成所有 off-policy 方法的共同限制。[nrehiew](https://nrehiew.github.io/blog/sft_rl_opd/) 还引 Ross 等人：SFT 只见老师访问的状态，推理时一步走偏就会 compounding error；on-policy 数据聚合能减小这个错配。

## Sparse credit {#sparse-credit}

RL 在学生自己的 rollout 上学习，因此能直接惩罚自己会犯的错，状态是对的。代价是每条轨迹通常只有对错这一类稀疏信号：知道结局不好，很难知道是哪一步害的。这就是稀疏 credit assignment。

[Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/) 的信息论说法：RL 每 episode 教 \(O(1)\) bit，蒸馏教 \(O(N)\) bit，\(N\) 是 token 数。OPD 保留学生采样，但把老师的 token 级分数换成 dense 监督，用来补这个洞。

## OPD：学生走，老师评

OPD 保留学生采样，把老师当成逐步评分器。直觉是棋手自己下棋，引擎给每步标好坏，而不是只看终局，也不是只看大师棋谱。

[Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/) 的默认目标是 [sampled-token reverse KL](/wiki/on-policy-distillation/teacher-signal-granularity/#sampled-token)：在学生自己的前缀上，最小化 \(\log\pi_\theta-\log\pi_{\text{teacher}}\)。它 [mode-seeking](/wiki/on-policy-distillation/teacher-signal-granularity/#kl-direction)，期望在学生分布下计算，因此是 on-policy。他们取 discount 为 0，只优化当前 token；实践中 \(\gt 0\) 的 discount 没有带来收益。[Tinker cookbook](https://tinker-docs.thinkingmachines.ai/cookbook/recipes/distillation/) 同样表示提高 `kl_discount_factor` 一般无帮助。这只是 [比较粒度](/wiki/on-policy-distillation/teacher-signal-granularity/#granularity) 里的一格，不是 OPD 的唯一定义。

实现上可以是 RL trainer 的一行改动：把 KL regularizer 的 reference 换成老师，把 per-token advantage 设为负的 reverse KL，再用 importance sampling 更新学生。老师只需一次 `compute_logprobs`，不必反传。轨迹由较小的学生生成。

Floating Bytes 把同一 loop 写成：无梯度采样 → 学生再前向拿带梯度 logprob → 老师前向打分 → reverse KL。采样不可微，所以学生需要第二次前向。

## On-policy 数据为什么承重 {#on-policy-data}

[nrehiew](https://nrehiew.github.io/blog/sft_rl_opd/) 把 2×2 再写成分布几何：后训练是在改序列分布，关键问题是目标分布是什么、拉得有多直接。

SFT 把模型拉向训练前就固定好的外部数据集。负对数似然不在乎起点，所以目标可以离原策略任意远；这是 catastrophic forgetting 的自然失败模式，也解释了它适合需要大改输出格式的冷启动。RL 没有任意外部目标：梯度只通过当前策略采到的样本走，主要改模型已经会去的高概率区域。

常见「SFT 是 forward KL、RL 是 reverse KL」只能解释一部分。nrehiew 认为它过度依赖对 reference 的显式 KL：RLVR 里 KL 约束往往比 RLHF 弱很多，抗遗忘却还在。SFT 每个 demonstrated token 都被均匀推高（任务关键 token 和 “therefore” 一类 style token 一视同仁）；RL 的 group-normalized advantage 会在不确定时缩小更新。这些观察对 SFT vs RL 成立，但不足以单独说明 OPD。

他更认同的解释来自 Shenfeld 等人。用最简单的 REINFORCE、二元 0/1 奖励：reward=1 才给正信号，reward=0 相当于丢掉，很像 rejection sampling。最优策略有很多，但 on-policy 采样迫使训练去拟合**离当前策略最近**的那个可解任务策略，每一步目标都隐式低 KL。SFT 的目标可以很远；OPD 的老师给信号，状态分布仍是学生的，所以能继承 RL 的抗遗忘，而不必靠显式 KL 惩罚。

Brown 2026 把后训练写成能力 vs 相对先验的 KL 预算。nrehiew 的结论是：任何还想停在这条 Pareto 前沿、又比 RL 更省算力的算法，都必须靠 on-policy 数据。OPD 和 RL 走到相近的地方，说明承重件是 on-policy，不是「RL 这个算法本身」。剩下的旋钮仍是 [稀疏 credit](/wiki/on-policy-distillation/sft-rl-opd/#sparse-credit)：outcome reward 太疏，老师 logit 太密但有偏。

## 证据：老师可以 SFT overtrain，学生仍少忘 {#sft-teacher-opd}

最小代码编辑任务上，nrehiew 先分别 SFT / RL 出两个老师，再对同一学生做 OPD。评测是 out-of-domain 损坏类型上的最小编辑，以及 LiveCodeBench v6 上的遗忘。

| Model | Pass@1 ↑ | Norm. Levenshtein ↓ | Added CC ↓ | LiveCodeBench v6 ↑ |
| --- | --- | --- | --- | --- |
| SFT teacher | 0.775 | 0.450 | 0.450 | 0.286 |
| RL teacher | 0.792 | 0.063 | 0.206 | 0.320 |
| OPD from SFT teacher | 0.800 | 0.059 | 0.206 | 0.297 |
| OPD from RL teacher | 0.787 | 0.055 | 0.228 | 0.314 |

两个 OPD 学生几乎一样：都明显好于 SFT teacher，并略超 RL teacher；LiveCodeBench 上的遗忘都轻于 SFT teacher，即使老师自己已经忘了。若老师分布是主因，SFT teacher 的学生应继承更多遗忘，但没有。原文：[nrehiew](https://nrehiew.github.io/blog/sft_rl_opd/)。

这不是「老师从不重要」的普遍定理，实验只在最小编辑上。它说明**学生状态从哪来**和**老师分布长什么样**是两支旋钮。Rethinking 管后者，见 [更高分数不等于新能力](/wiki/on-policy-distillation/when-opd-works/#new-capability)。实践含义是：可以先把专家 SFT overtrain，再用 OPD 把能力迁回通用模型，少带上 forgetting。

学生超过老师也不是孤例。Agarwal 等人的 OPD 原作里，蒸出的学生在 GSM8K 上超过老师。nrehiew 的假说：监督打在学生自己会去的 prefix 上；KL matching 也不是 reward 最大化，老师分布里的 style、不确定性和备选续写可以改采样行为，而不必复现老师的 greedy 输出。OPD 的 entropy collapse 比 RL 更陡，和他预期的 reverse KL mode-seeking 一致。

## 证据：算力效率 {#compute-evidence}

Qwen3 技术报告 Table 21（Thinking Machines 转引）：off-policy 蒸馏 AIME’24 55.0%、GPQA-Diamond 55.6%；加 RL 后 67.6% / 61.3%，17,920 GPU hours；加 OPD 后 74.4% / 63.3%，1,800 GPU hours。

Thinking Machines 自己的复现：Qwen3-8B-Base 在 OpenThoughts-3 上 SFT 400K prompts 后 AIME’24 为 60%。继续 off-policy 外推到 70% 约需 2M prompts。从该 400K checkpoint 做 OPD，约 150 steps（约 77K prompts，每 prompt 4 samples）到 70%。相对 SFT-2M，他们报 9-30x 的 compute efficiency，取决于是否把老师采样成本算进 off-policy 基线。

同初始化、把 RL 训出的老师再蒸回 base 时，OPD 大约用 7-10x 更少的 gradient steps 追上老师，综合算力他们估计 50-100x：蒸馏可用更短 context，强 SFT 初始化下 batch 也可以更小。

Tinker cookbook 把学生/老师换成 Qwen3.5-9B-Base / Qwen3.5-9B 后：rank-128 LoRA SFT 3000 steps 约 65% AIME’24；再 OPD 200 steps、16k-token rollouts 约 76.7%。评测用 `temperature=1.0`、`top_p=1.0`、`max_tokens=64000`。

LoRA 在大规模 SFT 上落后更明显；Thinking Machines 写 rank = 32 时，SFT 后 LoRA 落后 full finetuning 13%，OPD 后只落后 6%。

## 证据：行为恢复 {#behavior-recovery}

域数据 mid-train 会打掉指令跟随。Qwen3-8B 的 IF-eval 从 85% 降到纯文档 mid-train 的 45%；70% 文档 + 30% chat 混训后是 79%。再用**旧版 Qwen3-8B** 当老师、在 Tulu3 prompt 上 OPD，IF-eval 回到 83%，internal QA 从 36% 升到 41%。Tinker 写 IF-eval 大约 100 steps 内恢复。

对模型自己 `temperature = 1.0` 的样本做 SFT，即使期望 KL 为 0，实用学习率仍会让 IF-eval 下降：有限 batch 使训练逐渐变成 off-policy。锁行为要用老师固定的 OPD，而不是 self-SFT。

这些数字说明 OPD 可以当行为恢复工具，但不单独构成一条与 2×2 平级的设计轴。

## See Also

- [On-Policy Distillation 问题地图](/wiki/on-policy-distillation/overview/)
- [sampled-token、top-k 与 full-vocab](/wiki/on-policy-distillation/teacher-signal-granularity/)
- [老师信号何时可靠](/wiki/on-policy-distillation/when-opd-works/)
