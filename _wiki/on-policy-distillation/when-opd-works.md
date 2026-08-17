---
title: "老师信号何时可靠"
topic: on-policy-distillation
summary: "OPD 要老师与学生共享 thinking pattern、老师带有可迁移的新能力；否则 dense 监督只是更密的噪声。"
lang: zh-CN
updated: 2026-08-17
order: 4
sources:
  - title: "On-Policy Distillation"
    url: "https://thinkingmachines.ai/blog/on-policy-distillation/"
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
  - raw/on-policy-distillation/tinker-cookbook-distillation.md
  - raw/on-policy-distillation/2026-04-14-rethinking-on-policy-distillation.md
  - raw/on-policy-distillation/2026-03-26-revisiting-on-policy-distillation.md
  - raw/on-policy-distillation/zhihu-opd-deep-dive-v2.md
  - raw/on-policy-distillation/2026-05-10-sft-rl-opd-distributional-lens.md
---

## Overview

Thinking Machines 把 OPD 写成几乎免费的 dense 监督。后续工作把这句话拆开：监督密，不等于老师在学生前缀上的分数可学。失败可以来自老师选错（pattern 不一致、没有新能力），也可以来自比较粒度选错。后者是独立设计轴，见 [sampled-token、top-k 与 full-vocab](/wiki/on-policy-distillation/teacher-signal-granularity/)。本页管老师何时值得听。原文：[Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/)、[Rethinking OPD](https://arxiv.org/abs/2604.13016)。

## Thinking-pattern consistency {#thinking-pattern}

[Rethinking OPD](https://arxiv.org/abs/2604.13016) 的第一条成功条件：更强老师不保证更好蒸馏。用 Qwen3-1.7B-Base 当学生时，Qwen3-4B-Base-GRPO 弱于或接近 Qwen3-4B（Non-thinking）的榜，但初始 top-k overlap 更高，蒸出来也更好。后期 overlap 曲线会靠拢，性能差距却留着：早期 pattern 错配造成的损失后面补不回。

成功 run 的 token 级签名：overlap 从 72% 升到 91%，共享 top-k 集中 97%–99% 的概率质量，熵差收窄。失败 run 从一开始 overlap 就卡住。只在 overlap token 上算监督，效果可以对齐 full top-k，说明梯度主要打在这组高概率交集上。

## 更高分数不等于新能力 {#new-capability}

同家族、同配方、只是更大的老师（如 R1-Distill-7B 蒸 R1-Distill-1.5B）对学生几乎不可区分，OPD 增益有限。对同一 checkpoint 再做 RL 的老师（Skywork-OR1-Math-7B、Qwen3-4B-Non-Thinking-RL-Math）才能明显拉高 gap recovery rate。原文：[Rethinking OPD](https://arxiv.org/abs/2604.13016)。

这支旋钮是**老师分布里有没有可迁移的新能力**。另一支是**训练状态是不是学生自己的**。[nrehiew](https://nrehiew.github.io/blog/sft_rl_opd/) 在最小编辑上发现：SFT teacher 和 RL teacher 蒸出的 OPD 学生几乎一样，而且学生忘得比已经退化的 SFT teacher 更少。他把这读成：数据来源（on-policy）比老师分布更重要。这不是否定 Rethinking，实验域也不同；数字和几何解释在 [老师可以 SFT overtrain](/wiki/on-policy-distillation/sft-rl-opd/#sft-teacher-opd)。

## 冷启动和 prompt {#cold-start}

两条 complementary 策略：

1. **Off-policy cold start。** 先用老师轨迹 SFT，把 overlap 拉起来再 OPD。Rethinking 用 Qwen3-4B 在 OpenThoughts 子集上生成 200K 回复，SFT Qwen3-1.7B-Base，再用剩下约 30K 去重 prompt 做 OPD；明显好于从 base 直接 OPD。这与 Thinking Machines / Tinker 的「OpenThoughts3 SFT → DeepMath OPD」是同一条。
2. **Teacher-aligned prompt。** 用老师自己后训练过的 prompt，能把高概率 token 对齐做尖；学生熵会掉，需要混 OOD prompt。

Thinking Machines 还写：SFT（forward KL）先给新 token 加 support，reverse KL 再在 support 里 mode-seeking。老师策略若不在学生 support 里，需要显著更大的 batch。他们 math 实验用 4 samples / prompt；cookbook 里 DeepMath OPD 的 `groups_per_batch=512`，personalization 用 64。

## Sampled-token 比较为什么脆 {#sampled-token-failure}

TML 默认用 sampled-token reverse KL：只在学生抽出的那个 token 上算 log-ratio。[Revisiting OPD](https://arxiv.org/abs/2603.25562) 认为这在长轨迹上有三个失败模式。

1. **单 token 信号极不均衡。** 多数 sampled token 得负分，优化被少数局部正 advantage 带着走，容易强化语气词或犹豫标记。
2. **学生前缀上老师不可靠。** 轨迹一旦进入老师少见的状态，老师仍可能给重复循环、自我重置、畸形续写高概率。log-ratio 随位置变宽、极值变多。
3. **Tokenizer / special token 错配。** 一 token 比较会把切分差异当成语义分歧。给 sampled-token 基线加 special-token mask 后，AIME24 从 10.0 到 26.7，平均分从 36.4 到 40.7。

估计器上还有 bias–variance：sequence-level reverse KL 把未来 reward 耦进来，token-level 丢掉这些项，相对有偏，但 worst-case 方差上界是 \(O(T^{2})\) 而不是 \(O(T^{4})\)。增大 \(\gamma\) 会抬高梯度方差。这与 TML「discount = 0」一致：长序列上应保持局部更新，但局部比较本身要改。

## 比较方式也会让 dense 变成噪声

把分布差收成单个 sampled token，本身就会让老师分数不可用；改成老师 top-K 局部分布后，Revisiting 的单任务数学平均从 36.4 升到 41.5。方法细节在 [sampled-token、top-k 与 full-vocab](/wiki/on-policy-distillation/teacher-signal-granularity/)。

即便粒度选对，老师在长后缀上仍可能劣化。Rethinking 观察到不稳从后缀往前传：中等长度（3K 和 7K）最好，10K 和 15K 变差。这限制了「越长越能蒸」的假设。知乎的综合判断是：OPD 的核心风险，是密集监督在错误 prefix、错误 token 粒度、错误 KL 方向下变成密集噪声。OPD 也不替代 RL——它迁移老师已有的能力，不负责发现老师不会的策略。

## OPSD：同模型、特权前缀 {#opsd}

On-Policy Self Distillation（OPSD）里，老师和学生是同一个模型，但算「老师」logprob 时把参考答案当作 prefix。特权信息提供学习信号，不必另训一个更强老师。

[nrehiew](https://nrehiew.github.io/blog/sft_rl_opd/) 转述：因为师生是同一套权重，多数 token 上输出很接近；per-token KL 分析里，"wait" / "alright" 一类 style 或 pivot token 的 KL 高于 "power" / "exponent" / "logarithm" 一类 math token。更新太猛会在不重要的 token 上塌缩，所以要 per-token clipping。这让 OPSD 更接近 RLHF（信号有偏，需要 trust-region），而不是低偏的 RLVR。相对 [稀疏 credit](/wiki/on-policy-distillation/sft-rl-opd/#sparse-credit) 的 \(O(1)\) bit / episode，OPSD 走到另一极端：每个 token 都有「奖励」，但每步噪声和偏差也更大。本 topic 没有摄入 OPSD 原论文，以上只来自这篇整理。

## 操作要点

把上面收成可执行约束，而不是另一套算法：

- 学生和老师尽量同家族、同 chat/think 模板；不要只看榜单高几分。
- 老师应带 RL 或其它后训练得到的新能力，而不是同数据放大号。
- 先 SFT 再 OPD。Tinker：LoRA SFT lr `1e-3`，OPD lr `1e-4`；full FT 的 OPD 用 `5e-5`。
- Support 不够就加大 `groups_per_batch`；DeepMath 用 512，行为恢复用 64。
- Token 级、discount 0；不要为了「更数学正确」上 sequence-level return。比较对象用 top-K 或 full-vocab，而不是单个 sampled token。
- 环境 token（system、user、tool response、assistant header）全部 mask，只在学生生成 token 上算 KL。
- Prompt 可以反复用。TML 甚至用 1 条 prompt、每步 256 条、共 20 step（5120 条打分序列）去逼近老师的 AIME’24；RL 在同样设定更容易背答案。
- 多轮 tool-use 不要混环境奖励，除非明确要做 hybrid；Tinker Harbor recipe 用 `zero_reward`，只留老师 KL。
- 监控 overlap 是否在涨、熵差是否在收。overlap 卡住，先查 pattern mismatch 和冷启动，而不是加学习率。
- 同模型 OPSD：style token 的 KL 可能高于任务 token，不要按 KL 大小无差别猛更新；需要 clipping。详见 [OPSD](/wiki/on-policy-distillation/when-opd-works/#opsd)。

## See Also

- [On-Policy Distillation 问题地图](/wiki/on-policy-distillation/overview/)
- [SFT、RL 与 OPD](/wiki/on-policy-distillation/sft-rl-opd/)
- [sampled-token、top-k 与 full-vocab](/wiki/on-policy-distillation/teacher-signal-granularity/)
