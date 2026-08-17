---
title: "老师信号何时可靠"
topic: on-policy-distillation
summary: "OPD 要老师与学生共享 thinking pattern、老师带有可迁移的新能力，并且不要把 dense 监督收成单个 sampled token。"
lang: zh-CN
updated: 2026-08-17
order: 3
sources:
  - title: "On-Policy Distillation"
    url: "https://thinkingmachines.ai/blog/on-policy-distillation/"
  - title: "Distillation (Tinker Cookbook)"
    url: "https://tinker-docs.thinkingmachines.ai/cookbook/recipes/distillation/"
  - title: "Rethinking On-Policy Distillation of Large Language Models"
    url: "https://arxiv.org/abs/2604.13016"
  - title: "Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes"
    url: "https://arxiv.org/abs/2603.25562"
raw:
  - raw/on-policy-distillation/2025-10-27-on-policy-distillation.md
  - raw/on-policy-distillation/tinker-cookbook-distillation.md
  - raw/on-policy-distillation/2026-04-14-rethinking-on-policy-distillation.md
  - raw/on-policy-distillation/2026-03-26-revisiting-on-policy-distillation.md
---

## Overview

Thinking Machines 把 OPD 写成几乎免费的 dense 监督。后续工作把这句话拆开：监督密，不等于老师在学生前缀上的分数可学。失败来自两类问题——老师和学生根本不在同一套 reasoning 上，或局部比较方式把分布差收成一个 sampled token。

## 老师和学生要先对上

Rethinking OPD 给了两条成功条件。

**Thinking-pattern consistency。** 更强老师不保证更好蒸馏。用 Qwen3-1.7B-Base 当学生时，Qwen3-4B-Base-GRPO 弱于或接近 Qwen3-4B（Non-thinking）的榜，但初始 top-k overlap 更高，蒸出来也更好。后期 overlap 曲线会靠拢，性能差距却留着：早期 pattern 错配造成的损失后面补不回。

**更高分数 ≠ 新知识。** 同家族、同配方、只是更大的老师（如 R1-Distill-7B 蒸 R1-Distill-1.5B）对学生几乎不可区分，OPD 增益有限。对同一 checkpoint 再做 RL 的老师（Skywork-OR1-Math-7B、Qwen3-4B-Non-Thinking-RL-Math）才能明显拉高 gap recovery rate。

成功 run 的 token 级签名：overlap 从 72% 升到 91%，共享 top-k 集中 97%–99% 的概率质量，熵差收窄。失败 run 从一开始 overlap 就卡住。只在 overlap token 上算监督，效果可以对齐 full top-k，说明梯度主要打在这组高概率交集上。

## 冷启动和 prompt 怎么救

两条 complementary 策略：

1. **Off-policy cold start。** 先用老师轨迹 SFT，把 overlap 拉起来再 OPD。Rethinking 用 Qwen3-4B 在 OpenThoughts 子集上生成 200K 回复，SFT Qwen3-1.7B-Base，再用剩下约 30K 去重 prompt 做 OPD；明显好于从 base 直接 OPD。这与 Thinking Machines / Tinker 的「OpenThoughts3 SFT → DeepMath OPD」是同一条。
2. **Teacher-aligned prompt。** 用老师自己后训练过的 prompt，能把高概率 token 对齐做尖；学生熵会掉，需要混 OOD prompt。

Thinking Machines 还写：SFT（forward KL）先给新 token 加 support，reverse KL 再在 support 里 mode-seeking。老师策略若不在学生 support 里，需要显著更大的 batch。他们 math 实验用 4 samples / prompt；cookbook 里 DeepMath OPD 的 `groups_per_batch=512`，personalization 用 64。

## Sampled-token 比较为什么脆

TML 默认用 sampled-token reverse KL：只在学生抽出的那个 token 上算 log-ratio。Revisiting OPD 认为这在长轨迹上有三个失败模式。

1. **单 token 信号极不均衡。** 多数 sampled token 得负分，优化被少数局部正 advantage 带着走，容易强化语气词或犹豫标记。
2. **学生前缀上老师不可靠。** 轨迹一旦进入老师少见的状态，老师仍可能给重复循环、自我重置、畸形续写高概率。log-ratio 随位置变宽、极值变多。
3. **Tokenizer / special token 错配。** 一 token 比较会把切分差异当成语义分歧。给 sampled-token 基线加 special-token mask 后，AIME24 从 10.0 到 26.7，平均分从 36.4 到 40.7。

估计器上还有 bias–variance：sequence-level reverse KL 把未来 reward 耦进来，token-level 丢掉这些项，相对有偏，但 worst-case 方差上界是 \(O(T^{2})\) 而不是 \(O(T^{4})\)。增大 \(\gamma\) 会抬高梯度方差。这与 TML「discount = 0」一致：长序列上应保持局部更新，但局部比较本身要改。

## 局部 support 匹配

Revisiting 的修法是 teacher top-K local support matching：在每个前缀上取老师 top-K，把师生分布限制在这个集合上 renormalize，再算 truncated reverse KL。仍然是 token-level，但比较的是老师认为可行的局部支持，而不是一个 sampled token。

配套稳定化：

- **Support 上必须 renormalize**，否则两边质量不可比，训练会不稳。
- **Rollout 用 top-\(p\)。** 他们默认 \(p=0.9\)。无截断采样会打出极低概率 token，老师信号变差。
- **Mask special token。** 对 sampled-token 基线帮助很大；support matching 对它不那么敏感。

单任务数学上，完整方法平均 41.5，高于 sampled-token 的 36.4 和加 mask 后的 40.7。Ablation（AIME24 avg@32）：sampled-token 20.4；只加 teacher top-K 掉到 17.7；top-K + top-\(p\) 才到 23.6。Top-K 本身不够，还要管采样。

Rethinking 另外观察到 dense 监督随轨迹变深而退化，不稳从后缀往前传。中等长度（3K 和 7K）最好，10K 和 15K 变差。这限制了「越长越能蒸」的假设。

## 操作要点

把上面收成可执行约束，而不是另一套算法：

- 学生和老师尽量同家族、同 chat/think 模板；不要只看榜单高几分。
- 老师应带 RL 或其它后训练得到的新能力，而不是同数据放大号。
- 先 SFT 再 OPD。Tinker：LoRA SFT lr `1e-3`，OPD lr `1e-4`；full FT 的 OPD 用 `5e-5`。
- Support 不够就加大 `groups_per_batch`；DeepMath 用 512，行为恢复用 64。
- Token 级、discount 0；不要为了「更数学正确」上 sequence-level return。
- 长推理用 top-\(p\) rollout，并对 special token 做 mask；有条件则用老师 top-K truncated KL 替代纯 sampled-token。
- 环境 token（system、user、tool response、assistant header）全部 mask，只在学生生成 token 上算 KL。
- Prompt 可以反复用。TML 甚至用 1 条 prompt、每步 256 条、共 20 step（5120 条打分序列）去逼近老师的 AIME’24；RL 在同样设定更容易背答案。
- 多轮 tool-use 不要混环境奖励，除非明确要做 hybrid；Tinker Harbor recipe 用 `zero_reward`，只留老师 KL。
- 监控 overlap 是否在涨、熵差是否在收。overlap 卡住，先查 pattern mismatch 和冷启动，而不是加学习率。

## See Also

- [On-Policy Distillation 问题地图](/wiki/on-policy-distillation/overview/)
- [SFT、RL 与 OPD](/wiki/on-policy-distillation/sft-rl-opd/)
