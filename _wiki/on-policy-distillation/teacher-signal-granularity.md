---
title: "sampled-token、top-k 与 full-vocab"
topic: on-policy-distillation
summary: "OPD 在每个学生 prefix 上可以比较一个 sampled token、老师 top-k 局部分布，或整个词表；这决定信息量、成本和稳定性。"
lang: zh-CN
updated: 2026-08-17
order: 3
sources:
  - title: "On-Policy Distillation"
    url: "https://thinkingmachines.ai/blog/on-policy-distillation/"
  - title: "Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes"
    url: "https://arxiv.org/abs/2603.25562"
  - title: "OPD深度解析：从数学推导到DeepSeek V4、SWIFT与verl实践"
    url: "https://zhuanlan.zhihu.com/p/2033212181823608430"
  - title: "SFT, RL, and On-Policy Distillation Through a Distributional Lens"
    url: "https://nrehiew.github.io/blog/sft_rl_opd/"
raw:
  - raw/on-policy-distillation/2025-10-27-on-policy-distillation.md
  - raw/on-policy-distillation/2026-03-26-revisiting-on-policy-distillation.md
  - raw/on-policy-distillation/zhihu-opd-deep-dive-v2.md
  - raw/on-policy-distillation/2026-05-10-sft-rl-opd-distributional-lens.md
---

## Overview

Prefix 已经是学生自己的之后，还要决定老师在这个 prefix 上给什么。Thinking Machines 默认只比较学生采出的那一个 token；Revisiting OPD 改成老师 top-K 上的局部分布；知乎对 DeepSeek V4 / SWIFT / verl 的整理把 full-vocab reverse KL 写成第三条、信息最完整也最贵的选项。这是设计轴，不是某一篇的修 bug 附录。

老师是否值得听，见 [thinking-pattern consistency](/wiki/on-policy-distillation/when-opd-works/#thinking-pattern)。

## 每个 prefix 比较什么 {#granularity}

[知乎文](https://zhuanlan.zhihu.com/p/2033212181823608430) 把三种粒度收成一句话：sampled-token 是老师评价学生这一步实际说出的 token；top-k 是老师给出它最看好的 K 个候选；full-vocab 是老师把完整分布都给学生。

| 形式 | 老师返回 | 每个 prefix 的信息 | 成本 | 稳定性 |
| --- | --- | --- | --- | --- |
| sampled-token | 该 token 的 logprob | 低 | 低 | 差一些 |
| top-k | top-k ids + logprobs | 中 | 中 | 较好 |
| full-vocab | full logits | 高 | 高 | 最好但昂贵 |

### sampled-token {#sampled-token}

学生抽出 \(y_t\) 后，advantage 是 \(\log q(y_t)-\log\pi_\theta(y_t)\)。不需要 top-k 或 full logits。[Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/) / verl 的 `k1` + policy gradient 就是这一格：把 KL regularizer 换成老师。失败模式见 [sampled-token 比较为什么脆](/wiki/on-policy-distillation/when-opd-works/#sampled-token-failure)：单 token 信号失衡、OOD prefix 上老师不可靠、tokenizer 错配。

### top-k {#top-k}

取老师 \(S_t=\mathrm{TopK}_q(c_t)\)，在支持集上比较。[Revisiting OPD](https://arxiv.org/abs/2603.25562) 要求两边 renormalize 再算 truncated reverse KL；只加 top-K、不改采样会更差。知乎指出 top-k 仍有截断偏差：忽略老师 top-k 外的质量，即使重归一化也改了 full-vocab 目标。

### full-vocab {#full-vocab}

每个 prefix 对整个词表做 \(D(P_S\|P_T)\)。[知乎](https://zhuanlan.zhihu.com/p/2033212181823608430) 转述 DeepSeek V4：多教师、学生自己的 generated trajectories、full-vocabulary reverse KL logit distillation，明确反对把 full-vocab KL 收成 sampled-token gap 再当 advantage。V4 能负担是因为缓存 last-layer hidden、用 prediction head 重构 logits、按 teacher index 排序、专用 kernel 算 exact KL。本 topic 尚未摄入 V4 报告原文，以上只来自知乎对报告的阅读。

## KL 方向和怎么反传 {#kl-direction}

粒度选定之后，还有两个正交选择。

**方向。** forward KL（\(P_T\|P_S\)）mode-covering，老师有质量的地方学生不能给太低；reverse KL（\(P_S\|P_T\)）mode-seeking，适合推理/代码这种要走一条可靠路径的任务，但可能多样性坍塌；JSD 是折中。SWIFT 的 `beta`：`0` 为 forward KL，`1` 为 reverse KL，`0.5` 接近对称 JSD。

**反传。** 可以直接对 KL 求和反传（GKD / SWIFT），也可以把负 KL 当 advantage 做 policy gradient（TML / verl `use_policy_gradient`）。MiniLLM 走接近 sequence-level reverse KL 的 PG，要 single-step decomposition、teacher-mixed sampling、length normalization；Revisiting 给出 worst-case 方差 \(O(T^2)\) vs \(O(T^4)\)，所以长序列应保持 token-level（\(\gamma=0\)），但比较对象不要收成单个 sampled token。

只有 sampled-token logprob 时，k1 / k2 / k3 是单样本 reverse KL 估计器：k1 无偏、可正可负、作 PG advantage 最自然；k2 是平方近似，非负、方差低、有偏；k3 无偏且非负。verl 的 `k3+` 前向走 k3、反向走 k2。能拿到 top-k 或 full logits 时，应在支持集或词表上显式求和，而不是继续用 k1/k2/k3。

## 操作要点

把实现映射回上面的格子，而不是给 SWIFT / verl 单独建系统页。

- 要 TML 那一格：verl `loss_mode=k1` 且 `use_policy_gradient=true`，关掉 task reward。
- sampled-token 但想低方差 direct loss：verl `k3`，`use_policy_gradient=false`。
- 要 Revisiting 那一格：支持集内重归一化 reverse KL。SWIFT `lmbda=1,beta=1,gkd_logits_topk=64`（gather 后 `log_softmax`）；verl-recipe/gkd 的 `rkl`。verl 主仓 `forward_kl_topk` 是截断 forward KL，**不**重归一化。
- 最接近知乎所写的单教师 V4 目标：SWIFT `lmbda=1,beta=1,gkd_logits_topk=None`。多教师、hidden-state 重构不在这些开源 recipe 里。
- SWIFT 的 `lmbda` 还会改 prefix 来源：`1` 才是纯学生 rollout；`0` 加 `seq_kd` 是老师轨迹；否则是 dataset。那是 [状态来源和监督密度](/wiki/on-policy-distillation/sft-rl-opd/#sampling-density) 的轴，不要和粒度混成一个开关。
- 大规模 teacher server 常见 `--n-logprobs 256`，仍是 top-k，不是 full-vocab。
- 自己跑时的粗超参（知乎建议，不是 TML 默认）：rollout top-p `0.8-0.95`、temperature `0.7-1.0`、top-k 做 `32/64/128` ablation；response 先走 3K-7K。
- 同模型 [OPSD](/wiki/on-policy-distillation/when-opd-works/#opsd)：style token 的 KL 可能高于任务 token，不要按 KL 大小无差别更新；需要 per-token clipping。原文：[nrehiew](https://nrehiew.github.io/blog/sft_rl_opd/)。

## See Also

- [On-Policy Distillation 问题地图](/wiki/on-policy-distillation/overview/)
- [SFT、RL 与 OPD](/wiki/on-policy-distillation/sft-rl-opd/)
- [老师信号何时可靠](/wiki/on-policy-distillation/when-opd-works/)
