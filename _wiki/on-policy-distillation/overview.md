---
title: "On-Policy Distillation"
topic: on-policy-distillation
summary: "OPD 是学生自采样轨迹，老师在这些前缀上做 per-token reverse KL；最便宜的是只比较采到的 token，扩到 top-k 或全词表主要贵在 V 维分布的显存。clip 不是这条损失。训练上没有统一配方。"
lang: zh-CN
updated: 2026-08-18
order: 1
sources:
  - title: "On-Policy Distillation"
    url: "https://thinkingmachines.ai/blog/on-policy-distillation/"
  - title: "Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes"
    url: "https://arxiv.org/abs/2603.25562"
  - title: "Rethinking On-Policy Distillation of Large Language Models"
    url: "https://arxiv.org/abs/2604.13016"
  - title: "OPD深度解析：从数学推导到DeepSeek V4、SWIFT与verl实践"
    url: "https://zhuanlan.zhihu.com/p/2033212181823608430"
raw:
  - raw/on-policy-distillation/2025-10-27-on-policy-distillation.md
  - raw/on-policy-distillation/2026-03-26-revisiting-on-policy-distillation.md
  - raw/on-policy-distillation/2026-04-14-rethinking-on-policy-distillation.md
  - raw/on-policy-distillation/zhihu-opd-deep-dive-v2.md
---

## Overview

[OPD](#opd) 是：学生用当前策略采样轨迹，老师只在这些前缀上给 token 级监督。基本损失是每一步的 [reverse KL](#reverse-kl)。最便宜的实现是 [sampled-token](#sampled-token)：只比较实际采到的那个 token。扩到 [top-k](#top-k) 或 [full-vocab](#full-vocab)，前面层的反传形状基本不变，多出来的是 $V$ 维老师分布的显存和访存。[clip](#clipfrac) 不是这条蒸馏损失里的项。训练上没有统一配方：师生 top-k token 大量重合、高概率质量接近时，sampled-token 往往够用；rollout / 轨迹变长、前缀偏离老师 support 时，更常改成在局部支持集上比分布。

## OPD 在优化什么 {#what-is-opd}

容易和两条邻居混：SFT 学的是老师写好的序列；常见 RL 则整条轨迹只拿一个对错分。

### On-Policy Distillation {#opd}

轨迹来自学生自己的当前策略 $\pi_\theta$。老师只在学生走到的前缀 $c_t=(x,y_{<t})$ 上提供下一步监督。

基本损失是 per-token [reverse KL](#reverse-kl)：

$$
\mathrm{KL}\bigl(\pi_\theta \Vert \pi_{\mathrm{teacher}}\bigr)
=
\mathbb{E}_{x\sim\pi_\theta}
\Bigl[
\log\pi_\theta(x_{t+1}\mid x_{1:t})
-
\log\pi_{\mathrm{teacher}}(x_{t+1}\mid x_{1:t})
\Bigr]
$$

### reverse KL {#reverse-kl}

reverse KL 是 mode-seeking：只在学生当前会走到的前缀上，匹配老师下一步分布的众数。学生 support 里没有老师那些 token 时，没有有效梯度。所以常见做法是先 SFT、再 OPD。折扣取 0，只看当前下一步。

## sampled-token 成本为什么像 RL 正则 {#sampled-token-cost}

容易把「算力结构和 RL 的 KL 正则一样」听成「作用也是正则」。结构和角色要分开看。

### sampled-token {#sampled-token}

每个位置只比较学生实际采到的那个 token 的 logprob。老师对这条轨迹做一次 forward，取出 $\log\pi_{\mathrm{teacher}}(y_t)$。实现上常设 $A_t=-\mathrm{KL}$，走现成的 importance-sampling 更新。在带 KL regularization 的 RL 脚本上，往往只把 regularizer 模型换成 teacher。

算力结构相同：多一个模型，对学生已采 token 做 `compute_logprobs`。角色相反：RL 里这项把 $\pi_\theta$ 约束在 $\pi_{\mathrm{ref}}$ 附近；OPD 里它是主监督，目标是更强的 teacher。

### top-k {#top-k}

在老师或学生概率最高的 $K$ 个 token 上比较截断后的分布，通常在这个集合内重归一化。

### full-vocab {#full-vocab}

对整个词表做 KL。容易以为支持集变大，前面层反传也会变重。三者 backbone FLOPs 几乎一样。sampled-token 的 $\partial\log\pi(y)/\partial z$ 和 full-vocab reverse KL 的 $\partial\mathrm{KL}/\partial z$ 都是 $V$ 维，再乘同一个 $W_{\mathrm{lm}}$，前面层看到的上游梯度一直是 $[B,T,d]$。多出来的算术几乎只在最后一层附近对 $\pi_T$ 做一次 $V$ 维计算，相对 LM head GEMM 约是 $1/d$。

真正增加的是显存和访存：要不要把完整 $V$ 维老师分布写到显存、按位置对齐后算 loss。可以不缓存完整 logits，只留 last-layer hidden，算 loss 时再乘 head。

## top-p、top-k、clip 分别管什么 {#knobs}

容易把三个量当成一件事。它们分别管采样、loss 支持集、外层 RL 的 IS 截断统计。

### top-p {#top-p-rollout}

只管采样，不管 loss。温度保持 1，用 nucleus sampling 丢掉低概率尾巴，避免采到极低概率 token，让后续前缀落到老师 logprob 已经不可信的区域。目的是让 rollout 留在典型续写上，不是升温探索。只改 loss 支持集、不用 top-p，消融里比 sampled-token 更差。

[top-k](#top-k) 管每个前缀的 loss 在哪几个 token 上比。容易以为 $K$ 必须取老师的、或按当步 overlap 动态调。$K$ 是固定超参。主方法用老师的 $\mathrm{TopK}_q(c_t)$，师生在这个集合上重归一化再算截断 reverse KL。学生 top-$K$、以及老师 top-$K$ 并上学生 sampled token，都能用。关键是不要只比一个 token，不是必须用老师的 $K$。overlap 上的概率质量占比是事后诊断，不是选 $K$ 的算法。

### clip {#clipfrac}

名字像损失里的 clip。它不是 OPD 公式里的项。clipping-boundary fraction 是外层 RL 管线里 $r_t=\pi_\theta/\pi_{\mathrm{old}}$ 打到 PPO/GRPO 截断边的比例。蒸馏公式没有 $\epsilon$。若实现走 $A_t=-\mathrm{KL}$ 再做 IS，clip 会挡住一部分更新；若直接对 KL 反传，这层 clip 往往不存在。更低的 clipfrac 只是「这次更新更少撞上截断边」的诊断，不是又加了一个蒸馏损失项。

## sampled-token / top-k / full-vocab 训练上怎么选 {#estimator-choice}

没有统一配方。现有结论在「sampled-token 够不够」上不一致。

一边是：师生 top-k token 高度重合、共享 token 已占绝大部分概率质量时，sampled-token 与 $k\in\{4,16,64\}$ 的 top-k 下游接近；明显更差的是 Top-1（argmax），不是「只看一个按 $\pi_\theta$ 抽出的 token」。再把 $k$ 加大收益很小。

另一边是：rollout / 轨迹较长、前缀容易偏离老师 support、tokenizer 切分或特殊 token 干扰单点比较时，sampled-token 的监督偏斜、不稳定。改成 top-$K$ 截断 reverse KL，再加上 top-p rollout，梯度 norm 和 clipfrac 更低。其中不少增益来自 special-token mask，不完全是 estimator 本身。截断 reverse KL 是 surrogate，不等于 full-vocab。

可以先按设定选：

- 师生同族、学生轨迹已经落在老师 support 里、top-k 重合高：默认 sampled-token。不要用 Top-1。
- 轨迹变长、agent、tokenizer 切分不一致或特殊 token 多：top-k 截断 KL + 集合内重归一化 + top-p rollout，必要时再 mask 特殊 token。$K$ 不用很大。
- full-vocab：显存、通信和 kernel 都准备好、又在意方差时才值得。同一设定下三档还没有头对头比较。

## Open Questions

- sampled-token 何时会不稳定：是师生 top-k overlap 起不来，还是 tokenizer / 特殊 token 把单点 KL 比较扭曲了。
- 同一设定下 full-vocab 是否稳定地优于 top-k。

## Sources

- [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
- [Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes](https://arxiv.org/abs/2603.25562)
- [Rethinking On-Policy Distillation of Large Language Models](https://arxiv.org/abs/2604.13016)
- [OPD深度解析：从数学推导到DeepSeek V4、SWIFT与verl实践](https://zhuanlan.zhihu.com/p/2033212181823608430)
