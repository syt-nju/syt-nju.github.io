---
title: "On-Policy Distillation"
topic: on-policy-distillation
summary: "OPD 是在学生自采样前缀上用 reverse KL 把学生分布拉向老师；最便宜的是 sampled-token，top-k 与全词表主要贵在显存和搬运，训练上没有统一配方。"
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

[OPD](#opd) 的操作定义是：学生用当前策略采样轨迹，老师只在这些学生前缀上给 token 级监督，把 $\pi_\theta$ 往 $\pi_{\mathrm{teacher}}$ 拉。最基本、也最便宜的损失是 [sampled-token reverse KL](#sampled-token)。扩到 [top-k](#top-k) 或 [full-vocab](#full-vocab) 几乎不增加前面层的反传算力，贵的是 $V$ 维分布的物化和搬运。[clip](#clipfrac) 不是这条蒸馏损失的一部分。训练效果上没有统一配方：师生已经对上高概率区时，sampled-token 往往够用；长程一漂，才值得换成局部支持集。

## OPD 在优化什么 {#what-is-opd}

**On-Policy Distillation（OPD）** {#opd} 指：轨迹来自学生自己的当前策略，老师在学生走到的前缀 $c_t=(x,y_{<t})$ 上提供监督。它不是 SFT 那种学老师写好的序列，也不是整条轨迹只拿一个对错分的 RL。

最基本的损失是 per-token [reverse KL](#reverse-kl)。Thinking Machines Lab 写成：

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

**reverse KL** {#reverse-kl} 是 mode-seeking：在学生已经会走到的状态里追老师的模式。学生 support 里根本没有老师那些 token 时，这一项 steers 不动，所以他们默认先 SFT 再 OPD。折扣取 0，只看当前下一步。原文：[On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)。

## sampled-token 成本为什么像 RL 正则 {#sampled-token-cost}

**sampled-token** {#sampled-token} 每个位置只比较学生实际采到的那个 token。老师对这条轨迹做一次 forward，取出 $\log\pi_{\mathrm{teacher}}(y_t)$ 即可。实现上常把 $A_t=-\mathrm{KL}$ 丢进现成的 importance-sampling 更新。TML 的说法是：在带 KL regularization 的 RL 脚本上，往往只改一行——把 regularizer 模型换成 teacher。

算力形态像 RL 的 KL 正则：多一个模型，对学生已采 token 做 `compute_logprobs`。作用相反。RL 里这项通常拉住 $\pi_{\mathrm{ref}}$（多半是自己的起点）；OPD 里同一项是主信号，teacher 是更强的目标。

**top-k** {#top-k} 在老师（或学生）的局部支持集上比一小块分布，通常还要在集合内重归一化。**full-vocab** {#full-vocab} 对整个词表做 KL。三者 backbone FLOPs 几乎一样。sampled-token 的 $\partial\log\pi(y)/\partial z$ 和 full-vocab reverse KL 的 $\partial\mathrm{KL}/\partial z$ 都是 $V$ 维，再乘同一个 $W_{\mathrm{lm}}$，前面层看到的上游梯度一直是 $[B,T,d]$。多出来的算术几乎只在最后一层旁边扫 $\pi_T$，相对 LM head GEMM 约是 $1/d$。

真正涨的是显存和访存：要不要把 $V$ 维老师分布写出来、搬走、对齐着留下来。DeepSeek V4 能做 full-vocab，靠的是 teacher forward 不缓存完整 logits，只留 last-layer hidden，loss 时再乘 head。这条工程转述来自知乎对 V4 报告的整理，V4 原文尚未入库。

## top-p、top-k、clip 分别管什么 {#knobs}

这是三个旋钮，不要拧成一个。

**top-p 管采样，不管 loss。** {#top-p-rollout} Revisiting 默认温度不变，用 nucleus sampling 砍掉尾部，避免抽出极低概率 token、造出老师信号已经不可信的前缀。它是把 rollout 按回典型续写，不是升温去探索。只改 loss 支持集、不用 top-p，他们的消融里比 sampled-token 还差。

**top-k 管每个前缀的 loss 在哪几个 token 上比。** $K$ 是固定超参，不是按当步 overlap 动态调。主方法取 **teacher** 的 $\mathrm{TopK}_q(c_t)$，师生都在这个集合上重归一化再算截断 reverse KL。消融里 student top-$K$、以及 teacher top-$K$ 并上学生 sampled token，都能用；没有唯一赢家。关键是「别只比一个 token」，不是「必须用老师的 $K$」。Rethinking 里 overlap 上的质量占比是事后诊断，不是选 $K$ 的算法。

**clip 不是 OPD 损失里的项。** {#clipfrac} Revisiting 报告的 clipping-boundary fraction，是外层 RL 管线里 importance ratio $r_t=\pi_\theta/\pi_{\mathrm{old}}$ 打到 PPO/GRPO clip 边的比例。蒸馏公式本身没有 $\epsilon$。若实现走 TML 那种 $A_t=-\mathrm{KL}$ + IS，clip 会挡住一部分更新；若走直接对 KL 反传，这层 clip 往往不存在。他们拿更低的 clipfrac 当「更好训」的诊断，不是给 OPD 又加了一个损失项。

## sampled-token / top-k / full-vocab 训练上怎么选 {#estimator-choice}

没有统一标准做法。现有来源在「sampled-token 够不够」上直接分歧。

> **Status: Disputed**
> Rethinking 在师生已经能对上高概率 token 的设定里，认为 sampled-token 与 $k\in\{4,16,64\}$ 的 top-k 下游接近；明显更差的是 Top-1（argmax），不是「只看一个按 $\pi_\theta$ 抽出的 token」。共享 top-k 集合能攒住绝大部分概率质量，再把 $k$ 加大收益很小。
> Revisiting 在长程、前缀容易漂、tokenizer 会拧的设定里，认为 sampled-token 信号一边倒。他们的 top-$K$ 截断 reverse KL 加上 top-p rollout 后更稳，梯度 norm 和 clipfrac 更低。很大一块增益其实来自 special-token mask，不完全是 estimator。他们自己写：这是截断 surrogate，不等于 full-vocab。

可以按设定选，而不是认一个配方：

- 师生同族、学生已经会走到老师 support 里：sampled-token 做默认。不要用 Top-1。
- 长程、agent、tokenizer 容易拧：top-k 截断 KL + 集合内重归一化 + top-p rollout，必要时再 mask 特殊 token。$K$ 不用很大。
- full-vocab：只有显存、通信和 kernel 都准备好、又特别在意方差时才值得。没有已入库来源在同一设定里把头对头打过三档。

原文：[Rethinking On-Policy Distillation](https://arxiv.org/abs/2604.13016)，[Revisiting On-Policy Distillation](https://arxiv.org/abs/2603.25562)。

## Open Questions

- sampled-token 何时会训崩：是 overlap 涨不起来，还是 tokenizer / 特殊 token 把单点比较打歪。
- 同一设定下 full-vocab 是否稳定地赢 top-k。V4 选全词表，但报告本身尚未入库。

## Sources

- [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
- [Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes](https://arxiv.org/abs/2603.25562)
- [Rethinking On-Policy Distillation of Large Language Models](https://arxiv.org/abs/2604.13016)
- [OPD深度解析：从数学推导到DeepSeek V4、SWIFT与verl实践](https://zhuanlan.zhihu.com/p/2033212181823608430)
