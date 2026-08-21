---
title: "On-Policy Distillation"
topic: on-policy-distillation
summary: "OPD 是学生自采样轨迹，老师在这些前缀上做 per-token reverse KL；最便宜的是只比较采到的 token，扩到 top-k 或全词表主要贵在 V 维分布的显存。训练上没有统一配方。"
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

## OPD 在优化什么 {#what-is-opd}

### On-Policy Distillation {#opd}

OPD 的轨迹来自学生当前策略 $\pi_\theta$。老师只在学生已经生成的前缀 $c_t=(x,y_{<t})$ 上给下一步的分布监督。

和 SFT 的差别是数据从哪来：SFT 的序列是老师写的。和常见 outcome RL 的差别是监督有多密：那里整条轨迹只有一个对错标。

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

reverse KL 是 mode-seeking：只在学生 rollout 里实际出现的前缀上，匹配老师下一步分布的众数。学生 support 里没有的 token 没有有效梯度，所以常见做法是先 SFT、再 OPD。折扣取 0，只看当前下一步。

## sampled-token v.s. RL 正则项 {#sampled-token-cost}

### sampled-token {#sampled-token}

每个位置只比较学生实际采到的那个 token 的 logprob。老师对这条轨迹做一次 forward，取出 $\log\pi_{\mathrm{teacher}}(y_t)$。实现上常设 $A_t=-\mathrm{KL}$，走现成的 importance-sampling 更新。带 KL regularization 的 RL 脚本往往只把 regularizer 模型换成 teacher：多一个模型，对学生已采 token 做 `compute_logprobs`。

计算路径和 RL 的 KL 正则相同；角色不同。RL 里这项把 $\pi_\theta$ 约束在 $\pi_{\mathrm{ref}}$ 附近；OPD 里它是主监督，目标是更强的 teacher。

### top-k {#top-k}

在老师或学生概率最高的 $K$ 个 token 上比较截断后的分布，通常在这个集合内重归一化。

### full-vocab {#full-vocab}

对整个词表做 KL。支持集变大，并不等于前面层反传变重。三者 backbone FLOPs 几乎一样。sampled-token 的 $\partial\log\pi(y)/\partial z$ 和 full-vocab reverse KL 的 $\partial\mathrm{KL}/\partial z$ 都是 $V$ 维，再乘同一个 $W_{\mathrm{lm}}$，前面层看到的上游梯度一直是 $[B,T,d]$。多出来的算术几乎只在最后一层附近对 $\pi_T$ 做一次 $V$ 维计算，相对 LM head GEMM 约是 $1/d$。

真正增加的是显存和访存：要不要把完整 $V$ 维老师分布写到显存、按位置对齐后算 loss。可以不缓存完整 logits，只留 last-layer hidden，算 loss 时再乘 head。

## top-p v.s. top-k {#knobs}

### top-p {#top-p-rollout}

top-p 约束的是 rollout 怎么采，不改每个位置 loss 比哪些 token。温度保持 1，用 nucleus sampling 丢掉低概率尾巴，避免采到极低概率 token，否则后续前缀上老师的 logprob 已经不可信。目的是 rollout 仍是典型续写，不是升温探索。只改 loss 支持集、不用 top-p，消融里比 sampled-token 更差。

[top-k](#top-k) 管每个前缀的 loss 在哪几个 token 上比。$K$ 是固定超参，不是按当步 overlap 动态选。主方法用老师的 $\mathrm{TopK}_q(c_t)$，师生在这个集合上重归一化再算截断 reverse KL。学生 top-$K$、以及老师 top-$K$ 并上学生 sampled token，都能用。关键是不要只比一个 token，不是必须用老师的 $K$。overlap 上的概率质量占比是事后诊断，不是选 $K$ 的算法。

## sampled-token / top-k / full-vocab 训练上怎么选 {#estimator-choice}

没有统一配方。现有结论在「sampled-token 够不够」上不一致。

观点不一致。一种结论是：师生 top-k 高度重合、共享 token 已占绝大部分概率质量时，sampled-token 与 $k\in\{4,16,64\}$ 的 top-k 下游接近；明显更差的是 Top-1，不是「只看一个按 $\pi_\theta$ 抽出的 token」。再加大 $k$ 收益很小。

另一种结论是：轨迹较长、前缀容易离开老师 support、tokenizer 或特殊 token 干扰单点比较时，sampled-token 监督偏斜、不稳定。改成 top-$K$ 截断 reverse KL，再加上 top-p rollout，梯度 norm 更低。其中不少增益来自 special-token mask，不完全是 estimator 本身。截断 reverse KL 是 surrogate，不等于 full-vocab。

可以先按设定选：

- 师生同族、学生轨迹已经在老师 support 里、top-k 重合高：默认 sampled-token。不要用 Top-1。
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
