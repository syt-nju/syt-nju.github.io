# Reading GSPO: routing replay, sequence-level clipping, and why the objective changes

> Source: https://zhuanlan.zhihu.com/p/1933217003654586554
> Author: Yuteng Shen
> Collected: 2026-07-30
> Published: Unknown
> License: Copyright Yuteng Shen
> Completeness: Partial
> Retrieval note: Zhihu blocked automated retrieval. The text below is the complete description preserved on this site's Writing page.

This post is a reading note on GSPO and its relation to GRPO and DAPO. I focus on why MoE training is sensitive to routing replay, how token-level clipping can create unstable updates when routing changes, and why GSPO moves clipping toward the sequence level. I also discuss a broader intuition: sequence-level clipping may reduce the built-in preference for reinforcing already high-probability tokens, which could make exploration paths easier to preserve during RLVR training.
