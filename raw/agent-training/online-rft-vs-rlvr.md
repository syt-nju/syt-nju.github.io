# Why online RFT falls short of RLVR: negative samples are the key

> Source: https://zhuanlan.zhihu.com/p/1941603697311342965
> Author: Yuteng Shen
> Collected: 2026-07-30
> Published: Unknown
> License: Copyright Yuteng Shen
> Completeness: Partial
> Retrieval note: Zhihu blocked automated retrieval. The text below is the complete description preserved on this site's Writing page.

This post starts from an online RFT experiment built on top of a modified DAPO pipeline and asks a practical question: if we keep amplifying hard cases and positive trajectories, can pass@1 approach pass@k without full RL-style exploration? The conclusion is no. Pure RFT quickly memorizes successful trajectories and reaches high training reward, but its evaluation reward falls behind. My main takeaway is that RL's exploration and generalization are both tightly connected to negative samples, because they keep correcting unstable reasoning paths instead of letting the model profit from lucky guesses and hallucinated successes.
