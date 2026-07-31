---
title: "Why Online RFT Falls Short of RLVR"
topic: agent-training
summary: "Negative samples help preserve exploration and correct unstable reasoning paths that positive-only RFT can reinforce."
lang: en
source_note: "当前示例页基于 Writing 页面中的完整摘要整理；知乎原文阻止了自动抓取。"
updated: 2026-07-30
order: 1
sources:
  - title: "Why online RFT falls short of RLVR: negative samples are the key"
    url: "https://zhuanlan.zhihu.com/p/1941603697311342965"
raw:
  - raw/agent-training/online-rft-vs-rlvr.md
---

## Overview

Online reinforcement fine-tuning can amplify hard cases and successful trajectories, but this alone does not reproduce the exploration behavior of reinforcement learning with verifiable rewards. In the described experiment, pure RFT learned successful trajectories quickly while its evaluation reward remained behind.

## The Positive-Only Limitation

The experiment asks whether repeatedly training on hard cases and positive trajectories can make pass@1 approach pass@k without full reinforcement-learning exploration. Its result suggests that successful trajectories alone are insufficient: the model can memorize paths that worked during training without learning how to recover from unstable reasoning.

## Why Negative Samples Matter

Negative samples provide corrective pressure on paths that fail, rely on lucky guesses, or contain hallucinated intermediate reasoning. This connects exploration and generalization: the model is not only rewarded for known successes but also trained away from unreliable alternatives.

The practical implication is that high training reward under pure RFT should not be treated as evidence that evaluation behavior has caught up with RLVR.

## See Also

- [Understanding GSPO from the Objective Level](/wiki/agent-training/gspo-objective/)
