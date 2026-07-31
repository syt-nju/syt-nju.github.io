---
title: "Understanding GSPO from the Objective Level"
topic: agent-training
summary: "GSPO moves clipping to the sequence level to reduce instability caused by routing changes in MoE training."
lang: en
source_note: "当前示例页基于 Writing 页面中的完整摘要整理；知乎原文阻止了自动抓取。"
updated: 2026-07-30
order: 2
sources:
  - title: "Reading GSPO: routing replay, sequence-level clipping, and why the objective changes"
    url: "https://zhuanlan.zhihu.com/p/1933217003654586554"
raw:
  - raw/agent-training/gspo-objective.md
---

## Overview

GSPO changes the clipping granularity used by GRPO-style training. The motivation is tied to mixture-of-experts routing: when routing decisions change, token-level updates can become unstable even when the sequence as a whole remains a useful training signal.

## Routing Replay in MoE Training

MoE models route tokens through experts, which makes training sensitive to routing replay. When routing changes, token-level clipping can produce unstable updates.

## From Token-Level to Sequence-Level Clipping

GSPO moves clipping toward the sequence level. Treating the sequence as the unit of control reduces the influence of isolated token-level changes caused by routing differences.

One broader intuition is that sequence-level clipping may weaken the tendency to reinforce tokens that already have high probability. Preserving lower-probability paths could make exploration easier to retain during RLVR training, although this remains an interpretation rather than a demonstrated conclusion in the available source summary.

## See Also

- [Why Online RFT Falls Short of RLVR](/wiki/agent-training/online-rft-vs-rlvr/)
