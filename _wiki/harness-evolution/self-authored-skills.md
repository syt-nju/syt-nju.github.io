---
title: "自写技能有效性争议"
topic: harness-evolution
summary: "Ratchet 将 SkillsBench 的 LLM 自写技能 null result 解释为 librarian 问题；MUSE-Autoskill 则显示全生命周期 agent 下自建技能仍有正增益。"
lang: zh-CN
updated: 2026-08-07
order: 9
redirect_from:
  - /wiki/skill-management/self-authored-skills/
sources:
  - title: "Ratchet: A Minimal Hygiene Recipe for Self-Evolving LLM Agents"
    url: "https://arxiv.org/abs/2605.22148"
  - title: "MUSE-Autoskill: Self-Evolving Agents via Skill Creation, Memory, Management, and Evaluation"
    url: "https://arxiv.org/abs/2605.27366"
raw:
  - raw/harness-evolution/2026-05-ratchet.md
  - raw/harness-evolution/2026-05-26-muse-autoskill.md
---

## Overview

“LLM 能不能自己写出有用技能”不是一个单一 yes/no 问题。Ratchet 和 MUSE-Autoskill 给出的结论看似冲突，实际比较的是不同系统栈、不同生命周期治理和不同计分口径下的自写技能。

本页把争议保留下来：Ratchet 侧强调 SkillsBench 的 null result 说明瓶颈在 librarian；MUSE-Autoskill 侧强调当创建、记忆、管理、评估与精炼放进同一运行时后，自建技能仍能带来正增益。

## Ratchet 的解释

Ratchet 引用 SkillsBench 原结果：human-curated skills deliver +16.2pp，LLM-self-generated skills deliver +0.0pp。作者据此认为，问题不在“LLM 不会写技能”，而在技能库缺少生命周期治理：近重复失败标签未归一，弱技能不退役，活跃集合无上限，作者先验不稳定。

因此 Ratchet 的处方不是训练 curator，而是在冻结 LLM 单 agent 回路中加入 hygiene：pattern canonicalisation、outcome-driven retirement、bounded active-cap、meta-skill authoring prior。其主张是：自写技能若没有 librarian，会被噪声、重复和过时条目拖垮。

## MUSE-Autoskill 的反例

MUSE-Autoskill 在四个 GPT-5.5-backed agent 的 SkillsBench common set 上重新评估技能创建与使用。MUSE 无技能准确率为 46.95%，人写技能为 59.67%，自建技能 all-task 口径为 53.42%。这说明自建技能仍低于人写技能，但不是零增益。

MUSE 进一步报告：在成功覆盖的子集上，自建技能为 85.24%，同子集人写技能为 81.17%。作者将主要瓶颈解释为生成覆盖率，而不是已生成技能完全无效。跨 agent 迁移中，MUSE 自建技能注入 Hermes 后达到 51.90%，高于 Hermes 人写技能的 48.02%。

## 如何并置这两个结论

两边不应被写成谁完全推翻谁：

- Ratchet 讨论的是朴素自写技能在 SkillsBench 原语境中的 null result，并把重点放在 librarian / hygiene。
- MUSE 讨论的是带运行时创建、skill-level memory、单测评估、更新与裁剪的完整 agent，在严格 all-task 口径下仍有正增益，但覆盖失败仍被计入分母。
- 双方都承认技能质量和生命周期治理有关；差别在于 Ratchet 把治理视为首要瓶颈，MUSE 进一步把覆盖率、评估门控和跨 agent 迁移纳入系统。

因此本 topic 的中性表述应是：**自写技能是否有效，取决于创建协议、覆盖率、评估门控与库治理；不能只用“LLM 自写 +0.0pp”或“覆盖子集超过人写”单独裁决。**

## See Also

- [Harness Evolution 概览](/wiki/harness-evolution/overview/)
- [技能生命周期](/wiki/harness-evolution/skill-lifecycle/)
- [MUSE-Autoskill](/wiki/harness-evolution/muse-autoskill/)
