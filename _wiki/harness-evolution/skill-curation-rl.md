---
title: "Skill Curation RL"
topic: harness-evolution
summary: "SkillOS 把冻结 executor 与可训练 curator 拆开，用 grouped task streams 与 composite rewards 从延迟反馈中学习长期技能策展策略。"
lang: zh-CN
updated: 2026-08-04
order: 4
redirect_from:
  - /wiki/skill-management/skill-curation-rl/
sources:
  - title: "SkillOS: Learning Skill Curation for Self-Evolving Agents"
    url: "https://arxiv.org/abs/2605.06614"
raw:
  - raw/harness-evolution/2026-05-07-skillos.md
---

## Overview

启发式技能操作与短视改编，很难从间接、延迟的执行反馈里学到长期策展策略。SkillOS 的处方是：不要端到端改写整个 agent，而是只训练 **skill curator**，让它维护外部 SkillRepo；**agent executor** 冻结，只负责按库检索与执行任务。

## 架构

- **Executor 𝜋L**：冻结；给定任务与检索到的技能完成求解。
- **Curator 𝜋S**：可训练（实现上以 Qwen3-8B 为基座，GRPO）；根据累积经验更新 SkillRepo（Markdown 技能文件）。
- **Grouped task streams**：按技能相关依赖组流式任务——先出现的轨迹更新库，后出现的相关任务评估这些更新，从而给策展决策提供可学习信号。
- **Composite rewards**：把下游 executor 的成效更准确地归因到策展动作，而不是只看即时一步。

推理训练数据从 DeepMath-103k 抽样 33,000 条；agentic 评测在 ALFWorld 与 WebShop，推理评测含 AIME24、AIME25、GPQA-Diamond。

## 主要证据

以 Qwen3-8B 为 curator 时：

- ALFWorld、Qwen3-8B executor：SkillOS 平均 SR 61.2，强记忆基线 ReasoningBank 为 55.7；相对 No Memory 的 47.9 提升更大，步数也更低（18.9 vs 基线更高步数档）。
- 同一 curator 可接到更强 executor：Gemini-2.5-Pro 在 ALFWorld 上从 66.4 到 80.2。
- WebShop 与推理平均准确率同样抬升；例如 Qwen3-8B executor 上推理 Avg. Acc 到 73.8，Qwen3-32B executor 上到 79.7，Gemini-2.5-Pro executor 上到 88.6。
- 直接用 Gemini-2.5-Pro 当 curator（SkillOS-gemini）往往不如训练过的 8B curator，说明「更强生成 ≠ 更好策展」，需要与 executor 能力对齐的策展行为。

分析还报告：学到的 curator 技能使用更有针对性；SkillRepo 随时间形成更丰富结构与更高层 meta-skills。

## 在主题中的位置

相对 AutoSkill 的训练无关 merge 规则，以及 Ratchet 的固定 hygiene 配方，SkillOS 把「如何改库」本身当成策略学习问题。代价是需要可分组的任务流与可归因奖励；收益是 curator 可跨 executor / 域迁移，而不必为每个执行骨干重训整条 agent。

## See Also

- [Harness Evolution 概览](/wiki/harness-evolution/overview/)
- [Skill Library](/wiki/harness-evolution/skill-library/)
- [技能生命周期](/wiki/harness-evolution/skill-lifecycle/)
- [MUSE-Autoskill](/wiki/harness-evolution/muse-autoskill/)（training-free 全生命周期对照）
