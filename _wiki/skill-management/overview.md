---
title: "技能管理概览"
topic: skill-management
summary: "技能管理关注冻结 LLM 如何把经验外置为可检索、可组合、可治理的资产，以及 CE/harness 元优化与任务技能库的边界。"
lang: zh-CN
updated: 2026-08-07
order: 1
sources:
  - title: "Voyager: An Open-Ended Embodied Agent with Large Language Models"
    url: "https://arxiv.org/abs/2305.16291"
  - title: "AutoSkill: Experience-Driven Lifelong Learning via Skill Self-Evolution"
    url: "https://arxiv.org/abs/2603.01145"
  - title: "SkillOS: Learning Skill Curation for Self-Evolving Agents"
    url: "https://arxiv.org/abs/2605.06614"
  - title: "Ratchet: A Minimal Hygiene Recipe for Self-Evolving LLM Agents"
    url: "https://arxiv.org/abs/2605.22148"
  - title: "MUSE-Autoskill: Self-Evolving Agents via Skill Creation, Memory, Management, and Evaluation"
    url: "https://arxiv.org/abs/2605.27366"
  - title: "Meta Context Engineering via Agentic Skill Evolution"
    url: "https://arxiv.org/abs/2601.21557"
  - title: "Meta-Harness: End-to-End Optimization of Model Harnesses"
    url: "https://arxiv.org/abs/2603.28052"
raw:
  - raw/skill-management/2023-05-25-voyager.md
  - raw/skill-management/2026-03-01-autoskill.md
  - raw/skill-management/2026-05-07-skillos.md
  - raw/skill-management/2026-05-ratchet.md
  - raw/skill-management/2026-05-26-muse-autoskill.md
  - raw/skill-management/2026-01-29-meta-context-engineering.md
  - raw/skill-management/2026-03-30-meta-harness.md
---

## Overview

冻结 LLM 的长期能力不一定来自更新权重，也可以来自外部资产：代码技能、自然语言规则、Agent Skills 包、可训练的 SkillRepo，或更靠近 harness 的上下文工程程序。技能管理关心的不是“又出现了哪个系统”，而是这些资产如何表示、如何被检索和组合、谁负责策展、如何避免库漂坏，以及哪些能力其实已经越过了任务技能库边界，进入 CE / harness 元优化。

当前 topic 的主骨架是问题地图；系统页只作为证据锚点。`MUSE-Autoskill`、`MCE`、`Meta-Harness` 等页面保留，是因为它们会被多个问题反复引用，而不是因为每篇来源都应该变成系统页。

## 问题地图

### 技能资产长什么样

技能不是一种固定格式。Voyager 把技能做成可执行代码；AutoSkill 把对话经验抽成 `SKILL.md`；MUSE-Autoskill 采用带 `tests/`、脚本和 `.memory.md` 的 Agent Skills 包；SkillOS 把外部 SkillRepo 交给 curator 维护。关键问题是：资产要保留多少程序性结构、多少自然语言规则，以及是否能被独立测试和迁移。详见 [Skill Library](/wiki/skill-management/skill-library/)。

### 库如何不漂坏

库增长本身会制造噪声、冲突、重复与过时条目。AutoSkill 给出 add / merge / discard 与版本化合并；MUSE 把创建、评估、更新、合并和裁剪放进同一运行时；Ratchet 把瓶颈明确为 librarian / hygiene，并强调 retirement、active cap 与 meta-skill authoring prior。详见 [技能生命周期](/wiki/skill-management/skill-lifecycle/)。

### 策展能否被学习

启发式 librarian 只能表达人的先验。SkillOS 把 executor 冻结，只训练 skill curator，并用 grouped task streams 与 composite rewards 从延迟反馈中学习长期策展策略。这个方向把「如何改库」本身变成可优化策略，而不是固定规则。详见 [Skill Curation RL](/wiki/skill-management/skill-curation-rl/)。

### LLM 自写技能到底有没有用

这是本 topic 的主要争议。Ratchet 引用 SkillsBench 原文，把“LLM 自写技能无增益”解释为生命周期治理瓶颈；MUSE-Autoskill 在全生命周期 agent 中报告自建技能仍有正增益，并认为瓶颈更多在生成覆盖率。这个分歧不能靠单一分数裁决，必须保留协议、agent 栈、覆盖口径和技能治理差异。详见 [自写技能有效性争议](/wiki/skill-management/self-authored-skills/)。

### CE / harness 元优化是否属于本主题

MCE 和 Meta-Harness 已经不只是“任务技能库怎么管”，而是在优化 context engineering 或完整 harness 本身。它们仍放在本 topic，是因为都共享一个更高层主张：冻结模型的能力可以通过外部可检视资产增长。但阅读时应把它们当作**边界问题**：MCE 演化 CE skill，Meta-Harness 搜索任务侧 harness 程序；二者不是 task skill librarian。详见 [Meta Context Engineering](/wiki/skill-management/meta-context-engineering/) 与 [Meta-Harness](/wiki/skill-management/meta-harness/)。

## 共同主张与边界

共同主张：能力增长可以发生在外部技能资产或外部 harness 上，而不必更新权重。真正的分歧在于资产边界与反馈回路：技能是代码、Markdown、带测试的包，还是完整 harness；维护靠启发式、单测、RL curator，还是 coding agent 搜索；评估看任务成功、覆盖率、迁移性，还是搜索/上下文成本。

本 topic 暂时把 CE / harness 元优化作为相邻边界保留。若后续来源主要围绕 harness search、context optimization 或 workflow search 增长，应考虑拆出独立 topic，而不是继续扩张“技能管理”的含义。

## See Also

- [Skill Library](/wiki/skill-management/skill-library/)
- [技能生命周期](/wiki/skill-management/skill-lifecycle/)
- [Skill Curation RL](/wiki/skill-management/skill-curation-rl/)
- [自写技能有效性争议](/wiki/skill-management/self-authored-skills/)
- [MUSE-Autoskill](/wiki/skill-management/muse-autoskill/)
- [Meta Context Engineering](/wiki/skill-management/meta-context-engineering/)
- [Meta-Harness](/wiki/skill-management/meta-harness/)
