---
title: "Harness Evolution 概览"
topic: harness-evolution
summary: "Harness Evolution 关注冻结模型如何通过技能库、context、workflow、harness code 与评估闭环积累外部能力。"
lang: zh-CN
updated: 2026-08-07
order: 1
redirect_from:
  - /wiki/skill-management/overview/
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
  - title: "Self-Harness: Harnesses That Improve Themselves"
    url: "https://arxiv.org/abs/2606.09498v1"
  - title: "Harness Engineering for Self-Improvement"
    url: "https://lilianweng.github.io/posts/2026-07-04-harness/"
raw:
  - raw/harness-evolution/2023-05-25-voyager.md
  - raw/harness-evolution/2026-03-01-autoskill.md
  - raw/harness-evolution/2026-05-07-skillos.md
  - raw/harness-evolution/2026-05-ratchet.md
  - raw/harness-evolution/2026-05-26-muse-autoskill.md
  - raw/harness-evolution/2026-01-29-meta-context-engineering.md
  - raw/harness-evolution/2026-03-30-meta-harness.md
  - raw/harness-evolution/2026-06-08-self-harness.md
  - raw/harness-evolution/2026-07-04-harness-engineering-for-self-improvement.md
---

## Overview

冻结 LLM 的长期能力不一定来自更新权重，也可以来自围绕模型的外部系统：技能库、结构化 context、workflow、tool/runtime 协议、文件系统记忆、评估器、权限边界，以及能被搜索或自我修订的 harness code。Harness Evolution 关心的不是“又出现了哪个系统”，而是这些外部资产如何让同一个模型在更长任务、更复杂反馈和更多历史经验中变强。

当前 topic 的主骨架是问题地图；系统页只作为证据锚点。`MUSE-Autoskill`、`MCE`、`Meta-Harness`、`Self-Harness` 等页面保留，是因为它们会被多个问题反复引用，而不是因为每篇来源都应该变成系统页。Lilian Weng 的综述把这些线索整理成更大的 RSI 近端路径：先优化模型外部的运行系统，再让更强的运行系统支持更好的研究、训练或部署循环。

## 问题地图

### 技能资产长什么样

技能是 harness evolution 的一个资产层，而不是唯一对象。Voyager 把技能做成可执行代码；AutoSkill 把对话经验抽成 `SKILL.md`；MUSE-Autoskill 采用带 `tests/`、脚本和 `.memory.md` 的 Agent Skills 包；SkillOS 把外部 SkillRepo 交给 curator 维护。关键问题是：资产要保留多少程序性结构、多少自然语言规则，以及是否能被独立测试和迁移。详见 [Skill Library](/wiki/harness-evolution/skill-library/)。

### 库如何不漂坏

库增长本身会制造噪声、冲突、重复与过时条目。AutoSkill 给出 add / merge / discard 与版本化合并；MUSE 把创建、评估、更新、合并和裁剪放进同一运行时；Ratchet 把瓶颈明确为 librarian / hygiene，并强调 retirement、active cap 与 meta-skill authoring prior。详见 [技能生命周期](/wiki/harness-evolution/skill-lifecycle/)。

### 策展能否被学习

启发式 librarian 只能表达人的先验。SkillOS 把 executor 冻结，只训练 skill curator，并用 grouped task streams 与 composite rewards 从延迟反馈中学习长期策展策略。这个方向把「如何改库」本身变成可优化策略，而不是固定规则。详见 [Skill Curation RL](/wiki/harness-evolution/skill-curation-rl/)。

### LLM 自写技能到底有没有用

这是本 topic 的主要争议。Ratchet 引用 SkillsBench 原文，把“LLM 自写技能无增益”解释为生命周期治理瓶颈；MUSE-Autoskill 在全生命周期 agent 中报告自建技能仍有正增益，并认为瓶颈更多在生成覆盖率。这个分歧不能靠单一分数裁决，必须保留协议、agent 栈、覆盖口径和技能治理差异。详见 [自写技能有效性争议](/wiki/harness-evolution/self-authored-skills/)。

### Harness 到底优化什么

Weng 把 harness optimization 的对象序列概括为 instruction prompts → structured context → workflow → harness code → optimizer code。MCE 演化 CE skill，Meta-Harness 用外部 coding agent 搜索任务侧 harness 程序，Self-Harness 让目标模型依据自身失败轨迹提出 bounded harness edits；三者都不是 task skill librarian，而是把“如何运行模型”本身变成可评估、可修改的对象。详见 [Harness Engineering](/wiki/harness-evolution/harness-engineering/)、[Meta Context Engineering](/wiki/harness-evolution/meta-context-engineering/)、[Meta-Harness](/wiki/harness-evolution/meta-harness/) 与 [Self-Harness](/wiki/harness-evolution/self-harness/)。

### 评估与权限边界如何不被优化器破坏

自我改进 loop 优化的永远是给定信号。若信号来自单元测试、judge model 或公开 benchmark，系统可能学到 test overfitting、judge hacking 或 benchmark artifact。Harness evolution 因此不能只看 pass rate；评估器、trace audit、held-out tests、权限控制和 human review 需要在关键决策点留在可演化 loop 外部。

## 共同主张与边界

共同主张：能力增长可以发生在外部资产或外部 harness 上，而不必先更新权重。真正的分歧在于资产边界与反馈回路：技能是代码、Markdown、带测试的包，还是完整 harness；维护靠启发式、单测、RL curator、外部 coding agent 搜索，还是目标模型自提 harness edit；评估看任务成功、覆盖率、迁移性、搜索/上下文成本、held-out 非回归，还是长期可维护性。

本 topic 暂时把“外部能力演化”作为共同边界：任务技能库、context engineering 与 harness code search 都属于它；直接改模型权重、纯 self-play 训练或通用 continual learning 只在与 harness loop 联合优化时进入本 topic。

## See Also

- [Skill Library](/wiki/harness-evolution/skill-library/)
- [技能生命周期](/wiki/harness-evolution/skill-lifecycle/)
- [Skill Curation RL](/wiki/harness-evolution/skill-curation-rl/)
- [自写技能有效性争议](/wiki/harness-evolution/self-authored-skills/)
- [MUSE-Autoskill](/wiki/harness-evolution/muse-autoskill/)
- [Harness Engineering](/wiki/harness-evolution/harness-engineering/)
- [Meta Context Engineering](/wiki/harness-evolution/meta-context-engineering/)
- [Meta-Harness](/wiki/harness-evolution/meta-harness/)
- [Self-Harness](/wiki/harness-evolution/self-harness/)
