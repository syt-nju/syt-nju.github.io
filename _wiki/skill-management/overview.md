---
title: "技能管理概览"
topic: skill-management
summary: "冻结 LLM 靠外部技能资产做终身学习：任务技能库之外，MCE 演化 CE 技能，Meta-Harness 再用全历史诊断搜索完整 harness 代码。"
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

开放世界里的 LLM agent 若只做单次规划，跨任务知识无法沉淀。一条持续出现的路线是：权重冻结，把可复用行为外置为技能资产，再在检索与治理上下功夫。本主题按问题组织来源增量：库如何长出来、技能长什么样、谁来策展、库如何不漂坏、创建—记忆—评估如何收成统一生命周期、如何把 *Context Engineering 策略本身* 也当成可演化技能，以及如何用 coding agent 外环直接搜索完整 harness 程序。

## 七条演进线

1. **Voyager（奠基）**：Minecraft 上的终身探索 agent。automatic curriculum + 可执行代码 skill library + iterative prompting（环境反馈 / 执行错误 / self-verification）。160 次 iteration 内发现 63 个独特物品，约为对照的 3.3 倍；路程约 2.3 倍；木制节点约快 15.3 倍。库默认可增长，几乎不做 outcome-driven 退役。
2. **AutoSkill（抽取与版本）**：从对话轨迹抽象 **SKILL.md**，模型无关 plug-in；online hybrid 检索注入，异步 judge 做 add / merge / discard 与 versioned merge。WildChat-1M（>8 turns）四子集共 N=1858 技能；English GPT-3.5 子集 10,243 对话 / 631 skills；案例 `professional_text_rewrite` 至版本 0.1.34。勿与下方 MUSE-Autoskill 混淆。
3. **MUSE-Autoskill（全生命周期 + 迁移）**：ByteDance 的训练无关框架，五阶段 creation / memory / management / evaluation / refinement；运行时 `skill_create`、`.memory.md` skill-level memory、单测门控注册。SkillsBench 75-task 上人写技能 59.67%（+12.72pp），自建 all-75 53.42%（+6.47pp），覆盖子集 85.24%（同子集人写 81.17%）；自建技能转入 Hermes 达 51.90%。详见 [MUSE-Autoskill](/wiki/skill-management/muse-autoskill/)。
4. **SkillOS（学策展）**：冻结 executor + 可训练 curator，对外部 SkillRepo 做 RL（GRPO）；用 grouped task streams 与 composite rewards 归因延迟反馈。ALFWorld 上相对 ReasoningBank，Qwen3-8B executor 的平均 SR 从 55.7 到 61.2；同一 curator 可泛化到更强 executor（如 Gemini-2.5-Pro 从 66.4 到 80.2）。
5. **Ratchet（hygiene / librarian）**：重述 SkillsBench 原文：人写技能 +16.2pp，LLM 自写 +0.0pp，瓶颈在生命周期而非写作。四类机制——pattern canonicalisation、outcome-driven retirement、bounded active-cap、meta-skill authoring prior。MBPP+ hard-100 上 rolling-mean gain +0.328，late-window 0.584（peak 0.658）；SWE-bench Verified 有 +0.22 peak lift。消融显示 retirement 与 meta-skill 承重，显式去重可被 meta-skill 吸收。
6. **MCE（演化 CE 技能）**：Meta Context Engineering 把 ACE / GEPA 等固定 CE harness 视为设计空间中的单点；meta-agent 用 **agentic crossover** 演化 CE skills，base-agent 把 context 建成 files/code。五域 offline Avg. Rel. Gain 89.1（ACE 70.7），online 74.1（ACE 41.1）；相对 SOTA 相对改进 5.6–53.8%（均值 16.9%）；context 长度约可在 1.5K–86K tokens 间按任务调节。详见 [Meta Context Engineering](/wiki/skill-management/meta-context-engineering/)。
7. **Meta-Harness（搜索完整 harness）**：外环 coding agent 经 filesystem 访问历次源码、分数与执行轨迹，直接改写任务侧 harness。在线文本分类（GPT-OSS-120B）平均准确率 48.6，相对 ACE 高 7.7 points、相对 MCE 高 8.6 points，context 11.4K；200 道 IMO 级题上检索 harness 五模型平均高 4.7 points；TerminalBench-2 上 Opus 76.4% / Haiku 37.6%。与 MCE 同属元优化，但不是双层 CE skill，而是最小外环 + 全历史诊断。详见 [Meta-Harness](/wiki/skill-management/meta-harness/)。

## 共同主张与分歧

共同主张：能力增长可以发生在外部技能资产或外部 harness 上，而不必更新权重。分歧在于「谁写技能、谁管库、反馈从哪来」——启发式 merge（AutoSkill）、完整 lifecycle + 单测（MUSE）、RL 策展（SkillOS）、冻结作者加 hygiene（Ratchet）、把 *学习 context 的算法* 也当作可演化技能（MCE），或用 coding agent 对完整 harness 做端到端代码搜索（Meta-Harness）。Voyager 证明库有用；后续路线分别补抽取标准、运行时生命周期、策展学习、防漂移、CE skill 元优化，以及 harness 级程序搜索。

> **Status: Disputed**（SkillsBench 上「LLM 自写技能」是否无效）
> Ratchet 以 SkillsBench 原论文的人写 +16.2pp / LLM 自写 +0.0pp 定调瓶颈在 librarian。MUSE-Autoskill 在 GPT-5.5 四 agent 设定下报告自建技能仍有 all-75 增益（+6.47pp），覆盖子集甚至超过人写；但 all-75 自建仍低于人写，且未覆盖任务计 0。争议应读作协议与系统栈差异，而非单一分数对决；细节见 [MUSE-Autoskill](/wiki/skill-management/muse-autoskill/) 与 [技能生命周期](/wiki/skill-management/skill-lifecycle/)。MCE / Meta-Harness 评测的是垂直域 CE 或 harness 搜索，不直接裁决该争议。

## See Also

- [Skill Library](/wiki/skill-management/skill-library/)
- [技能生命周期](/wiki/skill-management/skill-lifecycle/)
- [Skill Curation RL](/wiki/skill-management/skill-curation-rl/)
- [MUSE-Autoskill](/wiki/skill-management/muse-autoskill/)
- [Meta Context Engineering](/wiki/skill-management/meta-context-engineering/)
- [Meta-Harness](/wiki/skill-management/meta-harness/)
