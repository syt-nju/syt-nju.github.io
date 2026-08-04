---
title: "技能管理概览"
topic: skill-management
summary: "冻结 LLM 靠外部技能库做终身学习：Voyager 奠基可增长库，AutoSkill 做抽取与版本演化，SkillOS 用 RL 学策展，Ratchet 用 hygiene 防库漂移。"
lang: zh-CN
updated: 2026-08-04
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
raw:
  - raw/skill-management/2023-05-25-voyager.md
  - raw/skill-management/2026-03-01-autoskill.md
  - raw/skill-management/2026-05-07-skillos.md
  - raw/skill-management/2026-05-ratchet.md
---

## Overview

开放世界里的 LLM agent 若只做单次规划，跨任务知识无法沉淀。一条持续出现的路线是：权重冻结，把可复用行为外置为技能库，再在检索与治理上下功夫。本主题按问题组织四篇来源的增量：库如何长出来、技能长什么样、谁来策展、库如何不漂坏。

## 四条演进线

1. **Voyager（奠基）**：Minecraft 上的终身探索 agent。automatic curriculum + 可执行代码 skill library + iterative prompting（环境反馈 / 执行错误 / self-verification）。160 次 iteration 内发现 63 个独特物品，约为对照的 3.3 倍；路程约 2.3 倍；木制节点约快 15.3 倍。库默认可增长，几乎不做 outcome-driven 退役。
2. **AutoSkill（抽取与版本）**：从对话轨迹抽象 **SKILL.md**，模型无关 plug-in；online hybrid 检索注入，异步 judge 做 add / merge / discard 与 versioned merge。WildChat-1M（>8 turns）四子集共 N=1858 技能；English GPT-3.5 子集 10,243 对话 / 631 skills；案例 `professional_text_rewrite` 至版本 0.1.34。
3. **SkillOS（学策展）**：冻结 executor + 可训练 curator，对外部 SkillRepo 做 RL（GRPO）；用 grouped task streams 与 composite rewards 归因延迟反馈。ALFWorld 上相对 ReasoningBank，Qwen3-8B executor 的平均 SR 从 55.7 到 61.2；同一 curator 可泛化到更强 executor（如 Gemini-2.5-Pro 从 66.4 到 80.2）。
4. **Ratchet（hygiene / librarian）**：重述 SkillsBench：人写技能 +16.2pp，LLM 自写 +0.0pp，瓶颈在生命周期而非写作。四类机制——pattern canonicalisation、outcome-driven retirement、bounded active-cap、meta-skill authoring prior。MBPP+ hard-100 上 rolling-mean gain +0.328，late-window 0.584（peak 0.658）；SWE-bench Verified 有 +0.22 peak lift。消融显示 retirement 与 meta-skill 承重，显式去重可被 meta-skill 吸收。

## 共同主张与分歧

共同主张：能力增长可以发生在外部技能资产上，而不必更新权重。分歧在于「谁写技能、谁管库、反馈从哪来」——启发式 merge（AutoSkill）、RL 策展（SkillOS）、还是冻结作者加 hygiene 规则（Ratchet）。Voyager 证明库有用；后三者分别补抽取标准、策展学习与防漂移。

## See Also

- [Skill Library](/wiki/skill-management/skill-library/)
- [技能生命周期](/wiki/skill-management/skill-lifecycle/)
- [Skill Curation RL](/wiki/skill-management/skill-curation-rl/)
