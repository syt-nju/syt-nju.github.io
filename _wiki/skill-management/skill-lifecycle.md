---
title: "技能生命周期"
topic: skill-management
summary: "技能库的瓶颈常在 librarian：AutoSkill 用 add/merge/discard 与版本合并做维护，Ratchet 用退役、容量上限与 meta-skill 先验抑制 library drift。"
lang: zh-CN
updated: 2026-08-04
order: 3
sources:
  - title: "AutoSkill: Experience-Driven Lifelong Learning via Skill Self-Evolution"
    url: "https://arxiv.org/abs/2603.01145"
  - title: "Ratchet: A Minimal Hygiene Recipe for Self-Evolving LLM Agents"
    url: "https://arxiv.org/abs/2605.22148"
raw:
  - raw/skill-management/2026-03-01-autoskill.md
  - raw/skill-management/2026-05-ratchet.md
---

## Overview

会写技能不等于库长期有用。SkillsBench 的对照被 Ratchet 用来定调：人写技能相对无技能基线 +16.2pp，LLM 自写技能 +0.0pp——瓶颈在版本、冲突与废弃等生命周期管理。本页整理两类「librarian」方案：AutoSkill 的训练无关维护回路，以及 Ratchet 的 hygiene 最小配方。

## AutoSkill：add / merge / discard

在线路径做 hybrid 检索与 Top-K 注入；演化路径从交互抽候选技能，再经 management judge 在 **add、merge、discard** 中决策。merge 不是拼接，而是 versioned evolution：保留可复用约束、合并增量并 bump 版本。案例：`professional_text_rewrite` 到 0.1.34，说明高频生产力技能可反复精炼；低频技能可停在 0.1.0。SkillBank 统计（WildChat-1M，>8 turns，四子集 N=1858）显示库以编程与写作为中心，但仍覆盖多样沟通类技能。

AutoSkill 强调可检视与可编辑，但对「按任务贡献退役」与「硬容量上限」着墨有限；Ratchet 的对照表也将其标为缺少 outcome-driven retirement 与 bounded active-cap。

## Ratchet：四类 hygiene

单 agent 回路里，冻结 LLM 同时写、取、管、退自然语言技能。四个候选机制：

1. **Pattern canonicalisation**：近重复失败标签归一，避免同 bug 生两技能。
2. **Outcome-driven retirement**：按贡献分与证据门槛降级弱势技能。
3. **Bounded active-cap**：活跃槽有限，迫使竞争进入 Router shortlist。
4. **Meta-skill authoring prior**：约束 Synthesizer 风格，隐式去重。

技能多从 failure cluster 合成，偏 pitfall / 负向约束。MBPP+ hard-100（Claude Opus 4.7，100 rounds，3 seeds）上 Default 相对 round-0 的 0.258 基线，late-window rolling mean 到 0.584，gain +0.328，peak 0.658；无技能对照几乎不涨。SWE-bench Verified 上有 +0.22 peak lift。

消融要点：去掉 skill injection 增益消失；去掉 meta-skill 损伤大；harsh retirement 可掉到无技能地板以下；显式 canonicalisation / cover-guard 在此规模可被 meta-skill 吸收；更频繁 meta-skill refresh（A8）多花约 55% wall-time 换边际收益。Proposition 1：有限 cap 与 retirement threshold 一起保证期望表现不会无界掉到无技能地板之下。

## 对照

| 维度 | AutoSkill | Ratchet |
| --- | --- | --- |
| 技能来源 | 对话/轨迹偏好与工作流 | 失败簇合成 |
| 维护动作 | add / merge / discard + 版本 | retirement + active-cap + meta prior |
| 训练 | 训练无关 plug-in | 权重冻结，无 RL 更新 curator |
| 实证形态 | SkillBank 统计与案例 | MBPP+ / SWE-bench 定量增益与消融 |

二者都承认库需要治理；AutoSkill 偏标准化工件与持续合并，Ratchet 偏结果驱动的裁剪与容量竞争。

## See Also

- [技能管理概览](/wiki/skill-management/overview/)
- [Skill Library](/wiki/skill-management/skill-library/)
- [Skill Curation RL](/wiki/skill-management/skill-curation-rl/)
