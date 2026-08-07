---
title: "技能生命周期"
topic: skill-management
summary: "技能库瓶颈常在 librarian 与评估：AutoSkill / Ratchet / MUSE 管任务技能；MCE 与 Meta-Harness 是相邻的 CE/harness 元优化边界。"
lang: zh-CN
updated: 2026-08-07
order: 3
sources:
  - title: "AutoSkill: Experience-Driven Lifelong Learning via Skill Self-Evolution"
    url: "https://arxiv.org/abs/2603.01145"
  - title: "Ratchet: A Minimal Hygiene Recipe for Self-Evolving LLM Agents"
    url: "https://arxiv.org/abs/2605.22148"
  - title: "MUSE-Autoskill: Self-Evolving Agents via Skill Creation, Memory, Management, and Evaluation"
    url: "https://arxiv.org/abs/2605.27366"
  - title: "Meta Context Engineering via Agentic Skill Evolution"
    url: "https://arxiv.org/abs/2601.21557"
  - title: "Meta-Harness: End-to-End Optimization of Model Harnesses"
    url: "https://arxiv.org/abs/2603.28052"
raw:
  - raw/skill-management/2026-03-01-autoskill.md
  - raw/skill-management/2026-05-ratchet.md
  - raw/skill-management/2026-05-26-muse-autoskill.md
  - raw/skill-management/2026-01-29-meta-context-engineering.md
  - raw/skill-management/2026-03-30-meta-harness.md
---

## Overview

会写技能不等于库长期有用。技能库会遇到版本、冲突、重复、闲置和失效问题，因此需要 librarian：什么时候 add、merge、discard、retire，什么时候通过测试或任务反馈阻断注册。AutoSkill、Ratchet 与 MUSE-Autoskill分别给出不同生命周期处方。关于“LLM 自写技能是否有效”的冲突证据，单独见 [自写技能有效性争议](/wiki/skill-management/self-authored-skills/)。

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

## MUSE-Autoskill：运行时生命周期与评估门控

MUSE 把五阶段（creation / memory / management / evaluation / refinement）放进同一 ReAct agent。关键差别是：**skill_create 在执行回路内调用**，代码型技能用 `tests/` unit tests 门控注册，失败则 `update_skill` 再检；管理侧支持精炼、合并与裁剪。记忆上除短/长期层外，每技能有 `.memory.md` 积累跨任务经验。

SkillsBench 75-task（GPT-5.5）上，MUSE 人写技能 59.67%（相对无技能 +12.72pp），自建技能严格 all-75 为 53.42%（+6.47pp）；覆盖 47/75 时覆盖子集 85.24% vs 同子集人写 81.17%。跨 agent：MUSE 自建技能转入 Hermes 达 51.90%，高于 Hermes 人写 48.02%。作者将主瓶颈定为生成覆盖率。完整数字与争议语境见 [MUSE-Autoskill](/wiki/skill-management/muse-autoskill/)。

## 对照

| 维度 | AutoSkill | Ratchet | MUSE-Autoskill |
| --- | --- | --- | --- |
| 技能来源 | 对话/轨迹偏好与工作流 | 失败簇合成 | 运行时成功轨迹蒸馏（`skill_create`） |
| 维护动作 | add / merge / discard + 版本 | retirement + active-cap + meta prior | 单测门控、update、merge、prune |
| 训练 | 训练无关 plug-in | 权重冻结，无 RL curator | 训练无关；强调跨 agent 迁移实验 |
| 实证形态 | SkillBank 统计与案例 | MBPP+ / SWE-bench 定量增益与消融 | SkillsBench / SkillLearnBench + Hermes 迁移 |

三者都承认库需要治理；AutoSkill 偏标准化工件与持续合并，Ratchet 偏结果驱动裁剪，MUSE 偏创建—评估闭环与可迁移技能包。

## MCE：元层技能选择（对照）

MCE 的「生命周期」发生在 CE skill 上：每轮 agentic crossover 生成 offspring，base 执行后写入 \(\mathcal{H}\)，再按 \(J_{\mathrm{val}}\) 做 \((1+1)\)-ES 式保留。meta-agent 可监测 train/val 并抑制过拟合，但没有 AutoSkill 式 merge/discard 目录，也没有 Ratchet 的 active-cap / retirement 配方。对象是 harness（如何学 context），不是任务技能库 librarian；评测在 FiNER 等 CE 域，不裁决 SkillsBench 争议。细节见 [Meta Context Engineering](/wiki/skill-management/meta-context-engineering/)。

## Meta-Harness：候选 harness 种群（对照）

[Meta-Harness](/wiki/skill-management/meta-harness/) 也不做任务技能库的 add/merge/retire。它维护已评估 harness 的种群与 Pareto 前沿，把每次评估的代码、分数与 traces 追加进 filesystem；proposer 自行决定读哪些历史、做局部编辑还是重写。选择信号来自 search-set（及多目标时的 Pareto），外环本身几乎不写死亲本规则。对象是任务侧 harness 程序，评测覆盖分类、数学检索与 TerminalBench-2 discovery 设定。

## See Also

- [技能管理概览](/wiki/skill-management/overview/)
- [Skill Library](/wiki/skill-management/skill-library/)
- [自写技能有效性争议](/wiki/skill-management/self-authored-skills/)
- [MUSE-Autoskill](/wiki/skill-management/muse-autoskill/)
- [Meta Context Engineering](/wiki/skill-management/meta-context-engineering/)
- [Meta-Harness](/wiki/skill-management/meta-harness/)
- [Skill Curation RL](/wiki/skill-management/skill-curation-rl/)
