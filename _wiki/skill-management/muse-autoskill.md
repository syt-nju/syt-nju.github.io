---
title: "MUSE-Autoskill"
topic: skill-management
summary: "ByteDance 的训练无关技能生命周期框架：运行时创建、skill-level memory、单测评估与跨 agent 迁移，在 SkillsBench / SkillLearnBench 上验证。"
lang: zh-CN
updated: 2026-08-04
order: 5
sources:
  - title: "MUSE-Autoskill: Self-Evolving Agents via Skill Creation, Memory, Management, and Evaluation"
    url: "https://arxiv.org/abs/2605.27366"
raw:
  - raw/skill-management/2026-05-26-muse-autoskill.md
---

## Overview

MUSE-Autoskill（Memory-Utilizing Skill Evolution）把技能当作可长期治理的资产，而不是一次性生成物。它在同一 agent 回路里覆盖五阶段生命周期——**creation、memory、management、evaluation、refinement**——并强调 skill-level memory、单测评估与跨 agent 可迁移的技能包。名称含 Autoskill，但与 ECNU 的 [AutoSkill](/wiki/skill-management/skill-lifecycle/)（对话轨迹抽象 SKILL.md）是不同系统。

## 五阶段生命周期

技能包遵循 Anthropic Agent Skills 目录约定（含 `SKILL.md`；可选 `tests/` 与脚本）。主循环是 ReAct 式 Planning / Action / Observation，功能尽量外置为技能；内置能力包括 `skill_create` 与 `web_search`。

1. **Creation**：在运行时通过 `skill_create` 从成功轨迹蒸馏技能，消解「先离线造技能、再另场景用」的 creation–usage mismatch。
2. **Memory**：多层记忆——短期上下文、跨会话长期笔记，以及每技能旁路的 **skill-level memory**（`.memory.md`，加载时与 `SKILL.md` 一并浮现）。
3. **Management**：目录注入 system prompt（progressive disclosure）；维护含 refinement、合并近重复、裁剪持续失败或闲置条目。
4. **Evaluation**：代码型技能优先跑 `tests/` unit tests，未通过则阻断注册；无测试时回退 sandbox / 轨迹反馈。
5. **Refinement**：失败触发 `update_skill` 补丁再检，形成 create → evaluate → register 回路。

配套基础设施还包括对话 DAG 上的两级自适应上下文压缩（单节点摘要 → 链段合并），以及可跨会话持久化的状态。

## SkillsBench 与 SkillLearnBench 证据

四个 GPT-5.5 骨干 agent：Hermes、Codex、Claude Code、MUSE-Autoskill。SkillsBench 用 **75-task × 5-run** 严格分母；无可用自建技能的任务在自建设定下计 0。

| 设定 | Hermes | Codex | Claude Code | MUSE |
| --- | --- | --- | --- | --- |
| SkillsBench 无技能 | 37.24% | 44.80% | 42.43% | **46.95%** |
| SkillsBench 人写技能 | 48.02% | 57.58% | 56.15% | **59.67%**（+12.72pp） |
| SkillsBench 自建技能（all-75） | – | 47.52% | 44.27% | **53.42%**（+6.47pp） |
| SkillLearnBench 人写 / 自建 | 70.0% / – | 68.0% / 40.0% | 63.0% / 37.0% | **72.0% / 48.0%** |

覆盖子集上，MUSE 自建技能准确率 **85.24%**，高于同子集人写 **81.17%**（覆盖 47/75，未覆盖 28 计 0）。作者将主瓶颈定为**生成覆盖率**，而非已生成技能质量。

**跨 agent 迁移**（技能注入 Hermes，不做任务特化修改）：MUSE 自建技能使 Hermes 达 **51.90%**（相对无技能 +14.66pp），高于 Hermes 人写技能 48.02%；同覆盖规模下 Codex 自建技能转入 Hermes 为 37.01%（几乎无效），Claude Code 自建为 45.97%。

局限（作者自述）：自建技能常由单条成功轨迹蒸馏并在同一任务复评，可能高估 within-task 增益；跨 agent 主要验证了 MUSE→Hermes；早期版本还讨论过更小任务子集带来的高估风险。

## 与主题内其他路线的关系

相对 [AutoSkill](/wiki/skill-management/skill-lifecycle/) 的对话抽取与 add/merge/discard，MUSE 更强调运行时创建、单测与 skill-level memory。相对 [SkillOS](/wiki/skill-management/skill-curation-rl/) 的可训练 curator，MUSE 为 **training-free**。相对 [Ratchet](/wiki/skill-management/skill-lifecycle/) 的 hygiene / retirement，MUSE 更偏完整创建—评估—迁移栈，而非最小 librarian 配方。

> **Status: Disputed**（SkillsBench「LLM 自写无效」定调）
> [Ratchet](/wiki/skill-management/skill-lifecycle/) 引用 SkillsBench 原文：人写技能约 +16.2pp，LLM 自写约 +0.0pp，据此主张瓶颈在生命周期治理。MUSE 在全生命周期 agent 下报告自建技能 all-75 仍有 +6.47pp，且覆盖子集可超过人写；但 all-75 自建（53.42%）仍低于人写（59.67%），覆盖失败仍计 0。双方评测协议、agent 栈与「自写」含义不同，不宜直接等同或互相否定；见 overview 中的争议摘要。

## See Also

- [技能管理概览](/wiki/skill-management/overview/)
- [Skill Library](/wiki/skill-management/skill-library/)
- [技能生命周期](/wiki/skill-management/skill-lifecycle/)
- [Skill Curation RL](/wiki/skill-management/skill-curation-rl/)
