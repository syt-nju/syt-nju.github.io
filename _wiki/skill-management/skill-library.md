---
title: "Skill Library"
topic: skill-management
summary: "技能库是冻结 LLM 的外部程序性记忆；形态从 Voyager 可执行代码，到 SKILL.md / Markdown SkillRepo，再到带单测与 per-skill memory 的 Agent Skills 包。"
lang: zh-CN
updated: 2026-08-04
order: 2
sources:
  - title: "Voyager: An Open-Ended Embodied Agent with Large Language Models"
    url: "https://arxiv.org/abs/2305.16291"
  - title: "AutoSkill: Experience-Driven Lifelong Learning via Skill Self-Evolution"
    url: "https://arxiv.org/abs/2603.01145"
  - title: "SkillOS: Learning Skill Curation for Self-Evolving Agents"
    url: "https://arxiv.org/abs/2605.06614"
  - title: "MUSE-Autoskill: Self-Evolving Agents via Skill Creation, Memory, Management, and Evaluation"
    url: "https://arxiv.org/abs/2605.27366"
raw:
  - raw/skill-management/2023-05-25-voyager.md
  - raw/skill-management/2026-03-01-autoskill.md
  - raw/skill-management/2026-05-07-skillos.md
  - raw/skill-management/2026-05-26-muse-autoskill.md
---

## Overview

Skill library 把「做过一次的事」变成「以后还能调」的外部记忆。它不是原始对话回放，而是可检索、 ideally 可组合的程序性工件。表示形态随系统而变，但角色一致：在权重冻结时承载跨任务能力增长。

## Voyager：可执行代码技能

每个技能是通过 self-verification 的函数式程序。描述由 GPT-3.5 生成，embedding 作向量库 key，value 是程序本体。新任务时检索 top-5 相关技能，注入 GPT-4 代码生成上下文；复杂行为通过组合已有程序放大能力，并缓解 catastrophic forgetting。写入门槛高：环境反馈、解释器错误与 critic 式 self-verification 通过后才 commit。去掉 skill library 后后期探索趋于平台；完整系统相对基线约 3.3 倍独特物品、约 2.3 倍路程，并是唯一解锁 diamond 工具的方法。

## AutoSkill：SKILL.md 工件

技能是可编辑、可版本化的 **SKILL.md**（Agent Skill 标准）：身份、标签、触发器、提示与约束。SkillBank 按 Users / Common 持久化并用向量索引。相对 Voyager，这里的技能多为自然语言行为规范与工作流，面向对话助手的个性化与制度性偏好，而不是 Minecraft 电机级程序。

## SkillOS：Markdown SkillRepo

SkillOS 沿用社区「技能即文件夹 / Markdown 指令」设定，把外部 SkillRepo 交给可训练 curator 维护；冻结 executor 只负责检索与执行。分析表明，学习后的库会发展出更丰富内部结构与更高层 meta-skills，策展本身比单纯换更强生成模型更关键。

## MUSE-Autoskill：可测试的 Agent Skills 包

MUSE 采用 Anthropic Agent Skills 目录约定：`SKILL.md` 定义接口；可选 `scripts/`、`resources/`、`tests/`。执行时先读接口（progressive disclosure），再按需读资源或跑脚本；代码执行经 sandbox 工具隔离。每技能可附 **`.memory.md`**，跨任务追加失败模式、输入格式与性能备注，加载时与接口一并注入。注册前优先跑 unit tests；无测试则回退 sandbox / 轨迹检查。该表示同时服务本机复用与跨 agent 迁移实验（详见 [MUSE-Autoskill](/wiki/skill-management/muse-autoskill/)）。

## 共同边界

四种表示都依赖检索质量；库增长后噪声、冗余与过时条目会拖垮有效上下文。表示问题与治理问题应分开看：见生命周期与策展 RL 页。

## See Also

- [技能管理概览](/wiki/skill-management/overview/)
- [技能生命周期](/wiki/skill-management/skill-lifecycle/)
- [MUSE-Autoskill](/wiki/skill-management/muse-autoskill/)
- [Skill Curation RL](/wiki/skill-management/skill-curation-rl/)
