---
title: "Skill Library"
topic: harness-evolution
summary: "技能库是 harness evolution 的外部程序性记忆层；核心形态包括可执行代码、SKILL.md、SkillRepo 和带测试的 Agent Skills 包。"
lang: zh-CN
updated: 2026-08-07
order: 2
redirect_from:
  - /wiki/skill-management/skill-library/
sources:
  - title: "Voyager: An Open-Ended Embodied Agent with Large Language Models"
    url: "https://arxiv.org/abs/2305.16291"
  - title: "AutoSkill: Experience-Driven Lifelong Learning via Skill Self-Evolution"
    url: "https://arxiv.org/abs/2603.01145"
  - title: "SkillOS: Learning Skill Curation for Self-Evolving Agents"
    url: "https://arxiv.org/abs/2605.06614"
  - title: "MUSE-Autoskill: Self-Evolving Agents via Skill Creation, Memory, Management, and Evaluation"
    url: "https://arxiv.org/abs/2605.27366"
  - title: "Meta Context Engineering via Agentic Skill Evolution"
    url: "https://arxiv.org/abs/2601.21557"
  - title: "Meta-Harness: End-to-End Optimization of Model Harnesses"
    url: "https://arxiv.org/abs/2603.28052"
raw:
  - raw/harness-evolution/2023-05-25-voyager.md
  - raw/harness-evolution/2026-03-01-autoskill.md
  - raw/harness-evolution/2026-05-07-skillos.md
  - raw/harness-evolution/2026-05-26-muse-autoskill.md
  - raw/harness-evolution/2026-01-29-meta-context-engineering.md
  - raw/harness-evolution/2026-03-30-meta-harness.md
---

## Overview

Skill library 把「做过一次的事」变成「以后还能调」的外部记忆。它不是原始对话回放，而是可检索、 ideally 可组合、可治理的程序性工件。在 [Harness Engineering](/wiki/harness-evolution/harness-engineering/) 语境下，skill library 是 persistent state / tool protocol 的一个资产层；核心问题是：技能应写成可执行代码、自然语言规范、Markdown 仓库，还是带测试和资源的 Agent Skills 包；这些表示如何影响检索、组合、评估和迁移。

[MCE](/wiki/harness-evolution/meta-context-engineering/) 与 [Meta-Harness](/wiki/harness-evolution/meta-harness/) 使用相似的文件/代码资产，但它们的对象已经转向 CE skill 或完整 harness，属于本 topic 的边界问题，而非普通任务技能库形态。

## Voyager：可执行代码技能

每个技能是通过 self-verification 的函数式程序。描述由 GPT-3.5 生成，embedding 作向量库 key，value 是程序本体。新任务时检索 top-5 相关技能，注入 GPT-4 代码生成上下文；复杂行为通过组合已有程序放大能力，并缓解 catastrophic forgetting。写入门槛高：环境反馈、解释器错误与 critic 式 self-verification 通过后才 commit。去掉 skill library 后后期探索趋于平台；完整系统相对基线约 3.3 倍独特物品、约 2.3 倍路程，并是唯一解锁 diamond 工具的方法。

## AutoSkill：SKILL.md 工件

技能是可编辑、可版本化的 **SKILL.md**（Agent Skill 标准）：身份、标签、触发器、提示与约束。SkillBank 按 Users / Common 持久化并用向量索引。相对 Voyager，这里的技能多为自然语言行为规范与工作流，面向对话助手的个性化与制度性偏好，而不是 Minecraft 电机级程序。

## SkillOS：Markdown SkillRepo

SkillOS 沿用社区「技能即文件夹 / Markdown 指令」设定，把外部 SkillRepo 交给可训练 curator 维护；冻结 executor 只负责检索与执行。分析表明，学习后的库会发展出更丰富内部结构与更高层 meta-skills，策展本身比单纯换更强生成模型更关键。

## MUSE-Autoskill：可测试的 Agent Skills 包

MUSE 采用 Anthropic Agent Skills 目录约定：`SKILL.md` 定义接口；可选 `scripts/`、`resources/`、`tests/`。执行时先读接口（progressive disclosure），再按需读资源或跑脚本；代码执行经 sandbox 工具隔离。每技能可附 **`.memory.md`**，跨任务追加失败模式、输入格式与性能备注，加载时与接口一并注入。注册前优先跑 unit tests；无测试则回退 sandbox / 轨迹检查。该表示同时服务本机复用与跨 agent 迁移实验（详见 [MUSE-Autoskill](/wiki/harness-evolution/muse-autoskill/)）。

## 边界：CE skill 与 harness 程序

MCE 的技能也是 workspace 文件夹，但语义是 **CE 策略**：可含自然语言方法论、可执行脚本、结构化 context 模板、验证协议，以及按 query 过滤/组装的动态算子。Meta-Harness 的候选资产通常是任务侧 harness 程序及其执行痕迹。二者说明文件/代码资产可以被抬到更高层优化对象，但阅读时应与 task skill library 区分。

## 共同边界

上述表示都依赖检索或加载质量；库增长后噪声、冗余与过时条目会拖垮有效上下文。表示问题与治理问题应分开看：见生命周期、策展 RL，以及 MCE 用验证集做技能级选择、Meta-Harness 用全历史 traces 做程序搜索的对照。

## See Also

- [Harness Evolution 概览](/wiki/harness-evolution/overview/)
- [Harness Engineering](/wiki/harness-evolution/harness-engineering/)
- [技能生命周期](/wiki/harness-evolution/skill-lifecycle/)
- [MUSE-Autoskill](/wiki/harness-evolution/muse-autoskill/)
- [Meta Context Engineering](/wiki/harness-evolution/meta-context-engineering/)
- [Meta-Harness](/wiki/harness-evolution/meta-harness/)
- [Skill Curation RL](/wiki/harness-evolution/skill-curation-rl/)
