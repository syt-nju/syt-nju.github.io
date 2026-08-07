---
title: "Skill Library"
topic: skill-management
summary: "技能库是冻结 LLM 的外部程序性记忆；形态从 Voyager 代码、SKILL.md、Agent Skills、MCE 的 CE skill 文件夹，到 Meta-Harness 搜索的 harness 程序种群。"
lang: zh-CN
updated: 2026-08-07
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
  - title: "Meta Context Engineering via Agentic Skill Evolution"
    url: "https://arxiv.org/abs/2601.21557"
  - title: "Meta-Harness: End-to-End Optimization of Model Harnesses"
    url: "https://arxiv.org/abs/2603.28052"
raw:
  - raw/skill-management/2023-05-25-voyager.md
  - raw/skill-management/2026-03-01-autoskill.md
  - raw/skill-management/2026-05-07-skillos.md
  - raw/skill-management/2026-05-26-muse-autoskill.md
  - raw/skill-management/2026-01-29-meta-context-engineering.md
  - raw/skill-management/2026-03-30-meta-harness.md
---

## Overview

Skill library 把「做过一次的事」变成「以后还能调」的外部记忆。它不是原始对话回放，而是可检索、 ideally 可组合的程序性工件。表示形态随系统而变，但角色一致：在权重冻结时承载跨任务能力增长；在 [MCE](/wiki/skill-management/meta-context-engineering/) 中，同一文件夹抽象还被抬到「如何做 Context Engineering」一层；[Meta-Harness](/wiki/skill-management/meta-harness/) 则把可搜索资产做成任务侧 harness 程序及其执行痕迹。

## Voyager：可执行代码技能

每个技能是通过 self-verification 的函数式程序。描述由 GPT-3.5 生成，embedding 作向量库 key，value 是程序本体。新任务时检索 top-5 相关技能，注入 GPT-4 代码生成上下文；复杂行为通过组合已有程序放大能力，并缓解 catastrophic forgetting。写入门槛高：环境反馈、解释器错误与 critic 式 self-verification 通过后才 commit。去掉 skill library 后后期探索趋于平台；完整系统相对基线约 3.3 倍独特物品、约 2.3 倍路程，并是唯一解锁 diamond 工具的方法。

## AutoSkill：SKILL.md 工件

技能是可编辑、可版本化的 **SKILL.md**（Agent Skill 标准）：身份、标签、触发器、提示与约束。SkillBank 按 Users / Common 持久化并用向量索引。相对 Voyager，这里的技能多为自然语言行为规范与工作流，面向对话助手的个性化与制度性偏好，而不是 Minecraft 电机级程序。

## SkillOS：Markdown SkillRepo

SkillOS 沿用社区「技能即文件夹 / Markdown 指令」设定，把外部 SkillRepo 交给可训练 curator 维护；冻结 executor 只负责检索与执行。分析表明，学习后的库会发展出更丰富内部结构与更高层 meta-skills，策展本身比单纯换更强生成模型更关键。

## MUSE-Autoskill：可测试的 Agent Skills 包

MUSE 采用 Anthropic Agent Skills 目录约定：`SKILL.md` 定义接口；可选 `scripts/`、`resources/`、`tests/`。执行时先读接口（progressive disclosure），再按需读资源或跑脚本；代码执行经 sandbox 工具隔离。每技能可附 **`.memory.md`**，跨任务追加失败模式、输入格式与性能备注，加载时与接口一并注入。注册前优先跑 unit tests；无测试则回退 sandbox / 轨迹检查。该表示同时服务本机复用与跨 agent 迁移实验（详见 [MUSE-Autoskill](/wiki/skill-management/muse-autoskill/)）。

## MCE：Context Engineering skill 文件夹

MCE 的技能同样是 workspace 中的文件夹，但语义是 **CE 策略**：可含自然语言方法论、可执行脚本、结构化 context 模板、验证协议，以及按 query 过滤/组装的动态算子。Base 层产出的 context artifact 本身也是 files/code 目录，不受预定 itemized-list schema 约束。历史技能库 \(\mathcal{H}\) 保存 \((s,c,J_{\mathrm{train}},J_{\mathrm{val}})\)，供 **agentic crossover** 审议式重组。详见 [Meta Context Engineering](/wiki/skill-management/meta-context-engineering/)。

## Meta-Harness：被搜索的 harness 程序

[Meta-Harness](/wiki/skill-management/meta-harness/) 优化对象通常是 **单文件 Python harness**（提示构造、检索、memory、编排），不是 SKILL.md 库条目。外环把历次候选的源码、分数与执行轨迹存进 filesystem，供 coding agent 按需读取后再提案。资产形态更接近「可版本化的任务侧程序种群」，而不是任务技能库或 CE skill 文件夹；与 MCE 的联系在于同属对 CE/harness 的元优化。

## 共同边界

上述表示都依赖检索或加载质量；库增长后噪声、冗余与过时条目会拖垮有效上下文。表示问题与治理问题应分开看：见生命周期、策展 RL，以及 MCE 用验证集做技能级选择、Meta-Harness 用全历史 traces 做程序搜索的对照。

## See Also

- [技能管理概览](/wiki/skill-management/overview/)
- [技能生命周期](/wiki/skill-management/skill-lifecycle/)
- [MUSE-Autoskill](/wiki/skill-management/muse-autoskill/)
- [Meta Context Engineering](/wiki/skill-management/meta-context-engineering/)
- [Meta-Harness](/wiki/skill-management/meta-harness/)
- [Skill Curation RL](/wiki/skill-management/skill-curation-rl/)
