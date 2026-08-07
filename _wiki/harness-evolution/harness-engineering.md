---
title: "Harness Engineering"
topic: harness-evolution
summary: "Harness engineering 把模型外部的 workflow、context、工具、文件系统记忆、评估与权限控制视为可演化的能力层。"
lang: zh-CN
updated: 2026-08-07
order: 6
sources:
  - title: "Harness Engineering for Self-Improvement"
    url: "https://lilianweng.github.io/posts/2026-07-04-harness/"
  - title: "Meta Context Engineering via Agentic Skill Evolution"
    url: "https://arxiv.org/abs/2601.21557"
  - title: "Meta-Harness: End-to-End Optimization of Model Harnesses"
    url: "https://arxiv.org/abs/2603.28052"
  - title: "Self-Harness: Harnesses That Improve Themselves"
    url: "https://arxiv.org/abs/2606.09498v1"
raw:
  - raw/harness-evolution/2026-07-04-harness-engineering-for-self-improvement.md
  - raw/harness-evolution/2026-01-29-meta-context-engineering.md
  - raw/harness-evolution/2026-03-30-meta-harness.md
  - raw/harness-evolution/2026-06-08-self-harness.md
---

## Overview

Harness 是包裹 base model 的运行系统：决定模型如何计划、调用工具、读取和压缩 context、写入 artifact、保存历史、并用评估反馈推进下一步。相比早期 “LLM + memory + tools + planning + action” 的 agent 框架，harness engineering 更接近 runtime 和软件系统设计：它处理 workflow、permission controls、persistent state management 与 evaluation。

这使 harness 成为冻结模型能力增长的外部层。权重不变时，系统仍可以通过更好的上下文组织、更可靠的工具协议、更可恢复的文件系统记忆、更合适的 workflow 和更稳的评估门控提升长程任务表现。

## 三个基础设计模式

Weng 将近端 harness 设计概括为三个常见模式：

1. **Workflow automation**：给模型一个可执行、可测试、可迭代的 loop，例如 plan → execute → observe/test → improve → execute again。重点不只是 prompt，而是让模型能分析自身轨迹和失败案例，并通过 agent runtime 继续推进。
2. **File system as persistent memory**：长程 rollout 中的 logs、code diffs、paper summaries、error traces 与 trajectories 会远超上下文窗口。把状态落到文件系统，让模型按需读取，比把全部历史塞进 context 更可恢复。
3. **Sub-agent and backend jobs**：主 agent 可以显式启动并监控并行子任务、后台实验或假设搜索；关键是让 parallelism inspectable，并把输出保存为文件、日志和状态记录，而不是只留在短暂 chat context。

这些模式解释了为什么 [Skill Library](/wiki/harness-evolution/skill-library/) 只是 harness evolution 的一层：技能文件、`.memory.md`、测试、脚本和资源都属于更大的 persistent state / tool protocol 设计空间。

## 优化对象层级

Harness optimization 的对象可以从浅到深排列：

instruction prompts → structured context → workflow → harness code → optimizer code。

[Meta Context Engineering](/wiki/harness-evolution/meta-context-engineering/) 主要把 structured context 和 context-management mechanism 变成可演化对象；[Meta-Harness](/wiki/harness-evolution/meta-harness/) 直接搜索 harness code，并让外部 coding agent 读取候选源码、分数和 traces；[Self-Harness](/wiki/harness-evolution/self-harness/) 则要求目标模型基于自身 held-in 失败提出 bounded harness edits，再用 held-in / held-out 非回归规则接受。

因此，harness engineering 的核心问题不是“哪条 prompt 更好”，而是能否把运行系统本身做成可观测、可评估、可修改的对象。

## Self-Improvement 语境

Weng 把 harness engineering 放进 recursive self-improvement（RSI）的近端路径：现代系统短期内未必先让模型直接改写权重，而是让模型改善训练 pipeline 或 deployment system，再由更好的系统支持后续模型、研究或任务执行。

这一路线的含义是：外部 harness 可能先承担“认知机械”的一部分。手写 prompt trick 可能被更强模型内化，但目标、约束、上下文、工具和评估接口不会消失；它们会变成更稳定的系统边界。

## 评估、权限与长期目标

Harness evolution 的风险来自它优化的信号本身。若 reward 来自单元测试，agent 可能过拟合测试；若来自 judge model，可能学到 judge-specific reward hacking；若来自 benchmark，可能利用 benchmark artifacts。Weng 因此强调 evaluator 和 permission control 应尽量位于 harness-evolving loop 外部，并配合 held-out tests、trace audits 与关键节点的人类审查。

长期软件工程目标也难以用短期 sandbox reward 表达。Coding agent 可以完成当前任务，但标准 RLVR 风格训练通常捕捉不到 maintainability、ownership boundaries、migration cost、backwards compatibility 或未来 debugging burden。Harness evolution 若要服务真实工程环境，评估必须覆盖这些长期成本，而不只是 pass count。

## See Also

- [Harness Evolution 概览](/wiki/harness-evolution/overview/)
- [Skill Library](/wiki/harness-evolution/skill-library/)
- [技能生命周期](/wiki/harness-evolution/skill-lifecycle/)
- [Meta Context Engineering](/wiki/harness-evolution/meta-context-engineering/)
- [Meta-Harness](/wiki/harness-evolution/meta-harness/)
- [Self-Harness](/wiki/harness-evolution/self-harness/)
