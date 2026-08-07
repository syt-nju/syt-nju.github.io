---
title: "Meta-Harness"
topic: skill-management
summary: "Meta-Harness 用 coding agent 外环搜索 harness 代码，经 filesystem 访问历次源码、分数与执行轨迹，在分类、数学检索与 TerminalBench-2 上超过 ACE/MCE 与手工 harness。"
lang: zh-CN
updated: 2026-08-07
order: 7
sources:
  - title: "Meta-Harness: End-to-End Optimization of Model Harnesses"
    url: "https://arxiv.org/abs/2603.28052"
raw:
  - raw/skill-management/2026-03-30-meta-harness.md
---

## Overview

**Harness** 是包裹冻结 LLM 的有状态程序：决定每一步存什么、取什么、给模型看什么。作者指出同一 benchmark 上仅改 harness 可造成约 \(6\times\) 性能差距；实践中 harness engineering 仍多靠人工改启发式。

Meta-Harness 是搜索 harness 的外环：proposer 是 coding agent（实验中为 Claude Code + Opus-4.6），反馈通道是不断增长的 **filesystem**——每个候选目录含源码、评估分数与执行轨迹（prompts、工具调用、模型输出、状态更新）。proposer 用 grep / cat 等按需读取，而不是把历史压成标量分或短摘要。外环刻意极简：维护种群与 Pareto 前沿，但不硬编码亲本选择；诊断与改写交给 agent。典型跑法约 20 轮、共评估约 60 个 harness。

项目页与 TerminalBench-2 产物：[meta-harness](https://yoonholee.com/meta-harness/)、[artifact](https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact)。

## 与 text optimizer / MCE 的差别

Table 1 对照 OPRO、TextGrad、AlphaEvolve、GEPA、Feedback Descent、TTT-Discover 等：其单次评估可用上下文多在约 0.002–0.026 MTok，而 Meta-Harness 设定下单次评估诊断可达约 **10.0** MTok。主张是：harness 决策影响长程行为，压缩反馈会切断失败与早期设计选择之间的因果链。

相对 [MCE](/wiki/skill-management/meta-context-engineering/)：两者都在「元优化 CE / harness」线上，但机制不同。MCE 用双层优化共演化 **CE skill** 与 **context artifact**；Meta-Harness 直接在 **完整 harness 程序空间** 搜索，并把外环诊断信息暴露为可查询的全历史文件系统。ACE / MCE 在本文中是在线文本分类设定下的强手工基线，而非被重构的唯一设计点。

## 在线文本分类

设定沿用 ACE / MCE 工作：LLM 逐条接收带标签样本、更新 memory，再在 held-out 上评测。基座为 **GPT-OSS-120B**；数据集为 USPTO-50k、Symptom2Disease（S2D）、LawBench。搜索初始化含 zero-shot、few-shot、ACE、MCE；20 轮、每轮 2 候选，共 40 个 harness。

Table 2 测试集（Avg Acc / Ctx↓，Ctx 为额外 context tokens，千）：

| Harness | USPTO | S2D | Law | Avg Acc | Ctx ↓ |
| --- | --- | --- | --- | --- | --- |
| MCE† | 14.0 | 83.0 | 23.0 | 40.0 | 28.5 |
| ACE† | 16.0 | 77.8 | 29.0 | 40.9 | 50.8 |
| Meta-Harness | 14.0 | 86.8 | 45.0 | **48.6** | **11.4** |

相对 ACE 高出 7.7 points、相对 MCE 高出 8.6 points；context 约为 ACE 的 \(1/4\) 量级（11.4K vs 50.8K）。† 实现来自 Ye et al.。这些是 **本文化协议下的绝对平均准确率**，与 MCE 原文在 DeepSeek-V3.1 上的 Avg. Rel. Gain 口径不同，不宜直接对撞。

相对 text optimizer（同 proposer 配置、同评估预算、仅用 search-set 选优）：Meta-Harness 以约 \(0.1\times\) 评估次数追平 OpenEvolve / TTT-Discover 的最终水平，最终再高出 **10** 点以上。Table 4（search-set）：Meta-Harness median **50.0** / best **56.7**；OpenEvolve 39.1 / 43.3；TTT-Discover 34.1 / 45.6；GEPA 32.6 / 40.2。

Table 3 消融：Scores Only median 34.6 / best 41.3；Scores + Summary 34.9 / 38.7；Full（含 traces）**50.0 / 56.7**。作者据此认为 raw execution traces 是关键接口成分，摘要无法恢复诊断信号。

OOD 九数据集（Table 5）：Meta-Harness 平均 **73.1**（Ctx 7.3），ACE **70.2**（Ctx 11.7）；在 6/9 数据集上最高。

## 检索增强数学推理

在 \(\geq\) 500,000 题语料上搜索 retrieval harness（40 轮、109 候选；search set 250 道 Olympiad 难度题）。选出的单一 harness 在 **200** 道未见过的 IMO 级题上，于五个 held-out 模型平均相对无检索高出 4.7 points（Table 6：No Retriever 34.1 → Meta-Harness 38.8），并略高于 BM25（37.5）。同一 harness 在搜索未见的模型上仍为正增益。

## TerminalBench-2

在 89 题同一 benchmark 上做 discovery 式搜索与评测（作者说明为公开竞赛式 harness 迭代惯例，并用人工/正则检查任务特有串泄漏）。Table 7：

| 基座 | Meta-Harness | 对照要点 |
| --- | --- | --- |
| Claude Opus 4.6 | **76.4%** | 超 Terminus-KIRA 74.7；榜上仅 ForgeCode 81.8 更高 |
| Claude Haiku 4.5 | **37.6%** | 超 Goose 35.5；该模型族报告结果中 #1 |

附录定性：TerminalBench-2 跑中 proposer 每轮中位读 **82** 个文件（约 41% 源码、40% traces）；早期把结构修复与 prompt 改写绑在一起导致回归后，能显式诊断 confound 并转向更安全的增量修改。

## 局限与主题定位

- 外环依赖强 coding agent；作者未系统比较多种 proposer。
- TerminalBench-2 非严格 held-out 划分；应读作发现型基准优化，并依赖其过拟合检查叙述。
- 相对任务技能库路线（Voyager / MUSE / SkillOS / Ratchet），本文优化的是 **任务侧 harness 程序**（分类 memory、检索策略、agent 编排），不是 librarian 对 SKILL.md 库的 add/merge/retire。
- 相对 MCE：同属 CE/harness 元优化，但一个演化「如何做 CE 的 skill」，一个用全历史诊断搜索「整份 harness 代码」。

## See Also

- [技能管理概览](/wiki/skill-management/overview/)
- [Meta Context Engineering](/wiki/skill-management/meta-context-engineering/)
- [Skill Library](/wiki/skill-management/skill-library/)
- [技能生命周期](/wiki/skill-management/skill-lifecycle/)
