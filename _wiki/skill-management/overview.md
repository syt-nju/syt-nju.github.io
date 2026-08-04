---
title: "技能管理概览"
topic: skill-management
summary: "从 Voyager 起，冻结 LLM 通过可增长的外部技能库做终身学习；后续工作把焦点从‘会写技能’推进到检索、演化和治理。"
lang: zh-CN
updated: 2026-08-04
order: 1
sources:
  - title: "Voyager: An Open-Ended Embodied Agent with Large Language Models"
    url: "https://arxiv.org/abs/2305.16291"
raw:
  - raw/skill-management/2023-05-25-voyager.md
---

## Overview

开放世界里的 LLM agent 若只做单次规划，跨任务知识无法沉淀。Voyager 在 Minecraft 上给出一条可复用路线：用黑盒 GPT-4 做 in-context 学习，把成功行为固化成可执行程序，写入不断增长的 skill library，再靠 curriculum 推动探索边界。权重不更新，能力增长发生在外部技能库里。

本主题后续页面会在此基础上整理：技能如何表示与检索、如何从交互中抽取与演化、以及生命周期治理（退役、容量、去重）如何避免库质量漂移。

## Voyager 的三件套

Voyager 由三块机制组成：

1. **Automatic curriculum**：以「尽可能发现多样事物」为总目标，按当前状态与探索进度自下而上提出下一任务，可看作 in-context 的 novelty search。
2. **Skill library**：技能是可执行代码，而不是低层电机指令；描述的 embedding 作检索键，复杂技能通过组合更简单程序快速放大能力，并缓解 catastrophic forgetting。
3. **Iterative prompting**：环境反馈、代码解释器错误、以及 GPT-4 self-verification 共同驱动程序 refinement；验证通过后才写入技能库。卡住超过 4 轮代码生成则换任务。

code as action 的动机是：程序天然表达时序扩展与可组合动作，适合 Minecraft 一类长程任务。

## 实证锚点

在 MineDojo 上相对 ReAct、Reflexion、AutoGPT 等基线，Voyager 在 160 次 prompting iteration 内发现 63 个独特物品，约为对照的 3.3 倍；路程约 2.3 倍；木制工具节点约快 15.3 倍，石制约 8.5 倍，铁制约 6.4 倍，且是唯一解锁 diamond 级的方法。清空背包、换新世界后，已学技能库可零样本支撑未见任务；同一技能库甚至可即插即用地提升 AutoGPT。

消融显示：去掉 skill library 后后期探索趋于平台；curriculum、环境反馈、执行错误与 self-verification 各自贡献明显。

## 主题开放问题

Voyager 证明「可增长技能库」可行，但几乎不做 outcome-driven 退役与容量约束。后续工作是否仍以可执行代码为技能形态、如何从对话或轨迹中自动抽取自然语言技能、以及如何用 RL 或 hygiene 规则管理库质量，是本主题继续累积的主线。

## See Also

- [Skill Library](/wiki/skill-management/skill-library/)
