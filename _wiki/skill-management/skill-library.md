---
title: "Skill Library"
topic: skill-management
summary: "Voyager 把成功的可执行程序按描述 embedding 存入向量库，用 top-5 检索支撑组合式技能复用与跨任务泛化。"
lang: zh-CN
updated: 2026-08-04
order: 2
sources:
  - title: "Voyager: An Open-Ended Embodied Agent with Large Language Models"
    url: "https://arxiv.org/abs/2305.16291"
raw:
  - raw/skill-management/2023-05-25-voyager.md
---

## Overview

Skill library 是 Voyager 把「做过一次的事」变成「以后还能调」的外部记忆。技能不是对话摘要，而是通过 self-verification 的可执行程序；库随 curriculum 提出的任务不断写入，形成可组合、可解释、可迁移的行为资产。

## 表示与索引

每个技能是一段可复用函数式代码。写入时：GPT-3.5 生成程序描述，用其 embedding 作为向量库的 key，value 是程序本体。生成提示强调技能会被后续更复杂函数复用，因此应保持 generic。

## 检索与组合

面对 curriculum 的新任务时，先用 GPT-3.5 生成解题建议，再与环境反馈一起作为 query；检索返回 top-5 相关技能，连同 control primitive API 注入 GPT-4 的代码生成上下文。复杂行为通过组合已有程序实现，使能力随库增长而复合放大，并减轻其他 continual learning 方法中的 catastrophic forgetting。

## 写入门槛

技能并非每次尝试都入库。Iterative prompting 反复执行代码、收集环境反馈与解释器错误，并由另一 GPT-4 扮演 critic 做 self-verification：成功才 commit（如 `craftStoneShovel()`、`combatZombieWithSword()`）；失败则给出 critique 继续 refinement。超过 4 轮仍卡住则放弃当前任务、向 curriculum 要新目标。

## 证据：库是否关键

去掉 skill library 的变体在后期探索趋于平台，说明库不只是缓存，而是推动「新技能建立在旧技能之上」的结构。完整 Voyager 在科技树上是唯一解锁 diamond 工具的方法；在新世界清空库存后，同一库支撑未见任务的零样本求解，并可作为 plug-and-play 资产提升 AutoGPT。

相对基线的量级：约 3.3 倍独特物品、约 2.3 倍路程、木制节点约 15.3 倍更快。这些数字衡量的是整套系统，但消融把技能库标为后期不平台化的关键因素。

## 边界

Voyager 的库默认持续增长，检索靠相似度，缺少按任务贡献的退役与显式容量上限。这为后续「技能生命周期管理」留下缺口：库越大，噪声与冗余如何不拖垮有效检索，是 skill management 的下一层问题。

## See Also

- [技能管理概览](/wiki/skill-management/overview/)
