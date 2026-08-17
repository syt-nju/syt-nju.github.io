---
title: "测试时训练问题地图"
topic: test-time-training
summary: "本 topic 问的是：冻结后的任务技能为什么测不到智力，以及测试时改参数、加算力或做 ICL，各自对应技能获取效率的哪一段。"
lang: zh-CN
updated: 2026-08-17
order: 1
sources:
  - title: "On the Measure of Intelligence"
    url: "https://arxiv.org/abs/1911.01547"
raw:
  - raw/test-time-training/2019-11-05-on-the-measure-of-intelligence.md
---

## Overview

冻结模型在已知任务上刷高分，测到的是已经结晶的技能，不是智力。Chollet 把智力写成 [技能获取效率](/wiki/test-time-training/skill-acquisition-efficiency/#skill-acquisition-efficiency)：在给定任务范围上，相对先验、经验和泛化难度，把信息转成新技能的速率。原文：[On the Measure of Intelligence](https://arxiv.org/abs/1911.01547)。

本 topic 的主骨架是这张测量地图。后续来源才会填上测试时改参数、测试时算力、ICL 与冻结后训练的方法比较；当前证据只来自 2019 年这篇定义与 [ARC](/wiki/test-time-training/measuring-general-intelligence/#arc-2019) 测量物。

## 问题地图

### 测到的是技能还是获取效率

任务技能可以被无上限先验或无上限训练数据[买到](/wiki/test-time-training/skill-acquisition-efficiency/#buy-skill)，同时掩盖系统自己的泛化力。公平比较要控制 [scope、priors、experience、generalization difficulty](/wiki/test-time-training/skill-acquisition-efficiency/#priors-experience)。原文：[Chollet 2019](https://arxiv.org/abs/1911.01547)。

### 评价集对谁未知

只测系统没见过的样本，仍可能测到工程师写进程序里的智力。[developer-aware generalization](/wiki/test-time-training/measuring-general-intelligence/#developer-aware) 要求任务对系统和开发者都未知。原文：[Chollet 2019](https://arxiv.org/abs/1911.01547)。

### 先验如何对人机公平

通用基准应显式穷尽先验，并只假设人类 [Core Knowledge](/wiki/test-time-training/measuring-general-intelligence/#core-knowledge)。[ARC](/wiki/test-time-training/measuring-general-intelligence/#arc-2019) 是按这条契约做的测量物，不是又一个技能榜。原文：[Chollet 2019](https://arxiv.org/abs/1911.01547)。

## 共同主张

- 智力不是二进制属性，而是相对某个 scope 的谱：local / broad / extreme。原文：[Chollet 2019](https://arxiv.org/abs/1911.01547)。
- 技能是智力过程的结晶输出；要测的是把先验和经验转成新技能的效率。原文：[Chollet 2019](https://arxiv.org/abs/1911.01547)。
- 对人机公平的通用基准必须控制先验、经验，并测量 developer-aware generalization。原文：[Chollet 2019](https://arxiv.org/abs/1911.01547)。

## 开放接口

本稿已写出、但尚未展开的效率轴包括 skill program 的计算代价、训练时计算、时间、能量与风险。测试时改参数、测试时加算力、ICL 与冻结后训练的对照，等后续来源再切页。ARC 的竞赛协议与数据集修订同样留给后续 ARC Prize 报告。

## 边界

本 topic 不收录量化、不收录 On-Policy Distillation。冻结模型的外部资产演化属于 [Harness Evolution](/wiki/harness-evolution/overview/)：那边明确把直接改权重划出；这边问的是技能测量与后来会碰到的测试时适应。两边只做边界对照，不把同一篇来源倒进两个 topic。

## See Also

- [技能获取效率](/wiki/test-time-training/skill-acquisition-efficiency/)
- [通用智能测量](/wiki/test-time-training/measuring-general-intelligence/)
- [Harness Evolution 概览](/wiki/harness-evolution/overview/)
