---
title: "深度搜索问题地图"
topic: deep-search
summary: "深度搜索训练数据的核心问题是如何制造答案出现前的长证据前缀，而不是只把轨迹、hop 数或图结构做长。"
lang: zh-CN
updated: 2026-08-07
order: 1
sources:
  - title: "FORT-Searcher: Synthesizing Shortcut-Resistant Search Tasks for Training Deep Search Agents"
    url: "https://arxiv.org/abs/2606.12087"
raw:
  - raw/deep-search/2026-06-10-fort-searcher.md
---

## Overview

训练深度搜索智能体需要的不是「看起来很长」的任务，而是**答案出现前的长证据前缀**：求解器必须先收集足够证据，才有条件锁定答案。现有合成管线常靠加 hops、加图结构、加层级约束或加大证据分散来抬高表观难度，但结构复杂度不一定变成 agent 实际经历的长程证据获取——求解过程可能塌缩到更便宜的 identifying route。

当前主题以 FORT-Searcher 为起点。它还不是深度搜索训练的完整综述，而是一张问题地图：什么决定实现难度、shortcut 如何让任务变浅、轨迹里该看什么信号、以及构造期能怎样压住这些风险。

## 问题地图

### 什么才是“难”

深度搜索难度不由题面 hops 或证据图大小直接决定，而由最便宜的 identifying route 决定。求解器不必验证全部线索，只要某个 identifying subset 已能唯一锁定答案，任务就可能沿这条弱路径被解掉。详见 [实现难度与 Shortcut](/wiki/deep-search/realized-difficulty/)。

### 为什么任务会变浅

Shortcut 是让题目以少于设计意图的证据获取被解掉的机制。当前已归纳四类：single-clue selectivity、evidence co-coverage、exposed constants、prior-knowledge binding。前三类压低结构下界，第四类来自模型侧先验绑定。详见 [实现难度与 Shortcut](/wiki/deep-search/realized-difficulty/)。

### 怎样观察真实搜索负载

理论下界在开放网页搜索上难以全量计算，因此需要轨迹签名。Realized solving cost 看总查询数，answer hit time 看答案首次出现位置，prior-shortcut rate 看模型是否在证据前点名答案。真正有用的是长 pre-answer search，而不是长轨迹本身。详见 [轨迹签名](/wiki/deep-search/trajectory-signatures/)。

### 构造期如何压住风险

FORT 把上述诊断反过来用于数据合成：从图初始化、图扩展、问题表述到对抗精炼，持续压制过选择性线索、证据共覆盖、暴露常量和先验绑定。系统细节与训练结果见 [FORT-Searcher](/wiki/deep-search/fort-searcher/)。

## 共同主张

有用的深度搜索监督是「答案出现前的长前缀」，不是「总检索步数很长」。当前证据主要来自 FORT 一篇来源，因此本 topic 暂时把 FORT 作为系统证据锚点，而不把它写成领域全貌。待扩展问题包括：同类诊断在不同搜索后端和求解器上是否稳定、RL 是否能进一步利用这类数据、推理侧 context management 对 headline 分数的贡献占比有多大。

## See Also

- [实现难度与 Shortcut](/wiki/deep-search/realized-difficulty/)
- [轨迹签名](/wiki/deep-search/trajectory-signatures/)
- [FORT-Searcher](/wiki/deep-search/fort-searcher/)
