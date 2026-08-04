---
title: "实现难度与 Shortcut"
topic: deep-search
summary: "实现搜索难度由最便宜的 identifying route 决定；四类 shortcut 会把表观复杂任务塌成浅层检索或先验绑定。"
lang: zh-CN
updated: 2026-08-04
order: 2
sources:
  - title: "FORT-Searcher: Synthesizing Shortcut-Resistant Search Tasks for Training Deep Search Agents"
    url: "https://arxiv.org/abs/2606.12087"
raw:
  - raw/deep-search/2026-06-10-fort-searcher.md
---

## Overview

任务实例写成三元组：答案空间、问题表达的约束集、检索接口。问题良定，故完整约束集唯一确定 gold answer。关键观察是：求解器不必验证每条线索，只需为某个 *identifying subset*（已能把候选缩到唯一答案的约束子集）取得证据。因此实现难度由验证该子集的最便宜路线决定，而不是任务设计者设想的那条路线。

结构下界可写成所有 identifying subset 上的最小路线代价；无先验、不瞎猜的参考求解器的代价被该下界卡住。这是单边、weakest-link 性质：只要有一个 identifying subset 又浅又集中，整题就变简单，与底层证据图多大无关。极端情形下，若某 identifying subset 可用一条初始即可执行的查询验证，结构下界塌成单次检索。

## 决定下界的四个量

三个客观量控制下界与其上的缺口，第四个量依赖具体模型：

- **Subset selectivity**：只用部分线索后剩余候选池大小。它不直接下界路线长度，而是设定探索缺口；当小子集已足够 identifying，它决定哪些子集进入最小化。
- **Evidence dispersion**：在忽略查询可执行性时，验证 gold 满足某线索子集所需的最少检索次数。若同一网页同时陈述两个本意事实，一步即可，计数降为 1。
- **Dependency depth**：查询串行链中最长链的长度——后续查询需要的名字或中间事实只能由更早检索暴露。它纯来自查询可执行性。
- **Solver-side cost reduction**：具体模型相对无先验参考求解器省下的代价，典型来自参数记忆认出目标。

固定 identifying subset 的路线代价至少是 evidence dispersion 与 dependency depth 的较大者；具体求解器的实现代价是无先验代价减去其 solver-side reduction。四个量把缺口归因到具名因子。

## 四类 Shortcut 风险

Shortcut 指任何让题目以少于设计意图的证据获取被解掉的机制。前三类作用于结构下界，第四类作用于模型侧减量。

1. **Single-clue selectivity**：一条或一小撮线索就把候选缩到一个或少数。即便子集并非严格 identifying，高选择性也会在有限检索结果里过早浮出答案。诊断例：四条独立线索本应汇聚到同一年，求解器只用「彩色电视」线索一查就把该年当作主候选，其余线索沦为事后验证。
2. **Evidence co-coverage**：单次检索的页面或 snippet 同时验证多条本意约束，把多步计划压成一步。诊断例：一条搜索结果 snippet 同时暴露答案实体及其行业、区域持股等答案侧事实。
3. **Exposed constants**：问题表面露出本应靠更早检索发现的精确名字、字符串、日期或数字，使下游查询从一开始就可执行，缩短串行依赖。诊断例：把目标人物的独特公开引语嵌进题面，模型几乎原样复用该短语，结果直接点名答案。关键是露出的字符串是目标的唯一属性，而非泛化中间线索。
4. **Prior-knowledge binding**：求解器在检索证据锚定之前就提交 gold answer。它不降低结构下界，故对无先验求解器仍需长证据路线的题，对已记住实体的模型仍可平凡。诊断例：模型在任何工具调用前就点名答案，之后才检索「证据」。

## 与主题其他页的关系

理论量难以在开放网页搜索上大规模精确计算（需枚举答案空间、所有 identifying subset 与合法路线），因此实现效果用 [轨迹签名](/wiki/deep-search/trajectory-signatures/) 诊断；[FORT-Searcher](/wiki/deep-search/fort-searcher/) 把这些风险反转为构造期控制。

## See Also

- [深度搜索概览](/wiki/deep-search/overview/)
- [轨迹签名](/wiki/deep-search/trajectory-signatures/)
- [FORT-Searcher](/wiki/deep-search/fort-searcher/)
