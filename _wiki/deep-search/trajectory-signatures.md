---
title: "轨迹签名"
topic: deep-search
summary: "用 solving cost、answer hit time 与 prior-shortcut rate 诊断实现难度；长轨迹不等于长答案前前缀，FORT 显著拉长 pre-answer search。"
lang: zh-CN
updated: 2026-08-04
order: 3
sources:
  - title: "FORT-Searcher: Synthesizing Shortcut-Resistant Search Tasks for Training Deep Search Agents"
    url: "https://arxiv.org/abs/2606.12087"
raw:
  - raw/deep-search/2026-06-10-fort-searcher.md
---

## Overview

[实现难度框架](/wiki/deep-search/realized-difficulty/) 中的理论量在开放网页搜索上难以全量计算。FORT 一文因此在固定求解器与检索预算下，用三条可观测的轨迹签名诊断实现难度，并据此比较开源深度搜索数据与 FORT 合成数据。

## 三条签名

- **Realized solving cost**：成功轨迹上的平均检索查询数。单独看证明力弱——长轨迹可能只是绕路。
- **Answer hit time**：gold answer 或其规范化别名首次出现的步数，取检索观察与模型可见文本中较早者。更晚的 hit time 意味着更长的答案前搜索前缀，这是压住廉价 identifying route 后期望看到的行为。
- **Prior-shortcut rate**：成功轨迹中，模型在任何检索锚定答案之前就提到答案的比例。作者称其为保守代理，只捕捉可见的 answer-before-evidence 行为。

真正有诊断价值的是 **solving cost 与 answer hit time 的缺口**：总步数很长但答案很早出现，说明监督并不「搜索重」。

## 开源数据 vs FORT

同一强 agent、同一预算下重评六个开源深度搜索数据集：表观轨迹长度 ≠ 搜索重监督。OpenSeeker 的 solving cost 为 84.7，但 answer hit time 仅 9.3，prior-shortcut rate 达 31.9。REDSearcher 在开源基线中签名最强（cost 92.1、hit time 18.7），答案仍远早于总成本。FORT 达到 cost 141.0、hit time 46.9，同时 prior-shortcut rate 为 11.0。

FORT 的额外代价并非来自更多 prior-bound 行为：hit time 显著后移，prior-shortcut rate 更低。

## 训练对照：长轨迹 ≠ 有用难度

四组各 12K 例、同一配方训练：把开源数据平均 solving cost 从 40.0 经 85.0 抬到 140.0，BrowseComp 仅从 47.1 到 49.5。把 cost 固定在 140.0 但换成 FORT 数据（hit time 47.0、prior-shortcut rate 11.4，对比开源侧 22.3 与 18.1），BrowseComp 到 52.9，BrowseComp-ZH 到 60.3。有用难度是长 pre-answer 前缀，不是长轨迹本身。

## 标注代理与局限

另对开源与 FORT 各 200 条成功 question–trajectory 做因子到可观测代理的映射：相对开源，FORT 把「自身证据只留下 1–2 个合理候选」的线索占比从 55.2 降到 40.2，归一化 evidence dispersion 从 78.7 升到 90.2，最贵答案前依赖链的检索代价从 3.1 升到 5.9，prior-bound 占比从 27.0 降到 16.0。

局限：prior-shortcut rate 只检测口头提前点名答案的模型；轨迹代理是实现效果而非理论量估计；签名依赖固定求解器与检索后端，换模型或搜索后端诊断会漂移。

## See Also

- [深度搜索概览](/wiki/deep-search/overview/)
- [实现难度与 Shortcut](/wiki/deep-search/realized-difficulty/)
- [FORT-Searcher](/wiki/deep-search/fort-searcher/)
