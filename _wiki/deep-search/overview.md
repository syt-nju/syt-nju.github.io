---
title: "深度搜索概览"
topic: deep-search
summary: "深度搜索训练数据的关键不在 hops 多少，而在实现难度：最便宜的 identifying route，以及能否压住四类 shortcut。"
lang: zh-CN
updated: 2026-08-04
order: 1
sources:
  - title: "FORT-Searcher: Synthesizing Shortcut-Resistant Search Tasks for Training Deep Search Agents"
    url: "https://arxiv.org/abs/2606.12087"
raw:
  - raw/deep-search/2026-06-10-fort-searcher.md
---

## Overview

训练深度搜索智能体需要可验证问题：答案在搜集到足够证据之前不可用。现有合成管线常靠加 hops、加图结构、加层级约束或加大证据分散来抬高「表观难度」，但结构复杂度不一定变成 agent 实际经历的长程证据获取——求解过程可能塌缩到更便宜的 identifying route。

当前主题以 FORT-Searcher 为起点，按问题组织知识：什么决定实现难度、有哪些 shortcut、如何用轨迹签名诊断、以及如何在构造期把这些风险压下去。

## 四条问题线

1. **实现难度（而非表观结构）**：求解器不必验证全部线索，只需验证某个已能唯一锁定答案的 identifying subset；难度由验证该子集的最便宜路线决定，呈现 weakest-link 性质。详见 [实现难度与 Shortcut](/wiki/deep-search/realized-difficulty/)。
2. **四类 shortcut 风险**：single-clue selectivity、evidence co-coverage、exposed constants、prior-knowledge binding。前三者压结构下界，第四者走模型侧先验捷径。
3. **轨迹签名诊断**：理论量在开放网页搜索上难算全；用 realized solving cost、answer hit time、prior-shortcut rate 在固定求解器与预算下观察实现效果。开源数据常见「轨迹很长、答案却很早出现」。详见 [轨迹签名](/wiki/deep-search/trajectory-signatures/)。
4. **FORT 构造期控制**：把上述分析反转为合成管线——图初始化、图扩展、问题表述、对抗精炼——再仅用 SFT 训练 FORT-Searcher。详见 [FORT-Searcher](/wiki/deep-search/fort-searcher/)。

## 共同主张

共同主张：有用的深度搜索监督是「答案出现前的长前缀」，不是「总检索步数很长」。分歧与待扩展点主要在训练范式（当前仅 SFT）以及推理侧 context-management 对 headline 分数的贡献占比；这些写在系统页局限中，不在概览里裁决。

## See Also

- [实现难度与 Shortcut](/wiki/deep-search/realized-difficulty/)
- [轨迹签名](/wiki/deep-search/trajectory-signatures/)
- [FORT-Searcher](/wiki/deep-search/fort-searcher/)
