---
title: "FORT-Searcher"
topic: deep-search
summary: "FORT 在构造期控制四类 shortcut，用轨迹签名校准可解但搜索重的任务；仅 SFT 的 FORT-Searcher 在同规模开源搜索 agent 中总体最优。"
lang: zh-CN
updated: 2026-08-04
order: 4
sources:
  - title: "FORT-Searcher: Synthesizing Shortcut-Resistant Search Tasks for Training Deep Search Agents"
    url: "https://arxiv.org/abs/2606.12087"
raw:
  - raw/deep-search/2026-06-10-fort-searcher.md
---

## Overview

FORT（Framework of Shortcut-Resistant Training-Data Synthesis）把 [实现难度分析](/wiki/deep-search/realized-difficulty/) 反转为构造期控制：在问题落成文之前，用内部证据图作工作区，调控选择性、分散度与依赖深度。FORT-Searcher 是仅在所得轨迹上做 supervised fine-tuning 的搜索 agent。

难度框架与 [轨迹签名](/wiki/deep-search/trajectory-signatures/) 见对应概念页；本页聚焦合成管线、训练设定与实证。

## 四阶段合成

**1. Graph initialization**  
从 Wikidata 选长尾根实体：主题覆盖（去掉抽象概念）、实体冷门（偏好无英文维基页的稀有实体）、信息密度（轻量预搜索确认可解）。用预挖掘的 **cycle** 而非线性链做种子——线性种子容易把下游实体名逼进题面（exposed constants）；cycle 把根嵌进闭合局部结构，关系线索可不点名每个中间节点。cycle 写入 entity-to-cycle 倒排索引，去重节点集，并过滤 hub、冗余与大众协作结构。

**2. Graph construction**  
在深度与节点预算下扩展，优先扩展最深未处理节点，以保留多步引用链成为串行依赖。Enricher 从 Wikidata、开放网页、结构化库、Google Scholar、Google Maps 等异构源采原子事实，刻意避免从同一证据项抽多条入选事实；并构造 derived facts（coincidence bridging、count aggregation、numerical relation、meta-fact extraction），使其难以在单条检索中原文出现。再做来源一致性与实体一致性检查。Expander **偏好 generic 事实而非代表性事实**：过于特征的事实会让单线索过选择性，故保留「单独弱、组合才 identifying」的可靠事实。

**3. Question formulation**  
选答案节点，剪冗余，保留联合 identifying、个体 generic 的线索；中间节点名改成泛指指称；剩余字面量做 exact-value fuzzing（类别泛化、范围放宽、元属性描述、算术编码、对比排除）。Fuzzing 不是为了暧昧或不可验证，gold answer 必须保留。

**4. Adversarial refinement**  
强对抗 agent 在真实搜索设定下打草稿，按轨迹签名验收：答对、检索轮数够、答案出现够晚、且无证据前绑定。过快解开的草稿按路线级 shortcut 修（换 co-covered 证据、去过选择性事实、隐瞒/fuzz 暴露常量；若先验点名则换根或加强证据路径）。预算内解不出的草稿按过度模糊或欠指定修（收窄线索、去歧义、恢复约束）。

精炼两侧可见：易捷径草稿从 cost 33.9 / hit 12.4 到 82.7 / 31.4，prior-shortcut rate 从 17.0 到 12.0； initially 未解草稿精炼后可解，仍保留 cost 123.0 / hit 50.2。精炼是校准「可解但搜索重」，不是单纯加压。

## 训练与推理

基座：Qwen3-30B-A3B-Thinking-2507（MoE，推理约激活 3B/30B，256K 上下文）。仅 SFT：sequence packing、6 epochs、global batch 64、最大序列长度 262,144。

推理用 context-managed 协议：同一 rollout 内保留工具结果以复用证据；触达轮次上限仍无终答则清空交互史、从原问题重启。

## 主要证据

同规模开源 agent 中，FORT-Searcher 五基准总体平均 **66.2**，高于 MiroThinker-1.7-mini 的 64.6。BrowseComp **72.2**，BrowseComp-ZH **75.0**（所列开源含更大模型中最优），xbench-DeepSearch-2505 **80.8**，xbench-DeepSearch-2510 与同规模最优并列 **57.2**。例外是 Seal-0：**46.0** vs MiroThinker-1.7-mini 的 48.2（该基准偏噪声/冲突证据下的搜索增强推理，而非长程证据发现）。

累计消融（2K 题）：完整管线求解准确率 29.0（更高=更易），去掉全部控制升到 81.6；cost 从 141.9 落到 43.7，hit time 从 46.5 前移到 11.8，prior-shortcut rate 从 11.4 到 22.3。该去除顺序下，去掉 fuzzing 对难度打击最大；因是累计而非逐组件，排序依赖顺序。轨迹对照与标注代理数字见 [轨迹签名](/wiki/deep-search/trajectory-signatures/)。

## 局限

- 训练仅 SFT；与 RL 结合留待后续。
- Headline BrowseComp 很大一部分来自推理侧 context-management：关掉后 BrowseComp 从 72.2 到 55.9，BrowseComp-ZH 从 75.0 到 62.1，另三个基准变动小得多。与其他 agent 对比时应确认是否启用同类机制。
- Prior-shortcut rate 与轨迹代理的保守性、固定求解器依赖，见轨迹签名页。

## See Also

- [深度搜索概览](/wiki/deep-search/overview/)
- [实现难度与 Shortcut](/wiki/deep-search/realized-difficulty/)
- [轨迹签名](/wiki/deep-search/trajectory-signatures/)
- [Visual reading: FORT-Searcher](/files/visual-reading/fort-searcher-shortcut-resistant-search/)
