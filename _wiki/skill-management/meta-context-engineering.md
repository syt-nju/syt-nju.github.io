---
title: "Meta Context Engineering"
topic: skill-management
summary: "MCE 用双层优化共演化 CE skills 与 context artifacts：meta 层 agentic crossover，base 层以 files/code 做完全 agentic 上下文工程。"
lang: zh-CN
updated: 2026-08-07
order: 6
sources:
  - title: "Meta Context Engineering via Agentic Skill Evolution"
    url: "https://arxiv.org/abs/2601.21557"
  - title: "Meta-Harness: End-to-End Optimization of Model Harnesses"
    url: "https://arxiv.org/abs/2603.28052"
raw:
  - raw/skill-management/2026-01-29-meta-context-engineering.md
  - raw/skill-management/2026-03-30-meta-harness.md
---

## Overview

Meta Context Engineering（MCE）把 Context Engineering（CE）从「人工固定 harness + 预定 schema」改成可学习的 agentic 能力。它形式化为双层优化：外层演化规定「如何表示与学习 context」的 **CE skill**；内层在给定技能下优化 **context artifact**（files 与 code）。与本主题既有的任务技能库（Voyager / AutoSkill / MUSE 等）不同，MCE 的技能对象是 *做 CE 的策略本身*，而 ACE、GEPA 等静态 CE 管线被看作该设计空间里的单点。

代码仓库：[ace-agent/ace](https://github.com/ace-agent/ace)。

## 双层问题与编排

Context 函数 \(c\) 由静态组件 \(\rho\) 与动态算子 \(F\)（检索、筛选、格式化等）组成。MCE 引入技能 \(s\)，由 base-agent 执行得到 \(c_s\)，并求解：

\[
s^{*}=\arg\max_{s} J_{\mathrm{val}}(c_{s}^{*})\quad\text{s.t.}\quad c_{s}^{*}=\arg\max_{c_{s}} J_{\mathrm{train}}(c_{s};s).
\]

编排采用 history-informed \((1+1)\)-ES：每轮 meta-agent 生成一个 offspring skill → base-agent 产出 context → 用验证集与当前最优比较并保留更优者；技能历史 \(\mathcal{H}\) 记录 \((s,c,J_{\mathrm{train}},J_{\mathrm{val}})\)。meta / base 均使用通用工具集（Read / Write / Edit / Bash / Glob / Grep / TodoWrite），权限按角色与迭代限定。

## Meta：Agentic Skill Evolution

**Agentic crossover** 相对固定遗传重组：LLM agent 审议任务规格 \(\tau\)、任意检查历史技能文件夹与执行评估，再合成新技能。技能在 workspace 中是文件夹，实践中可含：自然语言方法论、可执行脚本、结构化模板、验证协议、以及按 query 过滤/组装的动态检索算子。

作者观察：演化技能会调节自主性与粒度（刚性工作流 vs 全权委托）、按任务与模型容量调节 verbosity，并利用 train/val 信号监测过拟合、偏向泛化。

## Base：Files/Code 上下文工件

Base-agent 在技能文件夹、上一轮最优 context、训练 rollouts 与可选 LLM/embedding 工具下执行 `Engineer`，把 context 写成目录中的静态知识与动态算子代码，而不预置 itemized list 等 schema。这与 ACE 的 generation–reflection–curation 固定管线形成对照：后者可被 MCE 重建，但不是唯一点。

## 实证要点

五域评测（FiNER、USPTO50k、Symptom2Disease、LawBench、Aegis2.0），主生成器 DeepSeek-V3.1（Aegis2.0 用 Qwen3-8B）。相对 base 的 Avg. Rel. Gain（Table 1）：offline **MCE 89.1**（ACE 70.7，GEPA 61.5）；online **MCE 74.1**（ACE 41.1）。相对 SOTA agentic CE 的相对改进为 **5.6–53.8%**（均值 **16.9%**）。相对 DeepSeek-V3.1 base，offline / online 平均相对提升分别为 89.1% / 74.1%，并分别高出 prior SOTA **18.4%** / **33.0%**（相对改进口径，见原文 Introduction）。

| Method | FiNER Acc.% | USPTO50k Acc.% | Symptom2Disease Acc.% | LawBench Micro-F1 | Aegis2.0 F1 | Avg. Rel. Gain% |
| --- | --- | --- | --- | --- | --- | --- |
| Base | 58.0 | 6.0 | 63.7 | 0.36 | 0.54 | – |
| ACE (offline) | 71.0 | 18.0 | 79.2 | 0.65 | 0.68 | 70.7 |
| MCE (offline) | 75.0 | 20.0 | 89.2 | 0.70 | 0.80 | 89.1 |
| ACE (online) | 64.0 | 13.0 | 62.3 | 0.63 | 0.57 | 41.1 |
| MCE (online) | 68.0 | 20.0 | 76.4 | 0.66 | 0.63 | 74.1 |

适应性：有效 context 长度可随任务约在 **1.5K–86K** tokens（FiNER 最优约 1.5K / 20K；LawBench / USPTO50k 可达 44K / 86K）。FiNER 效率：约 1.5K tokens 时 MCE-S 73% vs ACE Step-20 的 65%；MCE-L 20K tokens 达 75%，高于 ACE 5 epoch 的 70%（约 79K tokens）。训练：FiNER 上 MCE 5 epoch 约 **1.9** hours vs ACE **25.8** hours（约 \(13.6\times\)）；达 95% train acc 约 **450** rollouts vs ACE 峰值 94% 的 **2169**（约 \(4.8\times\) 更少）。强→弱迁移时，MCE 的 Avg. Rel. Drop 通常低于 ACE（如至 Qwen3-8B：Table 2 中 MCE 为 17.1，ACE 为 23.6）。

消融：online 的 MCE (w/o skills) Avg. Rel. Gain 为 Table 1 的 71.3，完整 MCE 为 74.1，说明技能演化有贡献但 base agentic CE 已很强。

## 局限与主题定位

作者自述：更利于领域知识与模式匹配；对已有强反思 harness 的推理密集型任务、以及超长复杂 trajectory 的细粒度 credit assignment 可能不占优。评测域是垂直 CE benchmark，**不是** SkillsBench；与 Ratchet / MUSE 关于「LLM 自写任务技能」的争议互补，不宜直接对撞。

相对本主题：MCE 把「技能」抬到 *学习算法 / harness* 层，用验证信号做技能级选择，而不是任务技能库的 librarian（add/merge/retire）。表示上仍落在可执行文件夹技能，与 [Skill Library](/wiki/skill-management/skill-library/) 中 Agent Skills 传统一脉。

## 与 Meta-Harness 的对照

后续工作 [Meta-Harness](/wiki/skill-management/meta-harness/) 在 GPT-OSS-120B 的在线文本分类协议下，把 MCE 与 ACE 一并作为手工 harness 基线：其 Table 2 报告 MCE 平均准确率 40.0（Ctx 28.5K）、ACE 40.9（50.8K）、所发现 harness 48.6（11.4K）。这是 **另一基座与绝对准确率口径** 下的比较，不能用来改写上文 DeepSeek-V3.1 的 Avg. Rel. Gain（offline 89.1 / online 74.1）。机制上：MCE 共演化 CE skill 与 context artifact；Meta-Harness 用 coding agent + 全历史 filesystem 直接搜索完整 harness 程序，并进一步评测数学检索与 TerminalBench-2。

## See Also

- [技能管理概览](/wiki/skill-management/overview/)
- [Meta-Harness](/wiki/skill-management/meta-harness/)
- [Skill Library](/wiki/skill-management/skill-library/)
- [技能生命周期](/wiki/skill-management/skill-lifecycle/)
- [MUSE-Autoskill](/wiki/skill-management/muse-autoskill/)
- [Skill Curation RL](/wiki/skill-management/skill-curation-rl/)
