---
title: "技能获取效率"
topic: test-time-training
summary: "智力被定义为在给定任务范围上，相对先验、经验与泛化难度，把信息转成新技能的效率；任务技能本身可以被先验或数据买到。"
lang: zh-CN
updated: 2026-08-17
order: 2
sources:
  - title: "On the Measure of Intelligence"
    url: "https://arxiv.org/abs/1911.01547"
raw:
  - raw/test-time-training/2019-11-05-on-the-measure-of-intelligence.md
---

## Overview

Chollet 把历史上两种智力观拆开：结晶技能（在已知任务上达到目标）与获取新技能的能力。当代 AI 评价默认站在前一侧，用棋类、电子游戏等固定任务上的技能当智力。问题是技能高度受先验和经验调制。原文：[On the Measure of Intelligence](https://arxiv.org/abs/1911.01547)。

## 技能可以被买到 {#buy-skill}

无上限先验（把解硬编码进程序）或无上限训练数据，都能在给定任务上抬高技能，而不增加系统处理未来不确定性的能力。两者都是在 prior/experience 平面上移动，与泛化轴正交。Deep Learning 在这篇里被放在 local-generalization 一端：概念上接近 locality-sensitive hashtable，要任意高技能就需要对输入–目标空间做稠密采样。原文：[Chollet 2019 II.1](https://arxiv.org/abs/1911.01547)。

因此，在开发者事先知道的任务上测技能，不能当作对 broad abilities 或 general intelligence 的测量。技能是智力过程的结晶输出，不是智力本身。

## 智力是技能获取效率 {#skill-acquisition-efficiency}

正式表述：一个系统的智力，是它在某组任务范围（scope）上的 skill-acquisition efficiency，并且必须相对 priors、experience 和 generalization difficulty 来读。示意贡献是

`Expectation[skill · generalization / (priors + experience)]`

再按任务价值加权、对 scope 取平均。白话版：智力是学习者把它的经验和先验，转成有价值、涉及不确定性与适应的新技能的速率。原文：[Chollet 2019 II.2](https://arxiv.org/abs/1911.01547)。

这一定义要求学习与适应。系统若一开始就能在评价情境上表现很好，则该任务的 [developer-aware generalization difficulty](/wiki/test-time-training/measuring-general-intelligence/#developer-aware) 很低，智力分数也会低。

## 必须控制的四个量 {#priors-experience}

- **Scope：** 智力相对任务范围；两个系统只有在共享 scope、且先验可比时才能比。
- **Priors：** 初始系统里与解相关的信息比例，不是系统体积。无关知识只付索引与检索开销。
- **Experience：** 课程中相关且对系统而言新颖的信息；不惩罚噪声课程，也不把已经吸收后的重复步骤算进快学习者。
- **Generalization difficulty：** 训练时最短最优解，要改多少才能成为评价时的充分解。最简单的训练时策略若已够用，则任务没有需要适应的不确定性。

人类既不是进化写死的专用程序集合，也不是空白石板。先天先验不是泛化的限制，而是其来源：No Free Lunch 要求从数据学习必须做假设。对人机公平比较，应把人类 [Core Knowledge](/wiki/test-time-training/measuring-general-intelligence/#core-knowledge) 当作参照先验。原文：[Chollet 2019 II.1.3](https://arxiv.org/abs/1911.01547)。

## 尚未展开的效率轴

II.2.2 还列出 skill program 的计算效率，以及训练时计算、时间、能量、风险。这些量是后续「测试时多算」与「测试时改参」的接口，本页不把它们写成已有方法比较。

## See Also

- [测试时训练问题地图](/wiki/test-time-training/overview/)
- [通用智能测量](/wiki/test-time-training/measuring-general-intelligence/)
- [测试时参数更新](/wiki/test-time-training/parameter-update/)
