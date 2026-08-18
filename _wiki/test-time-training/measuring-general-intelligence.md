---
title: "通用智能测量"
topic: test-time-training
summary: "通用基准应控制先验与经验、测量 developer-aware generalization，并只假设人类 Core Knowledge；ARC 是这条契约的 2019 年测量物。"
lang: zh-CN
updated: 2026-08-17
order: 3
sources:
  - title: "On the Measure of Intelligence"
    url: "https://arxiv.org/abs/1911.01547"
raw:
  - raw/test-time-training/2019-11-05-on-the-measure-of-intelligence.md
---

## Overview

[技能获取效率](/wiki/test-time-training/skill-acquisition-efficiency/#skill-acquisition-efficiency) 要变成可执行的反馈信号，基准必须让开发者无法用事先知道的任务、隐式先验或无限练习数据买分。Chollet 给出一组通用智能基准指南，并用 Abstraction and Reasoning Corpus（ARC）做第一次落地。原文：[On the Measure of Intelligence](https://arxiv.org/abs/1911.01547)。

## Developer-aware generalization {#developer-aware}

System-centric generalization 只问系统自己没见过的情境。把开发者算进系统之后，同一概念变成 developer-aware generalization：评价任务不能被测试系统或其开发者事先知道，否则测到的可能是工程师把解写进程序的智力。原文：[Chollet 2019 I.3.2](https://arxiv.org/abs/1911.01547)。

多任务榜若任务全部事先公开，仍不满足这条。CoinRun、Obstacle Tower 这类「新关卡」属于对已知分布的 local generalization，因为关卡生成器对开发者可见。

## 基准应控制的量

一篇通用基准至少应：

- 测量 broad abilities 与 developer-aware generalization，评价集不含事先已知任务
- 说明自己测的是 local、broad 还是 extreme generalization
- 控制训练时可获得的经验，避免用无限新数据买分
- 显式、穷尽地列出假设的先验
- 对人机公平：只假设人类具备的先验（例如 Core Knowledge），并只要求人类量级的练习时间或训练数据

原文：[Chollet 2019 II.3](https://arxiv.org/abs/1911.01547)。

## Core Knowledge 先验 {#core-knowledge}

ARC 把先验钉在 Spelke 意义上的 Core Knowledge，并避免语言、真实物体图片和需练习才能获得的常识。四类是：

- **Objectness：** 按颜色连续或空间邻接解析物体，物体在噪声或遮挡下持续，接触可产生作用
- **Goal-directedness：** 许多格子对可被理解为有意图过程的起止状态
- **Numbers and Counting：** 计数、排序、比较数量；涉及的量大约小于 10
- **Basic Geometry and Topology：** 线与矩形、对称、旋转平移、缩放、包含关系、投影与复制

原文：[Chollet 2019 III.1.2](https://arxiv.org/abs/1911.01547)。

## ARC 作为测量物 {#arc-2019}

ARC 同时被写成通用智能基准、程序合成基准和心理测量测验。训练集约 400 题，评价集约 600 题，其中公开评价 400、私有评价 200；训练与测试任务不相交。每题平均 3.3 个示范，测试例通常为 1 个。格子符号共 10 种，高宽在 1x1 与 30x30 之间。作答应从零构造输出格子；每道测试例允许 3 trials，反馈为对错；整题对全部测试例都对才算成功。分数是评价集上解出的任务比例。原文：[Chollet 2019 III.1.1](https://arxiv.org/abs/1911.01547)。

设计意图是：评价任务对开发者未知，经验被固定示范对卡住，先验被 Core Knowledge 穷尽。私有评价集用来在竞赛里强制 developer-aware generalization。人可以在无专门训练的情况下解其中多数题；2019 年的判断是，当时的机器学习（含 Deep Learning）无法有意义地逼近它。

本稿把 ARC 写成测量契约的证据，不建成系统页。网格规格、尝试次数和 400 / 600 划分会被后续竞赛协议改写。

## 已知弱点

Chollet 自己列出：泛化难度未量化、测验效度未建立、全集约 1,000 题且可能概念重叠、计分过于 0 / 1、交互不够、Core Knowledge 是否被正确捕捉仍开放。更贴近定义的格式会让应试者向例题生成器要新输入、提交假设并按所需反馈量计分。原文：[Chollet 2019 III.2](https://arxiv.org/abs/1911.01547)。

## See Also

- [测试时训练问题地图](/wiki/test-time-training/overview/)
- [技能获取效率](/wiki/test-time-training/skill-acquisition-efficiency/)
- [测试时参数更新](/wiki/test-time-training/parameter-update/)
