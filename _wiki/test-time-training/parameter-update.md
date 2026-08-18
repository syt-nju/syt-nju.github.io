---
title: "测试时参数更新"
topic: test-time-training
summary: "TTT 把当前无标签测试样本做成自监督问题，先更新共享特征再预测；辅助任务与主任务的梯度需要正相关。"
lang: zh-CN
updated: 2026-08-17
order: 4
sources:
  - title: "Test-Time Training with Self-Supervision for Generalization under Distribution Shifts"
    url: "https://arxiv.org/abs/1909.13231"
raw:
  - raw/test-time-training/2019-09-29-test-time-training.md
---

## Overview

Sun 等人把 Test-Time Training（TTT）定义成：测试样本 $x$ 没有主任务标签 $y$，但仍给出关于测试分布的提示；让参数 $\theta$ 依赖 $x$ 而不依赖 $y$。做法是把单张无标签图做成自监督问题，更新共享特征提取器，再用原来的主任务头预测。原文：[Sun et al. 2020](https://arxiv.org/abs/1909.13231)。

这条机制回答的是 [local generalization](/wiki/test-time-training/skill-acquisition-efficiency/#skill-acquisition-efficiency) 下的分布偏移，不是 [ARC](/wiki/test-time-training/measuring-general-intelligence/#arc-2019) 那种对开发者未知的新任务。辅助损失的配方会变，但「用当前测试输入上的可计算损失改权重」会被后续来源反复更新。

## 决策边界可以依赖 $x$ {#theta-of-x}

标准经验风险最小化在训练集 $P$ 上得到固定 $\theta$。TTT 允许 $\theta(x)$：先在 $x$ 上最小化辅助损失 $l_s(x)$，再在更新后的共享层上算主任务。网络做成 Y 形：共享底部 $\theta_e$，主任务分支 $\theta_m$，自监督分支 $\theta_s$。训练时在 $P$ 上联合优化 $l_m+l_s$（joint training）；测试时只对 $\theta_e$ 做辅助损失上的梯度步，$\theta_m$ 保持训练结束时的值。原文：[Sun et al. 2020 §2](https://arxiv.org/abs/1909.13231)。

对照实验必须包含「训练时也用自监督、测试时冻住」的 joint training。Hendrycks 等人已表明训练时自监督能提高鲁棒性；TTT 的增量是测试时还要再走一步。

## 标准版与在线版 {#standard-online}

- **标准 TTT：** 每张测试图从联合训练得到的 $\theta$ 重新初始化，在该图的增强副本上走若干步，预测后丢掉 $\theta_e^*$。
- **TTT-Online：** 测试样本顺序到达且分布相同或平滑变化时，$x_t$ 的优化从 $\theta(x_{t-1})$ 接着走，保留对已见测试分布的记忆。

本稿实验里的自监督任务是把图旋转 0 / 90 / 180 / 270 度并做四分类（Gidaris et al. 2018）。增强与训练时相同（随机裁剪与水平翻转），一个 batch 只含同一张图的增强副本。因此用 Group Normalization 而不是 Batch Normalization：单图小 batch 会让 BN 统计失真。原文：[Sun et al. 2020 §2–3](https://arxiv.org/abs/1909.13231)。

操作要点（只适用于这篇视觉设定，不是后续 LLM TTT 的默认超参）：测试时学习率取训练最后一轮的 0.001，weight decay 与 momentum 为 0；标准版每图 10 步，在线版每图 1 步。

## 何时会有帮助 {#gradient-correlation}

凸模型上的充分条件是：主任务损失与自监督损失在共享特征上的梯度正相关。直观上，两个任务要在同一方向上犯错，决策边界也要相关。非凸的 ResNet 实验里，15 种腐蚀 × 5 个等级共 75 个测试集上，梯度内积与误差改进的线性相关系数为 0.93（标准版）和 0.89（在线版）。原文：[Sun et al. 2020 §4](https://arxiv.org/abs/1909.13231)。

辅助任务必须既良定义又非平凡。VID-Robust 里 airplane 几乎不涨：多数图两侧有黑边，旋转可被投机解掉；天空中的飞机人也难判断是否旋转。

## 证据（分布偏移，不是新任务） {#ood-evidence}

相对「只做识别」和「joint training 但测试时冻住」：

- CIFAR-10 原测试集误差：识别 8.9%，joint training 8.1%，TTT 再降 0.2%。作者强调这与许多鲁棒方法用原分布性能换偏移鲁棒不同。
- CIFAR-10-C level 5：标准版相对 joint training 总有改进；TTT-Online 常再改进超过 10%，三种噪声上超过 24%，pixelation 上 38%，且相对 joint training 的伤害不超过 0.2%。
- 即使无监督域适应方法 UDA-SS 能在训练时看到整个无标签测试集，TTT-Online 在 15 种腐蚀中的 13 种以及原分布上仍然更好。解释是 TTT-Online 只需适应当前测试分布，可以忘掉训练分布表征。
- CIFAR-10.1（2000 张、对人几乎看不出差异的新测试集）：识别误差 17.4、joint training 16.7、TTT 15.9（表中单位为 %）。作者称这是当时第一个能把已有模型在该集上推高的方法；相对 joint training 的 0.8 个百分点相对原分布腰斩仍然很小。

原文：[Sun et al. 2020 §3](https://arxiv.org/abs/1909.13231)。

这些数字测的是已知任务、未知腐蚀或采集偏移，属于 Chollet 谱上的 local generalization。后续 ARC 式 TTT 会把同一「测试时改 $\theta$」接到有示范对的监督损失上；Tent 会把损失换成熵、把可更新参数缩到 BN affine。那些对照等对应来源再写入。

## See Also

- [测试时训练问题地图](/wiki/test-time-training/overview/)
- [技能获取效率](/wiki/test-time-training/skill-acquisition-efficiency/)
- [通用智能测量](/wiki/test-time-training/measuring-general-intelligence/)
