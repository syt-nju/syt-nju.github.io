---
title: "QLoRA"
topic: llm-quantization
summary: "QLoRA 用 NF4 冻结基座、高精 LoRA adapter 与 Double Quantization / Paged Optimizer，把大模型微调塞进单卡，但不是经典 QAT。"
lang: zh-CN
updated: 2026-08-10
order: 6
sources:
  - title: "QLoRA Deep Dive"
    url: "https://kuriko-iwai.com/research/qlora-efficient-llm-finetuning-nf4-double-quantization"
  - title: "第 7 章 量化：用更少的显存跑更大的模型"
    url: "https://inferloop.dev/llm-infra/quantization/"
raw:
  - raw/llm-quantization/qlora-nf4-double-quantization-deep-dive.md
  - raw/llm-quantization/inferloop-llm-infra-quantization.md
---

## Overview

QLoRA（Quantized LoRA）把基座存成 4-bit，只训练高精低秩 adapter。它回答的是「微调时如何塞得下」，不是「如何导出最优生产 INT4 权重」。

## 为什么需要混合精度

若整网用 4-bit 训练，极小梯度（如 $10^{-7}$ 量级）可能落在同一量化档，优化器失去有效更新。QLoRA 的折中：

- 基座权重：NF4 存储（省显存）
- LoRA adapter：16-bit 训练（保梯度）
- 量化常数：可用 Double Quantization 再压
- 优化器状态：Paged Optimizer 在显存尖峰时落到 CPU

目标能力表述：足以在单张 48GB GPU 上微调 70B 级模型；adapter 参数量通常不到全参的 1%。

## NF4

NF4（4-bit NormalFloat）假设权重大致正态，把 $-1$ 到 $1$ 分成 16 个非均匀档，零点附近更密，尾部更疏。相对均匀 INT4，对集中在 0 附近的权重更省量化误差。

反量化可写为 $w = n(Q_1) \times c_1$：$n(Q_1)$ 来自 LUT，$c_1$ 是 block 共享的 FP32 缩放。按每 block 32 个权重计，$c_1$ 均摊约 1 bit/param，叠加 4-bit 索引约 5 bit/param。

Block-wise 的意义：局部 outlier 不至于用一个全局 scale 毁掉整张量分辨率。

## Double Quantization

把 $c_1$（FP32）再量化成 8-bit 的 $Q_2$，并引入第二级常数 $c_2$：

$$
w = n(Q_1) \times (Q_2 \times c_2)
$$

从而再挤掉一级量化常数开销。

## Paged Optimizer

训练中激活 / 优化器状态会造成 VRAM 尖峰。Paged Optimizer 把溢出页到 CPU RAM，避免 OOM，用带宽换稳定性。

## 与 PTQ / QAT 的关系

| 路线 | 基座 | 训练什么 | 典型产物 |
|------|------|----------|----------|
| GPTQ / AWQ PTQ | 校准后写死低比特 | 通常不再训 | 可部署量化权重 |
| QAT | 伪量化后 convert | 全参或指定层 | 与 PTQ 同结构的更稳量化权重 |
| QLoRA | NF4 冻结 | LoRA adapter | adapter；推理可合并或保持 PEFT |

bitsandbytes 是 QLoRA 常见实现后端：`load_in_4bit` / NF4 即时量化利于实验与训练，但不像 AWQ 那样优先追求可分发的高速推理格式。

## 何时可以不用

显存充裕且追求最稳 adapter 质量时，BF16 LoRA 可能更简单。需要生产吞吐时，常在 BF16/FP8/AWQ 权重上部署，而不是直接把 QLoRA 训练时的 bnb 路径当最终 serving 格式。

## See Also

- [PTQ 与 QAT](/wiki/llm-quantization/ptq-vs-qat/)
- [LLM 上的 QAT](/wiki/llm-quantization/qat-for-llms/)
- [GPTQ 与 AWQ](/wiki/llm-quantization/gptq-awq/)
- [大模型量化问题地图](/wiki/llm-quantization/overview/)
