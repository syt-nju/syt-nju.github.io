---
title: "大模型量化问题地图"
topic: llm-quantization
summary: "大模型量化的核心问题是如何在降精度省显存的同时控制量化噪声，以及何时用校准式 PTQ、何时回到 QAT 或 QLoRA 训练。"
lang: zh-CN
updated: 2026-08-10
order: 1
sources:
  - title: "Quantization concepts"
    url: "https://huggingface.co/docs/transformers/en/quantization/concept_guide"
  - title: "Practical Quantization in PyTorch"
    url: "https://pytorch.org/blog/quantization-in-practice/"
  - title: "第 7 章 量化：用更少的显存跑更大的模型"
    url: "https://inferloop.dev/llm-infra/quantization/"
  - title: "Understanding Post-Training Quantization with LLM Compressor"
    url: "https://huggingface.co/blog/rishiraj/llm-compressor"
  - title: "Quantization-Aware Training for Large Language Models with PyTorch"
    url: "https://pytorch.org/blog/quantization-aware-training/"
  - title: "QLoRA Deep Dive"
    url: "https://kuriko-iwai.com/research/qlora-efficient-llm-finetuning-nf4-double-quantization"
  - title: "当谈论 FP8 训练的时候，我们到底在聊什么?"
    url: "https://qingkeai.online/archives/aaxcavut"
raw:
  - raw/llm-quantization/hf-quantization-concepts.md
  - raw/llm-quantization/2024-11-15-pytorch-quantization-in-practice.md
  - raw/llm-quantization/inferloop-llm-infra-quantization.md
  - raw/llm-quantization/2026-06-17-understanding-ptq-with-llm-compressor.md
  - raw/llm-quantization/2024-11-12-pytorch-qat-for-llms.md
  - raw/llm-quantization/qlora-nf4-double-quantization-deep-dive.md
  - raw/llm-quantization/2025-11-09-fp8-training-recipes.md
---

## Overview

大模型量化不是单一算法，而是一组互相耦合的选择：把哪些张量降到多少比特、用什么映射与粒度、误差在校准阶段还是训练阶段消化、以及最终服务场景是 GPU 推理、端侧部署、低显存微调，还是大规模预训练里的 FP8 混合精度。

本 topic 先回答五类问题：数值如何映射、PTQ 与 QAT 的流程边界、LLM 上主流 INT4 权重量化、后训练里的 QLoRA，以及训练期 FP8 recipe。

## 问题地图

### 精度怎么降下去

量化把浮点范围映射到更少的离散档位，引入 scale / zero-point，并在 per-tensor、per-channel、per-group 等粒度上折中精度与元数据开销。INT4 还常依赖 packing；FP8 则依赖新硬件。详见 [线性量化基础](/wiki/llm-quantization/linear-quantization/)。

### 误差在什么时候消化

PTQ 在训完后用校准数据估参数或做层内补偿；QAT 在训练前向插入伪量化，用 STE 让模型适应量化噪声。生产推理多走 PTQ；低比特或精度敏感时才回到 QAT。详见 [PTQ 与 QAT](/wiki/llm-quantization/ptq-vs-qat/)。

### LLM INT4 为什么不只是四舍五入

朴素 RTN 在 INT4 上常崩。GPTQ 用 Hessian 做误差补偿；AWQ 保护高激活幅度通道。常见配方是 W4A16，并常跳过 `lm_head`。详见 [GPTQ 与 AWQ](/wiki/llm-quantization/gptq-awq/)。

### 后训练怎么和量化一起用

真正的 QAT 是 prepare → finetune → convert。更常见的低显存微调是 QLoRA：NF4 冻结基座 + 高精 LoRA adapter，不是经典 QAT。详见 [LLM 上的 QAT](/wiki/llm-quantization/qat-for-llms/) 与 [QLoRA](/wiki/llm-quantization/qlora/)。

### 大规模训练里的 FP8 怎么做

预训练 FP8 是 TE/MCore recipe：加速 linear 的三个 GEMM，并处理 Hopper/Blackwell layout、primary weights 与 TP/EP 通信。它与推理 FP8 PTQ 不是同一条流水线。详见 [FP8 训练 Recipe](/wiki/llm-quantization/fp8-training/)。

## 共同主张

- 量化的主收益首先是显存与带宽，不一定等于算力加速；weight-only 场景尤其如此。
- 对多数 7B+ 服务端 INT4 推理，校准式 PTQ（尤其 AWQ / GPTQ）已够用。
- QAT 在更低比特或端侧 8da4w 等设定上能显著回收精度，但训练成本更高。
- QLoRA 解决的是「训练时塞得下基座」，与「导出可部署量化权重」不是同一条流水线。
- FP8 训练的显存/通信收益取决于 recipe 与是否启用 FP8 primary weights，不能默认减半。

## See Also

- [线性量化基础](/wiki/llm-quantization/linear-quantization/)
- [PTQ 与 QAT](/wiki/llm-quantization/ptq-vs-qat/)
- [GPTQ 与 AWQ](/wiki/llm-quantization/gptq-awq/)
- [LLM 上的 QAT](/wiki/llm-quantization/qat-for-llms/)
- [QLoRA](/wiki/llm-quantization/qlora/)
- [FP8 训练 Recipe](/wiki/llm-quantization/fp8-training/)
