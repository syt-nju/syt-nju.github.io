---
title: "LLM 上的 QAT"
topic: llm-quantization
summary: "torchao / torchtune 的 LLM QAT 通过 prepare 插入伪量化、微调后再 convert，可在 8da4w 等设定上大幅回收相对 PTQ 的精度损失。"
lang: zh-CN
updated: 2026-08-10
order: 5
sources:
  - title: "Quantization-Aware Training for Large Language Models with PyTorch"
    url: "https://pytorch.org/blog/quantization-aware-training/"
  - title: "Practical Quantization in PyTorch"
    url: "https://pytorch.org/blog/quantization-in-practice/"
raw:
  - raw/llm-quantization/2024-11-12-pytorch-qat-for-llms.md
  - raw/llm-quantization/2024-11-15-pytorch-quantization-in-practice.md
---

## Overview

面向 LLM 的 QAT 不是另起一套模型结构，而是在与目标 PTQ 相同的量化设定下，先用伪量化微调，再 convert 成可复用同一套 kernel 的量化模型。

## prepare → train → convert

torchao 流程两步：

1. **prepare**：在 linear 层插入 fake quantize，前向模拟量化数值，但不做低比特 dtype cast。
2. **convert**：训练结束后把伪量化换成真实 quantize / dequantize，得到与对应 PTQ quantizer 同结构的模型。

实验中常用设定是 int8 per-token dynamic activations + int4 grouped per-channel weights（简称 8da4w），动机来自端侧 kernel 可用性与 LLM 量化研究中「per-token 激活 + per-group 权重」质量较好的经验。

## 与后训练的衔接方式

预训练全量数据往往不可及，因此实践是在 finetune 阶段做 QAT。PyTorch blog 在 C4（en）上对 Llama2-7B / Llama3-8B 跑 5000 steps：batch size 2，lr \(2\times10^{-5}\)，Llama2 max seq 4096、Llama3 8192，FSDP + activation checkpointing；8da4w 权重 group size 256。

经验技巧：前 N 步关闭伪量化，让权重先稳定再引入量化噪声；实验统一用前 1000 steps 关闭。

## 能回收多少精度

相对同设定 PTQ：

- Llama3-8B 8da4w：hellaswag 归一化精度掉点可回收约 96%，相对未量化掉点可到 <1%；wikitext word / byte PPL 掉点分别回收约 68% / 65%。
- Llama2-7B 8da4w：hellaswag 掉点回收约 62%；wikitext word / byte PPL 掉点约 58% / 57%。
- 落到 XNNPACK 后：QAT 相对 PTQ wikitext word perplexity 低 16.8%（23.316 → 19.403），模型体积同为 3.881 GB，端侧速度同量级。

低比特 weight-only 更残酷。Llama3-8B 2-bit 上纯 PTQ 的 wikitext word PPL 可爆到 603336；跳过最敏感的前 3 层与后 2 层后降到 6766；再叠加 QAT 可到约 30。跳过敏感层时，QAT 相对 PTQ 可回收 hellaswag 掉点约 53%，wikitext word / byte PPL 掉点约 99% / 89%。3-bit 上即便不跳层 QAT 也有效，跳层仍更好。

## 成本

伪量化遍布整网。Llama3-8B 约有 \((32 \times 7) + 1 = 225\) 个 linear，每个至少有权重伪量化，还可能有激活伪量化。微基准：8da4w QAT finetune 约比普通 full finetune 慢 34%；开启 activation checkpointing 后每 GPU 显存约多 2.35 GB（中位峰值 67.501 → 69.850 GB；吞吐 546.314 → 359.637 tok/s）。

## 边界

- QAT ≠ 真低比特训练；后者实际 cast 到低比特，既往成功多停在 8-bit，而 QAT 在更低比特仍有效。
- 当前 torchtune 集成以 full finetune 为主；与 LoRA / QLoRA 组合仍是展望项。
- 多数服务端 INT4 仍可先 PTQ；QAT 更适合端侧、低比特或 PTQ 掉点不可接受时。

## See Also

- [PTQ 与 QAT](/wiki/llm-quantization/ptq-vs-qat/)
- [QLoRA](/wiki/llm-quantization/qlora/)
- [GPTQ 与 AWQ](/wiki/llm-quantization/gptq-awq/)
