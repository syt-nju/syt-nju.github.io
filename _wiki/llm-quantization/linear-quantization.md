---
title: "线性量化基础"
topic: llm-quantization
summary: "线性量化用 scale 与 zero-point 把浮点映射到低比特整数，粒度与对称性决定精度与算力折中。"
lang: zh-CN
updated: 2026-08-10
order: 2
sources:
  - title: "Quantization concepts"
    url: "https://huggingface.co/docs/transformers/en/quantization/concept_guide"
  - title: "Practical Quantization in PyTorch"
    url: "https://pytorch.org/blog/quantization-in-practice/"
  - title: "第 7 章 量化：用更少的显存跑更大的模型"
    url: "https://inferloop.dev/llm-infra/quantization/"
  - title: "模型量化算法详解：从 PTQ 到 AWQ 的大模型压缩实战指南"
    url: "https://developer.cloud.tencent.cn/article/2701881"
  - title: "当谈论 FP8 训练的时候，我们到底在聊什么?"
    url: "https://qingkeai.online/archives/aaxcavut"
raw:
  - raw/llm-quantization/hf-quantization-concepts.md
  - raw/llm-quantization/2024-11-15-pytorch-quantization-in-practice.md
  - raw/llm-quantization/inferloop-llm-infra-quantization.md
  - raw/llm-quantization/2026-07-02-tencent-ptq-to-awq-guide.md
  - raw/llm-quantization/2025-11-09-fp8-training-recipes.md
---

## Overview

线性量化是后续 PTQ / QAT / GPTQ / AWQ 的共同底座：先选定浮点裁剪范围，再映射到目标整数网格，推理时按需要反量化或直接做整数算子。

## 映射公式

对浮点值 \(x\)，常用仿射量化：

\[
q = \mathrm{round}(x / S + Z),\quad x \approx S \cdot (q - Z)
\]

- \(S\)（scale）：浮点范围与整数范围之比。
- \(Z\)（zero-point）：浮点 0 对应的整数，保证零值可精确表示。

映射误差 \(x - \tilde{x}\) 即量化噪声。目标是用合适的方案、粒度与校准，把噪声对任务指标的影响压到可接受范围。

## 对称与非对称

| 方案 | 范围假设 | 参数 | 直观取舍 |
|------|----------|------|----------|
| Symmetric | 关于 0 对称，\(Z=0\) | 主要靠 \(S\) | 算子更简单；偏斜分布会浪费网格 |
| Asymmetric / Affine | 用真实 \([\min,\max]\) | \(S\) 与 \(Z\) | 更贴分布；权重量化时算力开销更大 |

PyTorch 实践指出：非负激活更适合仿射；权重量化常偏好对称 per-channel，因为跨通道方差大时 per-tensor 表现差。

## 粒度

- **Per-tensor**：整张量共用一组 \(S,Z\)，实现简单，分布不均时误差大。
- **Per-channel**：每个输出通道一组参数，权重量化更常用。
- **Per-group / block**：例如每 128 个权重一组，是 LLM INT4 的常见折中：比 per-tensor 细，比 per-channel 全量元数据更省。

训练期 FP8 把同一直觉推到 tile：DeepSeek-V3-like blockwise 常用 activation/gradient 的 `1×128`（沿 token 的 1D）与 weight 的 `128×128`（2D）；MXFP8 更细到 `1×32` 并共享 E8M0 scale。粒度不仅影响精度，还约束 TP/EP 通信能否做数值等价的 FP8 传输。详见 [FP8 训练 Recipe](/wiki/llm-quantization/fp8-training/)。

InferLoop 给出经验数字：Qwen2.5-7B 约 72 亿参数时，FP16 参数约 14 GB，INT4（GPTQ）参数约 3.5 GB；实际推理显存还会加上 KV Cache 与激活。

## INT4 packing 与 FP8

INT4 只有 16 档。多数硬件不能原生存 4-bit，因此常把两个 INT4 pack 进一个 INT8 字节。即便没有原生 INT4 算力，带宽减半仍可加速 memory-bound 推理；但精度掉点通常大于 INT8，需要 GPTQ / AWQ 一类方法。

FP8 保留浮点结构（常见 E4M3 / E5M2），推理侧 A8W8 可同时压激活与权重；训练侧则是 TE/MCore recipe 把 linear GEMM 输入量化到 FP8（Hybrid 时 gradient 常用 E5M2）。高效执行依赖 Hopper/Blackwell 等硬件；无原生加速时收益有限。

## 校准在基础层做什么

校准（calibration）是选定裁剪范围 \([\alpha,\beta]\) 并由此算 \(S,Z\) 的过程。常见观察器包括 running min/max、滑动平均 min/max、直方图；也可用 KL / MSE / 分位数。不同 observer 给出的 qparams 不同，需要按任务实证。

## See Also

- [PTQ 与 QAT](/wiki/llm-quantization/ptq-vs-qat/)
- [GPTQ 与 AWQ](/wiki/llm-quantization/gptq-awq/)
- [FP8 训练 Recipe](/wiki/llm-quantization/fp8-training/)
- [大模型量化问题地图](/wiki/llm-quantization/overview/)
