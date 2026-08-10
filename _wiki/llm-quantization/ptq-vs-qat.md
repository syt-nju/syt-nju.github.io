---
title: "PTQ 与 QAT"
topic: llm-quantization
summary: "PTQ 在训后用校准或优化压精度；QAT 在训练中插入伪量化并用 STE 适应噪声，精度更高但成本更大。"
lang: zh-CN
updated: 2026-08-10
order: 3
sources:
  - title: "Quantization concepts"
    url: "https://huggingface.co/docs/transformers/en/quantization/concept_guide"
  - title: "Practical Quantization in PyTorch"
    url: "https://pytorch.org/blog/quantization-in-practice/"
  - title: "第 7 章 量化：用更少的显存跑更大的模型"
    url: "https://inferloop.dev/llm-infra/quantization/"
  - title: "模型量化算法详解：从 PTQ 到 AWQ 的大模型压缩实战指南"
    url: "https://developer.cloud.tencent.cn/article/2701881"
  - title: "Quantization-Aware Training for Large Language Models with PyTorch"
    url: "https://pytorch.org/blog/quantization-aware-training/"
raw:
  - raw/llm-quantization/hf-quantization-concepts.md
  - raw/llm-quantization/2024-11-15-pytorch-quantization-in-practice.md
  - raw/llm-quantization/inferloop-llm-infra-quantization.md
  - raw/llm-quantization/2026-07-02-tencent-ptq-to-awq-guide.md
  - raw/llm-quantization/2024-11-12-pytorch-qat-for-llms.md
---

## Overview

按误差消化时机，量化分成两条主路线：训练后量化（PTQ）与量化感知训练（QAT）。LLM 生产推理默认 PTQ；需要回收大幅精度损失时再上 QAT。

## PTQ：训完再压

PTQ 在模型已训完后施加量化，通常不需要完整训练数据与反向传播。

### 动态 / weight-only

权重预先量化；激活在推理时按当前输入动态标定。实现简单，对 LSTM / Transformer 一类 memory-bound 模型友好；但层间反复 float↔int 转换有开销。

### 静态 PTQ

权重预先量化，激活范围用代表性数据预先校准并固定，推理时激活可保持量化精度。PyTorch 流程可概括为：

1. fuse 可融合模块（如 Conv-BN-ReLU）
2. 插入 Quant / DeQuant stub
3. prepare：挂 observer
4. calibrate：喂代表性数据（文档称约 100 个 mini-batch 量级常够用；随机数校准会得到坏 qparams）
5. convert：换成真正量化模块

InferLoop 对 LLM PTQ 的操作版：

1. 加载 FP16 模型
2. 用校准集前向，收集权重 / 激活统计
3. 计算 scale、zero-point（或更复杂的层内补偿）
4. 写出低比特权重
5. 保存并评测

优点：快、便宜。缺点：INT2 等极端位宽或小模型上掉点更大。

## QAT：训练时假装已经被量化

QAT 在前向插入 fake quantization：先按低比特网格 round/clamp，再立刻反量化回浮点，从而把量化噪声写进 loss。权重仍以浮点存储与更新；反传通常用 Straight-Through Estimator（STE）穿过不可导的 round。

伪量化与真量化的差别可以写成：

```text
# PTQ / 真量化：cast 到低比特
x_q = (x / scale + zp).round().clamp(qmin, qmax).cast(int8)

# QAT 伪量化：仍保持 float，只模拟数值
x_fq = (x / scale + zp).round().clamp(qmin, qmax)
x_fq = (x_fq - zp) * scale
```

典型流程：插入伪量化 → 微调若干 epoch → convert 成与 PTQ 同结构的真实量化模型。

优点：尤其在低比特上精度通常优于纯 PTQ。缺点：需要训练数据与算力；开销可到数百 epoch（经典 DNN 文献语境），LLM 上则体现为更慢的 finetune。

## 怎么选

| 场景 | 更常见选择 |
|------|------------|
| 7B+ 服务端 INT4 推理 | PTQ（GPTQ / AWQ 等） |
| 小模型、INT2/INT3、端侧精度敏感 | 考虑 QAT |
| 只想低显存 SFT / 对齐 | 往往是 QLoRA，不是完整 QAT |

InferLoop 的实践判断：到 2025 年实践中 PTQ 仍是绝对主流；除非极端低比特或极高精度要求，否则先 PTQ。

## See Also

- [线性量化基础](/wiki/llm-quantization/linear-quantization/)
- [GPTQ 与 AWQ](/wiki/llm-quantization/gptq-awq/)
- [LLM 上的 QAT](/wiki/llm-quantization/qat-for-llms/)
- [QLoRA](/wiki/llm-quantization/qlora/)
