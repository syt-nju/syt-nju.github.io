---
title: "MXFP4 推理量化"
topic: llm-quantization
summary: "MXFP4 用 E2M1 四比特浮点加每 32 元一块的 block scale，让 gpt-oss 等模型以远低于 BF16 的显存跑推理，并依赖专用 Triton kernel。"
lang: zh-CN
updated: 2026-08-11
order: 8
sources:
  - title: "Tricks from OpenAI gpt-oss YOU can use with transformers"
    url: "https://huggingface.co/blog/faster-transformers"
raw:
  - raw/llm-quantization/2025-09-11-hf-faster-transformers-gpt-oss.md
---

## Overview

MXFP4 是面向推理的 **4-bit 浮点 + 块级缩放**格式。相对 GPTQ/AWQ 的校准式 INT4，它更像「权重以 FP4 microscaling 原生交付」；相对训练侧 FP8/MXFP8 recipe，它解决的是**能否把大模型塞进单卡并高效做带 scale 的 GEMM**，不是预训练混合精度流水线。

本页证据主要来自 Hugging Face 对 gpt-oss 在 `transformers` 中的工程说明；**精度掉点数字本源未给**。

## 数值格式

`MXFP4` 元素布局为 **E2M1**：1 sign + 2 exponent + 1 mantissa。单独的 E2M1 档位很粗，靠 **blockwise scaling** 补动态范围：

- 向量按 **32 个元素**分块
- 每块存一个共享 scale，反量化时恢复量级
- 块内 4-bit 值相对该 scale 表示

因此 matmul / fusion 必须感知 block scale；没有专用 kernel 时，难以同时保住显存收益与可接受吞吐。

## 显存与模型实例

在 MXFP4 路径开启时，HF 给出的量级是：

- GPT-OSS 20B：大约 **16 GB** VRAM
- GPT-OSS 120B：大约 **80 GB** VRAM

不满足 MXFP4 运行条件时，`transformers` 默认落到更高精度路径（常用 **bfloat16**），显存大约是 MXFP4 的 **约 4×**。可用 `Mxfp4Config(dequantize=True)` 显式走反量化、吃满显存的对照路径。

## transformers 路径

若配置里出现 `'quant_method': 'mxfp4'`，在支持的环境下会自动走 MXFP4 + Triton kernel。gpt-oss 示例里 **`modules_to_not_convert`** 包含：

- `model.layers.*.self_attn`
- `model.layers.*.mlp.router`
- `model.embed_tokens`
- `lm_head`

即 attention、router、embedding、lm head 保持更高精度；可量化主体主要是其余线性/专家权重。微调后也可直接以 MXFP4 格式存回 Hub。

### 运行依赖

1. 安装 `accelerate`、`kernels`、`triton>=3.4`（PyTorch 2.8 已带 triton 3.4；2.7 需手动装）
2. NVIDIA GPU compute capability **≥ 7.5**

MXFP4 感知的 Triton kernel 由 Hub 上的 `kernels-community/triton_kernels` 等仓库自动拉取，**不必**再传 `use_kernels=True`。

### 与其它 Hub kernels 的冲突

Liger RMSNorm、MegaBlocks MoE 等需 `use_kernels=True` 的自定义 kernel **与 mxfp4 不兼容**；启用后推理会落到 `bfloat16`。选路径时要在「MXFP4 省显存」与「其它自定义 MoE/RMSNorm kernel」之间自行 benchmark。

## 与相邻路线的边界

| 路线 | 典型形态 | 和 MXFP4 的差别 |
|------|----------|-----------------|
| GPTQ / AWQ | 校准式 INT4 权重量化（常 W4A16） | 整数网格 + 校准/补偿；不是 E2M1 + block FP scale |
| QLoRA / NF4 | 训练时冻结 4-bit 基座 | 目标是低显存微调，不是 MXFP4 推理交付格式 |
| FP8 训练 | TE/MCore 把 GEMM 输入打到 FP8 | 训练 recipe；MXFP8 的 `1×32` scale 是同族「microscaling」直觉，但场景不同 |

## 稳定主张

- MXFP4 的核心折中是：**极低比特浮点 + 32-block scale**，用元数据换动态范围。
- 工程上能否落地，取决于 **感知 block scale 的 GEMM kernel**，不只取决于权重文件是否 4-bit。
- 对 gpt-oss 一类模型，MXFP4 首先回答「单卡能不能装下」；吞吐还取决于 kernel 与是否误开不兼容的其它加速路径。

## See Also

- [线性量化基础](/wiki/llm-quantization/linear-quantization/)
- [GPTQ 与 AWQ](/wiki/llm-quantization/gptq-awq/)
- [FP8 训练 Recipe](/wiki/llm-quantization/fp8-training/)
- [大模型量化问题地图](/wiki/llm-quantization/overview/)
