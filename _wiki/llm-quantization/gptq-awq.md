---
title: "GPTQ 与 AWQ"
topic: llm-quantization
summary: "GPTQ 用 Hessian 做层内误差补偿，AWQ 保护高激活通道；两者都是面向 LLM 的校准式 INT4 权重量化，常见配方为 W4A16。"
lang: zh-CN
updated: 2026-08-10
order: 4
sources:
  - title: "Understanding Post-Training Quantization with LLM Compressor"
    url: "https://huggingface.co/blog/rishiraj/llm-compressor"
  - title: "第 7 章 量化：用更少的显存跑更大的模型"
    url: "https://inferloop.dev/llm-infra/quantization/"
  - title: "模型量化算法详解：从 PTQ 到 AWQ 的大模型压缩实战指南"
    url: "https://developer.cloud.tencent.cn/article/2701881"
raw:
  - raw/llm-quantization/2026-06-17-understanding-ptq-with-llm-compressor.md
  - raw/llm-quantization/inferloop-llm-infra-quantization.md
  - raw/llm-quantization/2026-07-02-tencent-ptq-to-awq-guide.md
---

## Overview

朴素 Round-to-Nearest（RTN）把权重丢进最近的低比特桶，INT4 上常直接毁掉推理质量，因为权重重要性不均。LLM 实用 INT4 依赖更聪明的校准式 PTQ；两条主流哲学是 GPTQ 与 AWQ。

## 为什么不是随便 round

权重与激活相乘。若某权重连着持续尖峰激活，其舍入误差会被放大并跨层传播。RTN 对所有权重一视同仁，因此容易失败。

## GPTQ：Hessian 补偿

GPTQ 逐层、按列量化，并问：量化某个权重引入误差后，如何调整同层剩余未量化权重来补偿？

做法依赖对输出损失相对权重的 Hessian（曲率 / 敏感度）。校准前向得到层输入激活 \(X\) 后，目标常写成最小化 \(\|XW - XW'\|^2\)，对应 \(H = X^\top X\)。用途包括：

- 敏感度：对角元大的权重扰动更伤输出
- 误差补偿：用 \(H^{-1}\) 估计量化掉某权重后其余权重应如何偏移

代价是压缩阶段需要更多显存来算 Hessian；收益是 INT4 下更稳。工具链上 AutoGPTQ 已归档，常见继任是 GPTQModel；vLLM 侧也可用 `llm-compressor` 的 GPTQ modifier。

## AWQ：激活感知缩放

AWQ 观察：少数通道对应的激活幅度特别大，这些通道的量化误差会被放大。流程是：

1. 用校准数据统计通道激活幅度
2. 识别重要通道
3. 量化前放大对应权重（提高有效精度），并相应缩小激活以保持数学等价
4. 其余权重可更激进地量化

相对 GPTQ，AWQ 通常更轻、更快（InferLoop 称常约 2–3 倍量化速度），且在同等比特下常有更好精度保持。生产 INT4 GPU 推理里，AWQ 常被视为默认选项之一；AutoAWQ 归档后，官方推荐转向 `llm-compressor` 等统一工具。

## W4A16 配方

`llm-compressor` 的常见 recipe：

- `scheme="W4A16"`：权重 4-bit，激活保持 16-bit。激活随 prompt 变化且含 outlier，全量化（如 W8A8）更难；weight-only 用显存换质量。
- `targets="Linear"`：参数大头在线性层。
- `ignore=["lm_head"]`：词表投影对选词极敏感，掉精度易出乱码，通常留在 16-bit。

校准样本不必海量：文档示例用 256 条；过少统计不稳，过几百条后边际收益很小而耗时陡增。`max_seq_length=4096` 用来覆盖长上下文下的激活行为。InferLoop / GPTQModel 实践常用 128–1024 条，C4 一类通用语料；垂直场景可混入领域文本。`group_size=128` 是常见折中。

## 体积为什么不是严格 4×

理论 BF16→INT4 可缩约 75%。实测 Llama-3-8B 示例：BF16 15.30 GB → W4A16 5.45 GB，约减 64%。缺口来自未量化的 embedding、layer norm、`lm_head`。模型越大，线性层占比越高，压缩比越接近理论上限；小模型 vocab head 占比高时整体缩减更有限。

## 质量怎么验

生成对比只能看「像不像」。更硬的指标是 holdout perplexity。同一篇 blog 的滑动窗口结果：Base BF16 PPL 14.22，W4A16 14.85（+0.63，约 +4.4%），换约 64% 体积下降。InferLoop 在 Qwen2.5-7B-Instruct 上也给出类似量级：FP16 PPL 6.42，GPTQ INT4 6.58，AWQ INT4 6.51；MMLU 从 70.2% 掉到约 69.1–69.5%。

## 与其他格式的定位

| 方法 | 设备 | 角色 |
|------|------|------|
| GPTQ | GPU | 优化式 INT4 PTQ |
| AWQ | GPU | 激活感知 INT4 PTQ，服务端常用 |
| GGUF | CPU / 混合 | llama.cpp 生态本地部署 |
| FP8 | Hopper+ | 硬件原生，精度掉点通常更小 |
| bitsandbytes | GPU | 即时量化，利于实验与 QLoRA，不如 AWQ kernel 利于生产吞吐 |

## See Also

- [PTQ 与 QAT](/wiki/llm-quantization/ptq-vs-qat/)
- [线性量化基础](/wiki/llm-quantization/linear-quantization/)
- [大模型量化问题地图](/wiki/llm-quantization/overview/)
