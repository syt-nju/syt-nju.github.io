---
title: "FP8 训练 Recipe"
topic: llm-quantization
summary: "FP8 训练靠 TE/MCore recipe 在 GEMM 上用 E4M3/E5M2 或 MXFP8；显存与通信收益取决于 primary weights 与并行策略，不等于推理侧 FP8 PTQ。"
lang: zh-CN
updated: 2026-08-11
order: 7
sources:
  - title: "当谈论 FP8 训练的时候，我们到底在聊什么?"
    url: "https://qingkeai.online/archives/aaxcavut"
raw:
  - raw/llm-quantization/2025-11-09-fp8-training-recipes.md
---

## Overview

大规模预训练里的 FP8，首先是 **训练期混合精度 recipe**：把 linear 的 fprop / dgrad / wgrad GEMM 输入量化到 FP8，用 Tensor Core 加速；不是 GPTQ/AWQ 那种训后 INT4 权重量化，也不同于只把推理权重压到 FP8。

工具链上，Transformer Engine（TE）管 layer 内（量化、GEMM、多数 fusion、SDPA backend、TP overlap、CP 等）；Megatron-Core（MCore）管 layer 上（PP/EP/DP、建模、optimizer、checkpoint）。

## FP8 格式与缩放

NVIDIA Ada/Hopper 起提供 FP8 Tensor Core。常见两种：

- **E4M3**：1 符号 + 4 指数 + 3 尾数；PyTorch 为 `torch.float8_e4m3fn`（`fn` 因 OCP E4M3 不保留 inf）
- **E5M2**：1 符号 + 5 指数 + 2 尾数；`torch.float8_e5m2`

宣称收益：FP8 Tensor Core 算力约为 BF16 的 2 倍；理想下 weight/activation 显存与通信量可减半——实际取决于是否去掉 BF16 副本、以及通信能否 FP8 化。

因范围与精度有限，训练必须有缩放：对 tensor 或 tile（sub-channel/group/block）取 amax，缩放到 FP8 可表示最大值。对比：BF16 训练通常不需要缩放；FP16 常用全局 loss scale。

OCP 另有 **MXFP8**：每 32 个 E4M3/E5M2 共享一个 E8M0 scale。

## Recipe 三维

一个 FP8 recipe 至少含：

1. **格式**：纯 E4M3；或 Hybrid（activation/weight 用 E4M3，gradient 用 E5M2 → 前向 E4M3×E4M3，反向两个 E4M3×E5M2 GEMM）
2. **粒度**：至少 per-tensor；更细则定 tile 大小与 1D/2D
3. **覆盖范围**：实践上几乎只量化 linear（qkv、projection、fc1、fc2）；embedding、lm head、SDPA、main gradients、optimizer states 仍高精度。无 reduction 的通信（AllGather、AlltoAll）理论上可 FP8；有 reduction 的保持高精度。FP8 attention 尚无生产模型落地。

### 常用三种

| Recipe | 格式 | 粒度 | 定位 |
|--------|------|------|------|
| Per-tensor current scaling | Hybrid | 每 tensor 一个 scale | TE `Float8CurrentScaling`；MCore `--fp8-format hybrid --fp8-recipe tensorwise` |
| Blockwise（DeepSeek-V3-like） | 纯 E4M3 | act/grad `1×128`，weight `128×128` | 当前最流行；TE `Float8BlockScaling`；CUDA 12.9+ |
| MXFP8 | 纯 E4M3 + E8M0 | 均 `1×32` | Blackwell 主推；仅该代 Tensor Core 原生支持 |

**Delayed scaling**（用历史 amax buffer）可打破 amax 与量化依赖、少读 global memory，但 amax 非当前真值；作者实验称 **>7B 即可观察到收敛问题**，当下可直接忽略。**Current / live scaling** 对当前 tensor 统计 amax 再量化，多一次读；Nemotron-H-56B 用其证明可收敛。

## 计算流：加速三个 GEMM

BF16 baseline：三个 GEMM 输入为 BF16；fprop/dgrad 输出 BF16，wgrad 输出常为 FP32（便于累加）。Tensor Core 内部累加是 FP32，输出精度只是最后 cast。

FP8 训练主要是把这三个 GEMM 的 **input** 量化到 FP8；output 一般仍是 BF16/FP32，不必再量化。

- **Hopper**：FP8 GEMM 仅 TN layout → 常需 cast / cast_transpose；同一步内 weight 可在首个 micro-batch 量化后缓存复用；前向后通常只保存一份 **colwise** FP8 input 供反向，以减激活显存。
- **Blackwell**：任意 layout 的 FP8 GEMM，常可省转置；若量化方向不同，仍可能需要 rowwise 与 colwise 两份数据。

Blockwise 在 Hopper 上实质需要两类 GEMM：`128×128 @ 1×128`（2D×1D）与 `1×128 @ 1×128`（1D×1D）。Blackwell 上可用 MXFP8 模拟 blockwise（如 `1×128` 拆成 4 个共享 scale 的 MXFP8）。MXFP8 若 weight 也是 1D，即使任意 layout 仍常要存 rowwise+colwise 两份；**weight 改 2D（如 32×32）可避免**。

## 显存：不会自动减半

默认路径从 **BF16 weight 再量化出 FP8**，训练中同时保留两者，常驻显存甚至高于纯 BF16。动机是 drop-in 替换与实现简单。

去掉 BF16、直接从 FP32 master weight 量化到 FP8（**FP8 primary weights** / 早期称 native FP8；MCore 参数名 `--fp8-param-gather`）才能把 weights/gradients/optimizer states 的常驻占用打平甚至略优。难点包括：

- FP8 对象需带 scale，且可能同时含 rowwise/colwise；TE 用继承 `torch.Tensor` 的 `QuantizedTensor`
- 与 Distributed Optimizer（ZeRO-1）兼容：master weight 被切到多个 DP rank 时，per-tensor 需 local amax → allreduce max → 量化 shard → FP8 AllGather

激活侧：各 recipe 通常只需存一份 colwise FP8 input，相对 BF16 可减半；MoE expert 激活因 top-k 膨胀，收益更明显。注意：SDPA 保存 BF16 output、projection 若再存 FP8 input，两者不再共享，可能变成约 **1.5×** 该处占用。开 TP（含 SP）时，要同时拿 FP8 AG 通信收益与 FP8 激活显存收益需额外处理。

作者举例：2048×80GB Hopper 训 DeepSeek-V3 时，FP8 能跑的并行配置下 BF16 会 OOM；BF16 若改并行则性能差很多。

## 通信：等价优先

原则：开/关 FP8 通信尽量数值等价；不做「量化→通信→反量化」换速度。

- **排除 CP/PP**：attention 仍高精度；PP 通信量小且易 overlap，强行 FP8 常无损性与收益双输。
- **DP**：在 primary weights 路径下，parameter AllGather 可 FP8，相对「BF16 AG 的 FP8 训练」无损。
- **TP AG**：可先在 TP group 内 allreduce amax，再对 shard 量化并 FP8 AG，同时保留 colwise 供反向。
- **EP alltoall**：DeepSeek-V3 / DeepEP 路径；前向「FP8 通信再 GEMM」与「BF16 通信再量化」可等价。反向常只传一份 rowwise，需昂贵的 dequant→另一方向 requant；Grace Blackwell 上作者因此未用 FP8 dispatch。且需 **E8M0 / power-of-2 scale**，使 double 与 single quantization 几乎等价。**Activation 沿 token 1D 量化** 才与按 token 发送吻合；per-tensor 或 activation 2D tile 几乎无法做等价 FP8 dispatch。

## 稳定主张

- 训练 FP8 的关键设计选择是：**activation 沿 token 1D，weight 用 2D**——利于通信与少存双份 weight。
- Blockwise（DeepSeek-V3-like）是当下生产背书最多的 recipe；MXFP8 绑 Blackwell。
- 推理侧「模型存成 FP8」与训练侧「GEMM 输入 FP8 + master/BF16 权重策略」不要混为一谈。
- 训练侧 MXFP8（tile + E8M0）与推理侧 [MXFP4](/wiki/llm-quantization/mxfp4/)（E2M1 + block-32）同属 microscaling 直觉，但场景与比特宽不同。

## See Also

- [线性量化基础](/wiki/llm-quantization/linear-quantization/)
- [PTQ 与 QAT](/wiki/llm-quantization/ptq-vs-qat/)
- [MXFP4 推理量化](/wiki/llm-quantization/mxfp4/)
- [大模型量化问题地图](/wiki/llm-quantization/overview/)
