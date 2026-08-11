# Tricks from OpenAI gpt-oss YOU 🫵 can use with transformers

> Source: https://huggingface.co/blog/faster-transformers
> Author: Aritra Roy Gosthipaty, Sergio Paniego, Vaibhav Srivastav, Pedro Cuenca, Arthur Zucker, Nathan Habib, Cyril Vallez (Hugging Face)
> Collected: 2026-08-11
> Published: 2025-09-11
> License: Unknown
> Completeness: Complete
> Retrieval note: Fetched via WebFetch; site chrome and community comments removed; article body including MXFP4 and other sections preserved.

# Tricks from OpenAI gpt-oss YOU 🫵 can use with transformers

Published September 11, 2025

- +183

Aritra Roy Gosthipaty ariG23498 Follow

Sergio Paniego sergiopaniego Follow

Vaibhav Srivastav reach-vb Follow

Pedro Cuenca pcuenq Follow

Arthur Zucker ArthurZ Follow

Nathan Habib SaylorTwift Follow

Cyril Vallez cyrilvallez Follow

OpenAI recently released their GPT-OSS series of models. The models feature some novel techniques like MXFP4 quantization, efficient kernels, a brand new chat format, and more. To enable the release of gpt-oss through`transformers`, we have upgraded the library considerably. The updates make it very efficient to load, run, and fine-tune the models.

In this blog post, we talk about all the upgrades in-depth, and how they become part of the transformers toolkit so other models (current and future) can benefit from them. Providing clean implementations of new methods in transformers also allows the community to quickly understand and adopt them. Frameworks such as MLX, llama.cpp or vLLM can use the transformers code as a reference to build their own implementations.

For this release, we worked on:

- Zero-build Kernels, downloadable from the Hub
- MXFP4 Quantization
- Tensor Parallelism
- Expert Parallelism
- Dynamic Sliding Window Layer & Cache
- Continuous Batching & Paged Attention
- Load larger models faster

Best part: Most of these features should work across all major models within`transformers`!

## Zero-build Kernels, downloadable from the Hub

A kernel is a specialized, compact program that runs on accelerators to execute tasks like matrix multiplications, activations, or normalizations. In eager PyTorch, operations trigger individual kernels sequentially, which is straightforward but can incur extra memory transfers and launch overheads. PyTorch 2.0's`torch.compile` with backends like`TorchInductor` addresses this by automatically fusing and optimizing kernels, delivering`2–10×` performance gains.

In addition, the community has created custom kernels for frequent combinations of operations, not just individual PyTorch ops like matmul. For example, Flash Attention was created to optimize the critical attention block that defines the transformers architecture, and is present in many models including most LLMs. By carefully combining all the attention operations inside a single kernel, memory transfers are minimized, memory use is reduced, and speedups can be achieved.

The problem is that all these various kernels are available in separate libraries, which creates a dependency bloat if they were to be added to the transformers library. Furthermore, these kernels are not just Python code, they consist of low-level cuda code, glued together with C++ and exposed through a Python layer. This means they have to be compiled in the target system, which in turn requires whatever build system is required by each kernel library.

The kernels package solves this problem by downloading pre-built binaries of supported kernels from the Hub. You just indicate the kernel you want to use, and`kernels` will look for a version compatible with your system and download it on first use.

### Custom Kernels for GPT-OSS

GPT-OSS, a Mixture of Experts (MoE) model, is a big user of Kernels from the Hub. It leverages several custom kernels:

1. Liger RMSNorm, used as@use_kernel_forward_from_hub("RMSNorm")`
2. Megablocks MoE kernels:@use_kernel_forward_from_hub("MegaBlocksMoeMLP")
3. Flash Attention 3 with support for attention sinks.
4. MXFP4 triton kernels (covered later)

Let's take a look at the first two ones.

Behind the scenes, the decorators (1 and 2) simply point to community-contributed kernels. For example,`RMSNorm` comes from liger_kernels, while the`MegaBlocksMoeMLP` kernel comes from megablocks. Depending on your device (CUDA or ROCm) and whether you’re training or running inference, the right kernel is pulled in automatically.

This design is both specific and general: the RMSNorm liger kernels are already being reused across multiple models, and the MoE kernel could be applied to future MoEs as well.

Because`kernels` pulls code from the Hub, you have to opt-in to this feature by passing`use_kernels=True` in your model instantiation, as shown below. We enable`INFO` logging in the example so you can easily verify that downloadable kernels are in use.

These kernels are not compatible with`mxfp4`, so inference will happen in`bfloat16` if you use them. Please, benchmark your system for the best combination in memory and throughput that suits your project!

```
from transformers import AutoTokenizer, AutoModelForCausalLM

import logging
logging.basicConfig(level=logging.INFO)

model_id = "openai/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype="auto",
    device_map="auto",
    use_kernels=True,
)

```

Running a quick generation yields log messages like

```
INFO:root:Using layer `LigerRMSNorm` from repo `kernels-community/liger_kernels`
INFO:root:Using layer `MegaBlocksMoeMLP` from repo `kernels-community/megablocks`

```

Figure 1 shows that, in the system we tested, these kernels work best for larger batch sizes. We always recommend to benchmark any performance-related changes as closely to your production conditions as possible.

| |
| --- |
| Figure 1: Benchmarking results of custom kernels |

You can explore and play with the benchmarking script here

### Flash Attention 3

OpenAI gpt-oss models use attention sinks, which improves quality and facilitates the use of longer contexts. The vLLM team added this feature to the latest version of Flash Attention (Flash Attention 3), and the resulting custom kernel is available on the Hub. Currently, this kernel is compatible with the Hopper architecture. If you have one, this is the way to enable it:

```
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype="auto",
    device_map="auto",
+    # Flash Attention with Sinks
+    attn_implementation="kernels-community/vllm-flash-attn3",
)

```

## MXFP4 Quantization

Large language models are memory-hungry. Quantization reduces memory footprint by storing weights (and sometimes activations) in lower-precision formats. For reference,`FP32` uses 32 bits per number and`BF16` uses 16. By reducing bit width, we trade some precision for smaller models and faster memory movement.

If you want a visual primer on quantization trade-offs, Maarten Grootendorst’s article is excellent: A Visual Guide to Quantization.

### What is MXFP4

| |
| --- |
| Figure 2: The E2M1 format used in the MXFP4 format |

`MXFP4` is a 4-bit floating format with E2M1 layout: 1 sign bit, 2 exponent bits, and 1 mantissa bit, as shown in Figure 2. On its own, E2M1 is very coarse. MXFP4 compensates with blockwise scaling:

- Vectors are grouped into blocks of 32 elements.
- Each block stores a shared scale that restores dynamic range when dequantizing.
- Inside each block, 4-bit values represent numbers relative to that scale.

This blockwise scheme lets`MXFP4` keep range while using very few bits. In practice, GPT-OSS 20B fits in roughly`16 GB` of VRAM and GPT-OSS 120B fits in roughly`80 GB` when`MXFP4` is active, which is the difference between “cannot load” and “can run on a single GPU.” The catch is that matrix multiplies now have to respect block scales. Doing this efficiently at scale requires dedicated kernels.

### MXFP4 in transformers

`transformers` now includes native support for MXFP4, leveraging optimized`triton`(MXFP4) kernels for enhanced performance. This builds on the community-driven kernel distribution discussed earlier, utilizing pre-compiled kernels from the Hub to simplify deployment.

Key implementation details:

- Quantizer logic: Found in the MXFP4 quantizer file, this handles the core quantization process for MXFP4.
- Integration hooks: The MXFP4 integration file enables seamless use of MXFP4 within the transformers framework.

To check if a model supports`MXFP4`, inspect its configuration:

```
from transformers import GptOssConfig

model_id = "openai/gpt-oss-120b"
cfg = GptOssConfig.from_pretrained(model_id)
print(cfg.quantization_config)

# Example output:
# {
#   'modules_to_not_convert': [
#     'model.layers.*.self_attn',
#     'model.layers.*.mlp.router',
#     'model.embed_tokens',
#     'lm_head'
#   ],
#   'quant_method': 'mxfp4'
# }

```

If`'quant_method': 'mxfp4'` is present, the model will automatically use the MXFP4 pathway with Triton kernels when supported.

Thanks to this pull request, you can fine-tune gpt-oss models and save them directly to the Hub in MXFP4 format, streamlining deployment with optimized performance.

### Requirements and fallbacks

To run`MXFP4` on GPU you need:

1. `accelerate`,`kernels`, and`triton>=3.4` installed. Note that`Pytorch 2.8` already comes with`triton 3.4`, so you only need to manually install triton if using`Pytorch 2.7`.
2. NVIDIA GPU with compute capability`≥ 7.5`. This goes all the way back to Tesla, so you can run`gpt-oss-20b` on the free tiers of Google Colab and Kaggle, and on many consumer GPUs.

If these constraints are not met,`transformers` falls back to a higher-precision path (`bfloat16` is used by default), which requires about 4× the memory of MXFP4.

The snippet loads GPT-OSS twice on CUDA: once with`Mxfp4Config(dequantize=True)`(memory intensive) and once in the default quantized path (memory efficient). Figure 3 shows the amount of used VRAM after each load so you can visualize the savings.

| |
| --- |
| Figure 3: Memory requirements for the quantized and dequantized models |

### Kernels for MXFP4

Efficient`MXFP4` requires kernels that understand 32-element blocks and their scales during GEMMs and fused ops. This is where Kernels from the Hub comes in again.`transformers` automatically pulls in the`MXFP4`-aware Triton kernels from the community repository when you load a model that needs them. The repository will appear in your local cache and will be used during the forward pass. For the`MXFP4` kernels one does not need to use the`use_kernels=True` parameter like before, it is set to default in`transformers`.

Quick sanity check with the Hugging Face cache CLI, after running`gpt-oss-20b` on a GPU compatible with the triton MXFP4 kernels:

```
hf cache scan

```

Sample output:

```
REPO ID                          REPO TYPE SIZE ON DISK
-------------------------------- --------- ------------
kernels-community/triton_kernels model           536.2K
openai/gpt-oss-20b               model            13.8G

```

This indicates the MXFP4 kernels were fetched and are available for execution.

Let's run some benchmarks and see how well the MXFP4 kernels perform. In Figure 4, we see that the`MXFP4` kernels are even better than the custom MoE and RMSNorm kernels for larger batches.

| |
| --- |
| Figure 4: MXFP4 kernel benchmark |

You can explore and play with the benchmarking script here

## Tensor Parallelism

| |
| --- |
| Figure 5: Explanation of tensor parallelism. |

Tensor Parallelism (TP) splits tensors inside a layer across multiple GPUs (as shown in Figure 5). Each GPU multiplies its shard in parallel, and then partial results are collected using all-gather or all-reduce operations. This reduces per-GPU memory and keeps all GPUs working on the same layer, which improves throughput as sequence length or batch size grow. TP is communication-intensive and generally works best on a single machine with fast intra-node links.

### What this enables in transformers

`transformers` implements TP directly in`from_pretrained`. You can start with the predefined plan:

```
# run with: torchrun --nproc-per-node 4 tp_gpt_oss.py
import torch
from transformers import PreTrainedTokenizerFast, GptOssForCausalLM

model_id = "openai/gpt-oss-120b"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id)
model = GptOssForCausalLM.from_pretrained(
    model_id,
    tp_plan="auto", # built in TP support
    dtype="auto",
).eval()

messages = [
    {"role": "system", "content": "Be concise."},
    {"role": "user", "content": "Explain KV caching briefly."},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    reasoning_effort="low",
).to(model.device)

with torch.inference_mode():
    generations = model.generate(**inputs, max_new_tokens=128)

print(tokenizer.decode(generations[0][inputs["input_ids"].shape[-1]:]))

```

If you don’t have the infrastructure to run the above, you can just spawn a process on our GPUs using Hugging Face Jobs!

```
hf jobs run --detach --flavor l4x4 ghcr.io/astral-sh/uv:debian /bin/bash -c \
  "uv venv .venv --python 3.12 && \
  source .venv/bin/activate && \
  uv pip install --upgrade torch numpy transformers accelerate triton kernels && \
  wget https://huggingface.co/datasets/ariG23498/distributed/raw/main/tp_gpt_oss.py && \
  torchrun --nproc-per-node=4 tp_gpt_oss.py"

```

hf jobs is available for all Hugging Face PRO & Enterprise users.

Under the hood,`tp_plan="auto"` selects a predefined sharding recipe for each layer and wires the necessary collectives. You can inspect the active plan with`print(model._tp_plan)` if you want to verify what is being sharded.

### When to reach for TP

Use TP when the model is too large for one GPU and you want parallel compute, not only memory placement. TP tends to scale throughput with more GPUs, especially for long sequences or larger batches.

If you are curious about how TP differs from`device_map="auto"`(memory placement), this short Stack Overflow answer explains the distinction and when to use each.

To learn more about TP, here are two must-read resources:

- transformers guide: Tensor parallelism, supported models, plans, and extension points.
- Ultra-Scale Playbook: background on TP and its relationship to other parallelism modes.

## Expert Parallelism

Expert Parallelism (EP) shards experts inside MoE layers across GPUs. Each token is routed to one or a few experts, so only those experts run their feed-forward pass. Since experts are independent MLPs, we can place different experts on different ranks and exchange only the hidden states for the routed tokens. This keeps the matrix multiplies intact on each rank and replaces tensor slicing with routing and collectives.

Run with multiple processes using`torchrun`. EP is enabled via the distributed configuration and works with GPT-OSS MoE layers out of the box in transformers.

```
# run with: torchrun --nproc-per-node 4 ep_gpt_oss.py
import torch
from transformers import PreTrainedTokenizerFast, GptOssForCausalLM
from transformers.distributed import DistributedConfig

model_id = "openai/gpt-oss-120b"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id)
model = GptOssForCausalLM.from_pretrained(
    model_id,
    distributed_config=DistributedConfig(enable_expert_parallel=True), # enabling EP
    dtype="auto",
).eval()

messages = [
    {"role": "system", "content": "Be concise."},
    {"role": "user", "content": "Explain KV caching briefly."},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    reasoning_effort="low",
).to(model.device)

with torch.inference_mode():
    generations = model.generate(**inputs, max_new_tokens=128)

print(tokenizer.decode(generations[0][inputs["input_ids"].shape[-1]:]))

```

Here is how you would run using`hf jobs`

```
hf jobs run --detach --flavor l4x4 ghcr.io/astral-sh/uv:debian /bin/bash -c \
  "uv venv .venv --python 3.12 && \
  source .venv/bin/activate && \
  uv pip install --upgrade torch numpy transformers accelerate triton kernels && \
  wget https://huggingface.co/datasets/ariG23498/distributed/raw/main/ep_gpt_oss.py && \
  torchrun --nproc-per-node=4 ep_gpt_oss.py"

```

When you enable Expert Parallelism, Tensor Parallelism is also activated. This means you enjoy the best of both worlds!

## Dynamic Sliding Window Layer & Cache

Many recent LLMs use sliding window attention, or a combination of sliding and global attention layers, as a means to save memory and reduce those expensive quadratic matmuls that grow with sequence length. However, the dynamic KV cache implementation in transformers used to continue to allocate space according to sequence length, without looking at the individual attention layers. You could always optimize memory using compilation (meaning, fixed shapes), but that's a separate scenario altogether.

`transformers` now has a DynamicSlidingWindowLayer and a config‑aware DynamicCache. If the model config declares sliding‑window or hybrid attention (both sliding and global attention layers are used), the cache stops growing past the window for the sliding layers. If you don’t pass the config, behavior stays as before (full, ever‑growing KV as sequence length grows).

For models that only use sliding window layers, such as Mistral 7B, cache memory stops growing when the sequence reaches the window size (4096, in this case). This makes sense, because the sliding layers can't look past the previous 4K tokens anyway.

OpenAI gpt-oss alternates between sliding and global attention layers, which results in total KV cache memory being halved, as we'll see, as sequence length increases. This provides us with:

- Much lower KV‑cache memory for models with sliding or hybrid attention (e.g. GPT‑OSS). Cache growth plateaus once the window is reached (e.g., 4K for Mistral; 128 for GPT‑OSS sliding layers), instead of scaling linearly with total generated tokens. (GitHub, Transformers)
- Speed/latency wins on long prompts/long generations: smaller KV tensors mean lighter attention reads/writes and less memory bandwidth pressure, especially after the window is hit. (This is the central motivation behind sliding‑window/hybrid LLMs.) (AI21, vLLM Blog)

### How to use it

The optimized cache is set by default, that means you don't have to make any changes to your existing code. If you want to create the`DynamicCache` explicitly here is how you would do it:

```
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

model_id = "openai/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype="auto",
    device_map="auto",
).eval()

messages = [
    {"role": "system", "content": "Always respond in riddles"},
    {"role": "user", "content": "What is the weather like in Madrid?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    reasoning_effort="low",
).to(model.device)

cache = DynamicCache(config=model.config) # create the cache with the model's config

generated = model.generate(
    **inputs,
    max_new_tokens=500,
    past_key_values=cache
)
print(tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:]))

```

Figure 6 showcases how much of a difference it makes for us to use the Dynamic KV Cache with sliding window attention.

| |
| --- |
| Figure 6: The memory analysis of dynamic cache with sliding window attention |

## Continuous Batching & Paged Attention

A typical autoregressive generation process looks like Figure 7. You input the prefill tokens, and the model predicts each new token one after the other until it predicts the EOS (End of Sequence) token.

| |
| --- |
| Figure 7: Autoregressive token generation |

Let’s see what the generation process looks like when we pass a batch of inputs. In Figure 8 you notice that some generations finish off earlier than the others. This mismatch of length underutilizes the GPUs.

| |
| --- |
| Figure 8: Static batching of sequences |

This type of batching sequences is called static batching. While this is simple and easy to understand, it inherently comes with inefficiencies. Only after each sentence is completely generated can we move on to the next batch.

To bypass this issue, we use dynamic batching (also known as continuous batching). Instead of waiting for all the generation to finish, we schedule incoming requests to the completed generations. That way, as soon as a generation in a batch is complete, we prefill the batch with the next request. The process looks like Figure 9.

| |
| --- |
| Figure 9: Continuous Batching of sequences |

Transformers supports continuous batching with the`generate_batch` API. This is not meant for production-grade model serving –frameworks like vLLM and SGLang are great at that–, but can be very helpful for evaluation and experimentation. Here is an example script that runs CB end to end on`Qwen/Qwen3-4B-Instruct-2507`.

We have also performed a benchmark between Continuous Batching and Static Batching with 100 samples. In Figure 9, we note that CB is quite faster than SB.

| |
| --- |
| Figure 9: Continuous vs Static Batching Tokens/Second |

You can play around with the benchmark here: SB, CB

## Load larger models faster

When you load a large model into your GPU, PyTorch needs to reserve GPU memory for each layer’s weights. Each of these requests (per layer) takes time, and for multi-billion-parameter models it can mean thousands of tiny memory allocations, adding up to a long wait before the model is ready. Instead of asking the GPU for new memory every single time, it can hold on to a big chunk once and then hand out slices from it quickly.

PyTorch allocators can do exactly this. The catch is that the allocator only gets fast after you’ve given it some memory to work with. If you don’t “stock the pantry” first, you still end up doing many slow trips to the market. This PR (🎉#36380) taught`transformers` to pre-stock the pantry before it starts copying model weights.

It:

- Looks at the`device_map`(where each layer will live).
- Pre-allocates a big enough block on each GPU.
- Then, as layers are copied in, they just slot neatly into this pre-reserved space.

You have to make no changes to your existing code, as this is default behaviour in`transformers`. If you use`device_map="auto"` or provide your own device map, your model will now load faster automatically. If you’re running with Tensor Parallel (`tp_plan="auto"`) and`torchrun` you also benefit from companion changes that make multi-GPU loading smarter.

## Conclusion

`transformers` moves quickly and it is community-first. The library evolves at the pace of the field because contributors shape it in the open. Pieces added for new models become part of the toolkit and are reused in future integrations.

This velocity enables day-zero integrations like the GPT-OSS series. As the stack becomes increasingly PyTorch-first, it trims bloat and doubles down on the PyTorch paths that matter in practice. The result is a cleaner core that unlocks new capabilities through community kernels, quantization, and parallelism plans, while also standardizing model definitions so that architectures supported in transformers are a reference and extend across the wider ecosystem.

This post is a one-time snapshot of a process we repeatedly iterate on towards the same direction: serve the needs of the community. To be up to date with the latest additions to transformers, check the docs and release notes. And please, keep sharing your feedback and releasing your models in transformers for the community to enjoy 🤗

## Read More

If you want to go further into particular topics, here is a list of links that one should visit:

1. Hugging Face GPT-OSS Recipes Repository
2. Welcome GPT OSS: OpenAI's New Open-Source Model Family
3. OpenAI Cookbook: GPT-OSS Topic
4. Transformers Documentation: Distributed Inference on Multiple GPUs
5. Matthew Carrigan's X Thread on GPT OSS Innovations
6. YouTube Video: OpenAI GPT OSS Announcement
7. Transformers PR #36380: Faster Model Loading on Accelerators
8. Transformers PR #36335: Update from_pretrained for Tensor Parallelism
9. Transformers PR #40039: New Dynamic Sliding Window Layer and Cache
10. HAN Lab Blog: How Attention Sinks Keep Language Models Stable

## Models mentioned in this article 3

Updated Mar 5 • 905 • 3

Updated Jun 4 • 1.42k • 5

Updated Jul 1 • 55.9k • 43

## Datasets mentioned in this article 2

Updated Sep 4, 2025 • 37

Updated Feb 2 • 64

## Spaces mentioned in this article 1

#### The Ultra-Scale Playbook

Running 3.97k🌌 3.97kThe ultimate guide to training LLM on large GPU Clusters

## Collections mentioned in this article 1

CollectionOpen-weight models designed for powerful reasoning, agentic tasks, and versatile developer use cases. • 2 items • Updated Aug 7, 2025 • 465

More Articles from our Blog

## Unlocking asynchronicity in continuous batching

transformerspytorchoptimization 65 May 14, 2026

## Continuous batching from first principles

transformerspytorchoptimization Hot 432 November 25, 2025

