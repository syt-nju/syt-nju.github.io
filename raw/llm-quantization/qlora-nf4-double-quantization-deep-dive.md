# QLoRA Deep Dive: Efficient LLM Fine-Tuning with NF4 & Paged Optimizers

> Source: https://kuriko-iwai.com/research/qlora-efficient-llm-finetuning-nf4-double-quantization
> Author: Kuriko IWAI
> Collected: 2026-08-10
> Published: Unknown
> License: Unknown
> Completeness: Complete
> Retrieval note: Fetched from author site HTML; navigation chrome removed.

QLoRA Deep Dive: Efficient LLM Fine-Tuning with NF4 & Paged Optimizers

---

# A Technical Guide to QLoRA and Memory-Efficient Fine-Tuning

## Master how QLoRA enables 70B model tuning on consumer GPUs, leveraging NF4, Double Quantization, and Paged Optimizers.

Machine LearningDeep LearningData SciencePythonLLM

By Kuriko IWAI

### Table of Contents

IntroductionWhat is QLoRA

Quantization Mechanics - Balancing Precision and Memory Footprint

Why Precision Accuracy is Key - Tackling the Vanishing Gradient Problem

How QLoRA Works

4-bit NormalFloat (NF4)

Double Quantization (DQ)

Paged Optimizer

VRAM Analysis - Overall VRAM Saving

Why the Difference Matters - The Hardware Choice

QLoRA in Action - Production-Ready QLoRA

Loading Base Model

Preparing for Training

Setting Up LoRA Adapters

Training LoRA Adapters

Wrapping Up

Strategic Trade-offs - When to Skip QLoRA

## Introduction

Fine-tuning massive Large Language Models (LLMs) used to require a supercomputer or a cluster of high-end A100 GPUs.

For the average developers, this VRAM requirement made the most powerful LLMs untouchable.

Quantized Low-Rank Adaptation (QLoRA) democratizes LLMs by shrinking their memory footprint so drastically that they can be tuned on consumer-grade GPUs.

In this article, I’ll deep dive into the core mechanisms that make QLoRA possible—including NF4 and Double Quantization—walk through a production-ready implementation, and highlight the critical trade-offs to be considered.

## What is QLoRA

Quantized Low-Rank Adaptation (QLoRA) is an efficient LLM fine-tuning technique that reduces memory usage enough to fine-tune a 70B parameter model on a single 48GB GPU.

The below diagram illustrates how QLoRA leverages quantization:

Figure A. Comparison diagram showing simplified memory architectures of QLoRA, LoRA, and Full Fine-Tuning.(Created by Kuriko IWAI)

QLoRA (Figure A, left) extends Low-rank Adaptation (LoRA) (middle) by utilizing 4-bit quantization to host the large base model to reduce memory overhead.

It can retain the core efficiency of LoRA by employing trainable LoRA adapters which represent less than 1% of the parameters required for full fine-tuning(right).

### ◼ Quantization Mechanics - Balancing Precision and Memory Footprint

In the context of Machine Learning, quantization is the process of reducing the numerical precision of a model’s weights and activations to make the model smaller, faster, and more energy-efficient.

Each model parameter (weight) in an LLM is a number that determines how much signal passes from one neuron to the next.

Taking a 70B parameter model (e.g., Llama 3-70B) for an example:

#### ▫ At 32-bit

Most accurate model: The model can distinguish tiny differences between 0.0000001 and 0.0000002.

Expensive VRAM: Requires 280GB for hosting 70B model (32 bits x 70B / 8 bits/byte)

#### ▫ At 4-bit

Compromised accuracy: The model cannot distinguish 0.0000001 and 0.0000002 due to the lack of bits to store the mantissa. It recognizes both as the same 0.0000.

Save VRAM: Requires only 35GB VRAM, 1/8 to the 32-bit precision (4 bits x 70B / 8 bits/byte)

Quantization compromises the precision accuracy, but offers significant memory saving benefits.

### ◼ Why Precision Accuracy is Key - Tackling the Vanishing Gradient Problem

When an LLM trains, an optimizer like Adam, AdamW or SGD computes gradients, a partial derivative of the loss function with respect to the weight parameters, during the backward pass to minimize the error.

These gradients are incredibly small like 0.0000001 or 0.0000002.

When the 4-bit precision recognizes these values as the same 0.00000, it would:

Break the gradient computation by the optimizer, and

Mislead the model to stop learning because there’s no gradient updates.

But using high precision like 32 bits in training can cause out-of-memory (OOM) problems where VRAM memory spikes exceed the GPU memory, causing the GPU to crash immediately.

As Figure A shows, QLoRA employs a hybrid approach to leverage benefits of high and low precisions:

Base model parameters: Hosts in low precision (NF4, 4 bits).

Base model quantization data: Quantizes in low precision (FP8, 8 bits), using double quantization.

LoRA adapters: Hosts in high precision (16 bits) with paged optimizer (Figure A, left, CPU).

As we saw in the example, hosting the large base model (e.g., 70B parameters) in low precision enables QLoRA to save significant VRAM.

On the other hand, hosting small LoRA adapters in high precision can tackle the precision accuracy challenge.

And the paged optimizer manages memory usages during training to avoid the OOM breakage.

The next section explains the details of these techniques.

## How QLoRA Works

QLoRA introduces three key innovations to maintain 16-bit performance levels while using 4-bit storage:

NF4 (4-bit NormalFloat): A 4-bit quantization format to host the base model.

Double quantization: A technique to quantize the constants of NF4 from 32 bits to 8 bits.

Paged optimizer: A memory-efficient optimizer which utilizes CPUs when GPU memory footprint spikes.

### ◼ 4-bit NormalFloat (NF4)

4-bit NormalFloat (NF4) is a 4-bit quantization format commonly applied to QLoRA.

NF4 maps 4-bit indices from 0 to 15 to specific values stored in NF4 Lookup Table (LUT) based on the assumption that weights are normally distributed.

The diagram below compares the mapping of NF4 and its INT family counterpart, INT4:

Figure B. Probability density graph showing NF4 vs INT4 quantization notch distribution (Created by Kuriko IWAI)

Both NF4 and INT4:

Normalize weights into the [-1, 1] range,

Divide the range into 16 notches with unique indices,

Search an index with the closest value to the original, normalized weight value, and

Return the index in 4 bits (e.g, index: 6 → 0110)

However, NF4 can achieve higher precision by concentrating more notches in the high-density region around zero, whereas INT4 utilizes fixed-width notches regardless of the data distribution.

For example, NF4 LUT looks like:

| Notch | Index in Binary (Q1) | Index in Decimal (d_in) | Normalized Weight (n(Q1)) | Gap to Next Notch |
| --- | --- | --- | --- | --- |
| 1 (The negative edge) | 0000 | 0 | -1.0000 | 0.4500 |
| 2 | 0001 | 1 | -0.5500 | 0.2000 |
| 3 | 0010 | 2 | -0.3500 | 0.1300 |
| 4 | 0011 | 3 | -0.2200 | 0.0800 |
| 5 | 0100 | 4 | -0.1400 | 0.0600 |
| 6 | 0101 | 5 | -0.0800 | 0.0500 |
| 7 | 0110 | 6 | -0.0300 | 0.0200 |
| 8 | 0111 | 7 | -0.0100 | 0.0100 |
| 9 (The center) | 1000 | 8 | 0.0000 | 0.0100 |
| 10 | 1001 | 9 | 0.0100 | 0.0200 |
| 11 | 1010 | 10 | 0.0300 | 0.0500 |
| 12 | 1011 | 11 | 0.0800 | 0.0600 |
| 13 | 1100 | 12 | 0.1400 | 0.0800 |
| 14 | 1101 | 13 | 0.2200 | 0.1300 |
| 15 | 1110 | 14 | 0.3500 | 0.6500 |
| 16 (The positive edge) | 1111 | 15 | 1.0000 | — |

Table 1. NF4 Look-Up Table (LUT) and Non-Linear Weight Mapping

We can find Gap to Next Notch (Table 1, the rightmost column) gets smaller from 0.45 to 0.01 as it gets closer to the center, zero.

On the other hand, INT4 evenly distributes the 16 notches from -1 to 1.

So, Gap to Next Notch (Table 2, the rightmost column) is the same 0.1333, much larger than NF4’s 0.01 around the center, resulting in a larger gap to the original weight value.

| Notch | Index in Binary | Index in Decimal | Normalized Weight | Gap to Next Notch |
| --- | --- | --- | --- | --- |
| 1 | 0000 | 0 | -1.0000 | 0.1333 |
| 2 | 0001 | 1 | -0.8667 | 0.1333 |
| 3 | 0010 | 2 | -0.7333 | 0.1333 |
| 4 | 0011 | 3 | -0.6000 | 0.1333 |
| 5 | 0100 | 4 | -0.4667 | 0.1333 |
| 6 | 0101 | 5 | -0.3333 | 0.1333 |
| 7 | 0110 | 6 | -0.2000 | 0.1333 |
| 8 | 0111 | 7 | -0.0667 | 0.1333 |
| 9 | 1000 | 8 | 0.0667 | 0.1333 |
| 10 | 1001 | 9 | 0.2000 | 0.1333 |
| 11 | 1010 | 10 | 0.3333 | 0.1333 |
| 12 | 1011 | 11 | 0.4667 | 0.1333 |
| 13 | 1100 | 12 | 0.6000 | 0.1333 |
| 14 | 1101 | 13 | 0.7333 | 0.1333 |
| 15 | 1110 | 14 | 0.8667 | 0.1333 |
| 16 | 1111 | 15 | 1.0000 | — |

Table 2. INT4 Linear Quantization and Weight Mapping.

QLoRA applies NF4 to host the large base model to maximize memory saving benefits.

### ◼ Double Quantization (DQ)

Double quantization (DQ) is the process that QLoRA quantizes the constants in NF4 to further squeeze the memory footprint.

#### ▫ The Standard NF4 Quantization

NF4 quantization needs to store the scaling factors (c_1) that map 4-bit indices (Q1, Table 1, Index in Binary column) back to real values.

This dequantization process is denoted:

w=n(Q1)×c1⋯(1.1)

Where:

w: The reconstructed FP32 or BF16 weight used for the actual matrix multiplication.

n(Q_1): The normalized weight value from the NF4 LUT, corresponding to the binary index Q_1.

c_1: The first quantization constant in 32-bit float (FP32), applied to a block of weights.

The first part of Eq. 1.1 costs 4 bits per parameter as Q_1 is stored in NF4.

The second part of Eq. 1.1 costs 1 bit per parameter because the constant c_1 is stored in FP32 and shared across the block with 32 weights (parameters):

32 bits ÷32 parameter per block =1.0 bits⋯(1.2)

So, in total, the standard NF4 quantization overhead costs 5 bits per parameter.

Developer Note: Block-wise Quantization

Standard quantization rescales all parameters in the weight tensor W with a single shared constant c.

This approach is simple, but can lead to significant precision loss if the weight distribution has extreme outliers because the rest is squeezed into the same notch.

NF4 utilizes block-wise quantization where the weight tensor W is divided into small blocks, ensuring that local outliers do not disproportionately affect the precision of the entire tensor.

This helps NF4 maintain higher model fidelity at 4-bit depths.

Figure C. Visualization of standard quantization (left) and block-wise quantization (right) handling weight outliers in tensors (Created by Kuriko IWAI)

#### ▫ NF4 Quantization with DQ

QLoRA quantizes c_1 (FP32) to a 8-bit value (Q_2) to further squeeze out the quantization overhead:

w=n(Q1)×(Q2×c2)⋯(2)

Where:

w: The reconstructed FP32 or BF16 weight used for the actual matrix multiplication.

n(Q_1): The normalized value from the NF4 LUT, corresponding to the binary index Q_1.

Q_2: The quantized scale factor in 8-bit integers (FP8).

c_2: The second quantization constant in 32-bit float (FP32).

The first part of Eq. 2 costs the same as the standard NF4 quantization, 4 bits per parameter.

But the second part of Eq. 2 only costs 0.127 (0.125 + 0.002) bits per parameter because:

Q2 in FP8 is shared across a block with 64 weights: 8 bits / block size 64 = 0.125 bits per parameter.

c_2 in FP32 is shared across 256 Q2 values, which contains 64 weights x 256 blocks = 16,384 weights (parameters): 32 bits / 16,384 parameters = 0.002 bits per parameter.

So, in total, QLoRA’s quantization overhead is 4.127 bits per parameter < 5 bits.

As Table 3 shows, because c_2 in FP32 is shared over 10K parameters, QLoRA can further squeeze the overhead (4.127 bits) compared to the standard NF4 quantization (5 bits).

| | Standard NF4 (Block size: 32) | QLoRA (Block size: 64) |
| --- | --- | --- |
| Binary index Q1 | 4.000 | 4.000 |
| Quantization scale | 1.000 (c_1 in FP32) | 0.125 (Q2 in FP8) |
| Master scale (c_2) | n.a. | 0.002 |
| Total Overhead | 5.000 bits per parameter | 4.127 bits per parameter |

Table 3. Quantization Overhead Comparison: Standard NF4 vs. QLoRA.

Developer Note: Choosing Optimal Block Size

An optimal block size for the block-wise quantization depends on the balance between memory savings and precision accuracy.

Larger block size saves more memory, while compromising precision accuracy.

For example, Eq. 1.1 and 1.2 mathematically prove that the quantization overhead for standard NF4 is reduced to 4.5 bits per parameter (4 + 32 / 64) when the block size is 64, instead of 32.

But as Figure C shows, because each block contains 64 parameters, the risk of local outliers skewing the notch allocation and increasing the overall quantization error is higher than 32.

For QLoRA, researchers selected the block size of 64 because:

Hardware alignment: GPUs process data in warps of 32 threads. 64 is a multiple of 32, making memory access very efficient.

Precision: If the block is too large (e.g., 256), the outlier weights with extremely high or low values skew the scale for the entire block, making the quantization less accurate.

Overhead: Smaller block size like 32 eats up the VRAM savings.

The table below summarizes other use cases by block size:

| Block Size | Common Use Case | Impact |
| --- | --- | --- |
| 32 | High-precision 4-bit | Better accuracy. |
| 64 | QLoRA | The industry standard for balancing speed, size, and logic. |
| 128 | GPTQ / AWQ | Common for inference-optimized models. |
| n.a. | Standard INT8 | Each entire row/column shares one constant. |

### ◼ Paged Optimizer

A paged optimizer is a memory-efficient implementation of standard optimizers such as Adam, Adam W, or SGD, inspired by virtual memory systems in operating systems.

#### ▫ The OOM Problem in Training

QLoRA successfully squeezes the base model’s weight tensors with NF4 and DQ.

However, the activation tensors are stored in FP32 (32 bits) or FP16/BF16 (16 bits), meaning that when the optimizer computes gradients during backward pass, its optimizer states are stored in 32 bits or 16 bits.

This spikes the memory usage during training especially when the input sequence (data) gets longer.

And when the input data is slightly larger than the GPU memory, the GPU immediately crashes with an OOM error.

#### ▫ How Paged Optimizer Works

Paged optimizer avoids this OOM breakage by automatically mapping the optimizer states to the CPU RAM (host memory) when the GPU VRAM is exhausted, and then paging them back into the GPU when they are needed for a calculation.

Figure D. Logic flow of a Paged Optimizer moving states between GPU VRAM and CPU RAM (Created by Kuriko IWAI)

With Paging, the training slows down slightly as data moves to the CPU, but it does not crash.

The paged optimizer works as a memory safety net during adapter training.

## VRAM Analysis - Overall VRAM Saving

Overall, QLoRA can save over 95% of VRAM footprint compared to full fine-tuning.

Here is the comparison of the VRAM requirements by tuning method, using a 70B-parameter model as a benchmark:

| | Full Fine-Tuning | LoRA | NF4 PEFT (Block Size: 32) | QLoRA (Block Size: 64) |
| --- | --- | --- | --- | --- |
| 1. Static Hosting | --- | --- | --- | --- |
| Weight Precision | FP32 | BF16 | NF4 (BS:32) | NF4 + DQ |
| Bits per Param | 32.0 bits | 16.0 bits | 5.0 bits | 4.127 bits |
| Base Model | 280.00 GB | 140.00 GB | 43.75 GB | 36.11 GB |
| LoRA Adapter | n.a. | 1.40 GB | 1.40 GB | 1.40 GB |
| a.Total VRAM (GB) | 280.00 GB | 141.40 GB | 45.15 GB | 37.51 GB |
| a’. Total VRAM (GiB) | 260.77 GiB | 131.69 GiB | 42.05 GiB | 34.93 GiB |
| 2. Active Training | --- | --- | --- | --- |
| b. Gradients | 280.00 GB (FP32) | 1.40 GB (BF16) | 1.40 GB (BF16) | 1.40 GB (BF16) |
| c. Optimizer States | 560.00 GB (FP32) | 5.60 GB (BF16) | 5.60 GB (BF16) | 2.80 GB (Paged 8-bit) |
| Max VRAM (GB) (a + b + c) | 1,120.00 GB | 148.40 GB | 52.15 GB | 41.71 GB |
| Max VRAM (GiB) | 1,043.08 GiB | 138.21 GiB | 48.57 GiB | 38.85 GiB |
| OOM Protection | None (static) | None (static) | None (static) | Paged Optimizer |
| 3. Inference | --- | --- | --- | --- |
| d. KV Cache / 1K Tokens | 0.65 GB | 0.65 GB | 0.65 GB | 0.65 GB |
| Total VRAM (GB) (a + d) | 280.65 GB | 142.05 GB | 45.80 GB | 38.16 GB |
| Total VRAM (GiB) | 261.38 GiB | 132.30 GiB | 42.66 GiB | 35.54 GiB |

Table 4: VRAM Consumption Benchmark: 70B Model (Full Fine-Tuning vs. LoRA vs. Standard NF4 vs. QLoRA)

The VRAM consumption has three separate parts:

Static Hosting: Just loading the model. Includes storing all model parameters of the base model (70B) and LoRA adapters.

Active Training: Includes storing gradients and optimizer states on top of the static hosting.

Inference: Perform inference. Includes storing KV cache on top of the static hosting.

In Full Fine-Tuning (Table 4, leftmost), the Max VRAM required for training is notoriously high (over 1,000 GiB) because the system has to track gradients and optimizer states for every single one of the 70 billion parameters, not just a small adapter.

LoRA (Table 4, third column) saves over 85% of VRAM compared to the Full Fine-Tuning, yet storing the base model in 16 bits requires VRAM over 100GiB.

QLoRA (Table 4, rightmost) saves over 95% of VRAM compared to the Full Fine-Tuning by compressing the base model, adapters, and optimizers.

Beside, the 2.8GB allocation to the optimizer states is just a soft upper bound because paged optimizers allow memory to overflow into CPU RAM if necessary, which would result in further VRAM saving.

And because QLoRA successfully compress the base model, VRAM required for inference is also minimized among other tuning methods.

### ◼ Why the Difference Matters - The Hardware Choice

The GB/GiB distinction in Table 4 clearly explains why some models fit in a specific GPU and others don't:

RTX 3090 / 4090 (24 GB):

Advertised as 24GB.

System sees 22.3GiB.

Result: A 70B QLoRA model (max. 39 GiB) will not fit on a single 24GB card. Needs two of them (48GB total).

RTX 6000 Ada / A6000 (48 GB):

Advertised as 48GB.

System sees 44.7GiB.

Result: A 70B QLoRA model fits comfortably with about 5 GiB left over for KV Cache.

A100 / H100 (80 GB):

Advertised as 80GB.

System sees 74.5GiB.

Result: Easily fits. Still has over 35GiB of room for massive batch sizes or long context windows (32k+ tokens).

Developer Note:

The total number of trainable parameters in the LoRA adapter is defined by its rank and targeted layers.

Generally speaking, these parameters account for less than 1% of the total parameters (70B):

70B x 1% x 16 bits per parameter / 8 bits/byte = 1.40 GB.

Learn more: The math of < 1% - How LoRA streamlines the number of trainable parameters

## QLoRA in Action - Production-Ready QLoRA

To implement QLoRA, the industry standard is using the bitsandbytes library alongside Hugging Face’s peft.

The workflow follows these steps:

Loading the base model.

Prepare for training.

Train LoRA adapters.

Merge or attach the adapters.

### ◼ Loading Base Model

I use BitsAndBytesConfig from the transformers library to trigger NF4 quantization with DQ:

```
1from transformers import AutoModelForCausalLM, BitsAndBytesConfig
2
3# configure 4-bit quantization
4bnb_config = BitsAndBytesConfig(
5    load_in_4bit=True,
6    bnb_4bit_quant_type="nf4", # use nf4
7    bnb_4bit_compute_dtype=torch.bfloat16, # computes in bf16 for speed
8    bnb_4bit_use_double_quant=True # dq
9)
10
11# load base model and tokenizer
12model = AutoModelForCausalLM.from_pretrained(
13    model_id, 
14    quantization_config=bnb_config, # apply bnb_config
15    device_map="auto"
16)
17
```

### ◼ Preparing for Training

The model is prepared for training using prepare_model_for_kbit_training from the peft library:

```
1from peft import prepare_model_for_kbit_training
2
3model = prepare_model_for_kbit_training(model)
4
```

### ◼ Setting Up LoRA Adapters

After defining LoRA hyperparameters using LoraConfig from the peft library,

```
1from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
2from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
3
4# lora config
5lora_config = LoraConfig(
6    r=16, # rank    
7    lora_alpha=32, # alpha
8    target_modules=["q_proj", "v_proj"], # target layers
9    lora_dropout=0.05,
10    bias="none", 
11    task_type="CAUSAL_LM"
12)
13
14# set up the base model w/ lora config
15model = get_peft_model(model, lora_config)
16
```

### ◼ Training LoRA Adapters

Lastly, using paged_adamw_8bit as an optimizer for the trainer, an SFTTrainer instance:

```
1from trl import SFTTrainer
2from transformers import TrainingArguments
3
4# define training params
5training_args = TrainingArguments(
6    output_dir=output_dir,
7    per_device_train_batch_size=4,
8    gradient_accumulation_steps=4,
9    learning_rate=2e-4,
10    logging_steps=10,
11    max_steps=100,
12    fp16=True, # fp16 or bf16 if hardware allows
13    optim="paged_adamw_8bit" # paged adamw for qlora
14)
15
16# instantiate the trainer
17trainer = SFTTrainer(
18    model=model,
19    args=training_args,
20    train_dataset=your_dataset,
21    dataset_text_field="text",
22    max_seq_length=512,
23)
24
25# train
26trainer.train()
27
```

## Wrapping Up

QLoRA has effectively bridged the gap between academic research and production deployment.

It proves that a supercomputer is not necessary any more to build a specialized AI.

### ◼ Strategic Trade-offs - When to Skip QLoRA

While QLoRA seems the best choice for memory saving, it isn't always the right tool:

When latency is the priority: Quantized models make inference slower than their native counterparts due to dequantization overhead (n, c’s).

Small models: For a model with < 1B parameters, standard full fine-tuning or basic LoRA is faster and easier to manage.

Abundant hardware: With a H100 cluster and plenty of VRAM, full fine-tuning offers the gold standard of weight updates without the extra complexity of quantization layers.

Written by Kuriko IWAI. All images, unless otherwise noted, are by the author. All experimentations on this blog utilize synthetic or licensed data.

## FAQ

1) What is QLoRA and how does it differ from standard LoRA?

👉 QLoRA (Quantized Low-Rank Adaptation) is an extension of LoRA that enables fine-tuning of large models (like 70B parameters) on consumer-grade GPUs. While standard LoRA typically loads the base model in 16-bit (FP16/BF16), QLoRA uses 4-bit NormalFloat (NF4) quantization, Double Quantization, and Paged Optimizers to reduce VRAM requirements by over 95% compared to full fine-tuning.

2) What are the core technical innovations introduced in QLoRA?

👉 QLoRA introduces three primary innovations: 1) 4-bit NormalFloat (NF4), a quantization format that assumes a normal distribution of weights for higher precision; 2) Double Quantization (DQ), which quantizes the quantization constants themselves to save additional memory; and 3) Paged Optimizers, which leverage CPU RAM as a buffer during VRAM spikes to prevent Out-of-Memory (OOM) errors.

3) How does 4-bit NormalFloat (NF4) maintain model accuracy compared to INT4?

👉 Unlike INT4, which uses fixed-width intervals, NF4 allocates more quantization 'notches' to high-density regions (around zero) of the weight distribution. By using a non-linear Look-Up Table (LUT), NF4 reduces the gap between the original weight and its quantized representation, minimizing information loss during the compression process.

4) What is the memory overhead benefit of Double Quantization (DQ)?

👉 Standard NF4 quantization adds roughly 1.0 bit per parameter in overhead due to 32-bit scaling factors. Double Quantization further compresses these scaling factors from 32-bit to 8-bit. This reduces the total quantization overhead from 5.0 bits per parameter to approximately 4.127 bits, effectively squeezing more performance out of the same VRAM footprint.

5) Can I fine-tune a Llama 3-70B model on a single 24GB consumer GPU using QLoRA?

👉 No. While QLoRA is highly efficient, a 70B model requires approximately 35-39 GiB of VRAM for the static weights and training states. A single 24GB card (like an RTX 4090) is insufficient; however, it can be tuned on two 24GB cards (48GB total) or a single 48GB card like the RTX 6000 Ada.

6) What role does the Paged Optimizer play in the QLoRA workflow?

👉 The Paged Optimizer acts as a fail-safe against OOM crashes. When the GPU VRAM is exhausted—often during long sequence processing or backward passes—it automatically offloads optimizer states to the CPU RAM (host memory) and pages them back to the GPU as needed. This prevents training interruptions at the cost of a slight decrease in processing speed.

7) How do I implement QLoRA in a Python environment?

👉 The industry standard involves using the `bitsandbytes` library for quantization and `peft` for the LoRA adapters. You configure a `BitsAndBytesConfig` with `load_in_4bit=True` and `bnb_4bit_use_double_quant=True`, load your model via Hugging Face Transformers, and use `paged_adamw_8bit` as your optimizer within the `SFTTrainer`.

8) When should I avoid using QLoRA for model fine-tuning?

👉 Skip QLoRA if: 1) **Latency is critical**, as dequantization overhead can slow down inference; 2) **Working with small models** (<1B parameters) where standard LoRA or full fine-tuning is faster; or 3) **VRAM is abundant** (e.g., H100 clusters), where full fine-tuning provides the most direct weight updates without the complexity of quantization layers.

## Shipping AI Systems?

I help teams design and deploy scalable ML / RAG / LLM pipelines and MLOps infrastructure.

Work with me Join the Newsletter

Or explore:

- Dive deeper 👉 Research Archive
- Learn by building 👉 AI Engineering Masterclass
- Try it live 👉 Playground

## Share What You Learned

Kuriko IWAI, "A Technical Guide to QLoRA and Memory-Efficient Fine-Tuning" in Kernel Labs

https://kuriko-iwai.com/qlora-efficient-llm-finetuning-nf4-double-quantization

## Continue Your Learning

If you enjoyed this blog, these related entries will complete the picture:

### Is 4-Bit All You Need? The Math Behind Modern LLM Compression

Master LLM quantization schemes. Learn how FP32, INT8, NF4, and MX formats impact VRAM, latency, and model accuracy for efficient AI deployment.

### Deconstructing LoRA: The Math and Mechanics of Low-Rank Adaptation

Master Low-Rank Adaptation (LoRA). Learn the math of rank decomposition, parameter allocation in Qwen-3, hyperparameter tuning (r, alpha), and security risks.

### The Definitive Guide to LLM Fine-Tuning: Objectivee, Mechanisms, and Hardware

Master LLM fine-tuning. Compare SFT vs. CPFT, PEFT vs. Full Fine-Tuning, and hardware requirements (VRAM/GPU) for Llama 3, Mistral, and more.

### Transformer Architecture: Self-Attention & MLOps Guide

Master the inner workings of Transformers. A technical walkthrough of self-attention, multi-head mechanisms, and positional encoding with vector math examples.

### Tokenization Strategies for LLM Applications

Master the mechanics of LLM tokenization. Compare word-based, character-based, BPE, WordPiece, and Unigram architectures, and learn how to choose the right tokenizer for memory and learning efficiency.

### Optimizing LLM Performance: Context Window Impact on RAG Accuracy

Does a larger context window always mean better results? Explore a technical deep dive into benchmarking Llama 3.1 8B, evaluating RAG performance across various token lengths using GPT-4o as a judge.

### Regularizing LLMs with Kullback-Leibler Divergence

Explore how Forward and Reverse KL Divergence regularize LLMs. Learn to balance exploration and exploitation in SFT and RLHF to prevent policy collapse.

## Related Books for Further Understanding

These books cover the wide range of theories and practices; from fundamentals to PhD level.

Linear Algebra Done Right

Foundations of Machine Learning, second edition (Adaptive Computation and Machine Learning series)

Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems

Machine Learning Design Patterns: Solutions to Common Challenges in Data Preparation, Model Building, and MLOps

Hands-On Large Language Models: Language Understanding and Generation

---
