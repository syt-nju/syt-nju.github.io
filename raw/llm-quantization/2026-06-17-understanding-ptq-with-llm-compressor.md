# Understanding Post-Training Quantization with LLM Compressor

> Source: https://huggingface.co/blog/rishiraj/llm-compressor
> Author: Rishiraj Acharya
> Collected: 2026-08-10
> Published: 2026-06-17
> License: Unknown
> Completeness: Complete
> Retrieval note: Fetched from Hugging Face blog HTML via WebFetch; navigation and community chrome removed.

We’ve all been there. You find a new 8B or 70B parameter model released in Hugging Face, or maybe you’ve just finished a long fine-tuning run. You excitedly write your inference script, hit execute, and... your GPU immediately throws a fatal `CUDA Out of Memory` error. The weights alone are demanding more VRAM than your hardware physically has.

So, you look into quantization. You know that if you can squash those 16-bit floating-point weights down to 4-bit integers, you can drastically reduce the memory footprint. But how exactly do you do that without lobotomizing the model in the process?

Today, we're going to open up a trace on exactly how post-training quantization (PTQ) works. We’ll use `llm-compressor`, the official production quantization toolkit from the vLLM project. We’ll take a standard full-precision model, compress it, and chase the answers to what exactly happens to the math, the file size, and the output quality along the way.

I learnt about these techniques from the Fast & Efficient LLM Inference with vLLM course by Red Hat on DeepLearning.AI and I'll try sharing my knowledge from there by asking some questions first and trying to find an answer to them so let’s start pulling on some threads.

---

### Wait, why can't we just round the numbers?

If you want to turn a 16-bit float (like `0.1423`) into a 4-bit integer, the most obvious engineering approach is just to round it to the nearest available integer bucket.

This algorithm actually exists. It’s called **Round-to-Nearest (RTN)**. It’s fast, you don't need any data to run it, and it executes almost instantly. But there’s a catch: if you use RTN to drop a model down to INT4, the model’s reasoning capabilities usually fall off a cliff.

Why? Because in a neural network, not all weights are created equal.

If we want to shrink the model without breaking it, we need a smarter algorithm. `llm-compressor` gives us a few choices, and they generally fall into two distinct philosophical camps:

**1. Activation-Aware Weight Quantization (AWQ):** AWQ makes a fascinating observation: a tiny percentage of weights are attached to activations that consistently spike to massive values during inference. If you introduce a rounding error to a weight that gets multiplied by a huge activation spike, that error blows up, cascades through the layers, and ruins the model's output.

AWQ solves this by running a small amount of calibration data through the model to monitor which activations spike. It leaves the weights associated with those spikes alone (or treats them with extremely high precision) and aggressively quantizes the rest. It’s fast, computationally light, and runs incredibly well on NVIDIA hardware.

**2. Generative Pre-trained Transformers Quantization (GPTQ):** GPTQ takes a more mathematically rigorous, optimization-based approach. It asks: *"If I introduce an error by rounding this specific weight, how can I mathematically adjust the remaining unquantized weights in this exact layer to compensate for my mistake?"*

To pull this off, GPTQ computes the Hessian matrix of the loss with respect to the weights. In plain English: it measures the "curvature" or sensitivity of the loss landscape. It figures out exactly how a change in one weight impacts the final output. It then works layer by layer, column by column, quantizing a weight, measuring the error, and updating the remaining weights to absorb that error. Computing these Hessians takes more VRAM during the compression phase, but it results in highly resilient, accurate models and is practically an industry standard.

_(Side note: `llm-compressor` also supports extreme sparsity modifiers like **Sparse-GPT** for 2:4 sparsity if you are lucky enough to have an NVIDIA H100, as well as transform modifiers like **SmoothQuant**, **QuIP**, and **SpinQuant**. These algorithms physically rotate or smooth out the activation math to make the model easier to quantize. But to keep things focused, we'll stick to standard GPTQ.)_

---

### Building the Recipe: Why W4A16?

In `llm-compressor`, compression happens via a straightforward API called `oneshot()`. It’s called "oneshot" because it performs the compression in a single pass over the data, no expensive retraining or backward propagation required.

To make it work, we have to pass the API a "recipe." Let's build it:

```python
from llmcompressor.modifiers.quantization import GPTQModifier

recipe = GPTQModifier(
    scheme="W4A16",
    targets="Linear",
    ignore=["lm_head"],
)
```

Let's break down exactly why we are passing these specific arguments.

**Why the `W4A16` scheme?** This means we are dropping the Weights (W) to 4-bit, but leaving the Activations (A) at 16-bit (FP16/BF16). Why not quantize the activations too, like `W8A8`? Because activations are notoriously difficult to quantize. While weights are static, activations change depending on the user's prompt. They frequently contain massive outliers. By keeping activations in 16-bit, we preserve the dynamic range of the model's thoughts, while squashing the static weights down to 4-bit. This yields massive VRAM savings with a very low impact on quality.

**Why target `Linear` and ignore `lm_head`?** We target the linear layers because that's where the vast majority of the model's parameters live. If you want a smaller file size, that's where the money is.

But we explicitly tell the algorithm to skip the `lm_head`. The `lm_head` is the final linear layer that projects the model's internal vector math back into our massive human vocabulary (often 128,000+ tokens). If you lose precision in the `lm_head`, the model loses its ability to precisely select the right word, resulting in immediate gibberish. We leave the `lm_head` at 16-bit.

---

### Wait, how does it know what to compensate for?

Earlier, we noted that GPTQ adjusts weights to compensate for rounding errors. But to know what an "error" looks like, the model actually has to read some text.

This is why the `oneshot` API requires a calibration dataset.

```python
from llmcompressor import oneshot

# Run the single-pass compression
oneshot(
    model="meta-llama/Meta-Llama-3-8B",
    dataset="wikitext",
    dataset_config_name="wikitext-2-raw-v1",
    recipe=recipe,
    output_dir="./Llama-3-8B-W4A16",
    max_seq_length=4096,
    num_calibration_samples=256,
)
```

We point the API at a model and a standard, dense dataset (like Wikipedia articles). But look at the parameters `num_calibration_samples` and `max_seq_length`.

We are only passing 256 samples through the model. Why not 10,000? Because calibration has steep diminishing returns. If you use 10 samples, the model doesn't get a good picture of the activation landscape. But past a few hundred samples, the accuracy gains become microscopic while the compute time skyrockets. 256 is the golden ratio.

We also set `max_seq_length=4096`. This ensures the model processes realistic, long-form context so the quantizer can see how weights behave during sustained generation, rather than just on short sentences.

---

### The File Size Mystery: Where did the math go wrong?

Once the compression finishes, we check our disk. We just took a model with 16-bit float weights and crushed them into 4-bit integer weights. That is a 4x reduction. Our model should theoretically be 75% smaller.

But when we check the directory sizes, we might see something like this:

```text
Model Size Comparison
=============================================
Original (BF16):    15.30 GB
Quantized (W4A16):  5.45 GB
Reduction:          64%
```

Wait. 64%? Why didn't it shrink by 75%? Look back at our recipe. Remember the `ignore=["lm_head"]` argument?

We only quantized the _linear_ layers. The embedding layers, the layer norms, and the `lm_head` all stayed at 16-bit. Because those layers remain uncompressed, they drag the overall compression ratio down.

This creates an interesting quirk: the larger the model, the better your compression ratio. In a tiny 0.5B parameter model, the vocab head makes up a massive percentage of the file size, so W4A16 might only yield a 40% overall reduction. But in a 70B parameter model, the linear layers absolutely dwarf the vocab head, and your reduction approaches that theoretical 75% maximum.

---

### Did we break the reasoning?

A smaller model is useless if it sounds like a drunken autocorrect. We need to test the outputs side-by-side using Hugging Face's `transformers` library.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "./Llama-3-8B-W4A16"
prompt = "The fundamental architecture of a transformer relies on"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu",
    dtype=torch.bfloat16
)

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**Base Model Output:**

> "...the mechanism of self-attention, which allows the model to weigh the importance of different words in a sequence regardless of their positional distance, eliminating the need for recurrent layers."

**Quantized W4A16 Output:**

> "...the self-attention mechanism, enabling the network to dynamically prioritize different parts of the input sequence contextually, which removes the necessity for sequential data processing."

They aren't perfectly identical, which is mathematically expected since we fundamentally altered the weights, but the quantized model is logically sound, factually correct, and retains its deep technical context.

But "vibes" aren't a metric we can take to production. We need a real number to prove the model hasn't degraded.

### Enter Perplexity (The Ultimate Lie Detector)

To truly know if our GPTQ compression worked, we calculate **Perplexity (PPL)**.

Perplexity measures how "surprised" a model is by a sequence of text. We feed the model a holdout set of data, and use a sliding window to calculate the cross-entropy loss between what the model _predicts_ the next token should be, versus what the token _actually_ is.

If we severely damaged the model's weights, it will start predicting the wrong tokens, its "surprise" will go up, and the perplexity score will spike. Lower is always better.

When we run this rigorous sliding-window test on both our base model and our compressed model, the output gives us the objective truth:

```text
Perplexity Comparison
========================================
Base (BF16):       14.22
Quantized (W4A16): 14.85
Difference:        +0.63 (+4.4%)
```

There it is. Our perplexity went up by about 4.4%.

Is that an acceptable trade-off? For almost every production engineering use case, absolutely.

We traded a negligible 4% hit in our perplexity score for a massive 64% reduction in memory footprint. We took a heavy, demanding floating-point matrix, mapped the structural load points using the Hessian loss, squashed the bulk of it into 4-bit integers, protected the output vocabulary head, and ended up with a model that is drastically cheaper to host without sounding any less intelligent.

And because we used `llm-compressor`, this W4A16 model is natively structured to be loaded directly into `vLLM`, an ultra-fast inference engine that can now run this model at incredibly high batch sizes.
