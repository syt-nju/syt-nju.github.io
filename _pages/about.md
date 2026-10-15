---
permalink: /
title: "Yuteng Shen"
excerpt: "Post-training algorithm engineer at Tencent FiT, focused on agent training, post-training, evaluation, and reliable model systems."
author_profile: true
---

I am a post-training algorithm engineer at Tencent FiT (Financial Technology), which I joined full-time after completing my bachelor's degree in Artificial Intelligence at Nanjing University (2022-2026). I am interested in building reliable AI systems that connect model capability with real-world deployment.

## Focus

- Agent training and tool-use behavior
- Post-training with reinforcement learning
- Practical AI engineering 

## Selected Experience

### Tencent FiT, Post-Training Algorithm Engineer

*July 2025 - Present*

**NeoData Financial Data Service**

- Main model contributor to NeoData, Tencent FiT's financial data service that turns user queries into retrieval over large-scale licensed financial data through financial entity extraction and MCP tool calls.
- Iterate on the pipeline's small models for financial entity retrieval and ranking, and for financial tool selection and calling. NeoData serves tens of millions of daily active users, is the sole financial data provider for Yuanbao, one of the main financial data interfaces for WorkBuddy, and supplies data capabilities to many other financial businesses.

**Fin-Multiturn Financial Search Agent**

- One of the main contributors. In the second half of 2025, before mature agent training frameworks were available, extended verl to support multi-turn tool-use training for a financial search agent.
- Trained a Qwen3-30B-A3B policy with multi-turn SFT, rejection fine-tuning, and process-aware RL; it surpassed DeepSeek V3.2 on FinSearchComp under the same business tool setting.

### Huawei, Algorithm Engineer Intern

*December 2024 - May 2025*

**DeepSeek-R1 Reproduction**

- Worked on the team's DeepSeek-R1 reproduction, where long RL runs suffered training collapse: reward dropped sharply mid-training and the model began producing incoherent output.
- Introduced DAPO into the team's internal RL training framework and added a dynamic sampler that prioritizes prompts with high reward variance, extending the number of stable training steps by about one third.

## Selected Projects

### [Slay2Agent](https://github.com/syt-nju/Slay2Agent)

A train-free agent framework that drives a cloud LLM to autonomously play *Slay the Spire 2* through the STS2MCP REST API—no screen reading, no local model training. Strategy knowledge accumulates across runs via a three-layer memory system (in-context history, a read-only skill library, and an oracle) that is updated offline from full game traces.

### [myTorch](https://github.com/syt-nju/myTorch)

A NumPy-based neural network framework that mimics parts of the PyTorch experience. It uses operator overloading to enter the computation graph, manages the graph lifecycle during backpropagation, and emphasizes reusable abstractions in the core design.

### Open Source Fixes

- Fixed a checkpoint-saving bug in the DAPO trainer of `verl` and contributed the patch upstream.
- Identified and fixed a typing issue in the `GRPO` trainer of `TRL`.

## Writing

I use writing to organize newly learned material, capture temporary ideas from work, and reflect on experiences that changed my understanding. Over time, this site will collect the notes that best reflect how I learn, think, and build.