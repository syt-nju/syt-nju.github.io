---
permalink: /
title: "Yuteng Shen"
excerpt: "Post-training algorithm engineer at Tencent FiT, focused on agent training, post-training, evaluation, and reliable model systems."
author_profile: true
---

I am a post-training algorithm engineer at Tencent FiT (Financial Technology). I received my bachelor's degree in Artificial Intelligence from Nanjing University (2022-2026), and I am interested in building reliable AI systems that connect model capability with real-world deployment.

I like widening the boundary of my understanding a little at a time, while focusing on work that can create real value in practice. My recent work has focused on agent training, post-training, and evaluation workflows for fast system iteration. Before joining Tencent full-time, I interned as an algorithm engineer at Tencent and Huawei, where I worked on practical model behavior, training stability, and testing infrastructure.

## Focus

- Agent training and tool-use behavior
- Post-training with reinforcement learning
- Practical AI engineering 

## Selected Experience

### Tencent FiT, Post-Training Algorithm Engineer

**Fin-Multiturn Financial Search Agent**

- Built an agent that answers complex financial questions by breaking them into dependent search steps—for example, identifying the relevant stocks before retrieving and combining their latest market data across 79 internal financial tools.
- Owned the end-to-end training pipeline, from mining real user queries and synthesizing multi-step tasks to training a Qwen3 30A3B policy with multi-turn SFT, rejection fine-tuning, and process-aware RL.
- Deployed the agent to online financial traffic and surpassed DeepSeek V3.2 on FinSearchComp under the same business tool setting.
- Started as an intern in July 2025 and joined full-time after graduating in 2026.

### Huawei, Algorithm Engineer Intern

- Improved an internal RL training framework by introducing DAPO and designing dynamic sampling for high-variance rewards.
- Increased training stability without extra compute by prioritizing informative samples during optimization.

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