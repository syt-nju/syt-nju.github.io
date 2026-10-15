---
title: "Experience"
permalink: /experience/
author_profile: true
---

## Tencent FiT

**Post-Training Algorithm Engineer**  
Shenzhen, China  
2026 - Present

**Algorithm Engineer Intern**  
July 2025 - 2026

### NeoData financial data service

- Main model contributor to NeoData, which turns user queries into retrieval over large-scale licensed financial data through financial entity extraction and MCP tool calls.
- Serves tens of millions of daily active users; the sole financial data provider for Yuanbao, one of the main financial data interfaces for WorkBuddy, and a data backbone for many other financial businesses.
- Iterate on multiple small models along the pipeline, covering financial entity retrieval and ranking, and financial tool selection and calling (tool choice and argument filling).

### Fin-Multiturn financial search agent

- One of the main contributors. In the second half of 2025, before mature agent training frameworks were available, extended verl to support multi-turn rollouts with tool calls and trained the agent end to end.
- Built the agent around a Qwen3-30B-A3B policy and 79 MCP tools, covering entity resolution, multi-intent function calling, dependency-aware planning, and recovery from empty results.
- Built three generations of training data: filtered 700K production queries into 1K multi-hop seeds, explored DAG-first generation, and converged on template-guided synthesis grounded in real user phrasing.
- Trained with multi-turn SFT, rejection fine-tuning, and process-aware RL using answer-, DAG-, trajectory-, turn-, and parameter-level feedback.
- Designed asynchronous query-aware compression for 128K rollouts, keeping the newest tool response intact while compressing prior observations off the rollout critical path.
- Reached parity with DeepSeek V3.2 on internal benchmarks and surpassed it on FinSearchComp under the same business tool setting.

## Huawei

**Algorithm Engineer Intern**  
Nanjing, China  
December 2024 - May 2025

### DeepSeek-R1 reproduction

- Worked on the team's DeepSeek-R1 reproduction. Long RL runs suffered training collapse: reward dropped sharply mid-training and the model began producing incoherent output.
- Introduced DAPO into the team's internal RL training framework to address the collapse.
- Added a dynamic sampler that prioritizes prompts with high reward variance, so each batch carries more learning signal; this extended the number of stable training steps by about one third.
