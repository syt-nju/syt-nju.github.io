---
title: "Experience"
permalink: /experience/
author_profile: true
---

## Tencent

**Algorithm Engineer Intern**  
Shenzhen, China  
July 2025 - Present

### Fin-Multiturn financial search agent

- Built an end-to-end financial search agent around a Qwen3 30A3B policy and 79 MCP tools, covering entity resolution, multi-intent function calling, dependency-aware planning, and recovery from empty results.
- Built three generations of training data: filtered 700K production queries into 1K multi-hop seeds, explored DAG-first generation, and converged on template-guided synthesis grounded in real user phrasing.
- Trained the agent with multi-turn SFT, rejection fine-tuning, and process-aware RL using answer-, DAG-, trajectory-, turn-, and parameter-level feedback.
- Designed asynchronous query-aware compression for 128K rollouts, keeping the newest tool response intact while compressing prior observations off the rollout critical path.
- Built the interaction and evaluation workflow for rapid iteration; the resulting agent reached parity with DeepSeek V3.2 on internal benchmarks and surpassed it on FinSearchComp under the same business tool setting.
- Received a return offer and continued working on production-facing model systems.

## Huawei

**Algorithm Engineer Intern**  
Nanjing, China  
December 2024 - May 2025

### RL training framework optimization

- Improved an internal reinforcement learning training framework to address crash and instability issues during post-training.
- Introduced DAPO into the framework and designed a dynamic sampling strategy that prioritizes samples with higher reward variance.
- Increased training stability without additional compute by implicitly improving effective batch quality through better sample selection.
