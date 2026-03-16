---
title: "Experience"
permalink: /experience/
author_profile: true
---

## Tencent

**Algorithm Engineer Intern**  
Shenzhen, China  
July 2025 - Present

### Financial function calling

- Trained a multi-intent financial function-calling component with multi-turn SFT and DAPO on top of Qwen3 30A3B.
- Reduced hallucinations around stock and fund codes by increasing parameter-level reward pressure and constructing positive samples for persistently mistaken groups during DAPO training.
- Received a return offer from the internship and continued to work on production-facing model systems.

### Financial agent training and evaluation

- Built the interaction workflow used in multi-turn financial search agent training, including the testing pipeline and the post-test iteration SOP.
- Supported a Qwen3 30A3B-based financial agent whose performance not only reached parity with DeepSeek V3.2 on internal financial search benchmarks, but also surpassed it on FinSearchComp under the same business tool setting.
- Stabilized evaluation by fixing tool images, lowering judge randomness, and iterating judge prompts, reducing score fluctuation from 10% to 1%.
- Used LLM-assisted analysis SOPs to speed up issue localization across tools, evaluation standards, and entity interfaces during rapid system iteration.

## Huawei

**Algorithm Engineer Intern**  
Nanjing, China  
December 2024 - May 2025

### RL training framework optimization

- Improved an internal reinforcement learning training framework to address crash and instability issues during post-training.
- Introduced DAPO into the framework and designed a dynamic sampling strategy that prioritizes samples with higher reward variance.
- Increased training stability without additional compute by implicitly improving effective batch quality through better sample selection.
