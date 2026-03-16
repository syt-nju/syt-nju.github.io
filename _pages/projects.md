---
title: "Projects"
permalink: /projects/
author_profile: true
---

## myTorch

**Repository:** [github.com/syt-nju/myTorch](https://github.com/syt-nju/myTorch)

`myTorch` is a NumPy-based neural network framework designed to reproduce a streamlined PyTorch-like workflow while keeping the implementation compact and readable.

### What I built

- Operator overloading for tensor-style user interaction
- Automatic entry into the computation graph during forward execution
- Graph lifecycle management during backpropagation
- Reusable abstractions built from base classes and decorators

### Why it matters

This project strengthened my understanding of code modularity, improved my engineering sense for building reusable abstractions, and gave me a much more concrete grasp of the details behind backpropagation.

## Open Source Contributions

### verl

- Fixed a bug in the DAPO trainer where confusing step semantics prevented checkpoints from being saved automatically.
- Contributed the fix upstream and merged it into the main branch.

### TRL

- Identified and fixed a type-annotation issue in the `GRPO` trainer.
- Used the contribution process to improve correctness in a commonly used post-training codebase.

## What I Like Building

- LLM agents with clear tool-use behavior
- Evaluation systems that make model iteration reliable
- Training and post-training pipelines with strong feedback loops
- Small but well-structured infrastructure for experimentation
