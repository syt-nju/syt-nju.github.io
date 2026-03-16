---
title: "Writing"
permalink: /writing/
author_profile: true
---

I treat writing as a way to organize my thoughts and force myself to understand what I have learned more clearly. For me, a good technical blog is less a polished conclusion than a structured trace of learning, where scattered intuition becomes something explicit enough to revisit and reuse. Because of that, much of what I write comes from newly learned material and temporary ideas that are still taking shape.

## What I Write About

- Reading notes and summaries on papers, blogs, and newly learned technical material
- Temporary work ideas and abstract reflections that emerge during day-to-day problem solving
- Retrospective notes on experiences, including what worked, what did not, and what changed my understanding

## Current Direction

I regularly share technical reflections outside formal project documentation, including public writing in Chinese. Across those posts, I have accumulated more than 400 followers and over 400,000 total reads. Most of them are notes written close to the learning process itself, which is why they often focus on fresh concepts, partial understandings, and ideas worth testing rather than only finished conclusions.

## Selected Posts

### Why online RFT falls short of RLVR

[Why online RFT falls short of RLVR: negative samples are the key](https://zhuanlan.zhihu.com/p/1941603697311342965)

This post starts from an online RFT experiment built on top of a modified DAPO pipeline and asks a practical question: if we keep amplifying hard cases and positive trajectories, can pass@1 approach pass@k without full RL-style exploration? The conclusion is no. Pure RFT quickly memorizes successful trajectories and reaches high training reward, but its evaluation reward falls behind. My main takeaway is that RL's exploration and generalization are both tightly connected to negative samples, because they keep correcting unstable reasoning paths instead of letting the model profit from lucky guesses and hallucinated successes.

### Understanding GSPO from the objective level

[Reading GSPO: routing replay, sequence-level clipping, and why the objective changes](https://zhuanlan.zhihu.com/p/1933217003654586554)

This post is a reading note on GSPO and its relation to GRPO and DAPO. I focus on why MoE training is sensitive to routing replay, how token-level clipping can create unstable updates when routing changes, and why GSPO moves clipping toward the sequence level. I also discuss a broader intuition: sequence-level clipping may reduce the built-in preference for reinforcing already high-probability tokens, which could make exploration paths easier to preserve during RLVR training.

## Why Writing Matters To Me

Writing forces me to turn vague intuition into explicit reasoning. In practice, that habit improves how I debug model behavior, design experiments, and communicate system changes with teammates. Over time, I want this page to become a curated set of the notes that best reflect how I learn, think, and build.
