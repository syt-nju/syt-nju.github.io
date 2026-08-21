# Wiki Log

## 2026-07-30 ingest | Why Online RFT Falls Short of RLVR
- Disposition: New
- Raw: raw/agent-training/online-rft-vs-rlvr.md
- Updated: _wiki/agent-training/online-rft-vs-rlvr.md

## 2026-07-30 ingest | Understanding GSPO from the Objective Level
- Disposition: New
- Raw: raw/agent-training/gspo-objective.md
- Updated: _wiki/agent-training/gspo-objective.md

## 2026-08-03 ingest | FORT-Searcher: Shortcut-Resistant Search Task Synthesis
- Disposition: New
- Topic: deep-search (new topic added to _data/wiki_topics.yml)
- Raw: raw/deep-search/2026-06-10-fort-searcher.md
- Updated: _wiki/deep-search/fort-searcher.md
- License note: arXiv perpetual non-exclusive license 1.0, not Creative Commons. Full text mirrored at the maintainer's explicit decision; redistribution is not expressly granted by the license.

## 2026-08-04 ingest | Voyager: An Open-Ended Embodied Agent with Large Language Models
- Disposition: New
- Topic: skill-management (new topic added to _data/wiki_topics.yml)
- Raw: raw/skill-management/2023-05-25-voyager.md
- Created: _wiki/skill-management/overview.md
- Created: _wiki/skill-management/skill-library.md
- Cross-links: overview ↔ skill-library
- Note: Full text mirrored by maintainer decision; License recorded as arXiv perpetual non-exclusive (metadata only).

## 2026-08-04 ingest | AutoSkill: Experience-Driven Lifelong Learning via Skill Self-Evolution
- Disposition: Update + New
- Topic: skill-management
- Raw: raw/skill-management/2026-03-01-autoskill.md
- Created: _wiki/skill-management/skill-lifecycle.md
- Updated: _wiki/skill-management/overview.md
- Updated: _wiki/skill-management/skill-library.md
- Cross-links: overview ↔ skill-library ↔ skill-lifecycle
- Note: Full text from arXiv PDF; License CC BY 4.0.

## 2026-08-04 ingest | SkillOS: Learning Skill Curation for Self-Evolving Agents
- Disposition: New + Update
- Topic: skill-management
- Raw: raw/skill-management/2026-05-07-skillos.md
- Created: _wiki/skill-management/skill-curation-rl.md
- Updated: _wiki/skill-management/overview.md
- Updated: _wiki/skill-management/skill-library.md
- Cross-links: overview ↔ skill-library ↔ skill-curation-rl ↔ skill-lifecycle
- Note: Full text from arXiv PDF; License recorded as arXiv non-exclusive (metadata only).

## 2026-08-04 ingest | Ratchet: A Minimal Hygiene Recipe for Self-Evolving LLM Agents
- Disposition: Update
- Topic: skill-management
- Raw: raw/skill-management/2026-05-ratchet.md
- Updated: _wiki/skill-management/overview.md
- Updated: _wiki/skill-management/skill-lifecycle.md
- Cross-links: overview ↔ skill-lifecycle (hygiene vs AutoSkill lifecycle)
- Note: Full text from arXiv PDF; extends workshop version arXiv:2605.19576.

## 2026-08-04 ingest | MUSE-Autoskill: Self-Evolving Agents via Skill Creation, Memory, Management, and Evaluation
- Disposition: New + Update + Disputed (partial)
- Topic: skill-management
- Raw: raw/skill-management/2026-05-26-muse-autoskill.md
- Created: _wiki/skill-management/muse-autoskill.md
- Updated: _wiki/skill-management/overview.md
- Updated: _wiki/skill-management/skill-lifecycle.md
- Updated: _wiki/skill-management/skill-library.md
- Cross-links: overview ↔ muse-autoskill ↔ skill-lifecycle ↔ skill-library ↔ skill-curation-rl
- Note: Full text from arXiv PDF v2; License CC BY 4.0. Disputed region: SkillsBench LLM-self-authored null result (Ratchet) vs MUSE self-created skill gains under lifecycle-managed protocol.

## 2026-08-04 maintain | Remove agent-training topic
- Disposition: Update
- Topic: agent-training (removed from _data/wiki_topics.yml)
- Deleted wiki: _wiki/agent-training/online-rft-vs-rlvr.md
- Deleted wiki: _wiki/agent-training/gspo-objective.md
- Deleted raw: raw/agent-training/online-rft-vs-rlvr.md
- Deleted raw: raw/agent-training/gspo-objective.md
- Updated: _data/wiki_topics.yml; tests/wiki-search.test.js
- Note: Topic was test content; wiki pages and raw removed together by maintainer decision.

## 2026-08-04 maintain | Restyle deep-search to skill-management pattern
- Disposition: Update
- Topic: deep-search
- Raw: raw/deep-search/2026-06-10-fort-searcher.md (unchanged)
- Created: _wiki/deep-search/overview.md
- Created: _wiki/deep-search/realized-difficulty.md
- Created: _wiki/deep-search/trajectory-signatures.md
- Updated: _wiki/deep-search/fort-searcher.md (zh-CN system page; concept split out)
- Cross-links: overview ↔ realized-difficulty ↔ trajectory-signatures ↔ fort-searcher
- Note: Reorganized from English paper-centric page into Chinese concept/system pages aligned with skill-management style.

## 2026-08-05 ingest | Meta Context Engineering via Agentic Skill Evolution
- Disposition: New + Update
- Topic: skill-management
- Raw: raw/skill-management/2026-01-29-meta-context-engineering.md
- Created: _wiki/skill-management/meta-context-engineering.md
- Updated: _wiki/skill-management/overview.md
- Updated: _wiki/skill-management/skill-library.md
- Updated: _wiki/skill-management/skill-lifecycle.md
- Cross-links: overview ↔ meta-context-engineering ↔ skill-library ↔ skill-lifecycle ↔ muse-autoskill
- Note: Full text from arXiv PDF via deepxiv (v2); License arXiv perpetual non-exclusive 1.0 (metadata only). Complementary to SkillsBench dispute (CE benchmarks, not task-skill self-authoring).

## 2026-08-07 ingest | Meta-Harness: End-to-End Optimization of Model Harnesses
- Disposition: New + Update
- Topic: skill-management
- Raw: raw/skill-management/2026-03-30-meta-harness.md
- Created: _wiki/skill-management/meta-harness.md
- Updated: _wiki/skill-management/overview.md
- Updated: _wiki/skill-management/meta-context-engineering.md
- Updated: _wiki/skill-management/skill-library.md
- Updated: _wiki/skill-management/skill-lifecycle.md
- Cross-links: overview ↔ meta-harness ↔ meta-context-engineering ↔ skill-library ↔ skill-lifecycle
- Note: Full text from arXiv PDF via pdftotext (v1); License CC BY 4.0. Complementary to MCE (different base model / absolute Acc vs Avg. Rel. Gain); TerminalBench-2 is discovery-style same-split search.

## 2026-08-07 maintain | Reframe skill-management overview inventory
- Disposition: Update
- Topic: skill-management
- Updated: _wiki/skill-management/overview.md
- Note: Replaced「七条演进线」with「已摄入代表系统」grouped by problem focus; clarify listed papers are not independent evolution directions.

## 2026-08-07 maintain | Restructure wiki topics as problem maps
- Disposition: Update
- Topic: deep-search; skill-management
- Created: _wiki/skill-management/self-authored-skills.md
- Updated: _wiki/deep-search/overview.md
- Updated: _wiki/skill-management/overview.md
- Updated: _wiki/skill-management/skill-library.md
- Updated: _wiki/skill-management/skill-lifecycle.md
- Updated: _wiki/skill-management/muse-autoskill.md
- Cross-links: skill-management overview ↔ self-authored-skills ↔ skill-lifecycle ↔ muse-autoskill
- Note: Applied new page-role rules: overview pages are topic problem maps; system pages remain evidence anchors; disputed self-authored skill claims moved out of overview.

## 2026-08-07 ingest | Self-Harness: Harnesses That Improve Themselves
- Disposition: New + Update
- Topic: skill-management
- Raw: raw/skill-management/2026-06-08-self-harness.md
- Created: _wiki/skill-management/self-harness.md
- Updated: _wiki/skill-management/overview.md
- Updated: _wiki/skill-management/meta-harness.md
- Updated: _wiki/skill-management/skill-lifecycle.md
- Cross-links: overview ↔ meta-harness ↔ self-harness ↔ skill-lifecycle
- Note: Full text from ar5iv HTML rendering of arXiv v1; License Unknown. Complements Meta-Harness by internalizing harness proposal into the target model with held-in / held-out non-regression validation.

## 2026-08-07 maintain | Rename skill-management to harness-evolution
- Disposition: Update
- Topic: harness-evolution
- Updated: _data/wiki_topics.yml
- Updated: _wiki/harness-evolution/overview.md
- Updated: _wiki/harness-evolution/skill-library.md
- Updated: _wiki/harness-evolution/skill-lifecycle.md
- Updated: _wiki/harness-evolution/skill-curation-rl.md
- Updated: _wiki/harness-evolution/muse-autoskill.md
- Updated: _wiki/harness-evolution/meta-context-engineering.md
- Updated: _wiki/harness-evolution/meta-harness.md
- Updated: _wiki/harness-evolution/self-authored-skills.md
- Updated: _wiki/harness-evolution/self-harness.md
- Cross-links: old /wiki/skill-management/* URLs preserved via redirect_from
- Note: Topic reframed from task skill management to the broader external harness evolution problem map.

## 2026-08-07 ingest | Harness Engineering for Self-Improvement
- Disposition: New + Update
- Topic: harness-evolution
- Raw: raw/harness-evolution/2026-07-04-harness-engineering-for-self-improvement.md
- Created: _wiki/harness-evolution/harness-engineering.md
- Updated: _wiki/harness-evolution/overview.md
- Updated: _wiki/harness-evolution/skill-library.md
- Updated: _wiki/harness-evolution/skill-lifecycle.md
- Updated: _wiki/harness-evolution/meta-context-engineering.md
- Updated: _wiki/harness-evolution/meta-harness.md
- Updated: _wiki/harness-evolution/self-harness.md
- Cross-links: overview ↔ harness-engineering ↔ skill-library ↔ skill-lifecycle ↔ meta-context-engineering ↔ meta-harness ↔ self-harness
- Note: Full text from original Lil'Log HTML via WebFetch; License Unknown. Provides the frame for harness design patterns, optimization depth, RSI relevance, and evaluator/permission boundaries.

## 2026-08-10 ingest | Quantization concepts (Hugging Face)
- Disposition: New
- Topic: llm-quantization (new topic added to _data/wiki_topics.yml)
- Raw: raw/llm-quantization/hf-quantization-concepts.md
- Created: _wiki/llm-quantization/overview.md
- Created: _wiki/llm-quantization/linear-quantization.md
- Created: _wiki/llm-quantization/ptq-vs-qat.md
- Cross-links: overview ↔ linear-quantization ↔ ptq-vs-qat
- Note: License Unknown; full Transformers docs page mirrored.

## 2026-08-10 ingest | Practical Quantization in PyTorch
- Disposition: Update
- Topic: llm-quantization
- Raw: raw/llm-quantization/2024-11-15-pytorch-quantization-in-practice.md
- Updated: _wiki/llm-quantization/linear-quantization.md
- Updated: _wiki/llm-quantization/ptq-vs-qat.md
- Cross-links: linear-quantization ↔ ptq-vs-qat
- Note: License Unknown; Published/Last updated 2024-11-15.

## 2026-08-10 ingest | 第 7 章 量化：用更少的显存跑更大的模型
- Disposition: Update + New
- Topic: llm-quantization
- Raw: raw/llm-quantization/inferloop-llm-infra-quantization.md
- Created: _wiki/llm-quantization/gptq-awq.md
- Updated: _wiki/llm-quantization/overview.md
- Updated: _wiki/llm-quantization/linear-quantization.md
- Updated: _wiki/llm-quantization/ptq-vs-qat.md
- Cross-links: overview ↔ gptq-awq ↔ ptq-vs-qat
- Note: License Unknown; Published date Unknown.

## 2026-08-10 ingest | Understanding Post-Training Quantization with LLM Compressor
- Disposition: Update
- Topic: llm-quantization
- Raw: raw/llm-quantization/2026-06-17-understanding-ptq-with-llm-compressor.md
- Updated: _wiki/llm-quantization/gptq-awq.md
- Updated: _wiki/llm-quantization/overview.md
- Cross-links: overview ↔ gptq-awq
- Note: License Unknown; Published 2026-06-17.

## 2026-08-10 ingest | 模型量化算法详解：从 PTQ 到 AWQ 的大模型压缩实战指南
- Disposition: Update
- Topic: llm-quantization
- Raw: raw/llm-quantization/2026-07-02-tencent-ptq-to-awq-guide.md
- Updated: _wiki/llm-quantization/linear-quantization.md
- Updated: _wiki/llm-quantization/ptq-vs-qat.md
- Updated: _wiki/llm-quantization/gptq-awq.md
- Note: License Unknown; Published 2026-07-02.

## 2026-08-10 ingest | Quantization-Aware Training for Large Language Models with PyTorch
- Disposition: New + Update
- Topic: llm-quantization
- Raw: raw/llm-quantization/2024-11-12-pytorch-qat-for-llms.md
- Created: _wiki/llm-quantization/qat-for-llms.md
- Updated: _wiki/llm-quantization/overview.md
- Updated: _wiki/llm-quantization/ptq-vs-qat.md
- Cross-links: overview ↔ qat-for-llms ↔ ptq-vs-qat
- Note: License Unknown; Published/Last updated 2024-11-12.

## 2026-08-10 ingest | QLoRA Deep Dive: Efficient LLM Fine-Tuning with NF4 & Paged Optimizers
- Disposition: New + Update
- Topic: llm-quantization
- Raw: raw/llm-quantization/qlora-nf4-double-quantization-deep-dive.md
- Created: _wiki/llm-quantization/qlora.md
- Updated: _wiki/llm-quantization/overview.md
- Updated: _wiki/llm-quantization/ptq-vs-qat.md
- Cross-links: overview ↔ qlora ↔ qat-for-llms ↔ ptq-vs-qat
- Note: License Unknown; Published date Unknown.

## 2026-08-10 ingest | 当谈论 FP8 训练的时候，我们到底在聊什么?
- Disposition: New + Update
- Topic: llm-quantization
- Raw: raw/llm-quantization/2025-11-09-fp8-training-recipes.md
- Created: _wiki/llm-quantization/fp8-training.md
- Updated: _wiki/llm-quantization/overview.md
- Updated: _wiki/llm-quantization/linear-quantization.md
- Updated: _wiki/llm-quantization/ptq-vs-qat.md
- Cross-links: overview ↔ fp8-training ↔ linear-quantization ↔ ptq-vs-qat
- Note: License Unknown; qingkeai mirror of Zhihu post; Published 2025-11-09. Figure-only memory tables not transcribed as precise Wiki claims.

## 2026-08-11 ingest | Tricks from OpenAI gpt-oss YOU can use with transformers
- Disposition: New + Update
- Topic: llm-quantization
- Raw: raw/llm-quantization/2025-09-11-hf-faster-transformers-gpt-oss.md
- Created: _wiki/llm-quantization/mxfp4.md
- Updated: _wiki/llm-quantization/overview.md
- Updated: _wiki/llm-quantization/linear-quantization.md
- Updated: _wiki/llm-quantization/gptq-awq.md
- Updated: _wiki/llm-quantization/fp8-training.md
- Cross-links: overview ↔ mxfp4 ↔ linear-quantization ↔ gptq-awq ↔ fp8-training
- Note: License Unknown; knowledge compile focused on MXFP4 section; TP/EP/cache chapters not compiled.

## 2026-08-17 ingest | On-Policy Distillation
- Disposition: New
- Topic: on-policy-distillation
- Raw: raw/on-policy-distillation/2025-10-27-on-policy-distillation.md
- Created: _wiki/on-policy-distillation/overview.md
- Created: _wiki/on-policy-distillation/sft-rl-opd.md
- Cross-links: overview ↔ sft-rl-opd
- Note: License Unknown; first source for this topic. Page cut is the 2×2 of sampling source vs supervision density, not the blog's experiment list.

## 2026-08-17 ingest | Training LLMs using Off-Policy vs On-Policy Distillation
- Disposition: Update
- Topic: on-policy-distillation
- Raw: raw/on-policy-distillation/2026-07-on-policy-distillation-floating-bytes.md
- Updated: _wiki/on-policy-distillation/sft-rl-opd.md
- Updated: _wiki/on-policy-distillation/overview.md
- Cross-links: overview ↔ sft-rl-opd
- Note: License Unknown; HTML-to-text of the blog. Folded the two-forward-pass loop into sft-rl-opd; Colab 0.5B/1.5B numbers were not compiled as TML results.

## 2026-08-17 ingest | Distillation (Tinker Cookbook)
- Disposition: Update
- Topic: on-policy-distillation
- Raw: raw/on-policy-distillation/tinker-cookbook-distillation.md
- Updated: _wiki/on-policy-distillation/sft-rl-opd.md
- Updated: _wiki/on-policy-distillation/overview.md
- Cross-links: overview ↔ sft-rl-opd
- Note: License Unknown; HTML parse of the cookbook recipe. Hyperparameters folded into concept-page operational notes; no Tinker system page.

## 2026-08-17 ingest | Rethinking On-Policy Distillation of Large Language Models
- Disposition: New + Update
- Topic: on-policy-distillation
- Raw: raw/on-policy-distillation/2026-04-14-rethinking-on-policy-distillation.md
- Created: _wiki/on-policy-distillation/when-opd-works.md
- Updated: _wiki/on-policy-distillation/overview.md
- Updated: _wiki/on-policy-distillation/sft-rl-opd.md
- Cross-links: overview ↔ sft-rl-opd ↔ when-opd-works
- Note: CC BY 4.0; arXiv 2604.13016 v2. Split the second field axis: when teacher signal is learnable (pattern consistency + new capability). Cold-start recipe folded into that page, not a separate SFT recipe page.

## 2026-08-17 ingest | Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes
- Disposition: Update
- Topic: on-policy-distillation
- Raw: raw/on-policy-distillation/2026-03-26-revisiting-on-policy-distillation.md
- Updated: _wiki/on-policy-distillation/when-opd-works.md
- Updated: _wiki/on-policy-distillation/overview.md
- Cross-links: overview ↔ sft-rl-opd ↔ when-opd-works
- Note: arXiv non-exclusive license; arXiv 2603.25562. Sampled-token failure modes and teacher top-K local support matching folded into when-opd-works rather than a new estimator page.

## 2026-08-17 ingest | OPD深度解析：从数学推导到DeepSeek V4、SWIFT与verl实践
- Disposition: No material
- Topic: on-policy-distillation
- Raw: raw/on-policy-distillation/zhihu-opd-deep-dive.md
- Note: Completeness Partial. Direct fetch HTTP 503; star-proxy 403 acl_denied; archive.org timeout; Jina reader empty. No body obtained; Wiki pages must not infer this source.

## 2026-08-17 ingest | OPD深度解析：从数学推导到DeepSeek V4、SWIFT与verl实践
- Disposition: New + Update
- Topic: on-policy-distillation
- Raw: raw/on-policy-distillation/zhihu-opd-deep-dive-v2.md
- Created: _wiki/on-policy-distillation/teacher-signal-granularity.md
- Updated: _wiki/on-policy-distillation/overview.md
- Updated: _wiki/on-policy-distillation/sft-rl-opd.md
- Updated: _wiki/on-policy-distillation/when-opd-works.md
- Cross-links: overview ↔ sft-rl-opd ↔ teacher-signal-granularity ↔ when-opd-works
- Note: License Unknown. Prior raw was Partial/empty; this v2 is a complete Jina retrieval (zhuanlan 403/503; Zhihu CLI had no Access Secret in this environment). Split the third field axis: sampled-token vs top-k vs full-vocab. SWIFT/verl/V4 recipes folded into operational notes, not system pages. V4 report itself not ingested.

## 2026-08-17 update | OPD wiki linking
- Disposition: Update
- Topic: on-policy-distillation
- Updated: _wiki/on-policy-distillation/overview.md
- Updated: _wiki/on-policy-distillation/sft-rl-opd.md
- Updated: _wiki/on-policy-distillation/teacher-signal-granularity.md
- Updated: _wiki/on-policy-distillation/when-opd-works.md
- Cross-links: overview terms and claims now deep-link to definition headings and original source URLs
- Note: Added `{#exposure-bias}`, `{#sparse-credit}`, `{#sampled-token}`, `{#top-k}`, `{#full-vocab}`, `{#kl-direction}`, `{#thinking-pattern}`, `{#new-capability}`, `{#cold-start}`, `{#sampled-token-failure}`. Footer sources are no longer the only citation path.

## 2026-08-17 ingest | SFT, RL, and On-Policy Distillation Through a Distributional Lens
- Disposition: Update
- Topic: on-policy-distillation
- Raw: raw/on-policy-distillation/2026-05-10-sft-rl-opd-distributional-lens.md
- Updated: _wiki/on-policy-distillation/overview.md
- Updated: _wiki/on-policy-distillation/sft-rl-opd.md
- Updated: _wiki/on-policy-distillation/when-opd-works.md
- Updated: _wiki/on-policy-distillation/teacher-signal-granularity.md
- Cross-links: overview ↔ sft-rl-opd ↔ when-opd-works ↔ teacher-signal-granularity
- Note: All rights reserved. On-policy data as load-bearing anti-forgetting constraint folded into sft-rl-opd; OPSD / privileged prefix into when-opd-works. Rethinking teacher-importance and nrehiew teacher-matters-less kept as two knobs, not a disputed verdict. No system pages for Minimal Editing, MiMo, GLM 5, or OPSD.

## 2026-08-17 ingest | On the Measure of Intelligence
- Disposition: New
- Topic: test-time-training (new topic added to _data/wiki_topics.yml)
- Raw: raw/test-time-training/2019-11-05-on-the-measure-of-intelligence.md
- Created: _wiki/test-time-training/overview.md
- Created: _wiki/test-time-training/skill-acquisition-efficiency.md
- Created: _wiki/test-time-training/measuring-general-intelligence.md
- Cross-links: overview ↔ skill-acquisition-efficiency ↔ measuring-general-intelligence; overview → harness-evolution/overview (boundary only)
- Note: arXiv perpetual non-exclusive license 1.0, not Creative Commons. Full text mirrored at the maintainer's explicit decision. ARC folded into measuring-general-intelligence; no paper-titled page and no ARC system page. TTT vs TTC left as an open interface.

## 2026-08-17 ingest | Test-Time Training with Self-Supervision for Generalization under Distribution Shifts
- Disposition: Update + New
- Topic: test-time-training
- Raw: raw/test-time-training/2019-09-29-test-time-training.md
- Created: _wiki/test-time-training/parameter-update.md
- Updated: _wiki/test-time-training/overview.md
- Updated: _wiki/test-time-training/skill-acquisition-efficiency.md
- Updated: _wiki/test-time-training/measuring-general-intelligence.md
- Cross-links: overview ↔ parameter-update ↔ skill-acquisition-efficiency ↔ measuring-general-intelligence
- Note: License CC0 1.0. Rotation-prediction recipe, CIFAR-10-C / CIFAR-10.1 numbers, and TTT-Online folded into parameter-update. No TTT vs TTA or TTT vs TTC page yet.

## 2026-08-18 reset | On-Policy Distillation frontend notes
- Disposition: No material
- Topic: on-policy-distillation
- Deleted: _wiki/on-policy-distillation/overview.md
- Deleted: _wiki/on-policy-distillation/sft-rl-opd.md
- Deleted: _wiki/on-policy-distillation/when-opd-works.md
- Deleted: _wiki/on-policy-distillation/teacher-signal-granularity.md
- Note: User requested a frontend reset before re-digest. All raw/on-policy-distillation/ sources kept. Topic remains in _data/wiki_topics.yml. No proof index existed.

## 2026-08-18 compile | OPD basic loss, cost, and estimator knobs
- Disposition: Update note
- Topic: on-policy-distillation
- Raw: raw/on-policy-distillation/2025-10-27-on-policy-distillation.md
- Raw: raw/on-policy-distillation/2026-03-26-revisiting-on-policy-distillation.md
- Raw: raw/on-policy-distillation/2026-04-14-rethinking-on-policy-distillation.md
- Raw: raw/on-policy-distillation/zhihu-opd-deep-dive-v2.md
- Note: _wiki/on-policy-distillation/overview.md#what-is-opd
- Note: _wiki/on-policy-distillation/overview.md#sampled-token-cost
- Note: _wiki/on-policy-distillation/overview.md#knobs
- Note: _wiki/on-policy-distillation/overview.md#estimator-choice
- Proof: _data/wiki_proofs/on-policy-distillation.yml
- Created: _wiki/on-policy-distillation/overview.md
- Created: _data/wiki_proofs/on-policy-distillation.yml
- Deleted: _wiki/on-policy-distillation/sft-rl-opd.md
- Deleted: _wiki/on-policy-distillation/when-opd-works.md
- Deleted: _wiki/on-policy-distillation/teacher-signal-granularity.md
- Note: Conversation-first recompile after frontend reset. No new raw. Rethinking vs Revisiting on sampled-token sufficiency kept as Disputed. V4 full-vocab engineering is Zhihu paraphrase; report itself not ingested.

## 2026-08-21 compile | Drop clip section; retitle cuts
- Disposition: Update note
- Topic: on-policy-distillation
- Note: _wiki/on-policy-distillation/overview.md#sampled-token-cost
- Note: _wiki/on-policy-distillation/overview.md#knobs
- Proof: _data/wiki_proofs/on-policy-distillation.yml
- Updated: _wiki/on-policy-distillation/overview.md
- Note: Incidental clip/clipfrac follow-up removed from frontend and proof. Headings follow user cuts (`sampled-token v.s. RL 正则项`, `top-p v.s. top-k`).

## 2026-08-21 compile | Rewrite OPD prose as rereadable notes
- Disposition: Update note
- Topic: on-policy-distillation
- Note: _wiki/on-policy-distillation/overview.md
- Updated: _wiki/on-policy-distillation/overview.md
- Note: Language pass only. Drop lecture openers and motion stand-ins (走到 / 落到 / 容易把…听成…). Nest distinctions in the argument. Same accepted model.

## 2026-08-21 compile | Drop leftover OPD definition heading
- Disposition: Update note
- Topic: on-policy-distillation
- Note: _wiki/on-policy-distillation/overview.md#what-is-opd
- Updated: _wiki/on-policy-distillation/overview.md
- Note: Removed `### On-Policy Distillation`; first question section starts in prose. `{#opd}` term heading was recreating the Overview slot.

## 2026-08-21 maintain | Drop reserved Overview chrome
- Disposition: Update note
- Topic: on-policy-distillation
- Updated: _sass/_wiki.scss
- Updated: _includes/wiki-heading-nav.html
- Note: Theme `.page__content h2` bar plus header gap left an empty Overview slot even with no Overview heading. First wiki h2 no longer draws that bar. TOC no longer skips Overview as a reserved first section.

## 2026-08-21 compile | GRPO to sampled-token OPD
- Disposition: Update note
- Topic: on-policy-distillation
- Note: _wiki/on-policy-distillation/overview.md#sampled-token
- Proof: _data/wiki_proofs/on-policy-distillation.yml#sampled-token-cost
- Updated: _wiki/on-policy-distillation/overview.md
- Note: sampled-token section now writes the GRPO loss surgery: keep ρ/clip/token-mean, replace group-norm verifier A with per-token A_{i,t}=-KL.

## 2026-08-21 compile | sampled-token v.s. RL 正则项
- Disposition: Update note
- Topic: on-policy-distillation
- Note: _wiki/on-policy-distillation/overview.md#sampled-token-cost
- Updated: _wiki/on-policy-distillation/overview.md
- Note: Kramdown/MathJax delimiter fix only. Keep GRPO → sampled-token OPD advantage swap. Split subscripted inline `$` so `_` cannot merge math spans.





