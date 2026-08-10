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
