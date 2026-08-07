---
title: "Self-Harness"
topic: skill-management
summary: "Self-Harness 让固定模型依据自身失败轨迹提出 bounded harness edits，并用 held-in / held-out 非回归门控筛选可推广的 harness 改动。"
lang: zh-CN
updated: 2026-08-07
order: 9
sources:
  - title: "Self-Harness: Harnesses That Improve Themselves"
    url: "https://arxiv.org/abs/2606.09498v1"
raw:
  - raw/skill-management/2026-06-08-self-harness.md
---

## Overview

Self-Harness 研究一个比 [Meta-Harness](/wiki/skill-management/meta-harness/) 更内化的 harness 改进设定：固定模型不依赖人类工程师，也不依赖更强外部 agent，而是在当前 harness 下读取自己的执行失败证据，提出下一版 harness 的 bounded edit。模型权重、评估器与任务协议保持固定，只有围绕模型的非参数 scaffold 被修改。

这使它成为本 topic 中 CE / harness 元优化边界的一条新证据线：能力增长不只来自任务技能库或外部搜索器，也可能来自 agent 对自身执行协议的受控修订。

## 三阶段闭环

Self-Harness 的一次迭代包含三步：

1. **Weakness Mining**：在 held-in 任务上运行当前 harness，收集执行轨迹与 verifier outcome。失败样本按 verifier-grounded failure signature 聚类，signature 同时记录终端 verifier cause、agent 行为的因果状态与可复用的 agent mechanism。这样做是为了避免把 timeout、missing artifact 等表层结果误当成同一种可修复机制。
2. **Harness Proposal**：同一个固定模型在 proposer 角色中读取 failure evidence bundle、当前 editable surfaces、已通过行为与历史失败编辑，生成多个 materially distinct、minimal 的 harness 修改。每个候选都必须绑定一个具体 failure mechanism 与可编辑 surface，而不是泛泛加长 prompt。
3. **Proposal Validation**：每个候选 harness 在 held-in 与 held-out split 上重新评测。接受规则要求至少一个 split 改进、另一个 split 不退化；若多个兼容候选通过，再合并进下一版 harness。被拒候选保留在日志中，不改变 active harness。

## 与 Meta-Harness 的关系

Self-Harness 与 Meta-Harness 都把 harness 当成可优化对象，但优化者位置不同：

| 维度 | Meta-Harness | Self-Harness |
| --- | --- | --- |
| Proposer | 外部 coding agent，实验中使用 Claude Code + Opus-4.6 | 被评估的同一固定模型 |
| 反馈接口 | filesystem 中的候选源码、分数与完整 traces | 当前模型自身 held-in 失败聚类与 proposal history |
| 搜索空间 | 完整 harness 程序空间，可局部编辑也可重写 | declared editable surfaces 上的 bounded edits |
| 接受逻辑 | search-set / Pareto 前沿等外环选择 | held-in / held-out 非回归规则 |

因此，Self-Harness 不是 Meta-Harness 的替代结论，而是补上另一个问题：如果没有更强外部优化器，目标 agent 能否把自己的失败机制转成可验证的 harness 改动。

## Terminal-Bench-2.0 证据

实验使用 Terminal-Bench-2.0 的固定 64-case subset，排除不稳定外部网页资源与初始 harness 不支持的 multimodal 任务。三种模型 backend 保持模型、解码配置、预算、工具集、benchmark 环境与 evaluator 不变，仅允许修改 harness。

held-out pass rate 均提升：

| 模型 | Initial | Self-Harness |
| --- | ---: | ---: |
| MiniMax M2.5 | 40.5 | 61.9 |
| Qwen3.5-35B-A3B | 23.8 | 38.1 |
| GLM-5 | 42.9 | 57.1 |

held-in 也同步提升：MiniMax M2.5 从 43.0 到 50.0，Qwen3.5-35B-A3B 从 15.1 到 36.0，GLM-5 从 47.7 到 57.0。论文据此强调，改动不是只修 held-in failure，而是在 held-out 上也保留了正收益。

## 模型特异的 harness 改动

Self-Harness 的定性分析显示，被接受的改动不是一条统一提示词：

- MiniMax M2.5 的 retained edits 主要处理 missing required artifacts、schema-invalid tool content 与 stalled tool-use loops，推动 agent 更早创建输出文件、谨慎处理结构化工具内容，并在长工具交互后重定向。
- Qwen3.5-35B-A3B 的改动强调 dependency precheck、避免重复失败命令、打破无效探索循环，以及在工具错误后恢复 required artifacts。
- GLM-5 的改动集中在 shell 会话间持久化环境设置，以及从长时间探索转向实现与测试。

共同模式是 artifact reliability：不同模型暴露出不同执行病理，但高价值 harness edit 往往能把失败轨迹里的机制性弱点转成可执行的运行约束。

## 局限

Self-Harness 仍是 bounded harness edits under fixed benchmarks，不是开放式递归自我改进。接受门主要基于 pass-rate non-regression；在高风险环境中，仅靠 held-in / held-out pass count 不足以证明安全性或鲁棒性。实验也依赖 verifier outcome 与 trace record 的质量，若评估器看不见关键失败机制，Weakness Mining 会缺少可靠证据。

## See Also

- [技能管理概览](/wiki/skill-management/overview/)
- [Meta-Harness](/wiki/skill-management/meta-harness/)
- [Meta Context Engineering](/wiki/skill-management/meta-context-engineering/)
- [技能生命周期](/wiki/skill-management/skill-lifecycle/)
