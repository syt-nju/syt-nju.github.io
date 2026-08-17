# Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes

> Source: https://arxiv.org/abs/2603.25562
> Author: Yuqian Fu, Haohuan Huang, Kaiwen Jiang, Yuanheng Zhu, Dongbin Zhao
> Collected: 2026-08-17
> Published: 2026-03-26
> License: arXiv.org perpetual non-exclusive license
> Completeness: Complete
> Retrieval note: Full paper markdown via deepxiv paper 2603.25562 --raw (arXiv 2603.25562v1). Site chrome removed; figure images remain as arXiv asset references.

arXiv:2603.25562v1 [cs.LG] 26 Mar 2026

Revisiting On-Policy Distillation:  Empirical Failure Modes and Simple Fixes
=============================================================================

Yuqian Fu Haohuan Huang11footnotemark: 1Kaiwen Jiang Yuanheng Zhu22footnotemark: 2Dongbin Zhao  
State Key Laboratory of Multimodal Artificial Intelligence Systems, CASIA  
School of Artificial Intelligence, UCAS  
{fuyuqian2022, yuanheng.zhu}@ia.ac.cnEqual contribution. †Corresponding authors. ‡Work in progress.

###### Abstract

On-policy distillation (OPD) is appealing for large language model (LLM) post-training because it evaluates teacher feedback on student-generated rollouts rather than fixed teacher traces. In long-horizon settings, however, the common sampled-token variant is fragile: it reduces distribution matching to a one-token signal and becomes increasingly unreliable as rollouts drift away from prefixes the teacher commonly visits. We revisit OPD from the estimator and implementation sides. Theoretically, token-level OPD is biased relative to sequence-level reverse-KL, but it has a much tighter worst-case variance bound; our toy study shows the same tradeoff empirically, with stronger future-reward coupling producing higher gradient variance and less stable learning. Empirically, we identify three failure modes of sampled-token OPD: an imbalanced one-token signal, unreliable teacher guidance on student-generated prefixes, and distortions caused by tokenizer or special-token mismatch. We address these issues with teacher top-$K$ local support matching, implemented as truncated reverse-KL with top-$p$ rollout sampling and special-token masking. Across single-task math reasoning and multi-task agentic-plus-math training, this objective yields more stable optimization and better downstream performance than sampled-token OPD.

| [Uncaptioned image] | [Code](https://github.com/hhh675597/revisiting_opd "") | [Uncaptioned image] | [Blog](https://www.notion.so/yuqianfu/Revisiting-On-Policy-Distillation-Empirical-Failure-Modes-and-Simple-Fixes-31dd5cc40dd181f89eead3de7181df1d "") |
| --- | --- | --- | --- |

1 Introduction
--------------

On-policy distillation (OPD) trains a student on its own rollouts while evaluating local feedback with a stronger teacher. This makes OPD attractive for long-horizon reasoning and agentic post-training, where the student quickly reaches prefixes that are rare or absent in fixed teacher traces*(Agarwal et al., [2024](#bib.bib1 "On-policy distillation of language models: learning from self-generated mistakes"); Gu et al., [2024](#bib.bib2 "MiniLLM: knowledge distillation of large language models"))*. The practical question is therefore not whether on-policy teacher supervision is useful in principle, but which objective remains reliable once training is driven by student-generated trajectories.

In current language-model pipelines, OPD is usually implemented as a sampled-token comparison: at each decoding step, the student is updated only through the log-ratio on its sampled token. This approximation is cheap, but brittle for at least three reasons. It turns a distribution-level discrepancy into a highly imbalanced one-token signal; it can over-trust the teacher on prefixes that are common for the student but atypical for the teacher; and it is easily distorted by tokenizer or special-token mismatch. There is a corresponding estimator tradeoff. A more sequence-coupled objective can recover information that token-level OPD discards, but stronger reward coupling can also make optimization much noisier.

We study this tradeoff first at the estimator level. Sequence-level reverse-KL couples each token update to future rewards, whereas token-level OPD drops those terms. Token-level OPD is therefore biased relative to the sequence-level objective, but it has a much tighter worst-case variance bound. Our toy experiment shows the same pattern empirically: as future-reward coupling increases, gradient variance rises and optimization becomes less stable. This suggests a simple design target for long-horizon post-training: keep supervision local enough to control variance, while making the local comparison less brittle than a one-token point estimate.

Motivated by this view, we replace sampled-token supervision with teacher top-$K$ local support matching. At each prefix, we compare teacher and student distributions on the teacher’s locally plausible support instead of rewarding only the sampled token. We implement this objective as truncated reverse-KL with top-$p$ rollout sampling and special-token masking. The resulting update is still local and inexpensive, but less sensitive to idiosyncratic sampled continuations and tokenization artifacts than sampled-token OPD.

#### Contributions.

Our main contributions are threefold.

* •

    We analyze the estimator tradeoff in OPD: token-level OPD is biased relative to sequence-level OPD, but its worst-case variance grows much more slowly with sequence length, which matters in long-horizon LLM post-training.

* •

    We identify three practical failure modes of sampled-token OPD: an imbalanced one-token signal, unreliable teacher guidance on student-generated prefixes, and distortions caused by tokenizer or special-token mismatch.

* •

    We propose teacher top-$K$ local support matching, implemented as truncated reverse-KL with top-$p$ rollouts and special-token masking, and show stronger optimization behavior and downstream performance than sampled-token OPD in both single-task math reasoning and multi-task agentic-plus-math training.

2 Related Work
--------------

Our work is most closely related to on-policy distillation for language models. Offline distillation matches teacher outputs or logits on fixed traces, whereas OPD-style methods evaluate teacher signals on student-generated prefixes*(Agarwal et al., [2024](#bib.bib1 "On-policy distillation of language models: learning from self-generated mistakes"); Gu et al., [2024](#bib.bib2 "MiniLLM: knowledge distillation of large language models"))*. We focus on a narrower question within this family: once supervision is computed on the student’s own rollouts, what local comparison rule remains stable in long-horizon training? Recent model reports from Qwen3*(Yang et al., [2025](#bib.bib5 "Qwen3 technical report"))*, MiMo-V2-Flash*(Xiao et al., [2026](#bib.bib6 "Mimo-v2-flash technical report"))*, GLM-5*(Zeng et al., [2026](#bib.bib7 "GLM-5: from vibe coding to agentic engineering"))*, and Thinking Machines Lab*(Lu and Lab, [2025](#bib.bib4 "On-policy distillation"))* suggest that this regime is becoming relevant in practice.

Another relevant line of work studies how to preserve useful supervision under rollout drift. Representative directions include EMA-anchor stabilization with top-$K$ KL*(Zhang and Ba, [2026](#bib.bib8 "EMA policy gradient: taming reinforcement learning for LLMs with EMA anchor and Top-k KL"))*, off-policy correction*(Liu et al., [2025](#bib.bib15 "When speed kills stability: demystifying RL collapse from the training-inference mismatch"))*, perturbation-based stabilization*(Ye et al., [2026](#bib.bib9 "Adaptive layerwise perturbation: unifying off-policy corrections for LLM RL"))*, and hybrid rollout mixing between teacher and student policies*(Zhang et al., [2026](#bib.bib10 "Learning from mixed rollouts: logit fusion as a bridge between imitation and exploration"))*. These methods stabilize training by changing the broader optimization procedure or rollout source. Our method is more local: we revisit the per-prefix OPD comparison itself and ask how to preserve informative teacher guidance once teacher and student begin to diverge on student-generated trajectories.

3 Understanding Sampled-token OPD: Tradeoffs and Failure Modes
---------------------------------------------------------------

### 3.1 From reverse-KL to token-level OPD

We begin with the sequence-level objective behind OPD. For a prompt $x$, the reverse-KL objective is

|  | $J_{\mathrm{OPD}}(\theta)\=\mathbb{E}_{x\sim D}\left[D_{\mathrm{KL}}\left(\pi_{\theta}(\cdot\mid x)\,\|\,q(\cdot\mid x)\right)\right].$ |  |
| --- | --- | --- |

where $\pi_{\theta}$ and $q$ are the student and teacher models, respectively. Using the score-function identity, its gradient can be written as

|  | $\nabla_{\theta}J_{\mathrm{OPD}}(\theta)\=\mathbb{E}_{x,\,y\sim\pi_{\theta}(\cdot\mid x)}\left[\big(\log\pi_{\theta}(y\mid x)-\log q(y\mid x)\big)\,\nabla_{\theta}\log\pi_{\theta}(y\mid x)\right].$ |  |
| --- | --- | --- |

For each decoding step $t$, let $c_{t}\=(x,y_{<t})$ denote the prefix context, and let

|  | $g_{t}\=\nabla_{\theta}\log\pi_{\theta}(y_{t}\mid c_{t}),\qquad r_{t}\=\log\frac{\pi_{\theta}(y_{t}\mid c_{t})}{q(y_{t}\mid c_{t})}.$ |  |
| --- | --- | --- |

Using the autoregressive factorization

|  | $\log\pi_{\theta}(y\mid x)-\log q(y\mid x)\=\sum_{t^{\prime}\=1}^{T}r_{t^{\prime}},\qquad\nabla_{\theta}\log\pi_{\theta}(y\mid x)\=\sum_{t\=1}^{T}g_{t},$ |  |
| --- | --- | --- |

we obtain the sequence-level estimator

|  | $\hat{g}_{\mathrm{seq}}\=\sum_{t\=1}^{T}\left(\sum_{t^{\prime}\=1}^{T}r_{t^{\prime}}\right)g_{t}.$ |  | (1) |
| --- | --- | --- | --- |

For $t^{\prime}<t$, we have $\mathbb{E}[r_{t^{\prime}}g_{t}]\=0$ because $r_{t^{\prime}}$ depends only on the prefix before step $t$, while

|  | $\mathbb{E}[g_{t}\mid x,y_{<t}]\=\sum_{y_{t}}\pi_{\theta}(y_{t}\mid c_{t})\,\nabla_{\theta}\log\pi_{\theta}(y_{t}\mid c_{t})\=0.$ |  |
| --- | --- | --- |

The same gradient can therefore be written in causal return-to-go form:

|  | $\mathbb{E}[\hat{g}_{\mathrm{seq}}]\=\mathbb{E}\left[\sum_{t\=1}^{T}\left(\sum_{t^{\prime}\=t}^{T}r_{t^{\prime}}\right)g_{t}\right].$ |  |
| --- | --- | --- |

A common approximation in LLM training keeps only the immediate term at each position:

|  | $\hat{g}_{\mathrm{tok}}\=\sum_{t\=1}^{T}r_{t}g_{t}.$ |  | (2) |
| --- | --- | --- | --- |

We refer to ([2](#S3.E2 "In 3.1 From reverse-KL to token-level OPD ‣ 3 Understanding Sampled-token OPD: Tradeoffs and Failure Modes ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes")) as token-level OPD.
This approximation removes future-reward coupling, so the update for token $y_{t}$ depends only on its immediate reward. Consequently, it is biased relative to the sequence-level reverse-KL estimator, but exhibits lower variance in long-horizon settings.
This difference is reflected in their variance scaling.
Under bounded rewards and bounded score-function gradients, the worst-case variance upper bound of token-level OPD scales as $O(T^{2})$, whereas the sequence-level estimator scales as $O(T^{4})$. We provide a detailed derivation in Appendix[B](#A2 "Appendix B Bias and variance analysis of token-level versus sequence-level OPD ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes").

To interpolate between these extremes, we consider the discounted return-to-go estimator

|  | $\hat{g}_{\gamma}\=\sum_{t\=1}^{T}\left(\sum_{t^{\prime}\=t}^{T}\gamma^{t^{\prime}-t}r_{t^{\prime}}\right)g_{t},\qquad\gamma\in[0,1].$ |  | (3) |
| --- | --- | --- | --- |

The case $\gamma\=0$ recovers token-level OPD, while $\gamma\=1$ recovers the causal sequence-level estimator.
We conduct a two-task toy experiment, where increasing $\gamma$ is observed to induce substantially higher gradient variance and less stable optimization; see Figure[1](#S3.F1 "Figure 1 ‣ 3.1 From reverse-KL to token-level OPD ‣ 3 Understanding Sampled-token OPD: Tradeoffs and Failure Modes ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes") for an illustration and Appendix[C](#A3 "Appendix C Toy experiment details ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes") for additional experimental details.

<img src='2603.25562v1/x2.png' alt='Refer to caption' title='' width='664' height='237' />

*(a) Gradient variance in the toy experiment. Larger $\gamma$ generally yields higher variance in both tasks.*

<img src='2603.25562v1/x3.png' alt='Refer to caption' title='' width='664' height='295' />

*(b) State visitation under $\gamma\in{0.0,0.5,1.0}$ in the toy environment. For $\gamma\=1.0$, the policy model fails to consistently move toward the target, and instead exhibits drifting behavior.*

*Figure 1: Effect of increasing $\gamma$ in the toy experiment. Larger $\gamma$ yields a higher and more persistent variance regime and, in the sequence-level limit, drifting policies in state space.*

### 3.2 Why sampled-token OPD is brittle in practice

Although token-level OPD is attractive from a bias–variance perspective, the sampled-token comparison can be brittle in practice. We isolate three distinct issues: (1) the distillation signal is highly imbalanced, (2) the teacher signal becomes less reliable on student-generated prefixes, and (3) tokenizer and special-token mismatch can further distort a one-token comparison.

#### A highly imbalanced sampled-token signal.

In sampled-token OPD, the update at step $t$ is driven by the log-ratio on a single sampled token:

|  | $\log q(y_{t}\mid c_{t})-\log\pi_{\theta}(y_{t}\mid c_{t}).$ |  |
| --- | --- | --- |

Negative rewards arise whenever the student assigns higher probability to a sampled token than the teacher.
As shown in Figure[2](#S3.F2 "Figure 2 ‣ A highly imbalanced sampled-token signal. ‣ 3.2 Why sampled-token OPD is brittle in practice ‣ 3 Understanding Sampled-token OPD: Tradeoffs and Failure Modes ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes"), most sampled tokens receive negative rewards, and the positive learning signal is concentrated on a relatively small subset of tokens with positive advantage.
The result is an imbalanced training signal in which optimization is disproportionately driven by a few locally favorable tokens. Training can then become sensitive to short continuations that the teacher locally prefers, such as fillers or hesitation markers, even when those tokens contribute little to overall trajectory quality.

<img src='2603.25562v1/x4.png' alt='Refer to caption' title='' width='290' height='282' />

*Figure 2: Scatter of token probabilities (student vs teacher). Sampled-token OPD at the first training iteration on Qwen2.5-7B-It*(Qwen et al., [2025](#bib.bib11 "Qwen2.5 technical report"))*, using OpenThinker3-7B*(Guha et al., [2025](#bib.bib12 "Openthoughts: data recipes for reasoning models"))* as the teacher model. The sampled-token signal is heavily skewed toward penalizing the current student token rather than providing a balanced reward.*

#### The teacher signal can become unreliable on student-generated prefixes.

Sampled-token OPD implicitly assumes that the probability the teacher assigns to a student-generated token is a useful proxy for trajectory quality. This assumption weakens when rollouts enter prefixes that are common under the student but uncommon for the teacher.
On such prefixes, the teacher may assign high probability to tokens that appear plausible, while the trajectory has already deviated from a desirable direction. In our logs, this behavior is associated with patterns such as repetition loops, self-resetting reasoning, and malformed continuations (Figure[3](#S3.F3 "Figure 3 ‣ The teacher signal can become unreliable on student-generated prefixes. ‣ 3.2 Why sampled-token OPD is brittle in practice ‣ 3 Understanding Sampled-token OPD: Tradeoffs and Failure Modes ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes"); Appendix[D](#A4 "Appendix D Qualitative OPD reward-hacking case study ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes")).
These observations suggest an objective-level mismatch: OPD encourages token-level agreement with the teacher, but such proxy does not necessarily correspond to trajectory-level quality, especially on prefixes that are out-of-distribution for the teacher.

<img src='2603.25562v1/x5.png' alt='Refer to caption' title='' width='498' height='300' />

*Figure 3: The student falls into a repetition loop, but the teacher model maintains highly aligned with the student model on the repeating tokens, indicating a lack of proper penalty for such behavior.*

We hypothesize that two factors amplify this issue. First, teacher distributions are often sharp, so even modest student-teacher disagreement can produce large log-ratio values. Second, differences between the teacher’s generation pattern and the student’s make student prefixes more likely to fall outside the teacher’s typical context.
The same failure also appears in how the teacher signal changes with position. Figure[4](#S3.F4 "Figure 4 ‣ The teacher signal can become unreliable on student-generated prefixes. ‣ 3.2 Why sampled-token OPD is brittle in practice ‣ 3 Understanding Sampled-token OPD: Tradeoffs and Failure Modes ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes") shows the distribution of teacher-student log-probability gaps across token positions; it is relatively concentrated at early positions and becomes progressively wider later in the sequence, with more extreme values on long rollouts.

<img src='2603.25562v1/x6.png' alt='Refer to caption' title='' width='664' height='291' />

*Figure 4: Distribution of teacher-student log-probability gaps across token positions. Later positions show wider distributions and more extreme values, indicating a noisier teacher signal on long student-generated rollouts.*

#### Tokenizer and special-token mismatch.

Sampled-token OPD compares the exact token generated by the student using the teacher distribution.
When the two models use different tokenizations, the same raw text can be segmented differently, so a student generated token may not correspond to a natural token under the teacher.
For example, the student may generate <think> as <, think, >, while the teacher expects <th, ink, >.
Then token < receives low probability from the teacher, even though both models produce the same semantic content. Similar mismatches arise for special tokens such as end-of-sequence markers.
In this setting, a one-token comparison confuses semantic disagreement with tokenizer mismatch.
Since supervision is applied on a single token, such artifacts can distort the reward signal.

<img src='2603.25562v1/x7.png' alt='Refer to caption' title='' width='498' height='147' />

*Figure 5: Token-level comparison can penalize semantically correct outputs due to tokenizer mismatch.*

These observations motivate moving beyond one-token supervision: instead of comparing only the sampled token, we compare teacher and student over a set of plausible next-token continuations at each prefix, while retaining token-level updates for stability.

4 Method
--------

Our method retains token-level OPD but replaces one-token supervision with a distribution-level comparison over a teacher-selected support set at each prefix.
This yields a truncated reverse-KL objective that maintains computing efficiency while improving the balance of the training signal.
Section[4.1](#S4.SS1 "4.1 Teacher top-K local support matching ‣ 4 Method ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes") introduces the objective, and Section[4.2](#S4.SS2 "4.2 Practical stabilization choices ‣ 4 Method ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes") describes the practical choices that ensure stable training.

### 4.1 Teacher top-K local support matching

Instead of comparing teacher and student on a single sampled token, we compare them over a teacher-defined local support. A natural starting point is the full-vocabulary reverse-KL at a prefix $c_{t}$:

|  | $\mathcal{L}_{\mathrm{full}}(c_{t})\=\sum_{v\in\mathcal{V}}\pi_{\theta}(v\mid c_{t})\log\frac{\pi_{\theta}(v\mid c_{t})}{q(v\mid c_{t})}.$ |  | (4) |
| --- | --- | --- | --- |

Sampled-token OPD can be viewed as a one-sample Monte Carlo approximation to this quantity:

|  | $\mathcal{L}_{\mathrm{sample}}(c_{t},y_{t})\=\log\frac{\pi_{\theta}(y_{t}\mid c_{t})}{q(y_{t}\mid c_{t})},\qquad y_{t}\sim\pi_{\theta}(\cdot\mid c_{t}).$ |  | (5) |
| --- | --- | --- | --- |

This approximation is computationally attractive, while concentrating entire update on a sampled-token.
We instead compare teacher and student over a teacher-supported token set at each prefix.

For each prompt $x$, we sample a group of outputs ${o_{i}}_{i\=1}^{G}$ using the student inference policy. Let $c_{i,t}\=(x,y_{i,<t})$ be the prefix at position $t$ of output $o_{i}$, and define the teacher support set

|  | $S(c_{i,t})\=\mathrm{TopK}_{q}(c_{i,t}),$ |  | (6) |
| --- | --- | --- | --- |

which contains the $K$ highest-probability tokens under the teacher at that prefix.

We then renormalize both teacher and student distributions inside this local support:

|  | $\hat{\pi}_{\theta}(v\mid c_{i,t})\=\frac{\pi_{\theta}(v\mid c_{i,t})}{\sum_{u\in S(c_{i,t})}\pi_{\theta}(u\mid c_{i,t})},\qquad\hat{q}(v\mid c_{i,t})\=\frac{q(v\mid c_{i,t})}{\sum_{u\in S(c_{i,t})}q(u\mid c_{i,t})}.$ |  | (7) |
| --- | --- | --- | --- |

Our training objective averages the truncated reverse-KL over all rollout positions:

|  | $\mathcal{L}_{\mathrm{LSM}}\=\mathbb{E}_{x,\,{o_{i}}\sim\pi_{\theta,\mathrm{infer}}}\left[\frac{1}{\sum_{i\=1}^{G}|o_{i}|}\sum_{i\=1}^{G}\sum_{t\=1}^{|o_{i}|}\sum_{v\in S(c_{i,t})}\hat{\pi}_{\theta}(v\mid c_{i,t})\log\frac{\hat{\pi}_{\theta}(v\mid c_{i,t})}{\hat{q}(v\mid c_{i,t})}\right].$ |  | (8) |
| --- | --- | --- | --- |

Relative to sampled-token OPD, this objective performs a distribution-level comparison inside the teacher-supported local region rather than rewarding or penalizing only one sampled token. The resulting update redistributes positive and negative adjustments across all teacher-supported candidates at a prefix, yielding a more balanced training signal while remaining far cheaper than full-vocabulary KL.

### 4.2 Practical stabilization choices

#### Support-set renormalization.

Renormalization is necessary because the objective is evaluated on a truncated support rather than the full vocabulary. Without it, optimization can become unstable because the teacher and student mass inside the support is not directly comparable.

#### Top-$p$ rollout sampling.

We generate rollouts with top-$p$ sampling. Unconstrained sampling occasionally produces extremely low-probability tokens, which in turn creates prefixes where the teacher distribution is less informative and the student distribution is already deteriorating. Top-$p$ sampling keeps trajectories closer to typical continuations and makes the teacher signal more reliable.

#### Special-token masking.

We mask problematic special tokens to reduce false negatives caused by incompatible tokenization conventions. This is an orthogonal engineering fix: in our experiments it materially helps the sampled-token OPD baseline, while our local support objective is much less sensitive to it. In principle, one could also merge multi-token marker variants or average over equivalent tokenizations, but we do not pursue those tokenizer-specific remedies here because masking is the simplest model-agnostic correction.

5 Experiments
-------------

### 5.1 Setup

We implement local support matching on top of an existing OPD training pipeline, using Qwen2.5-7B-Instruct*(Qwen et al., [2025](#bib.bib11 "Qwen2.5 technical report"))* as the student. We consider two settings: (1)a single-task math reasoning setting, where OpenThinker3-7B*(Guha et al., [2025](#bib.bib12 "Openthoughts: data recipes for reasoning models"))* serves as the teacher and training uses the English portion of DAPO-Math-17K*(Yu et al., [2025](#bib.bib13 "DAPO: an open-source LLM reinforcement learning system at scale"))* with a maximum context length of 16K; and (2)a multi-task setting that alternates between math reasoning and a multi-turn agentic task based on ALFWorld*(Shridhar et al., [2021](#bib.bib14 "ALFWorld: aligning text and embodied environments for interactive learning"))*, where math uses OpenThinker3-7B*(Guha et al., [2025](#bib.bib12 "Openthoughts: data recipes for reasoning models"))* and the agentic task uses the released GiGPO-Qwen2.5-7B-Instruct-ALFWorld checkpoint*(Feng et al., [2025](#bib.bib3 "Group-in-group policy optimization for LLM agent training"))* as the teacher.

All runs use batch size 128, mini-batch size 64, learning rate $2\times 10^{-6}$, and temperature 1 by default. Rollouts are sampled with top-$p\=0.9$.

We report pass@1 on the math benchmarks and success rate on ALFWorld, unless otherwise specified. In a small number of cases, we additionally report average@32 for math evaluation.

### 5.2 Single-task math reasoning

*Table 1: Single-task math reasoning results. Local support matching improves over sampled-token OPD, and the gain remains after adding special-token masking to the baseline.*

| Method | Math500 | AIME24 | AIME25 | Minerva | OlympiadBench | Avg. |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B-It | 68.2 | 13.3 | 0.0 | 26.5 | 32.9 | 28.2 |
| OpenThinker3-7B | 92.2 | 53.3 | 40.0 | 39.0 | 55.6 | 56.0 |
| Sampled-token OPD | 80.0 | 10.0 | 16.7 | 32.4 | 43.1 | 36.4 |
| Sampled-token OPD w/ mask | 81.4 | 26.7 | 16.7 | 34.2 | 44.7 | 40.7 |
| Ours w/o mask | 80.4 | 23.3 | 26.7 | 34.2 | 43.9 | 41.0 |
| Ours w/ mask | 82.0 | 23.3 | 23.3 | 34.9 | 43.9 | 41.5 |

Table[1](#S5.T1 "Table 1 ‣ 5.2 Single-task math reasoning ‣ 5 Experiments ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes") shows that local support matching improves over sampled-token OPD in single-task math reasoning. Sampled-token OPD already raises the average score from 28.2 to 36.4, but still trails the teacher by a large margin. Special-token masking alone further improves the sampled-token baseline to 40.7, which indicates that tokenization artifacts are a material part of the problem.

Our full method achieves an average of 41.5.
The improvement persists after applying the same masking fix to the baseline, indicating that it is not solely due to mismatch handling but also reflects a stronger local distillation signal.
By contrast, masking has only a modest effect on our method (41.0 vs. 41.5), consistent with distribution-level support matching being less sensitive to tokenizer mismatch than one-token supervision.

### 5.3 Multi-task agentic-plus-math training

*Table 2: Results for multi-task training that alternates between ALFWorld and math reasoning. Local support matching preserves strong ALFWorld performance while improving the math side of the mixture.*

| Method | ALFWorld | Math500 | AIME24 | AIME25 | Minerva | OlympiadBench | Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B-It | 21.9 | 68.2 | 13.3 | 0.0 | 26.5 | 32.9 | 28.2 |
| GiGPO-Qwen2.5-7B-It-Alfworld | 95.3 | – | – | – | – | – | – |
| OpenThinker3-7B | – | 92.2 | 53.3 | 40.0 | 39.0 | 55.6 | 56.0 |
| Sampled-token OPD | 90.6 | 74.8 | 13.3 | 13.3 | 32.1 | 40.5 | 34.8 |
| Sampled-token OPD w/ mask | 93.8 | 76.0 | 20.0 | 13.3 | 33.5 | 40.4 | 36.6 |
| Ours w/o mask | 95.3 | 82.0 | 33.3 | 16.7 | 32.7 | 44.0 | 41.7 |
| Ours w/ mask | 97.7 | 79.0 | 20.0 | 16.7 | 34.6 | 42.5 | 38.6 |

Table[2](#S5.T2 "Table 2 ‣ 5.3 Multi-task agentic-plus-math training ‣ 5 Experiments ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes") shows a more asymmetric pattern in alternating multi-task training. The sampled-token OPD baseline is already strong on ALFWorld, so the main room for improvement lies on the math side.

The unmasked version of our method improves Math500 from 76.0 to 82.0 and raises the average math score from 36.6 to 41.7 while remaining competitive on ALFWorld. The masked version achieves the best ALFWorld result at 97.7 but gives up some of the math gains. Taken together, these results suggest that local support matching helps most where long-horizon token-level supervision is most brittle, while preserving strong agentic performance.

### 5.4 Training dynamics and alignment

<img src='2603.25562v1/x8.png' alt='Refer to caption' title='' width='498' height='233' />

*Figure 6: Single-task training curves for math reasoning. Local support matching improves training reward and final evaluation over the course of training.*

<img src='2603.25562v1/x9.png' alt='Refer to caption' title='' width='830' height='236' />

*Figure 7: Multi-task learning curves for ALFWorld and math reasoning. The main gains appear on the math side while agentic performance remains strong.*

<img src='2603.25562v1/x10.png' alt='Refer to caption' title='' width='830' height='491' />

*(a) Single-task optimization statistics.*

<img src='2603.25562v1/x11.png' alt='Refer to caption' title='' width='830' height='430' />

*(b) Multi-task gradient norms.*

<img src='2603.25562v1/x12.png' alt='Refer to caption' title='' width='830' height='491' />

*(c) Response length statistics.*

<img src='2603.25562v1/x13.png' alt='Refer to caption' title='' width='830' height='430' />

*(d) Teacher-student log-probability gaps.*

*Figure 8: Optimization and alignment diagnostics. Relative to sampled-token OPD, local support matching yields smaller gradient norms, fewer clipping-boundary hits, shorter responses, and smaller teacher-student log-probability gaps.*

Figures[6](#S5.F6 "Figure 6 ‣ 5.4 Training dynamics and alignment ‣ 5 Experiments ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes"), [7](#S5.F7 "Figure 7 ‣ 5.4 Training dynamics and alignment ‣ 5 Experiments ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes"), and [8](#S5.F8 "Figure 8 ‣ 5.4 Training dynamics and alignment ‣ 5 Experiments ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes") provide a more detailed view of the optimization dynamics.

#### Better learning curves.

On math reasoning, our method improves both training reward and evaluation performance throughout learning rather than only at the final checkpoint. This pattern holds in both the single-task setting and the alternating multi-task setting.

#### More stable optimization.

Our method yields smaller gradient norms and lower clipping-boundary fractions while maintaining sufficient policy entropy, and this indicates more stable optimization. We also observe that special-token masking substantially reduces the clipping-boundary fraction of sampled-token OPD during early and middle training, while having only minor effects on our method.

#### Improved teacher-student alignment.

The teacher-student log-probability gap on sampled tokens also becomes smaller, suggesting that the truncated local support objective improves alignment even under the sampled-token diagnostic used by the baseline.

### 5.5 Ablations

*Table 3: Ablation on single-task math training using AIME24 avg@32. Restricting the loss to teacher top-$K$ support is not sufficient by itself; top-$p$ rollout sampling is also needed.*

| Method | AIME24 avg@32 |
| --- | --- |
| Qwen2.5-7B-Instruct | 10.0 |
| OpenThinker3-7B | 63.3 |
| Sampled-token OPD (point estimate) | 20.4 |
| + teacher top-$K$ (truncated reverse-KL) | 17.7 |
| + teacher top-$K$ + top-$p$ | 23.6 |

<img src='2603.25562v1/x14.png' alt='Refer to caption' title='' width='830' height='430' />

*(a) Support renormalization.*

<img src='2603.25562v1/x15.png' alt='Refer to caption' title='' width='830' height='430' />

*(b) Support size $K$.*

<img src='2603.25562v1/x16.png' alt='Refer to caption' title='' width='830' height='430' />

*(c) Rollout top-$p$.*

*Figure 9: Ablations of the main design choices. Renormalization is required for stability, very small support sets hurt learning, and unconstrained rollout sampling degrades optimization.*

Table[3](#S5.T3 "Table 3 ‣ 5.5 Ablations ‣ 5 Experiments ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes") and Figure[9](#S5.F9 "Figure 9 ‣ 5.5 Ablations ‣ 5 Experiments ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes") suggest that the gains arise from several design choices rather than any single modification. Teacher top-$K$ comparison alone is not sufficient: the rollout policy must also remain in a stable region, and adding top-$p$ sampling turns an initially weaker top-$K$ variant into a stronger configuration. Renormalization inside the truncated support is essential, as removing it leads to rapid collapse. Performance is not especially sensitive to the exact support size once $K$ is large enough, but training becomes unstable when the support is too small or rollouts are fully unconstrained.

#### Top-$K$ support variants.

Our main experiments define the truncated expectation on the teacher’s top-$K$ support. A natural question is whether this choice itself is critical, or whether nearby support definitions perform similarly. We therefore compare three variants: teacher top-$K$ (used in the main results), student top-$K$, and teacher top-$K$ augmented with the student sampled token.

*Table 4: Ablation on alternative support-set definitions. We report AIME24 avg@32 for the early-ablation metric and pass@1 for the remaining benchmarks; the final column averages only the pass@1 metrics.*

| Method | AIME24 avg@32 | Math500 | AIME24 | AIME25 | Minerva | OlympiadBench | Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Teacher top-$K$ | 23.6 | 80.4 | 23.3 | 26.7 | 34.2 | 43.9 | 41.0 |
| Student top-$K$ | 22.3 | 82.4 | 30.0 | 16.7 | 35.7 | 44.9 | 41.9 |
| Teacher top-$K$ + sampled token | 22.4 | 81.6 | 26.7 | 23.3 | 36.4 | 46.7 | 42.9 |

Table[4](#S5.T4 "Table 4 ‣ Top-𝐾 support variants. ‣ 5.5 Ablations ‣ 5 Experiments ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes") suggests that the benefit is fairly robust across nearby support definitions. No single choice dominates across all benchmarks: teacher top-$K$ remains competitive, student top-$K$ is strong on several individual datasets, and teacher top-$K$ augmented with the sampled token achieves the best average score in this preliminary comparison. This points to the main benefit coming from replacing single-token comparison with local distribution-level matching rather than from one uniquely optimal support-set choice. At the same time, the comparison is still preliminary, so a more systematic end-to-end study of support-set design remains important future work.

6 Discussion and Limitations
----------------------------

#### The current objective is still a truncated surrogate.

Our local-support loss is evaluated on a restricted token subset and on prefixes generated by a rollout policy such as top-$p$ sampling. It is therefore not equivalent to full-vocabulary reverse-KL, nor does it explicitly correct for the sampling process that produced the training prefixes. This limitation matters most in two places that remain underexplored in our study: how to incorporate the sampled token when augmenting teacher top-$K$ support, and whether importance-weighting-style corrections are needed when rollout and training policies differ. We therefore view the current formulation as a practical design point rather than a final answer to support-set construction.

#### The reward-hacking explanation is still a mechanism hypothesis.

Our qualitative cases make the failure mode concrete, but they do not isolate a complete causal mechanism. In particular, the hypothesis that sharp teacher distributions and off-distribution prefixes jointly create misleading local rewards should be treated as a plausible explanation supported by evidence rather than as a fully identified causal account.

#### Teacher matching remains an imperfect proxy for task success.

Even when OPD is well defined as a teacher-matching objective, the resulting reward can still diverge from the underlying notion of successful behavior. Our reward-hacking cases make this gap concrete: locally teacher-preferred continuations can remain rewardable even when the overall trajectory is already unhelpful or harmful. A noticeable gap to the teacher also remains in our experiments, which suggests that better local supervision is only one part of the distillation problem, especially when teacher and student differ substantially. Closing that gap may require stronger rollout control, better handling of distribution shift, better use of teacher uncertainty, and combinations with outcome-verifiable rewards.

7 Conclusion
------------

This paper revisits OPD in long-horizon post-training. The central tradeoff is straightforward: sequence-level coupling is closer to the underlying objective but can be much higher-variance, whereas sampled-token OPD is easy to optimize but often too brittle to provide reliable supervision. Teacher top-$K$ local support matching occupies the middle ground by keeping the objective local while replacing one-token supervision with a truncated distribution-level comparison. Across single-task math reasoning and alternating agentic-plus-math training, it improves optimization behavior and downstream performance over sampled-token OPD. The remaining gap between teacher matching and task success suggests that better local objectives should be paired with stronger control of rollout drift and teacher uncertainty.

References
----------

* R. Agarwal, N. Vieillard, Y. Zhou, P. Stanczyk, S. R. Garea, M. Geist, and O. Bachem (2024)On-policy distillation of language models: learning from self-generated mistakes.In The Twelfth International Conference on Learning Representations,External Links: [Link](https://openreview.net/forum?id=3zKtaqxLhW "")Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes"),[§2](#S2.p1.1 "2 Related Work ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes").
* L. Feng, Z. Xue, T. Liu, and B. An (2025)Group-in-group policy optimization for LLM agent training.In The Thirty-ninth Annual Conference on Neural Information Processing Systems,External Links: [Link](https://openreview.net/forum?id=QXEhBMNrCW "")Cited by: [§5.1](#S5.SS1.p1.1 "5.1 Setup ‣ 5 Experiments ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes").
* Y. Gu, L. Dong, F. Wei, and M. Huang (2024)MiniLLM: knowledge distillation of large language models.In The Twelfth International Conference on Learning Representations,External Links: [Link](https://openreview.net/forum?id=5h0qf7IBZZ "")Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes"),[§2](#S2.p1.1 "2 Related Work ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes").
* E. Guha, R. Marten, S. Keh, N. Raoof, G. Smyrnis, H. Bansal, M. Nezhurina, J. Mercat, T. Vu, Z. Sprague, et al. (2025)Openthoughts: data recipes for reasoning models.arXiv preprint arXiv:2506.04178.Cited by: [Figure 2](#S3.F2 "In A highly imbalanced sampled-token signal. ‣ 3.2 Why sampled-token OPD is brittle in practice ‣ 3 Understanding Sampled-token OPD: Tradeoffs and Failure Modes ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes"),[§5.1](#S5.SS1.p1.1 "5.1 Setup ‣ 5 Experiments ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes").
* J. Liu, Y. Li, Y. Fu, J. Wang, Q. Liu, and Z. Jiang (2025)External Links: [Link](https://richardli.xyz/rl-collapse "")Cited by: [§2](#S2.p2.1 "2 Related Work ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes").
* K. Lu and T. M. Lab (2025)On-policy distillation.Thinking Machines Lab: Connectionism.Note: https://thinkingmachines.ai/blog/on-policy-distillationExternal Links: [Document](https://dx.doi.org/10.64434/tml.20251026 "")Cited by: [§2](#S2.p1.1 "2 Related Work ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes").
* Qwen, A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Li, D. Liu, F. Huang, H. Wei, H. Lin, J. Yang, J. Tu, J. Zhang, J. Yang, J. Yang, J. Zhou, J. Lin, K. Dang, K. Lu, K. Bao, K. Yang, L. Yu, M. Li, M. Xue, P. Zhang, Q. Zhu, R. Men, R. Lin, T. Li, T. Tang, T. Xia, X. Ren, X. Ren, Y. Fan, Y. Su, Y. Zhang, Y. Wan, Y. Liu, Z. Cui, Z. Zhang, and Z. Qiu (2025)Qwen2.5 technical report.External Links: 2412.15115,[Link](https://arxiv.org/abs/2412.15115 "")Cited by: [Figure 2](#S3.F2 "In A highly imbalanced sampled-token signal. ‣ 3.2 Why sampled-token OPD is brittle in practice ‣ 3 Understanding Sampled-token OPD: Tradeoffs and Failure Modes ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes"),[§5.1](#S5.SS1.p1.1 "5.1 Setup ‣ 5 Experiments ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes").
* M. Shridhar, X. Yuan, M. Cote, Y. Bisk, A. Trischler, and M. Hausknecht (2021)ALFWorld: aligning text and embodied environments for interactive learning.In International Conference on Learning Representations,External Links: [Link](https://openreview.net/forum?id=0IOX0YcCdTn "")Cited by: [§5.1](#S5.SS1.p1.1 "5.1 Setup ‣ 5 Experiments ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes").
* B. Xiao, B. Xia, B. Yang, B. Gao, B. Shen, C. Zhang, C. He, C. Lou, F. Luo, G. Wang, et al. (2026)Mimo-v2-flash technical report.arXiv preprint arXiv:2601.02780.Cited by: [§2](#S2.p1.1 "2 Related Work ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes").
* A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, et al. (2025)Qwen3 technical report.arXiv preprint arXiv:2505.09388.Cited by: [§2](#S2.p1.1 "2 Related Work ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes").
* C. Ye, X. Zhang, Y. Hao, Z. Yu, Z. Zhang, A. Gullapalli, H. Chen, and T. Zhang (2026)Cited by: [Appendix A](#A1.SS0.SSS0.Px3.p1.1 "Relation to other stabilization directions. ‣ Appendix A Future Directions ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes"),[§2](#S2.p2.1 "2 Related Work ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes").
* Q. Yu, Z. Zhang, R. Zhu, Y. Yuan, X. Zuo, YuYue, W. Dai, T. Fan, G. Liu, J. Liu, L. Liu, X. Liu, H. Lin, Z. Lin, B. Ma, G. Sheng, Y. Tong, C. Zhang, M. Zhang, R. Zhang, W. Zhang, H. Zhu, J. Zhu, J. Chen, J. Chen, C. Wang, H. Yu, Y. Song, X. Wei, H. Zhou, J. Liu, W. Ma, Y. Zhang, L. Yan, Y. Wu, and M. Wang (2025)DAPO: an open-source LLM reinforcement learning system at scale.In The Thirty-ninth Annual Conference on Neural Information Processing Systems,External Links: [Link](https://openreview.net/forum?id=2a36EMSSTp "")Cited by: [§5.1](#S5.SS1.p1.1 "5.1 Setup ‣ 5 Experiments ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes").
* A. Zeng, X. Lv, Z. Hou, Z. Du, Q. Zheng, B. Chen, D. Yin, C. Ge, C. Xie, C. Wang, et al. (2026)GLM-5: from vibe coding to agentic engineering.arXiv preprint arXiv:2602.15763.Cited by: [§2](#S2.p1.1 "2 Related Work ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes").
* J. Zhang, A. Hans, J. Kirchenbauer, M. Goldblum, A. Panda, and T. Goldstein (2026)Learning from mixed rollouts: logit fusion as a bridge between imitation and exploration.Notion Blog.External Links: [Link](https://juzhengz.notion.site/logit-fusion "")Cited by: [Appendix A](#A1.SS0.SSS0.Px3.p1.1 "Relation to other stabilization directions. ‣ Appendix A Future Directions ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes"),[§2](#S2.p2.1 "2 Related Work ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes").
* L. Zhang and J. Ba (2026)EMA policy gradient: taming reinforcement learning for LLMs with EMA anchor and Top-k KL.arXiv preprint arXiv:2602.04417.Cited by: [Appendix A](#A1.SS0.SSS0.Px3.p1.1 "Relation to other stabilization directions. ‣ Appendix A Future Directions ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes"),[§2](#S2.p2.1 "2 Related Work ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes").

Appendix

Appendix A Future Directions
----------------------------

#### OPD versus RL in multi-task transfer.

Our multi-task results motivate a more direct comparison between OPD and RL as transfer mechanisms. In RL, positive or negative transfer can be read directly from environment reward across tasks. In OPD, the optimization target remains teacher-derived, so transfer is filtered through what the teacher regards as locally preferable behavior. This distinction may help explain why our multi-task gains are strongest on the math side and why nearby support-set definitions become less uniform in that setting. A matched-task, matched-compute comparison between OPD and RL would help clarify when teacher-guided transfer tracks environment-level generalization and when the teacher–reward gap becomes the bottleneck.

#### Continual learning as a testbed.

Continual learning is another natural setting for OPD. A teacher-guided on-policy objective could act as a retention mechanism while the student adapts to new tasks, but that regime would also stress exactly the issues surfaced in this paper: distribution shift, teacher staleness, and the accumulation of approximation error over long adaptation horizons. Testing OPD there would therefore probe not only whether local support matching mitigates forgetting, but also whether teacher-based objectives remain useful once the student keeps moving away from the teacher’s original domain.

#### Relation to other stabilization directions.

This work is complementary to directions such as reward-hacking mitigation, EMA-anchor stabilization with top-$K$ KL*(Zhang and Ba, [2026](#bib.bib8 "EMA policy gradient: taming reinforcement learning for LLMs with EMA anchor and Top-k KL"))*, perturbation-based off-policy correction*(Ye et al., [2026](#bib.bib9 "Adaptive layerwise perturbation: unifying off-policy corrections for LLM RL"))*, and logit-level fusion between teacher and student rollouts*(Zhang et al., [2026](#bib.bib10 "Learning from mixed rollouts: logit fusion as a bridge between imitation and exploration"))*. These methods address different parts of the same broader problem: how to keep teacher-derived learning signals useful once teacher and student policies begin to diverge. We view local support matching as one component in that larger toolbox, rather than as a replacement for those stabilization strategies.

Appendix B Bias and variance analysis of token-level versus sequence-level OPD
--------------------------------------------------------------------------------

### B.1 Bias of the token-level estimator

Recall the sequence-level estimator in causal return-to-go form

|  | $\hat{g}_{\mathrm{seq}}\=\sum_{t\=1}^{T}\left(\sum_{t^{\prime}\=t}^{T}r_{t^{\prime}}\right)g_{t}.$ |  |
| --- | --- | --- |

Expanding the inner sum gives

|  | $\hat{g}_{\mathrm{seq}}\=\sum_{t\=1}^{T}r_{t}g_{t}+\sum_{t\=1}^{T}\sum_{t^{\prime}\=t+1}^{T}r_{t^{\prime}}g_{t}.$ |  |
| --- | --- | --- |

Since the token-level estimator keeps only the first term,

|  | $\hat{g}_{\mathrm{tok}}\=\sum_{t\=1}^{T}r_{t}g_{t},$ |  |
| --- | --- | --- |

their expectation gap is

|  | $\mathbb{E}[\hat{g}_{\mathrm{seq}}]-\mathbb{E}[\hat{g}_{\mathrm{tok}}]\=\mathbb{E}\left[\sum_{t\=1}^{T}\sum_{t^{\prime}\=t+1}^{T}r_{t^{\prime}}g_{t}\right].$ |  |
| --- | --- | --- |

This makes explicit that token-level OPD removes the future-reward coupling terms and is therefore generally biased with respect to the sequence-level objective.

### B.2 Worst-case variance upper bounds

Assume there exist constants $B_{r},B_{g}>0$ such that

|  | $|r_{t}|\leq B_{r},\qquad\|g_{t}\|\leq B_{g}\quad\text{for all }t.$ |  |
| --- | --- | --- |

For the token-level estimator,

|  | $\|\hat{g}_{\mathrm{tok}}\|\leq\sum_{t\=1}^{T}|r_{t}|\,\|g_{t}\|\leq TB_{r}B_{g},$ |  |
| --- | --- | --- |

which implies

|  | $\mathbb{E}\|\hat{g}_{\mathrm{tok}}\|^{2}\leq T^{2}B_{r}^{2}B_{g}^{2}.$ |  |
| --- | --- | --- |

Using $\mathrm{Var}(X)\leq\mathbb{E}\|X\|^{2}$, we obtain

|  | $\mathrm{Var}(\hat{g}_{\mathrm{tok}})\=O(T^{2}).$ |  |
| --- | --- | --- |

For the sequence-level estimator, define

|  | $R\=\sum_{t\=1}^{T}r_{t},\qquad G\=\sum_{t\=1}^{T}g_{t},\qquad\hat{g}_{\mathrm{seq}}\=RG.$ |  |
| --- | --- | --- |

Then

|  | $|R|\leq TB_{r},\qquad\|G\|\leq TB_{g},$ |  |
| --- | --- | --- |

so

|  | $\|\hat{g}_{\mathrm{seq}}\|\leq T^{2}B_{r}B_{g},\qquad\mathbb{E}\|\hat{g}_{\mathrm{seq}}\|^{2}\leq T^{4}B_{r}^{2}B_{g}^{2}.$ |  |
| --- | --- | --- |

Therefore,

|  | $\mathrm{Var}(\hat{g}_{\mathrm{seq}})\=O(T^{4}).$ |  |
| --- | --- | --- |

### B.3 Discussion

The sequence-level estimator is closer to the exact trajectory-level objective, but it couples each score term with many future rewards. In worst-case scaling, this changes variance growth from quadratic to quartic in sequence length. The argument is deliberately conservative, but it captures why stronger reward coupling can become problematic in long-horizon post-training.

Appendix C Toy experiment details
---------------------------------

### C.1 Environment

We use a two-task one-dimensional continuous-control environment to visualize how stronger return coupling changes OPD optimization. The student policy is a three-layer MLP with roughly 4K parameters. Its input is a three-dimensional vector containing task identity, current position, and normalized time step. The policy outputs the mean and standard deviation of a Gaussian action distribution, and the state transition is

|  | $s_{t+1}\=s_{t}+\delta,\qquad\delta\sim\mathcal{N}(\mu,\sigma).$ |  |
| --- | --- | --- |

The two tasks are mirror images of each other: the left task starts from $+2$ and targets $-3$, while the right task starts from $-2$ and targets $+3$. We first train separate teachers with REINFORCE and then distill them into a shared student with alternating-task OPD.

### C.2 Gradient variance estimation

At each training step, we split a batch of $B\=64$ trajectories into $M\=8$ micro-batches. For each micro-batch $m$, we compute a loss $\mathcal{L}_{m}$ and the corresponding gradient vector $\mathbf{g}_{m}$ on the output layer parameters. We then estimate gradient variance by

|  | $\mathrm{Var}(\mathbf{g})\=\frac{1}{M}\sum_{m\=1}^{M}\left\|\mathbf{g}_{m}-\bar{\mathbf{g}}\right\|^{2},\qquad\bar{\mathbf{g}}\=\frac{1}{M}\sum_{m\=1}^{M}\mathbf{g}_{m}.$ |  |
| --- | --- | --- |

We use this quantity only as a qualitative proxy, but it is sufficient for comparing relative variance across different $\gamma$ settings.

### C.3 Additional Results of Toy Experiments

Figure[A1](#A3.F1 "Figure A1 ‣ C.3 Additional Results of Toy Experiments ‣ Appendix C Toy experiment details ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes"), [A2](#A3.F2 "Figure A2 ‣ C.3 Additional Results of Toy Experiments ‣ Appendix C Toy experiment details ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes"), and [A3](#A3.F3 "Figure A3 ‣ C.3 Additional Results of Toy Experiments ‣ Appendix C Toy experiment details ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes") report gradient-variance curves and corresponding state-visitation heatmaps for different OPD estimators ($\gamma\in{0.0,0.25,0.5,0.75,1.0}$) across three random seeds. Although the exact magnitudes vary by seed, the qualitative pattern is consistent. All settings exhibit large variance spikes during early optimization, and larger $\gamma$ typically remains at a higher variance level later in training. In several runs, the variance under $\gamma\=0.75$ or $\gamma\=1.0$ stays one to several orders of magnitude above that of smaller $\gamma$ values.
Across runs, token-level OPD ($\gamma\=0$) consistently learns trajectories that move toward the target states for both tasks. Intermediate values of $\gamma$ remain qualitatively similar but become more diffuse. When $\gamma$ approaches the sequence-level case ($\gamma\=1.0$), the learned trajectories often deviate from the desired direction and stabilize around suboptimal regions of the state space.

<img src='2603.25562v1/x17.png' alt='Refer to caption' title='' width='747' height='267' />

<img src='2603.25562v1/x18.png' alt='Refer to caption' title='' width='747' height='332' />

*Figure A1: Toy experiment with random seed 42: gradient variance and state visitation.*

<img src='2603.25562v1/x19.png' alt='Refer to caption' title='' width='747' height='267' />

<img src='2603.25562v1/x20.png' alt='Refer to caption' title='' width='747' height='332' />

*Figure A2: Toy experiment with random seed 43: gradient variance and state visitation.*

<img src='2603.25562v1/x21.png' alt='Refer to caption' title='' width='747' height='267' />

<img src='2603.25562v1/x22.png' alt='Refer to caption' title='' width='747' height='332' />

*Figure A3: Toy experiment with random seed 2026: gradient variance and state visitation.*

Appendix D Qualitative OPD reward-hacking case study
-----------------------------------------------------

To complement the representative failures in the main text, we summarize a longer trajectory from multi-task training under sampled-token OPD. Read chronologically, the case exhibits the same pattern in several forms: the model keeps analyzing after it already has the answer, falls into repetition loops such as wait, drifts into malformed continuations, and still receives high local teacher probability on those tokens.

The failure first appears as *over-continuation*. Even after the answer is effectively available, the local signal continues to place substantial mass on generic reasoning fillers and connective tokens, encouraging the model to keep going instead of stopping cleanly. The same pattern later appears on prefixes such as confirm, where the local signal still favors additional verification rather than termination.
Some of this behavior may also reflect the teacher’s own output habits.
Figure[A4](#A4.F4 "Figure A4 ‣ Appendix D Qualitative OPD reward-hacking case study ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes") illustrates several representative cases.

<img src='2603.25562v1/x23.png' alt='Refer to caption' title='' width='830' height='659' />

*(a) High teacher probability on generic reasoning fillers (implies) at step 5.*

![Refer to caption](2603.25562v1/)

*(b) Teacher and student remain well aligned even after the answer is effectively available, so the model keeps analyzing instead of stopping at step 9.*

*Figure A4: Even after the student has effectively reached an answer, the teacher can still assign high conditional probability to meaningless continuations.*

The trajectory then develops into *hesitation loops and low-information continuations*. Repeated wait tokens, punctuation-heavy continuations, and other semantically weak fillers can remain locally rewardable even after the overall trajectory has become unproductive. This is consistent with the repetition-loop discussion in Section[3.2](#S3.SS2 "3.2 Why sampled-token OPD is brittle in practice ‣ 3 Understanding Sampled-token OPD: Tradeoffs and Failure Modes ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes"). We provide two similar cases in Figure[A5](#A4.F5 "Figure A5 ‣ Appendix D Qualitative OPD reward-hacking case study ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes").

<img src='2603.25562v1/x25.png' alt='Refer to caption' title='' width='830' height='324' />

*(a) The teacher may fail to penalize, and sometimes even reinforce, repetitive generation.*

<img src='2603.25562v1/x26.png' alt='Refer to caption' title='' width='830' height='497' />

*(b) Training can also produce overlong chain-of-thought traces with substantial low-quality content. This pattern is common in LLM reasoning and may partly reflect the teacher’s output style.*

*Figure A5: Loops, overlong CoT and punctuation can be locally rewardable.*

Finally, once the student drifts further off-distribution, the local signal can remain misleadingly positive rather than self-correcting. In the case study, this appears as degeneration and gibberish outputs, yet many tokens still receive high teacher probability. An example is shown in Figure[A6](#A4.F6 "Figure A6 ‣ Appendix D Qualitative OPD reward-hacking case study ‣ Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes").

<img src='2603.25562v1/x27.png' alt='Refer to caption' title='' width='830' height='630' />

*Figure A6: The teacher still assigns high probability to several tokens after the student drifts into nonsensical Chinese outputs.*


Instructions for reporting errors
---------------------------------

We are continuing to improve HTML versions of papers, and your feedback helps enhance accessibility and mobile
 support. To report errors in the HTML that will help us improve conversion and rendering, choose any of the
 methods listed below:


**Tip:** You can select the relevant text first, to include it in your report.

Our team has already identified [the following issues](https://github.com/arXiv/html_feedback/issues). We appreciate your time reviewing and reporting rendering errors we
 may not have found yet. Your efforts will help us improve the HTML versions for all readers, because disability
 should not be a barrier to accessing research. Thank you for your continued support in championing open access for
 all.

Have a free development cycle? Help support accessibility at arXiv! Our collaborators at LaTeXML maintain a [list of packages that need conversion](https://github.com/brucemiller/LaTeXML/wiki/Porting-LaTeX-packages-for-LaTeXML), and welcome [developer contributions](https://github.com/brucemiller/LaTeXML/issues).

BETA

