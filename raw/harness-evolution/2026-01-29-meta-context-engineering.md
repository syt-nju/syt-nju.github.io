# Meta Context Engineering via Agentic Skill Evolution

> Source: https://arxiv.org/abs/2601.21557
> Author: Haoran Ye, Xuning He, Vincent Arak, Haonan Dong, Guojie Song
> Collected: 2026-08-05
> Published: 2026-01-29
> License: arXiv.org perpetual non-exclusive license 1.0
> Completeness: Complete
> Retrieval note: Full text extracted via deepxiv from arXiv PDF (v2, updated 2026-02-11); abs URL canonicalizes to https://arxiv.org/abs/2601.21557v2. Code: https://github.com/ace-agent/ace. Categories: cs.AI, cs.NE.

## Abstract

The operational efficacy of large language models relies heavily on their inference-time context. This has established Context Engineering (CE) as a formal discipline for optimizing these inputs. Current CE methods rely on manually crafted harnesses, such as rigid generation-reflection workflows and predefined context schemas. They impose structural biases and restrict context optimization to a narrow, intuition-bound design space. To address this, we introduce Meta Context Engineering (MCE), a bi-level framework that supersedes static CE heuristics by co-evolving CE skills and context artifacts. In MCE iterations, a meta-level agent refines engineering skills via agentic crossover, a deliberative search over the history of skills, their executions, and evaluations. A base-level agent executes these skills, learns from training rollouts, and optimizes context as flexible files and code. We evaluate MCE across five disparate domains under offline and online settings. MCE demonstrates consistent performance gains, achieving 5.6--53.8% relative improvement over state-of-the-art agentic CE methods (mean of 16.9%), while maintaining superior context adaptability, transferability, and efficiency in both context usage and training.

## Authors

Haoran Ye, Xuning He, Vincent Arak, Haonan Dong, Guojie Song

## Body

*Figure 1: Conceptual overview of Meta Context Engineering (MCE). A cartoon representation of the bi-level optimization framework. The meta-agent drives skill evolution while the base-agent manages context optimization. This dual-layered approach ensures the agentic harness and context artifacts co-evolve for maximum performance.*

1 Introduction
--------------

The operational efficacy of large language models (LLMs) is governed by the curation and orchestration of their inference-time context *[mei2025survey]*. As LLM infrastructure evolves from monolithic chat interfaces toward compound agentic systems, context optimization has become both a critical bottleneck and a powerful lever for its enhancement. This establishes the discipline of Context Engineering (CE), which focuses on the principled LLM context optimization to maximize downstream utility and enable continuous self-improvement *[hua2025context_engineering_2]*.

Recent advances demonstrate CE’s efficacy across diverse objectives, such as enhancing vertical-domain performance *[zhang2025agentic, cai2025flex, agrawal2025gepa]*, solving long-horizon tasks *[zhang2025recursive_lm, sun2025scaling, hu2025hiagent, kang2025acon]*, coordinating multiple agents *[yuksekgonul2025optimizing, wu2024autogen, ma2026judgeflow]*, and self-evolution driven by experience *[ye2024reevo, lu2026llmasrnnrecurrentlanguagemodel, wei2025evomemory, gao2025survey]*. Unlike optimizing LLM parameters, optimizing their context provides distinct methodological advantages: (1) interpretability, by encoding experience in natural language rather than opaque weights; (2) efficiency, by enabling rapid deployment without costly updates of model parameters; (3) modularity, facilitating the composition and transfer of established contexts; and (4) robustness, ensuring immunity to catastrophic forgetting by decoupling capability acquisition from model weights.

However, current CE systems are fundamentally constrained by manually crafted agentic harnesses that impose characteristic inductive biases. At the context representation level, different context structures impose their own trade-offs:
case-based trajectories retain rich episodic traces but lack generalization *[zhou2025memento]*;
itemized lists accumulate abstract insights but remain flat and structurally inexpressive *[zhang2025agentic]*;
and graph-based hierarchies offer flexible organization but incur high latency without consistently outperforming naive retrieval *[xu2025mem, ai2025memorybench]*. At the context optimization level, existing methods exhibit opposing yet equally limiting biases.
On one hand, prompt-rewriting approaches such as GEPA *[agrawal2025gepa]* favor brevity, iteratively refining concise, high-level rules that fail in tasks requiring detailed strategies and deep domain knowledge *[zhang2025agentic]*.
On the other hand, additive-curation approaches *[zhang2025agentic, cai2025flex, suzgun2025dynamic, cai2025training]* favor verbosity, accumulating context through generation-reflection-curation pipelines that produce noisy context update (due to limited global visibility), incur excessive overhead (due to instance-level optimization), and cause context bloat (due to additive accumulation without holistic synthesis).
Ultimately, these heuristic choices restrict CE to a narrow design subspace, precluding the discovery of task-optimal strategies that lie beyond human intuition.

To address these limitations, we introduce Meta Context Engineering (MCE), a framework that supersedes static heuristics through the co-evolution of CE skills and context artifacts. MCE formalizes CE as a bi-level optimization problem, effectively decoupling the engineering strategy (how to represent and optimize context) from the resulting engineered artifact (what context is learned). At the meta-level, we propose agentic skill evolution, in which an agent iteratively refines CE skills *[anthropic2025agentskills_blog]*: executable instructions and code that govern the CE process. This evolution is driven by agentic crossover, an evolutionary operator that synthesizes superior skills by reasoning across task specifications, historical CE trajectories, and performance metrics. At the base-level, an agent executes these evolved skills to learn from training rollouts and constructs context, in a fully agentic manner.
Unlike prior methods that predefine context schemas, our base-agent leverages coding toolkits and file system access to instantiate and optimize context as flexible, programmatic artifacts.

In this manner, MCE replaces heuristic scaffolding with a generic design space. Prior state-of-the-art (SOTA) methods, such as the generation-reflection-curation workflow of agentic context engineering (ACE) *[zhang2025agentic]*, represent singular points within the vast manifold of possible CE skills. By granting AI the full agency and capabilities to generate arbitrary code, invoke other LLMs, and manipulate file structures, MCE can reconstruct such pipelines, discover novel CE architectures, and dynamically adjust CE strategies.

We evaluate MCE across five diverse domains (finance, chemistry, medicine, law, and AI safety), using four LLMs, and against SOTA CE methods. Under offline and online settings respectively, MCE achieves 89.1% and 74.1% average relative improvement over the DeepSeek-V3.1 base model, outperforming the prior SOTA by 18.4% and 33.0%.
In addition, MCE demonstrates superior 1) context adaptability: flexibly adjusting context length from 1.5K to 86K tokens across tasks, free from the brevity or verbosity biases of prior methods; 2) context efficiency: achieving better performance with fewer context tokens; 3) context transferability: exhibiting 4–7% lower performance degradation when transferring contexts from strong to weak models; and 4) training efficiency: accelerating training by $13.6\times$ and requiring $4.8\times$ fewer rollouts than ACE to achieve higher training accuracy.

We summarize contributions as follows.
❶ We propose MCE, a bi-level framework that co-evolves CE skills and context artifacts, superseding static CE harnesses with learnable skills.
❷ At the meta-level, we introduce agentic skill evolution, leveraging agent skills *[anthropic2025agentskills_blog]* as a novel abstraction for evolutionary agent optimization.
❸ At the base-level, we introduce fully agentic context optimization, leveraging coding toolkits and file system access to instantiate context as flexible, programmatic artifacts.
❹ We conduct comprehensive evaluations across five domains, four LLMs, and both offline and online settings. MCE achieves 5.6–53.8% relative improvement over state-of-the-art CE methods (mean of 16.9%), while demonstrating superior context adaptability, transferability, and efficiency in both context usage and training.

2 Background and Related Work
-----------------------------

### 2.1 Agentic Context Engineering

While LLMs possess strong zero-shot capabilities, post-deployment context engineering (CE) is crucial for domain adaptation and self-improvement *[khattab2023dspy, hu2025memory, fang2025comprehensive, shinn2023reflexion, yuksekgonul2025optimizing]*. Current SOTA CE methods rely on manually designed agentic harnesses: models accumulate experience through exploration-reflection workflows to update contexts with predefined schemas.

However, these harnesses impose characteristic inductive biases. At the context representation level, the trade-offs are distinct:
case-based trajectories retain rich episodic traces but lack generalization and abstraction *[zhou2025memento, wang2024agent]*;
itemized lists accumulate abstract insights but remain flat and structurally limited *[zhang2025agentic, suzgun2025dynamic]*;
and hierarchical, graph-based memories incur high latency and complexity without consistently outperforming naive retrieval *[chhikara2025mem0, ai2025memorybench, li2025memos, xu2025mem]*. At the context optimization level, existing methods exhibit opposing yet equally limiting biases.
On one hand, prompt-rewriting approaches such as GEPA *[agrawal2025gepa]* employ full LLM-based rewrites driven by reflections over sampled trajectories, favoring abstract, high-level rules over detailed domain knowledge (brevity bias).
On the other hand, additive-curation approaches such as Dynamic Cheatsheet (DC) *[suzgun2025dynamic]* and Agentic Context Engineering (ACE) *[zhang2025agentic]* orchestrate modular agents (generators, reflectors, and curators) to iteratively perform on-policy rollouts, textual reflections, and context updates. These methods utilize localized updates to itemized lists and can additively accumulate insights, but suffer from structural rigidity and context bloat *[cai2025flex, cai2025training]*.

We posit that no single agentic harness is universally optimal, motivating MCE’s dual-level optimization. Our instantiation of MCE is driven by two converging trends. First, agent architectures are shifting from rigid, multi-agent scaffolds toward unified, self-looping, and self-spawning frameworks with maximal agency and minimal yet general tools *[anthropic_claude_agent_sdk, manus_app, langchain_deepagents, bolin2026unrolling]*. In this architecture, domain specificity can be encapsulated in agent skills: organized instructions, scripts, and resources that agents discover and load dynamically *[anthropic2025agentskills_blog]*. This motivates our meta-level design: CE as a fully agentic process without restrictive scaffolding, with task specificity injected through learned skills.

Second, coding toolkits and computer (file system) access have became the essential harness of both general and vertical agents *[yang2025code, ClaudeCodeDocs2026, manus_app, cheng2026llm]*. The prevalence stems largely from the versatility of Turing-complete programming languages, which offer maximal design flexibility *[zhang2025darwin, zhang2025aflow]*, and their inherent verifiability, which facilitates robust training *[wei2025_asymmetry_verification, wang2024executable, yang2024swe]*. This insight informs our base-level design space: context artifacts consist of files and code, which are general, agent-native representations unconstrained by predefined schemas.

Ultimately, MCE represents a transition from manually crafted CE workflows to fully agentic meta- and self-learning systems, where domain specificity encapsulated with learned skills and context is managed via agent-generated files and code.

### 2.2 Evolutionary Computation with LLMs

Conventional evolutionary computation (EC) relies on manually designed operators, which struggle to capture complex solution structures and domain properties. Recent research demonstrates that LLMs can serve as intelligent genetic operators, leveraging semantic knowledge to generate meaningful variations without explicit rules *[wu2024evolutionary]*.

This paradigm shift enables the exploration of unstructured search spaces and facilitates optimization across varying levels of abstraction *[novikov2025alphaevolve, lange2025shinkaevolve]*.
Prior work in LLM-driven EC has primarily targeted the solution level (e.g., prompts *[guo2024connecting]*, textual solutions *[lee2025evolving]*, numerical parameters *[xu2025autoep]*), function level (e.g., heuristics *[romera2024mathematical, ye2024reevo, liu2024evolution]*, reward functions *[ma2024eureka]*), and program level (e.g., search algorithms *[novikov2025alphaevolve, hottung2025vrpagent]*, neural architectures *[liu2025alphago]*, agent workflows *[zhang2025evoflow]*).

In this work, we introduce agent skills *[anthropic2025agentskills_blog]* as a novel, integrated level of abstraction for evolutionary optimization. Agent skills are organized folders of instructions, scripts, and resources, that an agent can discover and utilize. We find that this abstraction offers three distinct advantages for meta-level agent optimization. First, by encapsulating instructions, resources, and scripts into a unified representation, skills enable the seamless co-evolution of different solution levels, overcoming the fragmented frameworks often seen in prior co-evolutionary approaches *[zhao2025understanding, cen2025beyond, xie2025co, guo2025nested]*. Second, skills provide a modular interface for agentic harnesses, decoupling the optimization target from the core agent architecture to enhance stability. Third, this representation facilitates more agentic genetic operators: agents can flexibly inspect and selectively recombine components from ancestor skill directories, enabling more granular and context-aware evolutionary updates.

<img src='x2.png' alt='Refer to caption' title='' width='660' height='335' />

*Figure 2: Methodological overview of Meta Context Engineering (MCE).*

3 Meta Context Engineering
--------------------------

We formalize Meta Context Engineering (MCE) as a bi-level optimization framework that co-evolves context engineering skills and context artifacts ([§˜3.1]). At the meta-level, an agent refines skills that guide context representation and optimization ([§˜3.2]); at the base-level, an agent executes these skills to optimize context as files and code ([§˜3.3]).
This dual-level optimization is orchestrated by a simple $(1+1)$-Evolution Strategy (ES) and instantiated with general agents that have access to coding toolkits and file systems ([§˜3.4]).
We present the methodological overview in [Fig.˜2].

### 3.1 Problem Formulation

We define a context function $c$ that maps each query $x\in\mathcal{X}$ to its context, specified by a tuple $(\rho,F)$:

|  | $c(x)\=(F_{k}\circ\cdots\circ F_{1})(x;\rho),$ |  | (1) |
| --- | --- | --- | --- |

where $\rho\={\rho_{1},\ldots,\rho_{m}}$ denotes static components (e.g., system prompts, knowledge bases, code libraries) and $F\={F_{1},\ldots,F_{k}}$ denotes dynamic operators (e.g., retrieval, selection, filtering, formatting, composition) that transform and assemble these components conditioned on query $x$.

Given an agent $f_{\theta}$ that produces output $\hat{y}\=f_{\theta}(x,c(x))$, the goal of CE is to find the optimal context function:

|  | $c^{*}\=\arg\max_{c\in\mathcal{C}}J(c),$ |  | (2) |
| --- | --- | --- | --- |

where $J(c)$ is a task-specific objective. This formulation encompasses supervised settings, where $J(c)\=-\sum_{i}\ell(f_{\theta}(x_{i},c(x_{i})),y_{i})$ measures prediction accuracy, and RL-based settings, where $J(c)\=\mathbb{E}_{x}[R(f_{\theta}(x,c(x)))]$ measures expected reward.

This formulation admits diverse instantiations along the spectra of dynamism and complexity:
static prompts correspond to constant functions $c(x)\=\rho_{0}$ with identity operators; retrieval-augmented systems implement $c(x)\=F_{\text{retrieve}}(x;\rho)$ over fixed resources; and agentic systems may incorporate pre/post-model hooks and chain multiple operators that dynamically generate, modify, or discard components during inference. The optimal design of $(\rho,F)$ depends on the task domain, data characteristics, and computational constraints.

Prior CE methods instantiate $c$ with predefined components and operators, then optimize $c$ directly using fixed procedures. For instance, ACE employs iterative generation-reflection-curation loops to optimize context as an itemized list *[zhang2025agentic]*. These procedures impose structural biases that may be suboptimal for specific tasks. MCE instead introduces a skill $s\in\mathcal{S}$ as an executable specification that defines how the context function should be represented and learned from data. Given a skill $s$, a base-level agent executes it to produce a context function $c_{s}\=(\rho_{s},F_{s})$. MCE then solves the bi-level problem:

|  | $s^{*}\=\arg\max_{s\in\mathcal{S}}J_{\text{val}}(c_{s}^{*})\quad\text{s.t.}\quad c_{s}^{*}\=\arg\max_{c_{s}}J_{\text{train}}(c_{s};s),$ |  | (3) |
| --- | --- | --- | --- |

where the inner optimization finds the best context function given skill $s$, and the outer optimization finds the skill that yields context with maximal validation performance.

MCE decouples what to learn (the context function $c_{s}$) from how to represent and learn it (the skill $s$). This decoupling is analogous to separating learned parameters from model architecture and training algorithms in machine learning (ML), except that MCE operates at a higher level of abstraction, where the ML models (LLMs) are already well-trained and frozen. Such decoupling has given rise to fields such as meta-learning, AutoML, neural architecture search, hyperparameter optimization, and meta-black-box optimization. By elevating this decoupling to CE, MCE opens a new design space for optimizing agentic AI.

### 3.2 Meta-Level: Agentic Skill Evolution

The meta-level agent maintains its skill database $\mathcal{H}_{k-1}\={(s_{i},c_{i},J_{i}^{\text{train}},J_{i}^{\text{val}})}_{i\=1}^{k-1}$, a folder that summarizes the history of skills, their resulting context functions, and evaluation metrics from all previous iterations. At iteration $k$, the meta-agent generates a new skill by performing agentic crossover:

|  | $s_{k}\=\textsc{Crossover}(\tau,\mathcal{H}_{k-1}),$ |  | (4) |
| --- | --- | --- | --- |

where $\tau$ denotes the task specification (e.g., task description, data format, evaluation criteria). Agentic crossover is an LLM agent-driven operator that synthesizes a new skill by selectively combining and refining elements from previous skills. Unlike prior LLM-driven evolutionary operators that apply fixed recombination rules (e.g., merging two programs into a better one), it is a flexible and deliberative process: the agent reasons over the task specification, arbitrarily inspects workspace folders, identifies successful and failure patterns, and composes an improved skill.

A skill $s\in\mathcal{S}$ is represented as a folder in the base-agent workspace. In practice, we find that it may include, but is not limited to, the following:
(1) a methodology describing the learning procedure in natural language (e.g., core philosophy, multi-phase approaches with error analysis, context pruning strategies); (2) executable scripts implementing the methodology (e.g., Python code for error pattern extraction, LLM-based context generation, embedding-based similarity analysis); (3) structured context templates (e.g., decision frameworks, disambiguation rules, gap-driven refinements); (4) validation protocols (e.g., functions to assess context quality and measure generalization); and (5) dynamic context operators such as retrieval functions that selectively filter and compose relevant context based on query features (e.g., keyword-based section extraction, embedding-based similarity matching, rule-based pattern detection for error-prone cases).
An example of a learned skill is provided in [Appendix˜C].

Analysis of the evolved skills reveals three key technical advantages. First, we observe that MCE dynamically adjusts autonomy, expressivity, and granularity: learned skills are found to specify rigid workflows or delegate full autonomy depending on the need, enabling either holistic batch-level synthesis or targeted instance-level updates. Second, evolving skills tailor context verbosity to task complexity and model capacity, generating concise rules for simple tasks or capacity-limited models, while providing detailed explanations otherwise.
Third, the meta-agent effectively monitors training and validation signals, detects overfitting, and steers skill evolution toward improved generalization.
Further analysis are provided in [Appendix˜C] and [Appendix˜D].

### 3.3 Base-Level: Fully Agentic Context Optimization

Given skill $s_{k}$ from the meta-level, the base-level agent executes it to produce a context function. The agent operates within a workspace containing: (1) its skill folder $s_{k}$, (2) prior best context function $c_{k-1}^{*}$ (warm-start), (3) training rollouts $\mathcal{R}_{k}\={(x_{i},\hat{y}_{i},\text{eval}_{i})}$ obtained by evaluating $c_{k-1}^{*}$ on $\mathcal{D}_{\text{train}}$, and (4) optional utility functions to invoke other AI models ([Appendix˜E]). The base-agent’s objective is to update the context function by learning from rollout feedback. The process is fully agentic and guided by the current skill:

|  | $c_{k}\=\textsc{Engineer}(\tau,s_{k};c_{k-1}^{*},\mathcal{R}_{k}).$ |  | (5) |
| --- | --- | --- | --- |

Here, a context function is instantiated as a collection of files in a designated directory, including both static and dynamic components.
Static components $\rho$ may include knowledge bases, decision rules, or examples, while dynamic operators $F$ may implement retrieval, filtering, or composition logic. The code and file-based representation imposes no structural constraints, enabling arbitrary computational procedures for context generation and manipulation. This stands in contrast to prior methods that specify rigid context representations and optimization workflows.

### 3.4 Algorithmic Orchestration

[Algorithm˜1] summarizes the iterative MCE procedure. Each iteration consists of three phases: (1) skill evolution, where the meta-agent generates $s_{k}$ by analyzing the task specification $\tau$ and skill history $\mathcal{H}_{k-1}$; (2) context optimization, where training rollouts are programmatically performed and the base-agent executes $s_{k}$ to produce context function $c_{k}$; and (3) evaluation, where $c_{k}$ is assessed on the validation dataset to update the skill database and track the best-so-far solution $c_{k}^{*}$.
Training data batching can be optionally adopted for the base-agent when dataset size is large.
Overall, the dual-level optimization implements a simple history-informed $(1+1)$-ES: at each iteration, agentic crossover can reference the entire skill history $\mathcal{H}_{k-1}$; a single offspring context $c_{k}$ is generated and compared against the current best $c_{k-1}^{*}$, with the better one retained. While we adopt this strategy for simplicity, more advanced search algorithms may further improve performance.

Both meta and base-level adopt fully agentic optimizations: each agent interacts with a programming environment through a standard tool set $\mathcal{T}$ \= {Read, Write, Edit, Bash, Glob, Grep, TodoWrite} and produces outputs by manipulating a file system workspace. The read/write permissions of both agents are strictly scoped according to their respective roles and the current iteration. This design is compatible with modern agentic frameworks such as Claude Agent SDK *[anthropic_claude_agent_sdk]* and LangChain DeepAgents *[langchain_deepagents]*, enabling straightforward integration into existing infrastructure. To facilitate programmatic invocation during rollouts and evaluations, we require the base-agent to implement callable interface(s) with predefined input-output signatures; these interfaces can be task-specific and flexibly defined. They are validated upon completion of each base-agent execution.

*Algorithm 1  Meta Context Engineering (MCE)*

0:Task specification $\tau$, data $\mathcal{D}\=\mathcal{D}_{\text{train}}\cup\mathcal{D}_{\text{val}}$, iterations $K$

1:Initialize skill database $\mathcal{H}_{0}\leftarrow\emptyset$, best context $c_{0}^{*}\leftarrow\emptyset$

2: for $k\=1,\ldots,K$ do

3: // Meta-level: evolve skill

4:$s_{k}\leftarrow\textsc{Crossover}(\tau,\mathcal{H}_{k-1})$

5: // Base-level: execute skill to produce context

6:$\mathcal{R}_{k}\leftarrow\textsc{Rollout}(c_{k-1}^{*};\mathcal{D}_{\text{train}})$

7:$c_{k}\leftarrow\textsc{Engineer}(\tau,s_{k};c_{k-1}^{*},\mathcal{R}_{k})$

8: // Evaluate and update database

9:$J_{k}^{\text{train}}\leftarrow J(c_{k};\mathcal{D}_{\text{train}})$, $J_{k}^{\text{val}}\leftarrow J(c_{k};\mathcal{D}_{\text{val}})$

10:$\mathcal{H}_{k}\leftarrow\mathcal{H}_{k-1}\cup{(s_{k},c_{k},J_{k}^{\text{train}},J_{k}^{\text{val}})}$

11:$c_{k}^{*}\leftarrow\arg\max_{c\in{c_{k-1}^{*},c_{k}}}J^{\text{val}}(c)$

12: end for

13: return Best context function $c_{K}^{*}$ and corresponding skill

4 Experiments
-------------

### 4.1 Experimental Setup

##### Tasks and Datasets.

We evaluate MCE across five benchmarks spanning different domains (financial, chemistry, medicine, law, and AI safety) to demonstrate its versatility and generality.
Unless otherwise specified, we adhere to the original train/val/test splits but use data subsets due to computational budget constraints. All experiments use the same datasets in the same order during sequential processing to ensure fair comparison across baselines. We include ground-truth labels for CE methods when applicable.
Details of the datasets and data processing are provided in [Appendix˜A]. We briefly describe the datasets below.
(1) FiNER (Financial) *[loukas2022finer]* requires labeling tokens in XBRL financial documents with its entity types. We report the pass@1 prediction accuracy.
(2) USPTO-50k (Chemistry) *[schneider2016s]* requires predicting precursor reactants from product molecules. We report the pass@1 exact match accuracy.
(3) Symptom2Disease (Medicine) *[gretelai_symptom_to_diagnosis_2023]* requires predicting diseases from patient symptom descriptions across 22 disease categories. We report pass@1 prediction accuracy.
(4) LawBench (Law) *[fei2024lawbench]* is a comprehensive Chinese legal benchmark. We use the criminal charge prediction subtask from LawBench, and report micro-F1 score.
(5) AEGIS2 (AI Safety) *[ghosh2025aegis2]* requires classifying user prompts as safe or unsafe and identifying specific violation categories. We report the F1 score.

##### Baselines.

We compare MCE against the following baselines. To ensure fair comparisons, we adhere to the suggested implementations in ACE *[zhang2025agentic]*, use the identical initial context or prompt templates, and allow a training budget no less than that used in MCE.
(1) Base Model: Zero-shot evaluation using the same prompt as MCE but without learned context.
(2) In-Context Learning (ICL): Provides task demonstrations in the prompt. Following *zhang2025agentic*, we include all training samples fitting within the context window.
(3) MIPROv2 *[opsahl2024optimizing]*: We implement the official DSPy implementation with auto\="heavy".
(4) GEPA *[agrawal2025gepa]*: We implement the official DSPy implementation with auto\="heavy" and match MCE’s total rollout budget.
(5) Dynamic Cheatsheet (DC) *[suzgun2025dynamic]*: We use the official implementation in cumulative mode (DC-CU).
(6) ACE *[zhang2025agentic]*: We use the official implementation with original parameter settings111[https://github.com/ace-agent/ace](https://github.com/ace-agent/ace "") and run offline training for 5 epochs unless playbook accumulation exceeds the context window or validation performance plateaus.

##### Models.

Following *zhang2025agentic*, we use DeepSeek V3.1 *[liu2024deepseek]* as the default generator (the model performing inference during training and testing) across all methods and benchmarks. For AEGIS2, safety guardrails necessitate lightweight models; we use Qwen3-8B *[qwen3technicalreport]* as the generator across all methods. The reflector model is consistently set to DeepSeek V3.1.
MCE additionally requires an agentic model; we use MiniMax M2.1 *[minimaxai2025minimaxm21]* by default.
MCE base-agents are allowed to invoke DeepSeek V3.1 during its execution.
All models are accessed via OpenRouter *[openrouter2025]*. In [§˜4.3], we show that MiniMax M2.1 does not improve ACE, ruling out knowledge transfer from the agentic model as the source of MCE’s gains.

##### MCE Instantiation.

In experiments, MCE optimizes context for five epochs. We instantiate the MCE agents using the Claude Agent SDK *[anthropic_claude_agent_sdk]*. To maintain methodological consistency with CE baselines and enable fair comparison, we define the context interface as a one-shot retrieval function: $\texttt{query}\rightarrow\texttt{context}$.
The system prompts for MCE agents under this instantiation are given in [Appendix˜B].

##### Evaluation Settings.

We evaluate all methods under two complementary settings *[zhang2025agentic]*.
(1) Offline: CE methods have access to a training set and iterate over it for multiple epochs to optimize context before evaluation on a held-out test set.
(2) Online: CE methods process the test set sequentially, with performance measured only on each instance’s first inference. For MCE, the base-level agent accumulates all processed instances in its file system for continuous learning. We evaluate two MCE variants: one using a fixed skill generated based on task specifications, and another where the base agent operates autonomously without skill guidance.

### 4.2 Main Results

#### 4.2.1 Context Performance

[Table˜1] presents our evaluation of context performance across methods and benchmarks. We summarize our observations below. We also refer to some of the results in [Table˜2], which we will revisit in [§˜4.2.2].

Obs.❶ MCE substantially improves base LLMs for domain adaptation. MCE achieves 89.1% average relative improvement over the base model (DeepSeek-V3.1) across the five benchmarks. The gains are even more pronounced for smaller models since they may lack basic domain knowledge without CE. As shown in [Table˜2], Gemma3-4B with MCE context achieves 172.6% average relative gain.

Obs.❷ MCE consistently outperforms all baselines across benchmarks and settings. MCE ranks first on all five benchmarks in the offline setting, with 89.1% average relative gain compared to 70.7% for ACE (second-best). In the online setting, MCE maintains leadership with 74.1% average gain versus ACE’s 41.1%. This consistent superiority across diverse domains demonstrates MCE’s robustness as a general-purpose CE framework.

Obs.❸ MCE adapts to task-specific requirements, while baselines exhibit inconsistent performance due to their fixed inductive biases. Baselines show inconsistent rankings across tasks because their hard-coded agentic harnesses suit only certain problem structures.
FiNER demands deep reflection and pattern abstraction; simple in-context imitation fails (ICL performs poorly), while ACE’s reflection-curation loop excels.
Symptom2Disease benefits from raw examples due to high semantic similarity between train and test instances; here, ACE underperforms even ICL, as its reflection overhead adds noise without benefit.
On Aegis2.0, the lightweight generator (Qwen3-8B) favors concise prompts, giving GEPA’s brevity bias an edge over ACE.
MCE’s robust top-rank across all tasks indicates its ability to discover task-appropriate strategies beyond any single heuristic. We provide detailed analysis in [Appendix˜D].

Obs.❹ MCE-enhanced general LLMs can surpass domain-specific vertical models. On benchmarks where specialized models are available, MCE-enhanced general LLMs outperform them.
LawBench reports that the best legal-specific model achieves 0.56 F1, while MCE reaches 0.70.
On Aegis2.0, Qwen3-8B with MCE attains 0.80 F1, surpassing Llama Guard 3 8B (0.72), a dedicated safety guardrail model. This suggests that learned context can substitute for expensive domain-specific fine-tuning.

*Table 1: Context performance on different benchmarks. Avg. Rel. Gain measures the mean relative improvement over the base model across all benchmarks. Best and second-best results are highlighted.*

| Method | FiNER | USPTO50k | Symptom2Disease | LawBench | Aegis2.0 | Avg. Rel. |
| --- | --- | --- | --- | --- | --- | --- |
|  | Acc.%$\uparrow$ | Acc.%$\uparrow$ | Acc.%$\uparrow$ | Micro-F1$\uparrow$ | F1$\uparrow$ | Gain%$\uparrow$ |
| Base Model | 58.0 | 6.0 | 63.7 | 0.36 | 0.54 | – |
| Offline Setting | | | | | | |
| ICL | 64.0 (+6.0) | 9.0 (+3.0) | 84.4 (+20.7) | 0.57 (+.21) | 0.59 (+.05) | 32.1 |
| MIPROv2 | 69.0 (+11.0) | 14.0 (+8.0) | 73.1 (+9.4) | 0.60 (+.24) | 0.59 (+.05) | 48.6 |
| GEPA | 66.0 (+8.0) | 15.0 (+9.0) | 70.8 (+7.1) | 0.69 (+.33) | 0.76 (+.22) | 61.5 |
| ACE | 71.0 (+13.0) | 18.0 (+12.0) | 79.2 (+15.5) | 0.65 (+.29) | 0.68 (+.14) | 70.7 |
| MCE | 75.0 (+17.0) | 20.0 (+14.0) | 89.2 (+25.5) | 0.70 (+.34) | 0.80 (+.26) | 89.1 |
| Online Setting | | | | | | |
| DC | 61.0 (+3.0) | 14.0 (+8.0) | 73.1 (+9.4) | 0.46 (+.10) | 0.53 (-.01) | 35.8 |
| ACE | 64.0 (+6.0) | 13.0 (+7.0) | 62.3 (-1.4) | 0.63 (+.27) | 0.57 (+.03) | 41.1 |
| MCE (w/o skills) | 67.0 (+9.0) | 18.0 (+12.0) | 76.9 (+13.2) | 0.70 (+.34) | 0.68 (+.14) | 71.3 |
| MCE | 68.0 (+10.0) | 20.0 (+14.0) | 76.4 (+12.7) | 0.66 (+.30) | 0.63 (+.09) | 74.1 |

*Table 2: Strong-to-weak context transferability. We train contexts using DeepSeek-V3.1 as the generator and transfer them to smaller models. Results show performance across three benchmarks (Aegis2 excluded as it already uses Qwen3-8B; USPTO50k excluded as both MCE and ACE produce contexts that exceed smaller models’ effective context length, and the task complexity prevents smaller models from producing valid answers). Avg. Rel. Drop measures the mean relative performance degradation when transferring from DeepSeek-V3.1 to smaller models within the same method. Best results are highlighted.*

| Model | Method | FiNER | Symptom2Disease | LawBench | Avg. Rel. | Avg. Rel. |
| --- | --- | --- | --- | --- | --- | --- |
|  |  | Acc.%$\uparrow$ | Acc.%$\uparrow$ | Micro-F1$\uparrow$ | Gain%$\uparrow$ | Drop%$\downarrow$ |
| DeepSeek-V3.1 | Base Model | 58.0 | 63.7 | 0.36 | – | – |
| | ACE | 71.0 (+13.0) | 79.2 (+15.5) | 0.65 (+.29) | 42.4 | – |
| MCE | 75.0 (+17.0) | 89.2 (+25.5) | 0.70 (+.34) | 54.6 | – |
| Llama3.3-70B | Base Model | 56.0 | 65.1 | 0.24 | – | – |
| | ACE | 71.0 (+15.0) | 68.4 (+3.3) | 0.19 (-.05) | 3.7 | 28.1 |
| MCE | 74.0 (+18.0) | 82.1 (+17.0) | 0.27 (+.03) | 23.6 | 23.6 |
| Qwen3-8B | Base Model | 59.0 | 65.6 | 0.27 | – | – |
| | ACE | 63.0 (+4.0) | 72.2 (+6.6) | 0.32 (+.05) | 11.8 | 23.6 |
| MCE | 71.0 (+12.0) | 80.2 (+14.6) | 0.45 (+.18) | 36.4 | 17.1 |
| Gemma3-4B | Base Model | 17.0 | 51.9 | 0.01 | – | – |
| | ACE | 64.0 (+47.0) | 51.4 (-0.5) | 0.00 (-.01) | 58.5 | 48.3 |
| MCE | 65.0 (+48.0) | 70.3 (+18.4) | 0.03 (+.02) | 172.6 | 43.4 |

<img src='x3.png' alt='[Uncaptioned image]' title='' width='317' height='225' />

*Figure 3: Context efficiency on FiNER. We plot context accuracy vs. context tokens. For MCE, we include the context generated in two iterations. For ACE, we include context at Step 20 of Epoch 1, end of Epoch 1, 2, and 5.*

<img src='x4.png' alt='[Uncaptioned image]' title='' width='317' height='224' />

*Figure 4: Training efficiency on FiNER. We plot best-so-far training set accuracy vs. the number of rollouts (for MCE, this includes both training and validation inference). We also indicate the total training duration.*

#### 4.2.2 Context Adaptability, Efficiency, and Transferability

Obs.❺ MCE is free from the inductive biases in context length imposed by prior methods. Prior CE methods exhibit inherent biases: GEPA *[agrawal2025gepa]* tends toward brevity, typically producing around 1–2K tokens, while ACE *[zhang2025agentic]* suffers from context bloat, reaching up to 80K tokens after 5 epochs of optimization (when using 200 training instances).
In contrast, MCE learns to produce context of suitable length for each task. The two most effective contexts on FiNER have 1.5K and 20K tokens ([Fig.˜3]), while those on LawBench and USPTO50k reach 44K and 86K tokens.
These results demonstrate that MCE adapts context length flexibly to task requirements, overcoming the inductive biases of prior methods.

Obs.❻ MCE produces more efficient contexts than ACE.[Fig.˜3] compares context efficiency on FiNER.
At comparable context lengths ($\sim$1.5K tokens), MCE-S achieves 73% accuracy versus 65% for ACE (Step 20).
Moreover, MCE-L reaches 75% accuracy with only 20K tokens, outperforming ACE even after 5 epochs of optimization (70% at 79K tokens).
We attribute MCE’s superior context quality to two factors: (1) the agent maintains a *global view* of accumulated context, enabling it to restructure and refine existing knowledge rather than blindly appending new items; and (2) MCE agentically processes *large batches* of training rollouts, aggregating feedback across many examples before updating the context.
Together, these properties yield coherent, non-redundant contexts, avoiding the redundancy and degradation observed in ACE’s additive curation.

Obs.❼ MCE contexts transfer better from strong to weak models. A practical desideratum for CE systems is the ability to transfer learned contexts from strong models to weaker ones, enabling a form of knowledge distillation.
As shown in [Table˜2], when transferring DeepSeek-V3.1-trained contexts to smaller models, MCE consistently exhibits lower performance degradation than ACE.
In contrast, ACE’s transferred contexts occasionally degrade performance below the base model (e.g., LawBench F1 decreases from 0.24 to 0.19 on Llama3.3-70B).
We attribute MCE’s superior transferability to: (1) context efficiency—smaller LLMs struggle with long-context processing, and MCE’s compact contexts mitigate this; and (2) generalizability—MCE’s fully agentic, batch-level optimization produces well-structured contexts less dependent on the training model, whereas ACE’s rollout-specific curation may overfit to the error patterns of the training model.

#### 4.2.3 Training Efficiency

We investigate the training efficiency of MCE on the FiNER benchmark ([Fig.˜4]).

Obs.❽ MCE significantly reduces training duration. On FiNER, MCE completes 5 training epochs in 1.9 hours versus 25.8 hours for ACE, which is a $13.6\times$ speedup. This speedup stems from MCE’s batch-level optimization: depending on the skill, the base agent either directly analyzes training data to curate context (reading incorrect predictions, identifying patterns, and updating files without heavy LLM scaffolding) or writes code to parallelize reflection and curation across training rollouts. This contrasts with ACE’s instance-by-instance optimization.

Obs.❾ MCE is rollout-efficient. The same architectural properties that improve context quality (global context view and batch-level optimization) also accelerate convergence.
On FiNER, MCE requires only 450 rollouts to reach 95% training accuracy, while ACE peaks at 94% after 2169 rollouts ($4.8\times$ fewer).

### 4.3 Ablation Studies

*Table 3: Ablating the bi-level design of MCE on FiNER.*

| Method | Offline | Online |
| --- | --- | --- |
| Base Model (zero-shot) | 58.0 | |
| ACE | 71.0 (+13.0) | 64.0 (+6.0) |
| MCE (w/o skills) | 73.0 (+15.0) | 67.0 (+9.0) |
| MCE (w/ a fixed skill) | 71.0 (+13.0) | 68.0 (+10.0) |
| MCE (full, w/ evolving skills) | 75.0 (+17.0) | - |

*Table 4: ACE performance with different reflector/curator models on FiNER. MiniMax M2.1 generates more verbose playbooks but achieves lower accuracy.*

| Checkpoint | DeepSeek V3.1 | | MiniMax M2.1 | |
| --- | --- | --- | --- | --- |
| | # Tokens | Acc (%) | # Tokens | Acc (%) |
| Epoch 1 | 12K | 71 | 38K | 69 |
| Epoch 2 | 23K | 69 | 90K | 69 |
| Final | 79K | 70 | 114K | 67 |

##### Ablating the Bi-level Design ([Table˜3]).

We conduct ablation studies on FiNER to validate MCE’s bi-level design. We compare three variants: (1) w/o skills: the base-agent operates without any skill guidance; (2) w/ a fixed skill: the base-agent follows a single skill generated by the meta-agent from task specifications alone, without iterative evolution; and (3) full: the complete MCE with evolving skills. Note that the full version is inapplicable to the online setting, as single-pass processing precludes iterative skill evolution.

The results reveal three findings. First, meta-level skill evolution provides a clear boost: full MCE achieves 75% versus 73% for the skill-less variant in the offline setting. Second, even without skill guidance, the base-agent alone outperforms ACE (73% vs. 71% offline, 67% vs. 64% online), indicating that fully agentic CE can be effective without manual scaffolding. Third, fixed skills exhibit high variance across tasks (see also [Table˜1]), as their quality depends on task specification alone without learning from validation performance.

##### Ruling Out Model Confounds ([Table˜4]).

MCE uses MiniMax M2.1 as the agentic model, raising the question of whether performance gains stem from superior model capabilities rather than MCE’s methodology. To address this, we replace the reflector and curator in ACE with MiniMax M2.1 and compare against the default DeepSeek V3.1.

The results show that MiniMax M2.1 degrades ACE’s performance.
Despite generating more verbose playbooks (114K vs. 79K tokens at completion), MiniMax M2.1 achieves lower final accuracy (67% vs. 70%). Training also terminates earlier (epoch 3, step 73) as the playbook exceeds the context window. This rules out knowledge transfer from the agentic model as the source of MCE’s gains. We conclude that MCE’s advantages stem from its methodological design, though these advantages require an agentic model capable of interacting with the programming environment, and producing valid files and code.

5 Conclusion and Discussion
---------------------------

This work presents Meta Context Engineering (MCE), a bi-level optimization framework that advances beyond static context representations and optimization procedures in prior CE methods.
MCE introduces learnable CE skills, represents context as files and code, employs fully-agentic bi-level optimization, orchestrates dual agents under an evolutionary framework, and co-evolves CE skills and context artifacts.
Our experiments across five domains demonstrate MCE’s consistent superiority.
MCE achieves 5.6–53.8% relative improvement over SOTA methods, with a mean gain of 16.9%. Additionally, MCE exhibits superior context adaptability, efficiency, and transferability, as well as substantial training speedups. These results validate that treating CE as a learnable agentic capability, rather than a fixed workflow with predefined schemas, unlocks a generic design space for optimizing agentic AI systems.

Limitations. MCE is particularly advantageous for tasks centered on domain knowledge acquisition and pattern matching, as its learnable skills effectively capture data characteristics and organize domain-specific structures. However, MCE may not offer advantages on reasoning-intensive tasks, as existing manually crafted agentic harnesses (characterized by iterative trials, error correction, and systematic reflection) are already well-suited for such problems. Second, MCE may struggle in scenarios where rollouts involve very long and complex trajectories. Such settings demand fine-grained credit assignment and detailed trajectory analysis, which batch-level fully agentic CE may fail to perform effectively. We note that this limitation stems from the capabilities of the underlying agent model rather than the MCE framework itself; as agentic models continue to improve, this constraint is expected to diminish.
We expect MCE to scale better with the advancement of agentic models.

Future Work. We identify three meaningful directions for future research.
First, the agentic skill evolution paradigm extends beyond CE. Prior evolutionary approaches target specific solutions, heuristic code, or search algorithms; MCE instead evolves skills, a higher-order and integrated abstraction essential for general AI. While static skills are well-recognized *[anthropics_skills, muratcankoylan_agent_skills_context_engineering]*, MCE is among the first to dynamically evolve them, bridging manual skill engineering and autonomous self-improvement. The paradigm of evolutionary skills could generalize to other agentic capabilities and task domains.
Second, our generator currently uses one-shot inference for consistency with CE baselines. Since MCE stores context as files, an agentic generator that interacts directly with these artifacts could be more effective. Extending MCE to co-evolve context utilization skills alongside context learning skills is a promising direction.
Third, progressive disclosure is native to agent skills: agents load skill details into context only when relevant. This enables MCE to compose and scale skills across domains with minimal overhead. Investigating skill transfer across tasks and emergent behaviors from skill composition are interesting open questions.

Taken together, MCE reformulates CE as a learnable agentic capability, unlocking a generic design space for optimizing general AI. We envision agents that not only execute tasks but continuously refine their learning algorithms and memory architectures, enabling open-ended evolution.

References
----------

Meta Context Engineering via Agentic Skill Evolution  
(Appendix)

Appendix A Experimental Details
-------------------------------

This section provides experimental details for the benchmarks. We also present task specifications for our bi-level agents and generator prompt templates used across all methods.

### A.1 FiNER

We utilize the FiNER dataset from the ACE official repository222[https://github.com/ace-agent/ace/tree/main/finance/data](https://github.com/ace-agent/ace/tree/main/finance/data ""). To create a challenging evaluation benchmark within our computational budget, we randomly sample a subset focusing on financial tags related to debt instruments, credit facilities, and loan-related financial reporting, with train/validation/test splits of 200/100/100 instances. Each instance is formulated as a named entity recognition task where the model must predict the appropriate financial tag for a given entity within its sentence context. Specifically, each query follows the format: “What is the best tag for entity ‘<value>’ in sentence: ‘<sentence>’?”, where the ground truth is the corresponding tag name.

The subset encompasses 12 debt and credit-related financial tags from the XBRL taxonomy:

1. 1.

    DebtInstrumentInterestRateStatedPercentage

2. 2.

    DebtInstrumentFaceAmount

3. 3.

    LineOfCreditFacilityMaximumBorrowingCapacity

4. 4.

    DebtInstrumentBasisSpreadOnVariableRate1

5. 5.

    DebtInstrumentCarryingAmount

6. 6.

    DebtInstrumentRedemptionPricePercentage

7. 7.

    LongTermDebtFairValue

8. 8.

    LongTermDebt

9. 9.

    LettersOfCreditOutstandingAmount

10. 10.

    LineOfCredit

11. 11.

    LineOfCreditFacilityCurrentBorrowingCapacity

12. 12.

    DebtInstrumentUnamortizedDiscount



### A.2 USPTO-50k

We utilize the USPTO-50k dataset *[schneider2016s]*, a widely-used benchmark for single-step retrosynthesis prediction in computational chemistry. Given a target product molecule represented in SMILES notation, the task requires predicting the precursor reactants needed to synthesize it. This task evaluates the model’s understanding of organic reaction mechanisms and chemical transformation patterns.

We follow *cai2025flex* in data preprocessing. We randomly sample 50 instances from the original training split for training, 30 instances from the training split for validation, and 100 instances from the test split for evaluation. Stratified sampling is used to uniformly cover all 10 reaction types, ensuring balanced representation across different chemical transformation classes.



### A.3 Symptom2Disease

We utilize the Symptom2Disease dataset from Hugging Face333<https://huggingface.co/datasets/gretelai/symptom_to_diagnosis>. Each instance consists of a natural language description of patient symptoms, and the task is to predict the correct medical diagnosis from 22 possible disease categories.

We preserve the original test split (212 samples) and perform stratified sampling on the original training split to create train/validation splits of 200/50 instances, maintaining balanced representation across disease categories. The disease categories include: diabetes, dengue, chicken pox, allergy, impetigo, arthritis, gastroesophageal reflux disease, typhoid, cervical spondylosis, hypertension, malaria, pneumonia, psoriasis, peptic ulcer disease, drug reaction, bronchial asthma, urinary tract infection, common cold, varicose veins, fungal infection, jaundice, and migraine.



### A.4 LawBench

We utilize the criminal charge prediction task from LawBench444[https://github.com/open-compass/LawBench](https://github.com/open-compass/LawBench ""), a comprehensive Chinese legal benchmark. Given case facts including prosecution descriptions, evidence summaries, and procedural information, the model must predict the applicable criminal charges. This task is challenging because (1) cases may involve multiple concurrent charges requiring comprehensive legal analysis, and (2) distinguishing between similar charges (e.g., theft vs. robbery, fraud vs. contract fraud) requires nuanced understanding of Chinese criminal law.

We randomly sample 200/50/100 instances for train/validation/test splits from the original dataset. Each instance follows the format: the model receives case facts and must output charges in the format “[罪名]charge1;charge2<eoa>”. Evaluation uses micro-F1 score to account for partial credit on multi-charge cases, following the original benchmark.



### A.5 Aegis2

We utilize the AEGIS2 dataset *[ghosh2025aegis2]*, a comprehensive AI safety dataset designed for training lightweight guardrail models. Each instance contains a user prompt with its corresponding safety label (safe/unsafe) and violated risk categories. The task is binary classification of user prompts, with unsafe as the positive class and F1 score as the evaluation metric.

To identify challenging categories, we evaluate LLaMA-3-SafeGuard *[dubey2024llama3herdmodels]* across all 24 risk subcategories using the system prompts specified in the original paper. Based on per-category F1 scores, we selecte the four worst-performing categories: Copyright_Trademark_Plagiarism, Political_Misinformation_Conspiracy, Unauthorized_Advice, and Immoral_Unethical.

For data splits, we include all 70 unsafe samples from the four selected categories in the test set, balanced with 70 randomly sampled safe instances (140 total). The validation set contains 64 unsafe samples from the original validation split and 64 safe instances (128 total). For training, we sample 75 instances from each of the four unsafe categories (300 total) and 100 safe instances, yielding 400 training samples.



Appendix B Prompts
------------------



Appendix C Learned Skills
-------------------------

To give an example of the learned skills, we present the best learned skill for the FiNER benchmark below.
To avoid an overly lengthy appendix, the complete collection of learned skills is available in the assets of our repository.

The learned skill provides an 8-phase systematic approach to context refinement. It organizes context into three files: reasoning-chains.md, semantic-principles.md, and tag-reference.md.
Notably, it orchestrates a workflow of three LLM calls to transform prediction errors into generalizable reasoning principles (see ## Implementation Guidance section): 1) using LLM for error analysis, 2) using LLM to generalize specific rules, and 3) using LLM to create reasoning chains.


Appendix D Case Studies and Analysis
------------------------------------

This section analyzes MCE’s performance compared to baselines across tasks, examines the skill evolution process, and overviews the optimal skills and context learned by MCE.

### D.1 FiNER

#### D.1.1 Task Characteristics and Baseline Limitations

FiNER presents a unique challenge that demands deep reflection, pattern abstraction, and exceptional domain expertise with acute sensitivity to nuanced rules. Its bottleneck lies in mastering high-density long-tail rules and leveraging extensive knowledge repositories. The critical factors for success are rule coverage and disambiguation logic.

Imitating the training examples alone is not enough; deep reflection and pattern abstraction beyond the training instances are required. Therefore, for this task, ICL, MIPROv2, and GEPA struggle to perform well. ICL and MIPROv2 rely on learning from training set demos, while GEPA fails because it compresses nuances into overly concise prompts. ACE excels because it leverages repetitive reflections to accumulate extensive knowledge and patterns in the playbook.

Unlike ACE’s monolithic playbook accumulation approach, MCE develops structured, hierarchical context files guided by evolving skills that progressively shift from generic methodologies to error-driven generalization.

#### D.1.2 Skill Evolution

The initial skill focuses on a systematic, phase-based approach for context curation:


After several iterations of meta-agent refinement based on validation performance feedback, the optimal skill shifts its core philosophy from pattern extraction to error-driven generalization:


The optimal skill introduces explicit anti-patterns, documenting what causes errors rather than just what the correct answers are:


#### D.1.3 Context Evolution

The curated context evolves from simple tag definitions to comprehensive disambiguation guides with explicit reasoning chains. The final context is organized into three specialized files:

1. 1.

    tag-reference.md: Per-tag decision rules with positive and negative indicators

2. 2.

    semantic-guidance.md: Critical distinctions for error prevention with decision trees

3. 3.

    common-patterns.md: 30+ specific error patterns with WRONG $\rightarrow$ RIGHT corrections

A key example of the evolved context’s sophistication is the facility type decision chain:


The retrieval function for FiNER uses the default full retrieval strategy that returns all context concatenated, which was found to be the most effective approach for this task.

### D.2 USPTO50k

#### D.2.1 Task Characteristics and Baseline Limitations

The USPTO50k task combines logical reasoning with deep domain knowledge retrieval.
In-Context Learning struggles with the vast chemical reaction space. Length constraints in MIPROv2 force trade-offs, typically favoring generic instructions while discarding rare reaction-specific rules critical for this hard-logic task. GEPA suffers from inherent brevity bias, problematic for USPTO where corner cases determine success. Its preference for conciseness systematically omits the detailed rules needed for edge scenarios. The incremental context updates and accumulation strategy of ACE are more effective for this task.

#### D.2.2 Skill Evolution

The initial skill adopts a comprehensive, taxonomy-driven approach to context building:


The optimal skill shifts focus from comprehensive coverage to differential learning, emphasizing both success pattern mining and failure pattern analysis:


A key innovation in the optimal skill is the explicit focus on workflow-oriented context rather than encyclopedic information:


#### D.2.3 Context Evolution

The curated context evolves into a structured knowledge system with multiple specialized components:

1. 1.

    workflow.md: Step-by-step decision process with embedded verification checks

2. 2.

    QUICK_REFERENCE.md: Critical error patterns distilled from training failures

3. 3.

    SUCCESS_PATTERNS.md: Validated patterns with 100% training accuracy

4. 4.

    Reaction-type guides: Individual files for each reaction class (cc_bond_formation.md, deprotections.md, etc.)

A distinguishing feature of the USPTO context is the integration of inline verification checkpoints within the workflow:


The context also explicitly documents success patterns that achieved 100% accuracy, providing positive exemplars:


This dual documentation of both error patterns (what to avoid) and success patterns (what to replicate) enables more robust generalization compared to error-only approaches. The workflow-centric organization ensures the base agent follows a systematic decision process rather than relying on pattern memorization, which is critical for the combinatorial complexity of chemical reaction space.

The retrieval function for USPTO50k uses the default full retrieval strategy that returns all context concatenated, which was found to be the most effective approach for this task.

### D.3 Symptom2Disease

#### D.3.1 Task Characteristics and Baseline Limitations

Symptom2Disease presents a fine-grained classification challenge (22 disease categories) where the core challenge lies in accurately mapping natural language symptom descriptions to specific medical labels.
Unlike FiNER and USPTO, this task exhibits minimal train-test distributional shift and does not require deep abstraction or multi-step reasoning—symptom descriptions in the test set closely mirror those in training, and classification relies primarily on pattern matching rather than logical inference. Consequently, many-shot ICL emerges as the strongest baseline: by providing 8–10 examples per disease category, ICL achieves sufficient distribution coverage for effective nearest-neighbor matching in the semantic space.

This property inverts the usual baseline hierarchy. Methods that attempt to inject abstraction and deep reflection—GEPA and ACE—underperform on this task. GEPA’s brevity bias causes context collapse, abstracting specific symptoms (e.g., “burning sensation” for peptic ulcer vs. “chest pain” for heart disease) into overly general rules that lose fine-grained distinctions. ACE’s monolithic playbook accumulation introduces noise: abstract heuristics can override the model’s pre-trained medical knowledge, and in the online setting, early misclassifications become entrenched and propagate errors.

Conversely, methods that leverage more examples perform better. DC achieves strong results by performing dynamic many-shot ICL with semantic retrieval, a natural fit for this task because semantic similarity directly corresponds to pragmatic similarity when classifying symptom descriptions. MIPROv2 also benefits from preserving distributional features through example selection rather than over-compressing knowledge into abstract instructions.

#### D.3.2 Skill Evolution

The initial skill adopts a three-phase approach: Pattern Extraction, Profile Synthesis, and Error-Driven Refinement:


The optimal skill represents a dramatic philosophical shift. Crucially, it learns from prior iterations that the current architecture is already near-optimal—explicitly referencing “88% val, 1% gap” as a baseline to preserve. The skill evolves from “build comprehensive context” to “conservative refinement with evidence-based addition”:


A key innovation is the introduction of explicit stop criteria—the skill recognizes when further refinement is counterproductive:


This represents a meta-learning insight: the skill has learned from iteration history that knowing when to stop is as important as knowing what to add.

#### D.3.3 Context Evolution

The context evolves into a comprehensive diagnosis guide organized around error prevention. Unlike the encyclopedic approach, the final context prioritizes decisive discriminators, which are rules that resolve specific confusion patterns:


The context also captures semantic symptom “essences”—generalized patterns that transcend specific wording:


The evolution demonstrates that for tasks with minimal distributional shift, the optimal strategy is conservative: preserve validated patterns, add discriminators only with strong evidence, and recognize when remaining errors are irreducible ambiguities rather than fixable gaps.

For symptom2disease, MCE retrieval function evolves into a sophisticated 1,440-line rule-based routing system. This extensive retrieval logic reflects the task’s fine-grained classification nature: different symptom combinations require different discriminators, and retrieving irrelevant rules can introduce noise.

The retrieval function implements a prioritized cascade of symptom pattern detectors, each triggering retrieval of specific context sections:


Each rule performs keyword-based symptom detection and returns only the relevant context sections:


The retrieval function also encodes learned discriminators for ambiguous cases:


This retrieval design embodies MCE’s core insight for this task: rather than accumulating monolithic context (like ACE’s playbook), the system learns when to apply which discriminators. The 1,440-line retrieval function effectively encodes a decision tree that routes each query to its most relevant context subset, preventing context interference while ensuring precise guidance for each symptom pattern.

### D.4 LawBench

#### D.4.1 Task Characteristics and Baseline Limitations

Criminal charge prediction task in LawBench presents a demanding legal classification task that combines three challenging properties: (1) long-context with detail sensitivity: inputs are detailed case fact descriptions where subtle distinctions determine the correct charge; (2) strict logical reasoning: crime determination depends on precise legal elements (e.g., distinguishing “theft” from “robbery” requires analyzing whether force or threat was used); and (3) high-precision classification: outputs are fixed crime labels from a closed set, demanding accurate matching between case facts and legal definitions.

This task inverts the baseline hierarchy observed in FiNER and USPTO. Here, ACE’s monolithic playbook accumulation strategy becomes a liability rather than an asset. In legal scenarios, ACE accumulates extensive bullets from historical errors, but when these bullets are retrieved and concatenated into context, they introduce context interference, overwhelming the current case’s key information with tangentially related past patterns. For instance, when classifying a “smuggling of ordinary goods” case, ACE geneartor may refer to strategies for “smuggling weapons” and “smuggling cultural artifacts,” which are noise that dilutes attention from the decisive factors. The model’s attention becomes dispersed across accumulated heuristics, causing it to overlook small but legally decisive details in the case facts.

Conversely, GEPA’s “brevity bias”—typically a weakness—becomes an advantage for this classification task. Rather than accumulating case-specific patterns, GEPA evolves a single, globally optimized instruction that provides clear procedural guidance for legal reasoning. Through Pareto optimization across the entire dataset, GEPA discovers prompts that generalize robustly across different case types without overfitting to specific examples. The evolved prompt structures the reasoning process explicitly: (1) comprehensive fact analysis, (2) strict format requirements for crime labels, (3) common error avoidance checklist, and (4) mandatory reasoning-before-answer workflow. This concise, globally-tuned instruction proves more effective than ACE’s playbook because it directs the model to focus on the current case’s facts and apply systematic legal reasoning, rather than pattern-matching against potentially misleading historical examples.

#### D.4.2 Skill Evolution

The initial skill adopts a pattern-based approach focused on charge definitions and distinction knowledge:


After several iterations of meta-agent refinement, the skill undergoes a fundamental philosophical shift. The meta-agent observes that pattern-based learning causes overfitting. Even “simplified” rules like “if property taken secretly $\rightarrow$ 盗窃” cause memorization because they match surface patterns rather than requiring deep analysis. The optimal skill introduces structural case decomposition:


A key innovation is the introduction of processing stages for error diagnosis. The skill categorizes by where in the processing pipeline the error occurred, rather than categorizing errors by charge type:


This represents a meta-learning insight: the skill learns that how errors occur matters more than which charges are confused.

#### D.4.3 Context Evolution

The context evolves into a structurally-organized system with 11 specialized files totaling approximately 30KB (reduced from 88KB in earlier iterations). Unlike pattern-based approaches, the context teaches a systematic processing framework:


The context also includes structural anti-patterns, explicit documentation of processing errors to avoid:


The retrieval function (177 lines) is notably simpler than symptom diagnosis (1,440 lines), reflecting the task’s different requirements. Rather than complex routing logic, it implements a trigger-based augmentation strategy:



This design reflects the task’s nature: legal classification requires systematic structural analysis of case facts rather than semantic pattern matching. The retrieval function ensures the model receives consistent structural guidance plus case-specific warnings, enabling generalization through process, in contrast to pattern accumulation observed in ACE.

### D.5 Aegis2.0

#### D.5.1 Task Characteristics and Baseline Limitations

Aegis2.0 requires classifying user prompts into five categories: Copyright Trademark Plagiarism, Political Misinformation Conspiracy, Unauthorized Advice, Immoral Unethical, and safe. We deploy Qwen3-8B as the generator for this task since safety guardrails require small models for efficient inference. This requirement poses unique challenges due to limited reasoning capacity and context window constraints of small models. Therefore, we observe that CE methods with brevity bias, such as GEPA, can outperform the others.

#### D.5.2 Skill Evolution

The initial skill adopts a meta-learning error synthesis approach, categorizing errors by root cause and synthesizing generalized patterns:


After iterations, the skill evolves to emphasize balanced precision control. The meta-agent discovers that small models are particularly prone to over-classification (flagging benign prompts as unsafe), and shifts focus to surgical fixes with explicit precision/recall trade-offs:


A key innovation is the introduction of critical distinctions that small models struggle with:


#### D.5.3 Context Evolution

The context evolves into 12 specialized files organized around violation categories and safe exclusions. The context emphasizes decision rules over examples, optimized for small model reasoning capacity.


The retrieval function (995 lines) implements a precision-first cascade with early safe returns for benign patterns:


The retrieval function’s early safe return mechanism is critical for preventing over-classification:


This design reflects the unique challenges of safety classification with small models: the retrieval function acts as a precision gate, preventing the model from seeing violation-related context when the prompt is clearly benign. This architectural choice reduces over-classification by ensuring the model only receives category-specific rules when patterns genuinely warrant scrutiny.

Appendix E Utility Functions
----------------------------

The utility functions to call LLMs (fixed to DeepSeek V3.1) and embedding models (fixed to OpenAI’s text-embedding-3-small) are provided below.
