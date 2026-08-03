---
title: "FORT-Searcher: Shortcut-Resistant Search Task Synthesis"
topic: deep-search
summary: "Realized search difficulty is governed by the cheapest identifying route, so FORT builds training tasks that suppress four shortcut risks instead of merely enlarging evidence graphs."
lang: en
updated: 2026-08-03
order: 1
sources:
  - title: "FORT-Searcher: Synthesizing Shortcut-Resistant Search Tasks for Training Deep Search Agents"
    url: "https://arxiv.org/abs/2606.12087"
raw:
  - raw/deep-search/2026-06-10-fort-searcher.md
---

## Overview

Training a deep search agent requires verifiable questions whose answers remain unavailable until enough evidence has been acquired. Most synthesis pipelines pursue this by enriching the intended solving structure, using more hops, richer graph shapes, hierarchical constraints, or greater evidence dispersion. This paper argues that structural complexity does not automatically transfer into realized search difficulty, because the intended search process can collapse through a cheaper identifying route.

FORT inverts the analysis: it takes the known collapse modes and turns them into construction-time controls applied while the task is being built. FORT-Searcher is the agent trained on the resulting trajectories using supervised fine-tuning only.

## Apparent Difficulty Is Not Realized Difficulty

A task instance is written as a triple of an answer space, the constraint set expressed by the question, and the retrieval interface. The question is well posed, so the full constraint set determines a unique gold answer.

The decisive observation is that a solver does not need to verify every clue. It only needs evidence for some *identifying subset*, meaning any subset of constraints that already narrows the candidate pool to the gold answer alone. Difficulty is therefore governed by the cheapest route that verifies such a subset, not by the route the task designer imagined. The paper writes this structural floor as the minimum route cost over all identifying subsets, and shows that the cost paid by an idealized no-prior, no-guessing solver is bounded below by it.

The bound is one-sided and behaves like a weakest-link property. If even one identifying subset happens to be shallow and concentrated, the whole question becomes easy regardless of how large the underlying evidence graph was. In the extreme case where some identifying subset can be verified by a single initially executable query, the structural floor collapses to a single retrieval step.

## Factors Behind the Bound

Three objective quantities control the floor and the gap above it, and a fourth quantity is specific to the model being used.

- **Subset selectivity** is the size of the candidate pool left after applying only part of the clues. It does not lower-bound route length directly. Instead it sets the exploration gap, and when a small subset becomes selective enough to identify the answer, it determines which subsets enter the minimization at all.
- **Evidence dispersion** is the minimum number of retrievals needed to verify that the gold answer satisfies a given clue subset, ignoring query executability. When one web page states two intended facts, a single evidence-acquisition step suffices and the count drops to one.
- **Dependency depth** is the length of the longest serial chain among queries, where a later query needs a name or intermediate fact that only earlier retrieval can expose. It arises purely from query executability.
- **Solver-side cost reduction** is the amount a concrete model saves relative to the no-prior reference solver, typically by recognizing the target from parametric memory.

The route cost for a fixed identifying subset is at least the larger of evidence dispersion and dependency depth, and the realized cost of a concrete solver is the no-prior cost minus its solver-side reduction. Because each gap is attributed to a named factor, the four together account for the difficulty.

## Four Shortcut Risks

A shortcut is any mechanism that lets a question be solved with less evidence acquisition than intended. The first three operate on the structural floor; the fourth operates on the model-specific reduction.

- **Single-clue selectivity.** One clue, or a small subset, narrows the candidates to one or a few. Even when the subset is not strictly identifying, high selectivity surfaces the answer early within finite retrieval results. In one diagnostic case a question converged four independent threads onto a single mid-century year, but the solver issued a query built only from the color-television thread and immediately treated the retrieved year as its main candidate, leaving the other threads as post-hoc verification.
- **Evidence co-coverage.** A single retrieved page or snippet verifies several intended constraints at once, compressing multiple planned steps into one retrieval. In the corresponding case, one search-result snippet simultaneously exposed the answer entity and several answer-side facts about its industry and regional holdings.
- **Exposed constants.** The question surface reveals an exact name, string, date, or number that should have been discovered by earlier retrieval, making downstream queries executable from the start and shortening the serial dependency. The diagnostic case embedded a distinctive public quotation from the target person into the question; the model reused the phrase almost verbatim and the results named the answer directly. What matters is that the exposed string was a unique attribute of the target rather than a generic intermediate clue.
- **Prior-knowledge binding.** The solver commits to the gold answer before retrieved evidence anchors it. This one does not reduce the structural floor at all, so a task can require a long evidence route for a no-prior solver and still be trivial for a model that already knows the entity. In the diagnostic case the model named the answer before any tool call, then retrieved evidence only afterwards.

## Trajectory Signatures

The theoretical quantities cannot be computed at scale in open-domain web search, since that would require enumerating the answer space, all identifying subsets, and all valid routes. The paper therefore diagnoses realized difficulty with three observable signatures measured under a fixed solver and retrieval budget.

- **Realized solving cost** is the average number of retrieval queries over successful trajectories. On its own it proves little, because a long trajectory may simply contain detours.
- **Answer hit time** is the first step at which the gold answer or a normalized alias appears, taking whichever comes first between the retrieved observations and the model's own visible text. A later hit time means a longer pre-answer search prefix, which is the behavior expected when cheap identifying routes have been suppressed.
- **Prior-shortcut rate** is the fraction of successful trajectories in which the model mentions the answer before any retrieval anchors it. The paper describes this as a conservative proxy, since it only captures visible answer-before-evidence behavior.

The gap between solving cost and answer hit time is the diagnostic of interest. Re-evaluating six open-source deep search datasets with the same strong agent under the same budget shows that apparent trajectory length is not the same as search-heavy supervision. OpenSeeker reaches a solving cost of 84.7 while its answer hit time is only 9.3 and its prior-shortcut rate reaches 31.9. REDSearcher has the strongest signatures among the open-source baselines, with cost 92.1 and hit time 18.7, yet its answer still surfaces far earlier than its total cost. FORT reaches cost 141.0 with hit time 46.9 while keeping its prior-shortcut rate at 11.0, indicating that the added cost is not driven by more prior-bound behavior.

## FORT: Turning the Analysis into Construction Controls

The difficulty framework is defined over finished questions, but synthesis must intervene before the question is verbalized. FORT uses an internal evidence graph as a construction workspace, where nodes are real-world entities and edges are verified facts that can later be phrased as clues. Selecting and verbalizing subgraphs from this workspace is what lets the pipeline control selectivity, dispersion, and dependency depth. The pipeline has four stages, each targeting specific risks.

**Graph initialization** selects a long-tail root entity from Wikidata under three criteria: topical coverage across broad semantic categories with abstract concepts removed, entity obscurity with a preference for rare entities lacking an English Wikipedia page, and sufficient information density confirmed by a lightweight pre-search so the task stays solvable. It then seeds the graph with a pre-mined cycle rather than a linear chain. The reasoning is that a purely linear seed tends to force downstream entity names into the question surface, which is exactly the exposed-constant failure; a cycle embeds the root in a closed local structure so relational clues can be phrased without naming every intermediate. Cycles are pre-mined into an entity-to-cycle inverted index, with duplicate node sets removed and hub-based, redundant, or mass-collaboration structures filtered out.

**Graph construction** expands the seed under a depth limit and a node budget, always expanding the deepest unprocessed node first so that multi-step referenced-entity chains survive to become serial dependencies. An enricher agent collects atomic facts from heterogeneous sources including Wikidata, open web pages, structured databases, Google Scholar, and Google Maps, deliberately avoiding drawing multiple selected facts from the same evidence item. It also constructs derived facts, which combine several atomic facts and are therefore unlikely to appear verbatim in any single retrieved item; the four constructors are coincidence bridging, count aggregation, numerical relation, and meta-fact extraction. Facts are then checked for source consistency and entity consistency, the latter guarding against similar names, abbreviation ambiguity, temporal or geographic drift, and series-versus-edition mismatches. Finally an expander agent prefers *generic* facts over representative ones. This is the counterintuitive step: a highly characteristic fact makes a single clue too selective, so the pipeline favors facts that are reliable but weak in isolation and identifying only in combination.

**Question formulation** picks the answer node, prunes redundant facts, and keeps clues that are jointly identifying while individually generic. Intermediate node names are withheld and rendered as generic referring expressions so downstream queries cannot execute from the initial question. Remaining literals are passed through exact-value fuzzing, which rewrites them into truthful but less directly searchable constraints using category generalization, range relaxation, meta-attribute description, arithmetic encoding, or contrastive exclusion. Fuzzing is explicitly not meant to make the question ambiguous or unverifiable; the gold answer must be preserved.

**Adversarial refinement** runs a strong adversary agent against each draft in a realistic search setting and inspects the per-trajectory signatures. A draft is accepted if the adversary answers correctly, spends at least a minimum number of retrieval turns, reaches the answer late enough, and does not bind the answer before evidence supports it. Drafts that fail are repaired in two directions. Drafts solved too quickly are treated as evidence of a route-level shortcut and repaired by replacing co-covered evidence, removing overly selective facts, or withholding and fuzzing exposed constants; if the model named the answer before evidence, the root entity is replaced or the evidence path strengthened. Drafts the adversary cannot solve within budget are treated as over-fuzzed or underspecified, and are repaired by narrowing clues, removing ambiguous facts, or restoring constraints.

This two-sided behavior is visible in the measurements. Shortcut-prone drafts move from cost 33.9 and hit time 12.4 to cost 82.7 and hit time 31.4, with the prior-shortcut rate falling from 17.0 to 12.0. Initially unsolved drafts produce no successful trajectory before refinement and afterwards become solvable while retaining cost 123.0 and hit time 50.2. Refinement is therefore a calibration step toward solvable-but-search-heavy, not simply a difficulty amplifier.

## Training and Inference

The base model is Qwen3-30B-A3B-Thinking-2507, a mixture-of-experts model that activates roughly 3B of 30B parameters at inference and supports a 256K context window. Training uses supervised fine-tuning only on the synthesized trajectories, with sequence packing, 6 epochs, global batch size 64, and maximum sequence length 262,144.

At inference the agent uses a context-managed protocol. Tool results are retained within a rollout so accumulated evidence can be reused, and if the turn limit is reached without a final answer the interaction history is cleared and the agent restarts from the original question.

## Results

Among comparable-size open-source agents, FORT-Searcher reaches an overall average of 66.2 across the five benchmarks, ahead of MiroThinker-1.7-mini at 64.6. It scores 72.2 on BrowseComp, 75.0 on BrowseComp-ZH, and 80.8 on xbench-DeepSearch-2505, and ties the best comparable-size result on xbench-DeepSearch-2510 at 57.2. Its BrowseComp-ZH score is the best open-source result among all agents listed, including substantially larger ones. The exception is Seal-0, where it scores 46.0 against 48.2 for MiroThinker-1.7-mini; that benchmark tests search-augmented reasoning under noisy or conflicting evidence rather than long-horizon evidence discovery.

Two analyses support the paper's central claim more directly than the headline numbers.

**Trajectory length alone is not useful difficulty.** Four training sets of 12K examples each were trained with the same recipe. Raising the average solving cost of open-source data from 40.0 through 85.0 to 140.0 moves BrowseComp only from 47.1 to 49.5. Holding cost fixed at 140.0 but switching to FORT data, where hit time is 47.0 and the prior-shortcut rate is 11.4 rather than 22.3 and 18.1 respectively, raises BrowseComp to 52.9 and BrowseComp-ZH to 60.3. Useful difficulty is the long pre-answer prefix, not the long trajectory.

**Every shortcut control contributes.** A cumulative ablation on 2K questions removes one component per row and reports the solving accuracy of the same strong agent, where higher accuracy means an easier question. Accuracy rises from 29.0 for the full pipeline to 81.6 with all controls removed, while solving cost falls from 141.9 to 43.7, hit time moves earlier from 46.5 to 11.8, and the prior-shortcut rate rises from 11.4 to 22.3. Under this removal order, dropping fuzzing produces the largest single drop in difficulty, which suggests obfuscation matters most for search difficulty. Because the ablation is cumulative rather than per-component, that ranking is conditional on the order used.

A separate annotation study on 200 successful question-trajectory pairs from each source maps the theoretical factors onto observable proxies. Compared with open-source data, FORT lowers the share of clues whose own evidence yields only one or two plausible candidates from 55.2 to 40.2, raises normalized evidence dispersion from 78.7 to 90.2, raises the retrieval cost of the most costly realized pre-answer dependency chain from 3.1 to 5.9, and lowers the prior-bound fraction from 27.0 to 16.0.

## Limitations

- Training is supervised fine-tuning only; the paper leaves the integration of reinforcement learning with FORT trajectories to future work.
- A large share of the headline BrowseComp result comes from the inference-time context-management protocol rather than the data. Disabling it drops BrowseComp from 72.2 to 55.9 and BrowseComp-ZH from 75.0 to 62.1, while the other three benchmarks move by much less. Comparisons against other agents should confirm whether the same mechanism is enabled.
- The prior-shortcut rate is a conservative lower bound that only detects models which verbalize the answer early.
- The trajectory-level proxies are explicitly described as observable effects rather than estimates of the theoretical quantities.
- All signatures are measured under one fixed solver and retrieval interface, so the diagnosis of a given dataset can shift with a different model or search backend.

## Further Reading

- [Visual reading: FORT-Searcher, an illustrated walkthrough](/files/visual-reading/fort-searcher-shortcut-resistant-search/)
