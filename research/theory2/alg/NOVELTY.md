# Novelty scan: a learning method for unlabelled time-unrolled data

Date: 2026-11-10. Scope: a possible new learning method for the ICML 2027 draft in `../icml_paper/` (read-only here).
All references below were checked today for existence, title and authors; the check for each one is logged in `REFS_VERIFY.md`, and the verified entries are in `refs.bib`.
Venues and years given in the tables are confirmed only where `REFS_VERIFY.md` says "confirmed"; the others are from memory and must be checked before they are cited in the paper.
No experiment was run. The design is in `DESIGN.md`, the pilot plan (DRAFT, not run) in `PILOT_PLAN.md`.

## 0. Summary

- **Candidate.** "Weight-tied DAG learning with latent temporal alignment": treat the pT anonymous columns as an unknown assignment to the (unit, slice) cells of an unrolled linear SVAR with one transition matrix A and one noise-variance vector D shared by all slices, and learn the assignment and (A, D) jointly by maximum likelihood.
- **Verdict on novelty: incremental to moderately new.** I found no paper that poses or solves this exact problem (hidden unit-and-slice labels of the columns, tied parameters, identifiability of the labels). Every ingredient exists: tied-parameter dynamic networks (1998), Gaussian DAG models with equality constraints on weights (2021–2025), latent permutations by Sinkhorn or graph matching (2015–2018), and learning dynamics when time labels are lost (2009–). The new part is the combination and an identifiability theory for the labels.
- **Verdict on the setting: mostly a benchmark construct.** I found no real data source in which the same columns are observed over i.i.d. replicates while their (unit, time) labels are unknown. The real problems nearby are different: unknown time of each *sample* (single-cell snapshots), or known labels with short panels.
- **Recommendation (one).** Keep the candidate, with a free initial covariance as the default law, and present it as the answer to the question that Reisach et al. leave open in their last sentence ("to what extent full or partial knowledge of the time order can and should be used for causal discovery on i.i.d. data"): *the time order of a time-unrolled benchmark is identifiable from the data, so hiding the labels hides little.* The method is the constructive half of that statement. As a stand-alone general-purpose causal-discovery method it would not be enough for ICML.
- **Two alternatives were scanned and rejected**: a start-law-aware likelihood for short labelled panels (already done, and no headroom: PCMCI+ is at 0.02), and an alignment-free weight-tying prior for static learners (already done in general form as coloured DAG search).

## 1. What Reisach et al. say themselves

Reisach, Chambaz, Blanchard, Weichwald (arXiv 2605.06288v1, read from the PDF today):
- Section 5 proposes generating i.i.d. observations "by repeatedly and independently simulating short time series trajectories" and unrolling a possibly cyclic summary graph in time.
- The discussion ends: "Prospectively, sampling data from independent trajectories of short time-unrolled DAGs also raises the question to what extent full or partial knowledge of the time order can and should be used for causal discovery on i.i.d. data."
- They give no method that uses or recovers the time order, and no statement on whether it is identifiable.

So the candidate addresses a question the source paper states as open. I found no follow-up paper that answers it (two web searches on 2026-11-10; a citation search on Semantic Scholar or Google Scholar was not done).

## 2. Scan by area

"Closeness" is to the candidate: 0 = unrelated, 3 = same problem.

### 2.1 Permutation- and ordering-based differentiable DAG learning (closeness 1)

| Work | What it does | What remains new |
|---|---|---|
| Mena, Belanger, Linderman, Snoek, ICLR 2018 (Gumbel–Sinkhorn) | Differentiable relaxation of latent permutations by the Sinkhorn operator with Gumbel noise | The tool only. The candidate can use it as its optimiser |
| Cundy, Grover, Ermon, NeurIPS 2021 (BCD Nets) | Variational posterior over DAGs as (permutation, triangular matrix); permutation by Gumbel–Sinkhorn | The permutation is a causal order of labelled variables. There is no tying and no hidden identity of variables |
| Charpentier, Kibler, Günnemann, ICLR 2022 (DP-DAG) | Samples DAGs as a differentiable permutation times a triangular mask | Same as above |
| Zantedeschi, Franceschi, Kaddour, Kusner, Niculae, ICLR 2023 (DAGuerreotype) | Optimises over the permutahedron with sparse relaxations | Same as above |
| Raskutti and Uhler, Stat 2018 (sparsest permutation) | Searches for the order whose DAG is sparsest | Same; gives the "sparsest I-map" argument used in `DESIGN.md` |
| Rolland et al., ICML 2022 (SCORE) | Order from the score function, nonlinear additive noise | Same |
| Zhao, Guo, Luk, Lupu, Dipojjwal, arXiv 2606.03227 (June 2026) | Gumbel–Sinkhorn permutation that triangularises the *instantaneous* matrix of an SVAR; time labels known | Closest in tools: a Sinkhorn permutation inside an SVAR likelihood. The permutation has a different role (order within a slice, labelled series). They state they are the first to use differentiable permutation learning in time-series discovery, so a method that only added "Sinkhorn + SVAR" would not be new |

In all of these the permutation is a topological order. In the candidate the permutation is an *identity map* from anonymous columns to (unit, slice) cells, and the order comes for free from the slices.

### 2.2 Layered DAG learning and equal-variance identifiability (closeness 1–2)

| Work | What it does | What remains new |
|---|---|---|
| Peters and Bühlmann, Biometrika 2014 | Equal error variances identify a linear Gaussian DAG | Not applicable as is: variances differ between units. But they are equal *within* a unit across slices |
| Chen, Drton, Wang, Biometrika 2019; Gao, Ding, Aragam, NeurIPS 2020; Gao, Tai, Aragam, AISTATS 2022 | Order or layer recovery under equal (or ordered) variances; layer-wise algorithms | Layers there come from a variance ordering. In the unrolled model the layers are slices and come from the replicated structure |
| Zhou, He, Zhong, Wang, arXiv 2111.01560; Zhao, He, Wang, arXiv 2111.00740 | "Topological layers" reconstructed bottom-up or top-down for QVF and non-Gaussian DAGs | Same remark; no tying across layers |
| Wu and Drton, arXiv 2308.08959 | *Partial* homoscedasticity: groups of nodes with equal error variance; characterises equivalence classes | Directly relevant: the tied D of the unrolled model is a partial-homoscedasticity pattern (one group per unit). The groups are assumed known there; here they are hidden |

### 2.3 Weight sharing, symmetry and template structure in graphical models (closeness 2; the closest static-model line)

| Work | What it does | What remains new |
|---|---|---|
| Højsgaard and Lauritzen, JRSS-B 2008 | Undirected Gaussian models with equality constraints on edges and vertices (RCON/RCOR), colouring given | Undirected, colouring known |
| Shah and Chandrasekaran, EJS 2012 | Covariance estimation under a known symmetry group | Group known |
| Makam, Reichenbach, Seigal, EJS 2023 (RDAG models) | Gaussian DAG models with coloured vertices and edges (same colour = same parameter); MLE existence and uniqueness | A tied unrolled SVAR with known labels **is** an RDAG model. Colouring and labels are given there; there is no structure or label learning |
| Boege, Kubjas, Misra, Solus, JRSS-B (arXiv 2404.04024) | Coloured Gaussian DAG models: Markov properties, faithfulness can fail, structural identifiability for homogeneous coefficients and for BPEC-DAGs; a GES analogue (GECS) that learns DAG and colouring | **Closest static prior work.** Their identifiability and their search are for *blocked* colourings: edges of one colour share the same head. Time-unrolled tying is the opposite case: the T−1 copies of a summary edge have different heads. So their theorems do not cover it, and their search space does not contain it. Our model is a coloured DAG model in their general sense, and their warning that coloured models may contain no faithful distribution applies to us (see `DESIGN.md`, assumption F) |
| Friedman, Murphy, Russell, UAI 1998 | Structure learning of dynamic probabilistic networks: a prior network B_0 for the first slice and a transition network B_→ with repeated structure and parameters; "learning DPNs is not quite the same as applying PN methods to the unrolled network" | **Closest dynamic prior work.** It is the candidate with the labels known, including a separate start law. With labels this is solved since 1998 |
| Segal, Pe'er, Regev, Koller, Friedman (module networks, arXiv 1212.2517) | Learns which variables share a conditional distribution (modules) and each module's parents | Learns a parameter-sharing partition from data, which is the same spirit. Sharing is between variables with the *same parents*; unrolled copies have *different* (shifted) parents |
| Niculescu, Mitchell, Rao, JMLR 2006 | Parameter equality constraints in Bayesian networks with known structure | Known structure and known sharing |
| Maier, Marazopoulou, Arbour, Jensen, UAI 2013 (RCD) | Causal discovery of relational (template) models | Template instances are labelled by the relational schema |
| Yeh, Hu, Hasegawa-Johnson, Schwing, AISTATS 2022; Zhou, Knowles, Finn, ICLR 2021 | Learn a parameter-sharing scheme (equivariance) of a neural layer from data | Same idea in deep learning ("discover the weight tying"); no graph, no identifiability of a causal structure |

### 2.4 Lost or unknown time: samples (rows) without time labels (closeness 2 in spirit, different object)

| Work | What it does | What remains new |
|---|---|---|
| Huang and Schneider, ICML 2009 | Learns a linear dynamical system from snapshots whose order in time is unknown | **Closest in spirit.** The unknown is the time of each *sample* (row); variables are labelled. In our setting the samples are whole trajectories and the unknown is the cell of each *column* |
| Huang and Schneider, NeurIPS 2011 | VAR estimation from a few sequences plus non-sequence snapshots, through the Lyapunov equation of the stationary covariance | Uses the same closed-form covariance Σ(A, D) that we use; labels of variables known |
| Stein and Worah, arXiv 2401.11858 | Linear dynamics (gene network) from non-sequential snapshots, given an approximate clock | Row-time again |
| Trapnell et al., Nat. Biotechnol. 2014; Saelens et al., Nat. Biotechnol. 2019; Magwene, Lizardi, Kim, Bioinformatics 2003 | Pseudotime: order cells or samples along a trajectory | Row-time; heuristic orderings |
| Schiebinger et al., Cell 2019; Lavenant, Zhang, Kim, Schiebinger, arXiv 2102.09204 | Trajectory inference from time-labelled population snapshots by optimal transport | Snapshot times known, pairing unknown |
| Guan et al., JMLR 2026 (arXiv 2410.22729, APPEX) | Identifies drift, diffusion and causal graph of a linear SDE from temporal marginals; alternates OT coupling and MLE | Structurally similar algorithm (alternate a matching step and a closed-form MLE). The matching is between *samples* of consecutive labelled marginals |
| El Rimawi-Fine et al., arXiv 2510.16656 | Structure learning for population dynamics from snapshots | Same family |
| Matsumoto et al., Bioinformatics 2017 (SCODE); Qiu et al., Cell Systems 2020 (Scribe); Pratapa et al., Nat. Methods 2020 (BEELINE) | Gene-network inference on pseudotime-ordered cells; Scribe and BEELINE report large losses when pseudotime replaces real time | Shows that the row-time problem is real and hard. It is not our problem |
| Fonteijn et al., NeuroImage 2012 (event-based model) | Orders biomarker events from cross-sectional patients | Orders *variables* in time from cross-sectional data, by a likelihood over permutations. Variables are distinct biomarkers, not replicated copies; no dynamics, no tying |
| Bystrova, Assaad, Si-moussi, Thuiller, bioRxiv 2024 | What a static method can recover from single-time-point observations of a dynamic system | Negative/qualified results for one slice; no label recovery |

### 2.5 Seriation and graph matching (closeness 1–2: these are the optimisation sub-problems)

| Work | What it does | What remains new |
|---|---|---|
| Atkins, Boman, Hendrickson, SIAM J. Comput. 1998 | Spectral seriation: recovers a hidden linear order from a Robinson similarity matrix | For a stationary window, slice distance is a (block) Toeplitz structure, so seriation of |corr| is a natural initialiser for the slice labels. It ignores units and direction |
| Fogel, Jenatton, Bach, d'Aspremont, NeurIPS 2013; Flammarion, Mao, Rigollet, Bernoulli 2019; Berenfeld, Carpentier, Verzelen, arXiv 2408.10004 | Convex relaxations and optimal rates for seriation, including Toeplitz matrices | Scalar Toeplitz; no blocks, no direction of time, no graph |
| Vogelstein et al., PLoS ONE 2015 (FAQ); Lyzinski et al., TPAMI 2016 | Frank–Wolfe on the relaxed quadratic assignment problem; why the indefinite relaxation is preferable | With (A, D) fixed, our alignment step **is** a quadratic assignment problem. FAQ is the natural solver |

### 2.6 Permutation-latent regression (closeness 1)

Pananjady, Wainwright, Courtade (arXiv 1608.02902); Unnikrishnan, Haghighatshoar, Vetterli (arXiv 1512.00115); Hsu, Shi, Sun (NeurIPS 2017); Abid, Poon, Zou (arXiv 1705.01342); Tsakiris and Peng (ICML 2019, homomorphic sensing).
- They study y = ΠXβ with an unknown permutation of the *responses*: statistical limits, NP-hardness in general, algebraic identifiability.
- Relation: the same warning applies to us (the exact alignment step is a hard combinatorial problem). The object differs: our permutation acts on both sides of a covariance, and the design is the model itself.

### 2.7 Time-series structure learning that uses or reconstructs the unrolled graph (closeness 1)

| Work | What it does | What remains new |
|---|---|---|
| PCMCI+ (Runge 2020), DYNOTEARS (Pamfil et al. 2020), VAR-LiNGAM (Hyvärinen et al. 2010) (all in the paper's refs) | Tied lagged structure with known labels | Labels known |
| Günther, Ninad, Runge, UAI 2023 (J-PCMCI+) | Multiple data sets / short series pooled | Labels known |
| Das, Chakraborty, Maulik, arXiv 2602.02830 (SC3D); Biswas et al., arXiv 2508.01920 (CITS); Sanchez et al., arXiv 2510.24639 (DOTS) | Recent methods that reason on the time-unrolled DAG of a labelled series | Labels known |
| Löwe, Madras, Zemel, Welling, CLeaR 2022 (amortized causal discovery) | Shares one inference model across series with different graphs | Different kind of sharing |
| Peters, Janzing, Gretton, Schölkopf, ICML 2009; Bauer, Schölkopf, Peters, ICML 2016; Weiss 1975; Chan, Ho, Tong, Biometrika 2006 | Direction of time: Gaussian (V)ARMA processes satisfy a model of the same class when reversed; non-Gaussian ones do not | Bears directly on our identifiability question (stationary Gaussian law, time reversal). What we add: with the *structural* restriction of a diagonal innovation covariance, a reversed Gaussian VAR(1) is generically outside the model class when p ≥ 2, so second moments can identify the direction; and exact characterisations of when they cannot (`DESIGN.md`, T4). I verified Bauer et al. at abstract level only; whether their paper discusses a diagonal-innovation restriction was not checked |

### 2.8 Tiered background knowledge (closeness 1–2)

Bang and Didelez (UAI 2023, arXiv 2306.01638; CLeaR 2025, arXiv 2503.21526); Petersen, Osler, Ekstrøm (Am. J. Epidemiol. 2021); Larsen, Ekstrøm, Petersen (arXiv 2502.06232).
- They show how a *known* partial time order (tiers, e.g. cohort waves) restricts the equivalence class and improves discovery. Reisach et al. cite this line as motivation.
- Relation: slices are tiers. The candidate *learns* the tiers, which is possible only because the dynamics are replicated. I found no method that learns tiers from data (one web search; not exhaustive).

### 2.9 Short labelled panels and the start law (the first alternative direction)

Anderson and Hsiao (J. Econometrics 1982); Hsiao, Pesaran, Tahmiscioglu (J. Econometrics 2002); Binder, Hsiao, Pesaran (Econometric Theory 2005); Friedman, Murphy, Russell (UAI 1998).
- The "initial conditions problem" of dynamic panel models with small T is a classical topic in econometrics: the likelihood must model the first observation (fixed, stationary, or free), and the estimator of the dynamics depends on that choice. Binder et al. treat short panel VARs by maximum likelihood.
- In the graphical-model line the same split is the prior network B_0 of Friedman et al.
- So "a start-law-aware likelihood for short labelled panels" is **already done**. It also has no headroom in our benchmark: PCMCI+ with labels is at SHD/|E| = 0.02 under both starts.

## 3. Verdict

**Rating: between "incremental" and "clearly new"; I would call it a new problem with an incremental method.**

Closest three prior works and the delta:

1. **Friedman, Murphy, Russell (UAI 1998).** Same statistical model (start law + tied transition model), labels known. Delta: the unit and slice labels are latent; identifiability of the labels; the start law as a nuisance.
2. **Boege, Kubjas, Misra, Solus (JRSS-B; arXiv 2404.04024)**, with Makam, Reichenbach, Seigal (EJS 2023) and Wu and Drton (2023). Static Gaussian DAG models with tied weights and variances; identifiability from tying; a search that learns a colouring. Delta: their identifiable class and their search are restricted to colour classes with a common head; the unrolled colouring is "one colour class per summary edge, T−1 heads", which they do not cover; and the colouring here is generated by a hidden grid, which gives a much smaller search space and a different theory (slices as a graph invariant).
3. **Huang and Schneider (ICML 2009)** (and the trajectory-inference line up to Guan et al., JMLR 2026). Learning linear dynamics when temporal information is lost, by alternating a matching/ordering step and a closed-form fit. Delta: they lose the time of samples; we lose the cell of variables, with complete replicate trajectories.

What a critical reviewer can say, and I think fairly:
- "The setting is self-made: nobody receives i.i.d. trajectories with shuffled, anonymous columns." (See Section 4.)
- "With labels the problem is trivial; the method only undoes a shuffle that the benchmark designer applied."
- "The algorithm is FAQ or Sinkhorn applied to a Gaussian likelihood."

What is defensible as new:
- The identifiability statements for the labels (which start law, which T, which graphs), with exact negative cases. None of the works above states them.
- The consequence for the benchmark proposal of Reisach et al.: replicated structure is a third exploitable regularity next to var- and rel-sortability, and it does not go away with a burn-in.

**Is it enough for ICML?** As the main contribution of a stand-alone methods paper: no. As a fifth result of this benchmark paper, it can raise Contribution if (a) the pilot shows a large gap to FLOP under the burn-in, and (b) the text presents it as a statement about the benchmark ("the labels are recoverable") and not as a new general causal-discovery algorithm. The risk that a reviewer still scores it as a solution to a self-made problem is real; I would put it near one in three.

## 4. Who has this problem for real?

Honest answer: **almost nobody has exactly this problem.**

| Setting | Does it match? |
|---|---|
| The benchmark construction of Reisach et al. (i.i.d. short trajectories handed to static methods) | Yes, by construction. It is the only exact match I found |
| Single-cell snapshots, cross-sectional data of a dynamic system, destructive measurements | No. Each sample is measured once, at an unknown or coarse time. The unknown is the time of the *row*. This is the pseudotime / trajectory-inference problem, which has a large literature (Section 2.4) |
| Reset experiments, e.g. Causal Chambers; short panels in econometrics or epidemiology | No. Trajectories are replicated, but the time labels are recorded. With labels, standard time-series methods are near-perfect |
| Cohort studies with waves | Labels (waves) are known and used as tiers. The variables of different waves are usually different measurements, so the tying assumption fails |
| Wide tables in which the wave or stage of a column is missing or not trusted (legacy data, merged sensor logs) | Possible but rare; I found no documented case |
| Processes unrolled in space instead of time: identical stations of a production line (cf. causalAssembly), sensors along a flow, layers of identical units | Partial. Here "slice" is a stage and the stage of a sensor may be unknown. The tying assumption (identical stages) is strong and I found no data set where stage labels are missing |

So the setting should be described in the paper as what it is: the threat model of a proposed benchmark, in which a "time-agnostic" method is supposed to get no help from time.
The value of the method is then a benchmark statement, in line with the paper: *a learner that is label-free is not time-agnostic if the labels can be reconstructed.*

### Alternatives considered

| Direction | Novelty | Headroom in our benchmark | Decision |
|---|---|---|---|
| (A) Start-law-aware likelihood for short **labelled** panels | Already done (Section 2.9) | None: PCMCI+ 0.02 | Rejected |
| (B) Alignment-free weight-tying prior for static learners (penalise the number of distinct edge weights in DAGMA/NOTEARS/FLOP) | The general idea is coloured DAG search (Boege et al.); a "fused" penalty on NOTEARS weights would be a small variation. Without the grid, the space of colourings is far larger than (pT)! / symmetries, and there is no identifiability theory for non-blocked colourings | Unknown; likely small, since which edges to tie is the hard part | Rejected as the main proposal; kept as an ablation idea |
| (C) Row-time version (unknown time of each sample) | Crowded (Section 2.4) | Not connected to the paper's theory | Rejected |
| (D) The candidate, with a free initial covariance and the label-recovery framing | New problem, incremental method (Section 3) | Large on paper: 0.52 (FLOP, burn-in) against 0.02 (labels) | **Recommended** |

## 5. Recommendation

Add **one** result to the paper: *time labels of a time-unrolled benchmark are identifiable, and a tied-likelihood learner recovers them* (working name TiDAL, "tied DAG learning with latent alignment"; `DESIGN.md`).

- Default law: free initial covariance. It contains the fixed start and the burn-in window as special cases, so the method does not depend on the start artefact. Input is standardised, so it cannot use var-sortability.
- Claimed novelty: the identifiability results and the benchmark consequence. The optimiser is stated to be standard (FAQ-type alternating minimisation).
- Condition: run the pilot of `PILOT_PLAN.md` first. If the kill criterion is not met, do not add the method; the identifiability statements T1–T4 of `DESIGN.md` can still go into the paper as a short section, since they hold without any algorithm.

## 6. Limits of this scan

- Searches: arXiv API, Crossref API, and about 20 web searches on 2026-11-10. No Semantic Scholar or Google Scholar citation crawl. dblp refused automated access (bot check).
- I read abstracts, and for Reisach et al. and Boege et al. parts of the full text. I did not read the full text of most papers. Statements such as "labels known" rest on abstracts and on my prior knowledge of the papers.
- A paper that poses the unlabelled-column problem under another name (for example in system identification or in sensor networks) may exist; the searches "unlabelled / shuffled / permuted variables + VAR / dynamical system" found none.
