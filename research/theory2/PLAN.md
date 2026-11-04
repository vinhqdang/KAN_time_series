# Theory 2: sortability of time-series causal models with acyclic summary graphs and temporal aggregation

Pre-registered by commit before any derivation is run or any data is touched.

## Motivation and verified prior work

- Reisach, Chambaz, Blanchard, Weichwald, "A Topological Sorting Criterion for Random Causal Directed Acyclic Graphs", arXiv 2605.06288 (May 2026). Full PDF text read (pdftotext). Key facts:
  - Defines relatives of a node (nodes joined to it by an open path, i.e. marginally d-connected) and rel-sortability s_rho; Theorem 1 monotonicity of relatives along the order; Theorem 2 lower bound for ER DAGs; Theorem 3 s_rho = 1 implies a singleton MEC.
  - Section 5 proposes time-unrolled DAGs as more realistic benchmarks. Theorem 4 gives only a SUFFICIENT condition: if every weakly connected component of the summary graph (with self-loops) is strongly connected, then lim_{T->inf} s_rho(G_T) = 1/2.
  - It does not treat (i) summary graphs that are not strongly connected (feed-forward systems such as river networks, supply chains, signalling cascades), (ii) temporal aggregation or subsampling, which create contemporaneous dependence, or (iii) var- or R2-sortability of time-series data.
- Herman, Wahl, Ninad, Runge, CLeaR 2025 (arXiv 2503.17037): discusses collider vs confounder (fork) R2 behaviour with an upper bound for the collider centre; must be read in full before claiming anything about fork/collider R2 results (the earlier theory scan in research/theory/ missed it, so its candidate B novelty claim is downgraded).
- Ormaniec et al., ICLR 2025 (iSCM): sortability under standardisation, i.i.d. setting.

## Targets

T1 (graph-only, main theorem). For any summary graph G_S with self-loops, give the exact value of lim_{T->inf} s_rho(G_T), both for the finite unrolled DAG of Reisach et al. and for the stationary (infinite-past) process. Conjecture to prove or refute: relatives of v are the nodes u whose set of source strongly connected components (sources of the condensation reaching u) intersects that of v; the limit equals the edge-average over non-self-loop summary edges (i,j) of 1[|rel(j)| > |rel(i)|] + 1/2 * 1[|rel(j)| = |rel(i)|], with self-loops contributing 1/2, all weighted as in the unrolled DAG. Corollaries to check: Theorem 4 of Reisach et al. as a special case; lim = 1/2 if and only if a stated condition holds (necessary and sufficient); in-forests with several sources (river networks) have lim > 1/2 whenever a confluence exists; out-trees from a single source give exactly 1/2.

T2 (aggregation, linear stationary SVAR). Observation by block mean over window D (and point subsampling every D steps, as a contrast). Characterise the contemporaneous covariance and the limits of var-sortability and R2-sortability as D grows, as functions of the summary graph and delays. Re-use and check the earlier derivation in research/theory/B (covariance b(D - tau)_+, O(1/D) convergence), restated with exact assumptions.

T3 (link). Show how T1 and T2 interact: which observation models (snapshot i.i.d. trajectories, long stationary series, aggregated series) make which sortability criteria informative, in a summary table with proofs.

## Pre-registered empirical predictions (tested only after T1 and T2 are written and committed)

On CausalRivers subgraphs (same loaders and subgraph sampler as research/ts/stage0, new seeds 7000-7029 per region and size, 24 h and 12 h):
- P1: the theorem-predicted s_rho computed from the true river subgraph exceeds 1/2 for subgraphs containing a confluence.
- P2: an estimated relatives-sort (number of significant marginal correlations, Fisher z, alpha 0.01, Bonferroni within subgraph) orients edges with accuracy above 0.5 (bootstrap CI excluding 0.5) on subgraphs with confluences, and with accuracy within 0.05 of 0.5 on subgraphs without confluences.
- P3: across subgraphs, the Spearman correlation between predicted and estimated per-subgraph rel-sortability is at least 0.3.
- Kill rule: if P2 fails on confluence subgraphs at both resolutions, the real-data link of T1 is reported as unsupported.

## Rules

Every result is labelled proved, sketched or conjectured. Every claimed theorem is checked by exact enumeration on random small graphs (at least 10,000 graphs, fixed seeds) and counterexamples are searched for actively. Novelty claims cite what was opened in full versus seen in snippets. No threshold above changes after results are seen; deviations are listed.
