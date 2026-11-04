# Theory 2: rel-sortability of time-unrolled DAGs and temporal aggregation

Pre-registered plan: `PLAN.md` (commit e24e4bf). Theory, proofs, novelty check and the Phase 2 operationalisation: `THEORY.md` (Phase 1, commit e92696e, pushed before any Phase 2 data were sampled). Verification code: `verify/`. Empirical code and outputs: `empirical/`. Raw CausalRivers data and the stage0 block-mean caches stay outside the repository (`/tmp/claude-0/dl/`); the repository holds only node ids, windows and derived statistics.

## Results that survived (full statements and proofs in THEORY.md)

Setting: Reisach et al. (arXiv 2605.06288), summary graph G_S with p nodes and a self-loop at every node; G_T is the unrolled DAG; relatives and s_ρ are as defined there. E_× is the set of summary edges between different strongly connected components (SCCs).

1. **Thm 1.3 (proved).** For every T, an edge (i(t), j(t+1)) has s_ρ = 1 iff some u satisfies d(u,j) ≤ t ≤ d(u,i), where d is shortest-path length and ∞ if there is no path. Otherwise s_ρ = 1/2.
2. **Thm 1.5 (proved).** lim_T s_ρ(G_T) = 1/2 + |E_×|/(2|E_S|), and 0 ≤ s_ρ(G_T) − lim ≤ (p−1)/(2(T−1)).
3. **Cor. 1.6 (proved).** The limit is 1/2 iff every weakly connected component is strongly connected. Reisach et al.'s Theorem 4 is the "if" half. If the condition fails, s_ρ(G_T) ≥ 1/2 + 1/(2|E_S|) for every T. For summary DAGs the limit is 1/2 + m/(2(m+p)), whatever the shape: chains, in-trees, out-trees and colliders with equal m and p give the same value.
4. **Rem. 1.8 (proved).** In G_T, the strict sorting of crossing edges whose endpoints have equal summary relatives comes from the fixed start at t = 1. It amounts to O(1) extra relatives out of O(T).
5. **Thm 1.9 / Cor. 1.10 (proved).** Stationary version: an infinite past, or a latent burn-in of at least p−1 slices. Then exactly, for every T, s_ρ = 1/2 + #{edges (i,j) whose head has a source SCC that the tail lacks}/(2|E_S|). The value is 1/2 iff every weak component has a single source SCC. Out-trees, forks and chains give 1/2. In-forests (river networks) give 1/2 + #{edges into confluences}/(2|E_S|), which is above 1/2 iff there is a confluence. The plan's conjecture is this result. It is false for the finite G_T.
6. **T2 (proved).** Under a stable linear SVAR with self-loops, block means converge at rate O(1/D) to the long-run SEM. That SEM has edge weights Σ_τ A_τ/(1−λ_j) and noise variance σ_j²/(1−λ_j)² (Prop. 2.3). The block-mean R²-sortability tends to 0 on out-stars and to 1 on in-stars. Point subsampling leaves every contemporaneous criterion unchanged by D: it stays at Γ(0). In both schemes the marginal-correlation support equals the summary relatives (Prop. 2.8). Corrections to research/theory/B: the (D−τ)_+ formula and the "all ties for small D" claim hold only without autocorrelation. Monotonicity in D on in-trees is refuted.

Verification: exhaustive enumeration of every summary graph on p ≤ 4 nodes, plus 10,000 random graphs (p ≤ 7, 46,724 (graph, T) cases, 1.9 M edges), found 0 failures of any stated result. The plan's conjecture disagreed with the exact G_T limit in 9,603 of the 10,000 random graphs.

## Phase 2: pre-registered predictions on CausalRivers

Design: the stage0 subgraph sampler, copied unchanged into `empirical/sample.py`, with seeds 7000–7029 for East Germany and Bavaria at d = 5 and d = 10. That gives 120 subgraphs and 791 induced edges. The prediction is the stationary T1 value computed from the full regional river graph, with unobserved stations treated as latent. The estimator is a relatives count from Fisher z tests at α = 0.01, Bonferroni-corrected within each subgraph, on standardised block-mean series at 12 h and 24 h. CIs use a cluster bootstrap over subgraphs (2,000 resamples, seed 0). The rules are fixed in THEORY.md Section 6, and all numbers come from `empirical/results_p.json` and `empirical/test_p.log`.

| | 12 h | 24 h |
|---|---|---|
| P1: confluence subgraphs with predicted s̃ > 1/2 | 112 / 114 | same graphs |
| P2 confluence subgraphs (114 subgraphs, 767 edges): accuracy [95% CI] | 0.501 [0.492, 0.509] | 0.498 [0.489, 0.507] |
| P2 non-confluence subgraphs (6 subgraphs, 24 edges): accuracy | 0.500 | 0.500 |
| share of subgraph pairs judged dependent (confluence subgraphs) | 0.984 | 0.978 |
| P3: Spearman(predicted s̃, estimated accuracy), n = 120 | −0.018 (p = 0.84) | 0.018 (p = 0.84) |

Verdicts:
- **P1: holds in 112 of 114 confluence subgraphs and fails as worded in 2.** The 2 exceptions (east_germany d5 seed 7022 and bavaria d5 seed 7025) are split-and-rejoin channels, where an upstream station feeds two branches that rejoin. The river graphs are DAGs but not in-forests: East Germany has 26 nodes with out-degree ≥ 2 and Bavaria has 13. In those graphs the theorem itself predicts 1/2 (single source), so the exceptions are consistent with T1. They are still exceptions to P1 as pre-registered. Mean predicted s̃ on confluence subgraphs is 0.824. All 6 non-confluence subgraphs have s̃ = 0.5.
- **P2 confluence: fails at both resolutions.** Both CIs include 0.5.
- **P2 non-confluence: passes at both resolutions** (0.500, within 0.05 of 0.5). The pass is trivial, because every pair is declared dependent and every edge is a tie. The test rests on only 6 subgraphs.
- **P3: fails at both resolutions** (ρ ≈ 0, threshold 0.3).
- **Kill rule: triggered.** The real-data link of T1 is reported as **unsupported**: the estimated relatives sort carries no orientation information on CausalRivers at 12 h or 24 h.

Why (exploratory, not pre-registered; `empirical/explore_corr.py`): station pairs that are not river relatives have mean correlation 0.64 (12 h) and 0.66 (24 h), against 0.73 and 0.74 for relatives. With series of 700 to 3,600 steps, almost every pair is significant, so every station has every other station as a "relative". This matches stage0's hidden-driver diagnostic (residual correlation 0.51 among unrelated stations). In terms of the theory, regional precipitation is a latent common ancestor of all stations. Adding it to the summary graph gives each weak component a single source SCC, and Cor. 1.10(i) then predicts rel-sortability exactly 1/2. The data are consistent with the theory applied to the augmented graph and inconsistent with the river-only graph. This explanation was found after seeing the data and has not been tested. Correcting the Fisher z sample size for autocorrelation (Bartlett-type, exploratory) does not help: confluence-subgraph accuracy is 0.509 [0.494, 0.523] at 12 h and 0.505 [0.491, 0.519] at 24 h, with 93% to 94% of pairs still significant.

## Deviations from the plan
1. The version of "theorem-predicted s_ρ" was not named in the plan. The stationary version (Thm 1.9) was chosen and recorded in THEORY.md Section 6 before any Phase 2 data were touched. Under the finite-G_T version P1 would hold trivially for all 120 subgraphs, and that version would predict s̃ > 1/2 for the non-confluence subgraphs as well.
2. Self-loops are excluded from the predicted edge average, because the estimator cannot orient them. This does not change on which side of 1/2 the prediction falls. The rule was recorded in Section 6.
3. Relatives for the prediction use the full regional graph (latent unobserved stations), and edges are the induced-subgraph edges, as in stage0. Recorded in Section 6.
4. The non-confluence group is small: only 6 of 120 subgraphs, because the sampler grows connected subgraphs and almost all of them contain a confluence. The plan did not fix a minimum size for this group.
5. Fixes made during Phase 1 to my own code, before any result was used: the Theorem 4 / converse check in `t1_corollaries.py` was rewritten because the first version was partly tautological, and zero-valued failure counters were made explicit in `t1_enum.py`.

## Files
- `THEORY.md`: all statements, proofs, the novelty check (what was opened and what was seen only in snippets) and the Phase 2 operationalisation.
- `verify/t1_enum.py`, `verify/t1_corollaries.py`, `verify/t2_aggregation.py`, `verify/t2_support.py`, each with a matching `*_results.txt`.
- `empirical/sample.py` produces `subgraphs_7000.json` and `sample.log`. `empirical/test_p.py` produces `results_p.json` and `test_p.log`. `empirical/explore_corr.py` produces `explore_corr.json`.
- Reproduce: `OMP_NUM_THREADS=1 /usr/bin/python3 <script>` from the script's folder (needs the stage0 caches in `/tmp/claude-0/dl/`). One core was used.
