# B1 analysis plan: does the start artefact of time-unrolled generators change how causal-discovery methods score and rank?

Written and committed before any simulation that produces an algorithm score. The only computations run before this commit are graph-only (population) ones: the exact s_rho of every test summary graph (`exact_curves.py predict`, output `results/exact_predictions.txt`) and a check that the exact formulas agree with brute-force enumeration (280 graph/T cases, 0 disagreements). Roadmap items addressed: B1 and B2 of `../icml_paper/review/00_editorial_decision.md` (consensus findings C4, C8).

## 0. Question and threat model

Reisach, Chambaz, Blanchard, Weichwald (arXiv 2605.06288, Section 5) propose benchmark DAGs obtained by unrolling a summary graph G_S (self-loop on every node) over T slices, with i.i.d. observations from independently simulated short trajectories. Their G_T starts every trajectory at slice 1 with noise only. THEORY.md shows that this fixed start makes rel-sortability exceed its stationary value: Thm 1.5 gives s_rho(G_T) exactly, Thm 1.9 gives the stationary (burn-in) value 1/2 + |E_up|/(2|E_S|). B1 asks whether the gap matters for algorithms, when the time label of every variable is hidden.

Threat model, stated in advance. With the time labels visible, every edge of G_T goes from slice t to slice t+1, so sorting by time orients every edge for free (order accuracy 1 for every graph, generator and method that uses it). The experiment is therefore only meaningful for the benchmark use that Reisach et al. propose, i.e. the pT variables are handed to an i.i.d. causal-discovery method as anonymous columns. We include one reference method that uses the labels (R0 below) only to make this explicit; it is not part of any hypothesis.

## 1. Design

### 1.1 Summary graphs (all with a self-loop on every node), p in {5, 10}

| family | definition | E_x (crossing edges) |
|---|---|---|
| symER | undirected G(p, 2/(p-1)) (expected degree 2), each edge replaced by both directions | empty (strongly connected weak components) |
| ring | directed cycle 0 -> 1 -> ... -> p-1 -> 0 (graph fixed; seeds vary weights and noise) | empty |
| chain | 0 -> 1 -> ... -> p-1 | all non-loop edges |
| outtree | random recursive tree, edges parent -> child | all non-loop edges |
| intree | random recursive tree, edges child -> parent (river-like, confluences) | all non-loop edges |
| ER1 | directed G(p, 1/p), cycles allowed | mixed |
| ER2 | directed G(p, 2/p), cycles allowed | mixed |

A graph with no non-loop edge is resampled (only symER, ER1, ER2 can produce one). Families symER and ring are the strongly connected (SC) families; the other five are the non-SC families.

Horizons T in {3, 5, 10}. Window size pT in {15, 25, 50, 30, 50, 100}.

### 1.2 SEM (fixed in advance)

- Linear Gaussian VAR(1): x_{t+1} = A^T x_t + e_{t+1}, A[i,j] != 0 iff (i,j) in E_S.
- Weights: every summary edge, self-loops included, gets Unif(0.5, 1) (Reisach et al. Section 3.1 range, positive). If the spectral radius of A exceeds 0.8, A is rescaled to spectral radius 0.8 (stability, fast mixing). Positive weights make every trek-connected pair correlated, so the population relatives equal the graph relatives (faithfulness of relatives; no cancellations).
- Noise: e_v ~ N(0, sd_v^2), constant over time.
- Variance regimes:
  - **R-het** (primary): sd_v ~ Unif(0.5, 1) per summary node; raw (unstandardised) data.
  - **R-eq**: sd_v = 1; raw data.
  - **R-std**: R-het data, every column standardised before the method sees it.
  rel-SortnRegress, R2-SortnRegress, Random-SortnRegress, FLOP (always given standardised data) and PC (Fisher z) are scale invariant, so in R-std only Var-SortnRegress and the estimated var-sortability are recomputed; the other entries equal R-het by invariance (this is a property of the code, not an assumption).
- n = 1000 i.i.d. trajectories per data set.

### 1.3 Generators and their ground truth

- **G1 (Reisach-style fixed start).** x_1 = e_1 (noise only), then T-1 steps. Ground truth: G_T (window = whole DAG). Causally sufficient. Exact population s_rho: Thm 1.3/1.5 (`srho_G1_exact`).
- **G2 (burn-in).** x_1 = e_1, run B = 50 latent steps, keep the next T slices. B >= max(p-1, Delta) = graph condition of Thm 1.9; with spectral radius <= 0.8, the initial-condition contribution to the covariance is of order 0.8^100 (checked numerically, sanity rule S3). Ground truth: the T-slice window of G_{B+T}. Its directed edges are a copy of G_T's edges; this copy is the target DAG for SHD/SID/order accuracy. **Causal sufficiency fails**: the latent slice B is a common parent of the first window slice. Because every edge goes one slice forward, latent projection adds bidirected edges only among first-window-slice nodes (pairs with a common summary ancestor), and no first-window node has an observed parent. The window distribution is therefore Markov to G_T plus a complete DAG on slice 1 in any order, but those slice-1 edges are not causal. We report SHD both on all pairs and excluding pairs inside the first window slice (SHD_x1), to separate the start artefact from this sufficiency cost. Exact population s_rho: Thm 1.9 (relatives = marginal d-connection in the full DAG, restricted to the window), 1/2 + |E_up|/(2|E_S|) for every T.
- **G2s (exact stationary initialisation).** x_1 ~ N(0, Sigma), Sigma the solution of Sigma = A^T Sigma A + D (`solve_discrete_lyapunov`), then T-1 steps. This is the B -> infinity limit of G2 in distribution. It keeps a DAG model over the window only in the sense above (G_T plus an arbitrary complete DAG on slice 1, whose edges have no causal meaning; the causal model is G_T plus slice-1 bidirected edges), so its ground truth and exact s_rho are those of G2 (infinite-past case of Thm 1.9). G2s is run only in R-het, as a replication of G2 (sanity rule S1); it is not used for H1-H3.

### 1.4 Seeds

- Test summary graphs: seeds 1000-1029 for every (family, p), i.e. 30 graphs per (family, p), 420 graphs. Graph and parameter RNG: `default_rng([family_index, p, seed])`; weights and noise s.d. are drawn from the same stream right after the graph and are shared by all T, generators and regimes, so G1 and G2 are paired by graph and parameters. Data RNG: `default_rng([family_index, p, seed, T, generator_index, regime_index, 7])`.
- Validation seeds 9000-9004 (5 graphs per (family, p)): used only to debug the pipeline, time the methods and check mixing. No hyperparameter is tuned on them, because every hyperparameter below is fixed in advance from the cited papers' defaults. Validation results are not reported as evidence.
- The test seeds are run once.

### 1.5 Methods (time labels hidden: columns randomly permuted per data set, pT anonymous variables)

- **M1 rel-SortnRegress** (Reisach et al. Alg. 1 + Alg. 2): count |corr| > eps per column, eps from the t-test at alpha = 0.05 with m = n; ascending order, ties broken uniformly at random; then Alg. 2 (standardise, OLS, adaptive LassoLarsIC(BIC), as in the sortnregress reference code).
- **M2 R2-SortnRegress**: ascending R^2 of each variable regressed on all others; Alg. 2.
- **M3 Var-SortnRegress**: ascending marginal variance of the data as given (raw in R-het/R-eq, standardised in R-std, where all ties are broken at random); Alg. 2.
- **M4 Random-SortnRegress**: uniformly random order; Alg. 2.
- **M5 FLOP** (Wienöbst, Henckel, Weichwald, ICLR 2026): PyPI `flopsearch` 0.3.0 (installed and tested before this plan; Rust implementation by the authors), BIC penalty 2.0, 10 restarts (Reisach et al.'s setting), standardised input. Output is a CPDAG.
- **M6 PC** (causal-learn 0.1.4.8): Fisher z, alpha = 0.01, stable. Output CPDAG. Because PC takes about 35 s per data set at pT = 100, PC is run only on cells with pT <= 50 (all except p = 10, T = 10), in R-het, for G1 and G2.
- **R0 Time-SortnRegress (labels visible; reference only)**: order by the known slice, random within slice; Alg. 2. Its order accuracy is 1 by construction.
- DAGMA/NOTEARS are not run (DAGMA requires torch, not installed; FLOP and PC are the standard learners).

### 1.6 Metrics (per data set)

- Exact population s_rho of the generator's ground truth (G1: Thm 1.5 exact; G2, G2s: Thm 1.9), and exact gap = s_rho(G1) - s_rho(G2).
- Estimated var-, R2- and rel-sortability (Reisach et al. eq. (4), edge-based, on the true window edges; rel uses the Alg. 1 counts).
- SHD against the true window DAG: each unordered pair counts 1 if its state (none, i->j, j->i, undirected) differs; undirected CPDAG edges always count as an error on a true edge. SHD_x1: the same, excluding pairs inside the first window slice.
- Order accuracy (sort-based methods): fraction of true edges (i,j) with i before j in the method's order.
- Orientation score (all methods): mean over true edges of 1 (output has i->j), 1/2 (undirected), 0 otherwise.
- SID (gadjid 0.1.0, Peters-Buehlmann SID; for CPDAG outputs the undirected edges are removed first, as in Reisach et al.).

## 2. Hypotheses and decision rules (thresholds fixed now)

All tests use the R-het regime and generators G1 vs G2 unless stated. Bootstrap CIs: percentile, 10,000 resamples over the stated units, seed 0. Wilcoxon signed-rank (scipy, `zero_method="wilcox"`).

**H1 (start artefact reaches rel-SortnRegress).** For each non-SC family, unit = summary graph (p, seed): d = mean over T in {3,5,10} of [OA_M1(G1) - OA_M1(G2)] (OA = order accuracy), 60 units per family. One-sided Wilcoxon (d > 0), Holm over the 5 families at alpha = 0.05.
- Supported if at least 4 of 5 families are Holm-significant and the pooled mean d over the 300 units is >= 0.02.
- Tracking: Spearman correlation between cell-mean d and cell-mean exact gap over all 42 cells (7 families x 2 p x 3 T) >= 0.5 with one-sided p < 0.05. Also reported: unit-level (graph x T) Spearman, OLS slope of d on exact gap.
- Kill: fewer than 3 of 5 Holm-significant, or pooled mean d < 0.01: "the start artefact does not reach rel-SortnRegress at n = 1000". Any non-SC family significant in the opposite direction (two-sided Wilcoxon, Holm) is reported as contradicting the theory.

**H2 (strongly connected families).** Graph-only prediction made before this plan (`results/exact_predictions.txt`): the exact gap on symER is 0.43/0.25/0.11 (p = 5, T = 3/5/10) and 0.44/0.32/0.14 (p = 10); on the ring it is 0.5 in every cell except p = 5, T = 10 (0.22). The reason is that the excess over the limit 1/2 decays only like Delta/(2(T-1)), with Delta the largest finite distance (Section 3), and T >= p does not make it small at T <= 10. So the review's expectation "small at T >= p" is predicted to be false at these horizons. Decision rules, per SC cell (family, p, T), unit = graph seed (30 per cell), d = OA_M1(G1) - OA_M1(G2):
- H2a (review's form): "small at T >= p" holds if |mean d| < 0.03 in every SC cell with T >= p (p = 5: T = 5, 10; p = 10: T = 10). Predicted to fail.
- H2b: "non-negligible at short T" holds if at least one SC cell with T < p has mean d >= 0.03 with the 95% CI excluding 0.
- H2c: on symER, d decreases with T: paired one-sided Wilcoxon over the 60 graphs of d(T=3) > d(T=10).
- The exact s_rho(G_T) vs T curve for T = 2..20 per family is reported (B2).

**H3 (ranking changes).** Primary metric SHD, methods M1-M5 in every cell, plus M6 where run. For each cell (family, p, T) and each method pair, the paired difference over the 30 seeds is tested under G1 and under G2 separately (two-sided Wilcoxon). A **significant rank swap** is a pair whose mean difference has opposite signs under G1 and G2 and is significant under both after Holm correction over all (pair x cell x generator) tests of the metric. H3 is supported if at least one significant swap exists on SHD.
- Also reported (not used for the verdict): the same for SID, orientation score, order accuracy (M1-M4); swaps by mean only; the number of cells in which the best method (lowest mean SHD) changes; Kendall tau between the G1 and G2 rankings per cell; and SHD_x1. A swap that disappears under SHD_x1 is attributed to the sufficiency cost of the latent burn-in, not to the start artefact.
- Negative rule: no significant swap means "no ranking change detected"; the benchmark-impact claim is then limited to absolute scores.

**Var- versus rel-sortability (descriptive, with predictions).** Var-SortnRegress should gain from G1 in R-het and R-eq (variance accumulates from the fixed start), with no gain in R-std; rel-SortnRegress is invariant across R-het and R-std by construction, and its G1-G2 difference should be similar in R-eq. Reported per family: G1-G2 differences of M1 and M3 order accuracy and of the three estimated sortabilities, in each regime, with CIs.

**Sanity rules (a failure is reported and limits interpretation).**
- S1: |mean OA_M1(G2) - OA_M1(G2s)| < 0.02 in every family (R-het). Otherwise the burn-in has not mixed.
- S2: mean estimated rel-sortability(G1) - (G2) > 0 in every non-SC family. Otherwise the relatives estimator does not see the artefact and H1 cannot be interpreted.
- S3: max |Sigma_B - Sigma| / max |Sigma| < 1e-3 over all test graphs, where Sigma_B is the covariance of x after B = 50 steps from x_1 = e_1 and Sigma the stationary covariance.

## 3. Sharpened bound (B2) and its check

Claim: for every T >= 2, 0 <= s_rho(G_T) - L <= min(Delta, T-1) |E_o| / (2(T-1)|E_S|), where Delta = max finite shortest-path distance in G_S (self-loops ignored). Proof idea: in the proof of Thm 1.5 an E_o copy at time t has value 1 only if t <= d(u,i) for some u with d(u,i) finite, and d(u,i) <= Delta. Likewise Thm 1.9 needs only B >= Delta (the proof uses only that every finite distance is at most B). Check by brute-force enumeration of relatives (no theorem used) on 3000 random summary graphs (p = 2..8; dense and sparse directed ER and the family generators) at T in {2,3,4,5,7,10,15}: count violations (any violation is reported with the counterexample), cases where the bound is attained, and cases where it is strictly sharper than the p-1 bound; and for the burn-in, whether B = Delta gives the stationary value and how often B = Delta - 1 does not.

## 4. Reporting

- `README.md`: verdicts with CIs and tests, deviations (none allowed for thresholds), a main-text paragraph with one table/figure, and appendix details. Per-data-set results are committed compressed (`results/b1_test.csv.gz`); large raw data are not saved.
- Any change to this plan after this commit is listed as a deviation with its reason.
