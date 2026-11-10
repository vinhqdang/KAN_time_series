# b1y: time-stamped plan for the decisive controls (matched-correlation counterfactual, variance-matched control, first-slice-primary scoring)

Roadmap items b1-b7 of `../mlj_paper/review/00_decision.md`. This is a time-stamped plan, not a registration: it is committed and pushed to the repository before any data are simulated and before any learner score is computed for any of the questions below. It is not posted to OSF.

## 0. What was computed before this commit (and what was not)

Computed and committed with this plan (graph-only or population quantities, scoring-convention unit tests, and dummy-data runs of the analysis code; no study data, no learner scores):
- `q6_burnin_exact.py` → `results/q6_burnin_exact.txt` (exact enumeration, §8).
- `q1_population_predictions.py` → `results/q1_population_predictions.{txt,csv.gz}` and `q1_dose_population.py` → `results/q1_dose_population.txt` (population correlations of every Q1 arm on the test graphs; the oracle predictions that the Q1 data are compared with).
- `test_b1y.py` (all pass): covariance identities, patterns of the counterfactual starts, exact covariance against Monte-Carlo simulation, variance-matching factors, scoring conventions (truth scores 1 and its transpose 0), exact-gap check against Thm 1.5/1.9, seed disjointness.
- `analyse_y.py --selftest` (passes): the decision rules below run on synthetic results with injected truths, and return the right verdict for each.

Not computed: any simulated data set, any rel-SortnRegress, FLOP, DAGMA, NOTEARS or other learner output for the seeds listed in §10, any validation-seed learner score. `run_q1.py` is written but has not been executed; `run_q2.py` and `run_q5.py` are stubs; `methods_y.py` conversions are untested on data.

## 1. Question, and what each outcome would mean (fixed now)

The paper's unified claim is: *the exact relatives gap of a time-unrolled DAG is the limit (n → infinity) of what the fixed start does to benchmark results; at finite n the realised effect is the detectable image of that gap, so its size depends on detectability (n, correlation strength), and the learner-ranking changes seen in B1x are a separate, variance-sortability effect.* B1 (+0.125 order-accuracy gain, slope 0.53 on the exact gap) and the post hoc oracle (cell R² 0.97, coefficient of the exact gap given the oracle −0.05) are compatible with this but do not test it. The oracle is the expectation of the same algorithm, so it cannot separate the graph-relatives mechanism from a general change in correlation strength. Q1 and Q2 are built to separate them by intervention.

Decision table, fixed before the run (U = unified-story verdict for the paper):

| Q1 result | | U |
|---|---|---|
| V2 = "detectable strength, not population pattern" and V4 supported | | **U-A**: the exact gap is the n → infinity ceiling; the effect at finite n is its detectable image. Paper: exact theory + finite-n detectability, with n- and correlation-dose curves. |
| V2 = "population pattern" | | **U-B**: the population relatives pattern drives the effect; the theory explains the mechanism directly. |
| V2 = "detectable strength" and V4 not supported | | **U-C**: the exact gap does not explain the effect even at n = 25,000 in the range tested. Paper: exact values and tight burn-in as a graph result, plus a cautionary benchmark section; the artefact claim is limited to var-sortability and detectability. |
| V2 = "mixed" | | **U-D**: both contribute; the exact gap bounds the effect, detectability modulates it. Report the retention numbers; no mechanism sentence beyond that. |
| V1 fails (negative control outside ±0.01) | | **STOP**: the design is invalid; no Q1 interpretation. |

Q2 verdicts feed a separate sentence: if HY1 = "variance channel", the DAGMA/NOTEARS vs FLOP change under the fixed start is attributed to var-sortability (with the manipulation check); if HY1 = "not variance", it is reported as unexplained by variance; if the precondition fails, the reversal is reported as not reproduced at tuned hyper-parameters.

## 2. Conventions common to all questions

- **Graphs, SEM, weights, noise:** B1's (`../b1/PLAN.md` §1.1-1.2): families symER, ring, chain, outtree, intree, ER1, ER2 (all with self-loops), positive weights Unif(0.5, 1) rescaled to spectral radius ≤ 0.8 (`weights_cap` with cap 0.8 reproduces `b1/common.weights` exactly), heterogeneous noise s.d. Unif(0.5, 1) (R-het), raw data, n = 1000, time labels hidden (columns randomly permuted per data set), as in B1 and B1x.
- **Generators:** G1 (x_1 = e_1), G2 (latent burn-in B = 100), G2s (exact stationary start). Ground truth is the T-slice window DAG G_T for all arms.
- **Primary scoring (Q3, applies to every question):** first-slice-excluded.
  - SHD_x1 = B1's SHD excluding the pairs that lie inside the first slice, divided by the number of true edges: **SHD_x1n**.
  - OA_x1 = order accuracy over the true edges whose tail is not in slice 1 (the edges leaving slice 1 are dropped), as in the B1x re-analysis (`orders_x1.csv.gz`).
  - Full-window SHD/|E| and OA are reported as secondary in the appendix. Reason: under G2 the latent burn-in makes slice-1 variables confounded and the slice-1 pairs are scored differently (0.044 vs 0.273 reference SHD with and without the first-slice pairs); excluding them removes the scoring-convention part of the G1-G2 gap.
- **Units and CIs:** the unit is the summary graph; the horizons T of one graph are averaged first. Percentile bootstrap over graphs, 10,000 resamples, RNG seed 0. Ratios use the same resample for numerator and denominator.
- **Retention** of an arm X on metric M: R_X = mean_g[M(X) − M(G2)] / mean_g[M(G1) − M(G2)], pooled over the non-strongly-connected families (chain, outtree, intree, ER1, ER2); SC families (symER, ring) are reported separately with the same formula.

## 3. Q1: matched-correlation counterfactual (roadmap b1)

### 3.1 Arms (all share the graph, weights and noise s.d. of the graph seed; start law differs)

The first slice is x_1 ~ N(0, Σ_1), then T − 1 steps. G1: Σ_1 = D (D = diag of noise variances). G2s: Σ_1 = Σ_stat. G2: after B = 100 latent steps (equal to Σ_stat up to the mixing error; sanity S1).

| arm | Σ_1 | population relatives pattern | detectable pattern at n = 1000 |
|---|---|---|---|
| G1 | D | G1's | G1's |
| **G1rep** (negative control) | D, a second independent data draw | G1's | G1's |
| G2, G2s | burn-in / Σ_stat | G2's | G2's |
| **CF1_c**, c ∈ {0.25, 0.5, 2, 4} | c·D (diagonal) | **G1's, exactly** | G1's pattern, correlation strength changes with c |
| **CF2_m**, m ∈ {1, 3, 10} | (1 − λ)D + λΣ_stat, λ = min(1, m·λ*) | **G2's, exactly** (any λ > 0) | m = 1: G1's up to near-threshold pairs (see S3); larger m: G2-like |

λ* is computed per (graph, T) from the exact population window covariance: the largest λ on a geometric grid (1e-7 to 1, 71 points) such that every window pair that is exactly uncorrelated under G1 has |ρ| ≤ 0.5·ε for every smaller λ on the grid (ε = 0.062, the Algorithm 1 threshold at n = 1000). So CF2_m1 has G2's population relatives (hence the same exact s_ρ as G2) but all newly created correlations are below half the detection threshold. Unit tests verify both pattern statements.

Graph-only predictions made before any data (`results/q1_population_predictions.txt`, 300 non-SC graphs, oracle = population correlations thresholded at ε): oracle retention CF2_m1 1.02, CF2_m3 0.99, CF2_m10 0.71; CF1_0.25 0.76, CF1_0.5 0.85, CF1_2 1.15, CF1_4 1.27. The pairs whose detectability flips between CF2_m1 and G1: mean 4.8, 95th percentile 20, in at most 6.7% of pairs, ≤ 1% of pairs in 85.5% and ≤ 2% in 95.5% of (graph, T) cases. These are the numbers the sample results are compared with; the verdict rules below are on the sample, not on the oracle.

### 3.2 Study design and size

- **Main study:** 7 families × p ∈ {5, 10} × seeds 4000-4029 (30 graphs per (family, p)) × T ∈ {3, 5, 10} × the 11 arms (G1, G1rep, G2, G2s, CF1 ×4, CF2 ×3): 13,860 data sets. Metric: order accuracy of rel-SortnRegress (rel-SnR order; OA_x1 primary, OA full secondary). SHD_x1n (Algorithm 2 on the rel-SnR order) only on arms G1, G2, CF1_0.25, CF1_4, CF2_m1 and T ∈ {3, 5}: 4,200 data sets.
- **Dose study (graph-level theory vs finite sample):** families ring, chain, ER2; p ∈ {5, 10}; seeds 4200-4219; T ∈ {3, 5}; arms G1 and G2s.
  - **n-dose:** n ∈ {200, 1000, 5000, 25000} at cap 0.8: 1,920 data sets.
  - **ρ-dose:** spectral-radius cap ∈ {0.5, 0.95} (0.8 is the n = 1000 row of the n-dose) at n = 1000: 960 data sets. Same raw weights as the cap-0.8 graph, rescaled to the cap.
  - Graph-only predictions (`results/q1_dose_population.txt`): oracle gain / exact gap, ring p = 5: n = 200: 0.43, 1000: 0.71, 5000: 0.88, 25000: 0.95, 10⁶: 1.00; cap 0.5 / 0.8 / 0.95: 0.25 / 0.71 / 0.85. Chain and ER2 rise in the same way (chain p = 10: 0.70 → 0.82 → 0.88 → 0.91; ER2 p = 10: 0.20 → 0.37 → 0.53 → 0.67).

### 3.3 Hypotheses, verdicts and thresholds (fixed now; metric OA_x1 unless stated; non-SC families pooled for V2, V3)

- **V1 negative control (validity).** d = OA_x1(G1rep) − OA_x1(G1), mean over graphs. Valid iff |mean d| < 0.01. If not valid: STOP (no other Q1 verdict is interpreted).
- **V2 population pattern at undetectable strength.** R = retention of CF2_m1 with graph-cluster bootstrap CI.
  - "Detectable strength, not population pattern": R ≥ 0.75 and CI lower bound ≥ 0.50.
  - "Population pattern": R ≤ 0.35 and CI upper bound ≤ 0.60.
  - Otherwise "mixed".
  - Retentions of CF2_m3 and CF2_m10 are reported (dose-response in detectable strength at G2's pattern).
- **V3 strength dose at G1's exact pattern.** Δ_c = OA_x1(CF1_4) − OA_x1(CF1_0.25). "Strength matters at fixed population pattern" iff the CI of Δ_c excludes 0 and |Δ_c| ≥ 0.25 × (mean OA_x1(G1) − OA_x1(G2)); otherwise "flat". Population relatives, and hence the exact s_ρ, are identical for all CF1 arms.
- **V4 n-dose (does the realised effect approach the exact gap?).** Q(n) = mean over graphs of [OA_x1(G1) − OA_x1(G2s)] divided by the mean exact s_ρ gap. Supported iff Q(25000) − Q(1000) ≥ 0.10 with the CI above 0 and Q(25000) ≥ 0.60.
- **V5 ρ-dose.** Q(cap 0.95) − Q(cap 0.5) ≥ 0.10 with the CI above 0.
- **Slope.** OLS slope of the graph-level gain (G1 − G2, OA_x1) on the exact gap, with family × p fixed effects removed (within-family slope) and a graph-cluster bootstrap CI (2,000 resamples); the pooled slope (B1: 0.53) is reported next to it. Reported with its CI, no threshold.
- **Oracle calibration (not a verdict):** observed OA of each arm against the oracle-predicted OA of that arm (R², slope, cell level); a consistency check of the simulation, not a mechanism claim.
- **SHD_x1n arms:** the same retention and dose statistics on SHD_x1n for the five SHD arms, reported without a threshold.
- **Sanity rules (a failure is reported and limits the interpretation):**
  - S1: |mean OA_x1(G2s) − OA_x1(G2)| < 0.02 (burn-in mixed).
  - S2: CF2 definable: the share of (graph, T) with λ* at the grid floor or degenerate ≤ 10% (population value 0.2%).
  - S3: detectability of CF2_m1 equals G1's up to near-threshold pairs: ≤ 2% of pairs flip in ≥ 90% of (graph, T) (population value 95.5%).
  - S4: the rel-SnR G1 − G2 gain on the non-SC families is positive with CI above 0 (B1's result reproduces on new seeds, OA_x1 metric).

### 3.4 Cost

13,860 OA-only data sets at about 0.1 s, 4,200 Algorithm-2 runs at about 0.3 s, 2,880 dose data sets at about 0.1 s: about 1.0 CPU-h. Cap 1.5 h.

## 4. Q2: variance-matched control and tuned learners (roadmap b2, b3)

### 4.1 Design

- **Cells:** families chain, intree, ring, ER2; (p, T) ∈ {(5,3), (5,5), (5,10), (10,3), (10,5), (20,3)} (d = 15, 25, 50, 30, 50, 60). Test seeds 4300-4311 (12 graphs per (family, p)); validation seeds 9810-9812 (3 per (family, p)).
- **Arms:** G1, G2, **VM_G1** (the G1 data with every column multiplied by the exact population factor sqrt(v_G2 / v_G1), so its marginal variances equal G2's), **VM_G2** (the G2 data rescaled to G1's variances). Rescaling does not change correlations: rel-SortnRegress, FLOP and PC are invariant (the code asserts it); raw DAGMA and NOTEARS are not. VM arms reuse the G1/G2 data sets, no new simulation.
- **Methods:** FLOP (flopsearch 0.3.0, BIC 2, 10 restarts, standardised input), DAGMA-linear (tuned), NOTEARS-linear (tuned; only cells with d ≤ 30: (5,3), (5,5), (10,3), because a d = 50 NOTEARS fit on a chain took 151 s in B1x), Var-SortnRegress, rel-SortnRegress, empty graph. GOLEM is dropped (p = 5 only, 15-40 s per fit).
- **Hyper-parameter tuning, validation seeds only:**
  - DAGMA: λ1 ∈ {0.01, 0.03, 0.1} × w_threshold ∈ {0.1, 0.3}; NOTEARS: λ1 ∈ {0.01, 0.1} × w_threshold ∈ {0.1, 0.3}.
  - Criterion: mean SHD_x1n over the validation graphs, pooled over families, cells and both generators G1 and G2 on raw data; one setting per method, used for all four arms. Pooling over the generators prevents tuning to the artefact. Ties go to the package default.
  - The choice is written to `results/q2_tuned.json` and committed before the test run.

### 4.2 Hypotheses and verdicts (SHD_x1n; unit = graph; A(arm) = SHD_x1n(FLOP) − SHD_x1n(CO), positive means the CO method is better)

- **Precondition** (per CO method, in-trees, the family of the B1x reversal): A(G1) − A(G2) ≥ 0.10 with CI above 0. If it fails, "reversal not reproduced at tuned hyper-parameters" and HY1 is not evaluated.
- **HY1 (is it variance?).** R_VM = [A(VM_G1) − A(G2)] / [A(G1) − A(G2)].
  - "Variance channel": R_VM ≤ 0.40 and CI upper bound ≤ 0.60.
  - "Not variance": R_VM ≥ 0.75 and CI lower bound ≥ 0.50.
  - Otherwise "mixed".
  - Injection: R_inj = [A(VM_G2) − A(G2)] / [A(G1) − A(G2)]; variance alone is sufficient iff R_inj ≥ 0.60.
- **HY2 (does the sign pattern survive tuning?).** For in-trees: A(G1) > 0 and A(G2) < 0, both with CI excluding 0. Reported for all four families.
- **HY3 (FLOP's own shift).** FLOP SHD_x1n(G1) − SHD_x1n(G2), paired by graph, with CI; and the interaction [CO − FLOP](G1) − [CO − FLOP](G2) per family with CI (both reported for SHD_x1n, SID and ancestral order accuracy).
- **Manipulation check (sanity):** Var-SortnRegress order accuracy under VM_G1 within 0.03 of its value under G2, and under VM_G2 within 0.03 of its value under G1. If it fails, the rescaling did not remove the variance information and HY1 is not interpreted.
- **Below-chance diagnosis (step 0, validation seeds, no verdict):** the B1x ancestral order accuracy of standardised DAGMA and GOLEM was 0.31-0.42. Unit tests (already passing) show the scoring maps the truth to 1 and its transpose to 0. On validation data sets report for standardised DAGMA the shares of true edges oriented forwards, backwards, or left incomparable; if backwards exceeds forwards, report it as a genuine reversed orientation and not as a scoring error.

### 4.3 Cost

Tuning: DAGMA 6 × 144 fits plus NOTEARS 4 × 72 fits on 3 validation seeds: about 1.0-1.5 CPU-h. Test: 576 base data sets × 4 arms; DAGMA about 2,300 fits, NOTEARS about 1,150 fits, the rest trivial: about 3.5 CPU-h. Cap 5.5 h in total. **Size rule fixed now:** if the validation timing projects the test above 4 h, the test seeds are cut to 4300-4307 before the run and this is logged as a deviation.

## 5. Q3: first-slice-excluded scoring as primary (roadmap b3)

- All Q1, Q2 and Q5 results use SHD_x1n and OA_x1 as primary, as in §2.
- **Post hoc re-analysis of committed data (no new data; separate from the studies above, labelled post hoc):** from `b1/results/b1_test.csv.gz` (columns `*_shdx1`, `n_true`, `*_oa`), `b1x/results/b1x_test.csv.gz` (`*_shdx1`) and `b1x/results/orders_x1.csv.gz` (`M*_oa_x1`), recompute the paper's Table 2 (pooled gains), Table 3 (rank-change cells and interactions) and Fig. 3 with the x1 metrics and graph-cluster CIs, and report old and new side by side. Written after approval; computed only from the columns named here.

## 6. Q4: run-level inference for the real-data tests (roadmap b5; analysis plan on existing results, no new data)

- **Unit rule.** The unit of independence is the independent run or station, not the segment: segments of one series are serially dependent and subgraphs share stations.
- **B3** (`b3/results/results_b3.json`): H1 has 3 runs (ref, pol_1_90, pol_12_30) of 10 segments each; H2 has 2 runs; S4 has 3. Exact one-sided sign test over runs: smallest attainable p is 1/8 (3 runs) and 1/4 (2 runs), so no run-level test can reach 0.05. Reported as "k of m runs agree", with the 10 segments inside each run as a descriptive within-run interval (block bootstrap over contiguous segments). The Holm family does not include the segment-level p-values of B3.
- **E1** (`realdata/results3.json`): one series per study (WT: one run; LT: one file), the units being its consecutive segments. No inferential test; descriptive interval by circular block bootstrap with block length at least the longest node autocorrelation time of the series, stated per study. Claims are written as "in this run".
- **CausalRivers** (`empirical2/results2.json`, `subgraphs_8000.json`: 120 subgraphs, node lists, regions): build the station-sharing graph (two subgraphs adjacent if they share a station) and report its size distribution. Two analyses: (a) station-disjoint subsets: random maximal subsets with pairwise disjoint station sets (200 random orderings, seed 0), the statistic and CI per subset reported as the median over orderings with the share of orderings in which the criterion holds; (b) a station bootstrap (resample stations; a subgraph enters with weight equal to the share of its stations drawn). The 12 h and 24 h columns are the same records: one pre-named horizon per claim (the paper's headline horizon), the other reported as a robustness row, never pooled as replicates.
- **K-sensitivity (roadmap b8):** the plug-in test of Section 5 of the paper on E1, study 3 and the river data with K ∈ {20, 50, 100, 300}; the number of decisions that change, as B3 already reports (0 of 2,730).

## 7. Q5: extra baselines as references (roadmap b6)

- **Data:** fresh seeds 4100-4111 (12 per (family, p)), the B1x design (families ring, chain, intree, ER2; p ∈ {5, 10}; T ∈ {3, 5}; G1, G2), so 384 data sets, with FLOP, rel-SnR, Var-SnR and the empty graph re-run on the same data.
- **Label-free added methods:** GES (causal-learn, BIC), GRaSP (causal-learn, depth 3), BOSS (causal-learn), DirectLiNGAM (lingam, pwling, only d ≤ 30). Outputs are CPDAGs: undirected edges count as in B1 (an undirected true edge pair is an error of 1).
- **Labelled references:** PCMCI+ (as in B1x; run again for this seed set) and a pooled lag-1 OLS threshold (`var_ols_labelled`).
- **Not available here, dropped from the plan:** DYNOTEARS (the only PyPI package, causalnex, does not install on Python 3.13), VAR-LiNGAM applied to i.i.d. short trajectories (not defined for 1,000 trajectories of length ≤ 5), Tetrad-based BOSS/GRaSP via py-tetrad (not installable). The lag-1 OLS reference stands in for the linear lag-1 labelled reference.
- **Timing gate (step 0, validation seed 9820):** a method enters a cell only if one fit takes at most 60 s at that d; the projected total must be at most 2 CPU-h, otherwise the d ≤ 30 restriction is extended to all added methods.
- **HB1:** "FLOP remains the best label-free method" iff FLOP has the lowest mean SHD_x1n among {FLOP, GES, GRaSP, BOSS, DirectLiNGAM where run, rel-SnR, Var-SnR} in at least 6 of the 8 (family, generator) groups. Otherwise the paper's "FLOP best" statement is replaced by the table. Descriptive; no p-value.

## 8. Q6: per-graph minimal burn-in and tightness for general Δ (roadmap b4) — computed now

`q6_burnin_exact.py` (exact enumeration with `rel_window_enum`, independent of the theorems; B_ref = Δ + 2p; results in `results/q6_burnin_exact.txt`). B_min is the smallest B whose window relatives counts (B_min_cnt) or window rel-sortability (B_min_s) equal the stationary value, for every B' ≥ B up to B_ref.
- **Directed chain with self-loops, p = 2..8 (Δ = p − 1), T = 2..5:** B_min = Δ in every case, for both measures. So B = Δ − 1 is insufficient for every Δ from 1 to 7 and every T from 2 to 5.
- **Directed ring with self-loops, p = 2..8:** B_min = ⌈Δ/2⌉ (= ⌈(p−1)/2⌉) for every T from 2 to 5.
- **Random summary graphs** (396 graphs, p = 3..7, T ∈ {3, 5}, 792 cases, fresh RNG key 499): no violation of B_min ≤ Δ; B_min = Δ in 406 of 792 (51%); the slack Δ − B_min is 0 / 1 / 2 / 3 / 4 in 406 / 250 / 114 / 20 / 2 cases. B_min_cnt and B_min_s coincide in every case.
- **A proof for every p (to be written into the paper and, if possible, Lean):** for the chain, at B = Δ − 1 = p − 2 the window node (first window slice, chain node 0) and the window node (first window slice, chain node p − 1) have no common ancestor: the common ancestors of the first are the earlier copies of node 0, and none of them reaches chain node p − 1 within p − 2 steps. In the stationary window they are related through the earliest copy of node 0. So the relatives counts differ from the stationary ones at B = Δ − 1, for every p ≥ 2 and every T ≥ 1. This is a one-line argument; the numbers above are a check, not the proof.

## 9. Budget, order and stop rules

| question | decision value | CPU-h (cap) | stop / kill rule fixed in advance |
|---|---|---|---|
| Q6 | tightness for general Δ | 0.01 (done) | none |
| Q1 | separates graph pattern from detectability; sets the story U | 1.5 | V1 fails → stop Q1 interpretation; S1-S3 failures reported and limit the verdicts |
| Q2 | learner-ranking change: tuned? variance? | 5.5 | precondition fails → HY1 not evaluated; size rule of §4.3 |
| Q3 | scoring convention | 0.1 (post hoc re-analysis) | none |
| Q4 | cluster inference for real data | 0.3 | none; claims are rewritten as run-level counts |
| Q5 | missing baselines | 2.2 | timing gate of §7 |
| **total** | | **9.6** (cap 12) | the total cap is fixed; if Q1 + Q2 exceed their caps by more than 20%, Q5 is dropped |

Order: Q1 first (cheap, decides the story), then Q2, then Q5; Q3 and Q4 are analyses of committed data and may run at any time. One core, `OMP_NUM_THREADS=1`.

## 10. Seeds and RNG streams (fresh; disjointness proved in `test_b1y.py::test_seed_disjointness`)

- Graph/parameter stream `default_rng([401, family_index, p, seed])`; data stream `default_rng([402, family_index, p, seed, T, arm_index, n, rho_index, 17])`.
- Seeds: Q1 main 4000-4029; Q1 dose 4200-4219; Q2 test 4300-4311; Q5 test 4100-4111; validation 9800-9804 (Q1), 9810-9812 (Q2), 9820 (Q5).
- Earlier streams and seeds (all disjoint from the above): B1 `[fi, p, s, ...]` (fi 0-6) with seeds 1000-1029 and 9000-9004; B1-NL prefixes 101, 102, 201, 202, 203 with seeds 2000-2024 and 9100-9102; B1x prefixes 301, 302 with seeds 3000-3011 and 9600.
- Each test seed is run once. Validation seeds are used for debugging, timing, and the Q2 hyper-parameter choice only; no hypothesis is tested on them.

## 11. Multiplicity

- **Claim family F (Holm at 0.05, one-sided bootstrap p-values, floor 1/10,001):** C1 = V3 (Δ_c ≠ 0), C2 = V4 (n-dose difference > 0), C3 = V5 (ρ-dose difference > 0), C4 = within-family slope > 0, C5 = DAGMA precondition (A(G1) − A(G2) > 0 on in-trees), C6 = NOTEARS precondition, C7 = FLOP's own G1 − G2 shift ≠ 0 (SHD_x1n, all families pooled). |F| = 7.
- Verdicts defined by thresholds on ratios or levels (V1, V2, HY1, HB1, S1-S4) are descriptive rules on CIs and are not Holm-adjusted.
- Secondary quantities (SID, ancestral order accuracy, SHD full-window, SC-family rows, per-family rows) are reported with CIs, not tested.
- The earlier Holm family of 21 claims (B1, B1-NL, B3, B1x) is not extended; these seven are reported as a separate family. The study sequence is adaptive (each plan followed the previous results); this is stated in the paper.

## 12. Reporting and deviations

- `README.md` (updated after the runs): the verdicts with CIs, every deviation with its reason, the U verdict, and the push time of this plan (recorded from the hosting server).
- Any change to this plan after its commit is listed as a deviation. Results are committed compressed; shard files are ignored.
- **Needed from the user before any run:** approval of the design and budget (about 9.6 CPU-h on one core), and, if wanted, an external registration (OSF) of this plan before Q1 and Q2 are run.
