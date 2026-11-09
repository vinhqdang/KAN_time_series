# B1-NL analysis plan: does the B1 start artefact survive nonlinear, signed and non-Gaussian parametrisations?

Pre-registered replication of B1 (`../b1/PLAN.md`, `../b1/README.md`). It answers the criticism that B1 used "only one linear-Gaussian parametrisation with positive weights". Written and committed (and pushed) before any simulation that produces an algorithm score. Computations run before this commit are parameter- or graph-only, with no method applied to data:
- exact s_rho of every new test summary graph (`results/exact_predictions.txt`);
- the coupling (mixing) check S3 on every test parameter set: max 5.6e-5 (NL), 6.8e-6 (SGN; SGNU has the same weights);
- the Lipschitz constant of g3 (1.2745) and, on the 3 validation parameter sets per (family, p) only, the squared correlation between x_i and g(x_i) at stationarity (strength of the nonlinearity): mean 0.92 (tanh), 0.81 (sin; minimum 0.004), 0.80 (g3).

The code (`common_nl.py`, `run_b1nl.py`) is committed with this plan. It imports the graph generators, exact s_rho, estimators, Alg. 2 and metrics from `../b1/common.py` unchanged; committed B1 files are not modified.

## 0. Question and threat model

Same as B1 §0. The pT window variables are handed to i.i.d. methods as anonymous columns (time labels hidden). With labels visible every edge is oriented by time; Time-SnR (R0) is reported as a reference only.

## 1. Design

### 1.1 Summary graphs, horizons

Identical to B1 §1.1: families symER, ring (strongly connected, SC), chain, outtree, intree, ER1, ER2 (non-SC), p in {5, 10}, self-loop on every node, T in {3, 5, 10}. Same generator code (`summary_graph`), new seeds and a new RNG stream (§1.4), so the graphs are new draws.

### 1.2 Parametrisations (fixed now)

Noise s.d. sd_v ~ Unif(0.5, 1) per summary node (B1's R-het). Spectral cap 0.8 as in B1.

- **NL (nonlinear additive VAR(1), Gaussian noise).** x_{t+1,j} = a_jj x_{t,j} + sum_{i != j, (i,j) in E_S} w_ij g_ij(x_{t,i}) + e_{t+1,j}. All a, w ~ Unif(0.5, 1) (positive, as B1, so that NL changes only the functional form). g_ij is drawn uniformly per edge from {tanh(x), sin(x), g3(x) = (x + x^2/2)/(1 + x^2/4)}: all bounded, slope 1 at 0; sin is non-monotone; g3 has an even ("quadratic-type") component and Lipschitz constant 1.27. Self-loops are linear. Stability: if the spectral radius of the Lipschitz bound matrix M (M_jj = a_jj, M_ij = w_ij Lip(g_ij)) exceeds 0.8, all weights are scaled to make it 0.8; then the map is a contraction in a weighted norm and the chain mixes geometrically.
- **SGN (linear, signed weights, Gaussian noise).** Every summary edge, self-loops included, gets weight s·Unif(0.5, 1) with s = ±1 with probability 1/2 (independent). Rescaled to spectral radius 0.8 if larger.
- **SGNU (linear, signed, uniform noise).** The same graphs and weights as SGN (paired), with e_v ~ Unif(−√3 sd_v, √3 sd_v) (same variance). This is the optional non-Gaussian configuration.

**Relatives under these parametrisations (stated in advance).**
- The exact s_rho used throughout is the *graph* value: Thm 1.5 (G1) and Thm 1.9 (G2), i.e. relatives = common ancestor in the unrolled DAG. Those theorems are graph-only. They equal the population "correlation relatives" only when every trek-connected pair is correlated.
- SGN/SGNU: with signed weights, contributions of different treks can cancel, so a graph relative can have zero or near-zero correlation (exact cancellation has measure zero in the weights, near-cancellation does not). The sample relatives found by the correlation-count estimator may therefore deviate from the graph relatives; we expect a modest attenuation relative to B1, same sign.
- NL: dependence passing through bounded, saturating or non-monotone functions (sin) is only partly visible to a correlation-count estimator, and it decays faster along long paths. The estimator should therefore detect the graph relatives only partially (lower recall than linear, especially for distant relatives). Expected direction: the G1 − G2 gain in rel-SnR order accuracy stays positive but is attenuated relative to B1 (we expect the OLS slope of d on the exact gap to fall below B1's 0.53). We report the recall and false-positive rate of the estimator against the graph relatives per data set (`rel_recall`, `rel_fpr`) to show how much of the graph relation the estimator sees.
- Alg. 2 (linear adaptive Lasso) and PC-type Gaussian scores are misspecified under NL; this is deliberate (the methods are those benchmarked by Reisach et al.), and the comparison is G1 vs G2 within the same method.

### 1.3 Generators

- **G1** fixed start x_1 = e_1, then T − 1 steps. Ground truth G_T.
- **G2** latent burn-in B = 100 (B1 after D1), keep the next T slices. Ground truth: copy of G_T (same sufficiency caveat as B1; SHD_x1 reported).
- **G2L** burn-in 500, used only for sanity rule S1 (rel-SnR order accuracy on an independent G2L data set, computed in the G2 row). G2L replaces B1's exact stationary start G2s, which has no closed form under NL or uniform noise.
- n = 1000 trajectories per data set.

### 1.4 Seeds and size (one core, about 2–3 h)

- Test graphs: seeds 2000–2024 (25 per (family, p)) for NL and SGN; seeds 2000–2009 (10 per (family, p)) for SGNU. Graph and parameter RNG `default_rng([key, family_index, p, seed])` with key 101 (NL) and 102 (SGN and SGNU, paired); data RNG `default_rng([data_key, family_index, p, seed, T, gen_index, 11])`, data_key 201/202/203. All distinct from B1 (seeds 1000–1029, keys without a prefix).
- Validation seeds 9100–9102 (3 per (family, p) and config): debugging and timing only; not reported as evidence; no tuning (every hyperparameter is fixed in advance from B1).
- Data sets: (25 + 25 + 10) × 7 × 2 × 3 × 2 = 5,040. Projected cost from B1's per-data-set timings without PC: about 2.1 h on one core.
- **Pre-stated size rule.** If the validation run projects more than 3.0 h single-core for the test run, the seed counts are cut to 20 / 20 / 8 (first seeds of each list) before the test run, and this is logged as a deviation.
- Reductions relative to B1, fixed now for the one-core budget: fewer graphs per cell (25 or 10 instead of 30); **PC is not run** (it dominated B1's run time); the R-eq regime and G2s are dropped (G2L replaces G2s). R-std is kept (it only changes Var-SnR and the estimated var-sortability; the other methods are scale invariant).
- The test seeds are run once, in resumable shards (one file per (config, family, p), written atomically).

### 1.5 Methods (time labels hidden; columns randomly permuted per data set)

M1 rel-SortnRegress, M2 R²-SortnRegress, M3 Var-SortnRegress (raw; std_M3 on standardised data), M4 Random-SortnRegress, M5 FLOP (flopsearch 0.3.0, BIC penalty 2, 10 restarts, standardised input), R0 Time-SnR (labels visible, reference only). Exactly B1's implementations (`../b1/common.py`, `run_b1.py`). No PC (§1.4).

### 1.6 Metrics

As B1 §1.6: exact s_rho and gap, estimated var-/R²-/rel-sortability, SHD, SHD_x1, order accuracy (OA), orientation score, SID (gadjid). New diagnostics: `rel_recall`, `rel_fpr` (§1.2), `coupling_err` (S3).

## 2. Hypotheses and decision rules (thresholds fixed now; identical to B1 where possible)

Each configuration c in {NL, SGN, SGNU} is analysed separately with the same rules. Bootstrap: percentile, 10,000 resamples, seed 0. Wilcoxon signed-rank, `zero_method="wilcox"`. Holm within configuration.

**H1-c (fixed start helps rel-SnR on non-SC families).** For each non-SC family, unit = summary graph (p, seed), d = mean over T of [OA_M1(G1) − OA_M1(G2)]: 50 units per family (NL, SGN), 20 (SGNU). One-sided Wilcoxon (d > 0), Holm over the 5 families, alpha = 0.05.
- Supported if ≥ 4 of 5 families are Holm-significant and the pooled mean d over the non-SC units is ≥ 0.02.
- Kill: < 3 of 5 significant or pooled mean d < 0.01 ("the start artefact does not reach rel-SnR under c"). Any non-SC family significant in the opposite direction (two-sided, Holm) is reported as contradicting the theory (reversal).
- In between: "not supported".

**H1-c tracking.** Spearman correlation between cell-mean d and cell-mean exact gap over the 42 cells (7 families × 2 p × 3 T) ≥ 0.5 with one-sided p < 0.05. Also reported: unit-level Spearman and the OLS slope (compared descriptively with B1's 0.53 and the attenuation prediction of §1.2).

**H3-c (ranking changes).** Methods M1–M5. Per cell and method pair, two-sided Wilcoxon of the paired SHD difference over the seeds under G1 and under G2 separately. A significant rank swap is a pair with opposite-sign mean differences that is significant under both after Holm over all (pair × cell × generator) tests of the configuration (840 tests). Supported if ≥ 1 significant swap on SHD. Also reported, not used for the verdict: SID, orientation, OA, SHD_x1 swaps; best-method changes; Kendall tau. Note: with 10 seeds (SGNU) the smallest two-sided Wilcoxon p is 0.002, which cannot survive Holm over 840 tests (needs < 6e-5), so **H3-SGNU is not testable** and is reported as such (descriptive only); H3 is decided for NL and SGN.

**H4-c (var- vs rel-artefact split; B1's descriptive analysis turned into a rule).** Pooled over all 7 families and T (unit = data-set pair):
- (a) Var-SnR OA gain G1 − G2 on raw data ≥ 0.05 with 95% CI > 0 (the fixed start creates var-sortability);
- (b) Var-SnR OA gain on standardised data has |mean| < 0.02 (it disappears under standardisation);
- (c) rel-SnR OA gain (scale invariant, so identical under standardisation) has 95% CI > 0.
The split holds if (a), (b) and (c) hold. (b) is close to automatic, since standardised variances are all equal and Var-SnR then orders at random; the content is in (a) and (c).

**Replication summary (fixed now).** "B1 replicates beyond the linear-Gaussian positive-weight case" if H1 and H1-tracking are supported in both NL and SGN; "partial" if in exactly one of them; "fails" otherwise. SGNU is reported as a robustness check (H1 and tracking with its own verdict; no H3).

**Secondary (descriptive, as B1 H2).** SC families per cell: mean d with CI; whether some SC cell with T < p has d ≥ 0.03 with CI > 0.

**Sanity rules (a failure is reported and limits interpretation).**
- S1: |mean OA_M1(G2) − OA_M1(G2L)| < 0.02 in every family and configuration.
- S2: mean estimated rel-sortability G1 − G2 > 0 in every non-SC family (per configuration). Under NL a failure would mean the estimator does not see the artefact; H1-NL is then uninterpretable.
- S3: max coupling error < 1e-3 (already checked on all test parameter sets: pass).

## 3. Reporting

`README.md` with verdicts, CIs and tests, deviations (no threshold changes), a main-text paragraph and a compact table, and an honest reading even if the effect vanishes or reverses. `results/b1nl_test.csv.gz` (merged shards) is committed; shard files are ignored by `.gitignore`. Any change to this plan after this commit is a deviation, logged with its timing (before or after the test run).

## 5. Notes and deviations recorded after the validation run and before the test run

- **Validation run** (seeds 9100–9102, all three configs, 756 data sets): pipeline debugged, no tuning. Timing 403–451 s per 3 seeds per config, i.e. a projected 2.3 h single-core for the test run (< 3.0 h), so the **full size of §1.4 is kept** (25 / 25 / 10 seeds). Validation scores are not evidence; with 3 seeds no H1/H3 test can reach Holm significance.
- **N1 (implementation, no effect on results).** `window_relatives` (diagnostic `rel_recall`/`rel_fpr`) was rewritten from an integer-matmul loop to a float-matmul transitive closure by repeated squaring (1.6 s → 0.02 s per call). Verified identical output on all 252 validation (graph, T, generator) cases (0 mismatches). Done between the NL and the SGN/SGNU validation runs.
- **N2 (power of S1, no rule change).** S1 compares two *independent* data sets (G2 vs G2L), so it carries sampling noise: on validation the per-pair s.d. of OA_M1(G2) − OA_M1(G2L) is 0.09 (NL), 0.07 (SGN, SGNU). At test size the standard error per family is about 0.007 (NL, SGN; 150 pairs) and 0.009 (SGNU; 60 pairs), so the 0.02 threshold is 2.2–2.7 SE and a chance failure in one of 7 families is plausible (especially SGNU) even with perfect mixing. The rule is kept unchanged; the per-family 95% CI is reported alongside, and mixing itself is checked directly by S3 (coupling).
- The analysis script `analyse_b1nl.py` implements §2 as written and is committed with this note.
