# BASIL (Bayesian spline additive DAG, order MCMC): results grid

Formulation and hypotheses: research/04_bayes_formulation.md (committed before any code ran).
Hyperparameters were NOT tuned for the headline numbers: g = n, pi = 2/(d-1), kmax = 3, 4 chains x (500 burn-in + 2500 iters). A validation sweep on seeds 0-9 (d=10, n=1000) over g in {n, n/5, n/20}, pi x {1,2} changed nothing material, so the a-priori default was kept.
Test seeds 1000+; reference baseline for all paired tests = SPADE-adaptive (the old heuristic pipeline). Raw: results/all_cells.csv, summary: results/all_cells_analysis.txt, scripts: bench.py, basil.py, aggregate.py.

## Cells actually run (planned grid was larger; cut for compute, see "Deviations")
d=10 n=1000: unif(30 datasets), gauss(21), laplace(20), hetero(20) | d=10 n=500, n=2000 unif (20 each) | d=20 n=500, n=2000 unif (10 each) | d=50 n=500 unif (6), n=2000 unif (1 only).
SCORE/NoGAM only on d=10 n=1000 unif, first 10 seeds, plus partial gauss/laplace.

## Headline (means; BASIL minus SPADE-adaptive, 95% bootstrap CI)
| cell | AUROC | AUPRC | orientation | curve slope |
|---|---|---|---|---|
| d10 n1000 unif | -0.018 [-0.037,+0.002] | +0.012 | -0.001 | +0.071 |
| d10 n500 unif | -0.019 [-0.043,+0.003] | +0.053 | +0.026 | +0.058 |
| d10 n2000 unif | **-0.045 [-0.072,-0.020]** | -0.020 | -0.003 | +0.049 |
| d10 n1000 gauss | +0.001 | **+0.097 [+0.054,+0.143]** | **+0.085 [+0.055,+0.117]** | +0.109 |
| d10 n1000 laplace | -0.006 | **+0.121 [+0.054,+0.191]** | **+0.138 [+0.089,+0.184]** | +0.238 |
| d10 n1000 hetero (ANM violated) | -0.021 [-0.046,-0.001] | +0.013 | +0.033 | +0.063 |
| d20 n500 unif | -0.022 [-0.042,-0.004] | 0.000 | -0.018 | +0.050 |
| d20 n2000 unif | **-0.039 [-0.055,-0.023]** | -0.028 | -0.017 | +0.039 |
| d50 n500 unif (6 datasets) | +0.070 | +0.301 | +0.093 | +0.938 |
BASIL absolute: slope 0.97-1.00 in every cell; 90%-band coverage 0.76-0.81; ECE 0.003-0.021 (probabilities are well calibrated).

## Verdicts on the pre-registered hypotheses
- H1 (AUROC >= SPADE-adaptive - 0.02, CI-based): **FAILS** in most cells (d10 unif at all n, d20, hetero). Loss up to -0.045. Passes only for Gaussian noise (and laplace by point estimate).
- H2 (orientation >= -0.03): holds by point estimate in every cell (range -0.018..+0.138); in d10 n1000 unif the CI lower bound is -0.035. Kill line (-0.05) never reached. Gains are significant under Gaussian/Laplace noise.
- H3 (calibrated curves, slope 0.85-1.10, coverage >= 0.8): slope **PASS everywhere** with no debiasing step. Coverage is 0.76-0.81: met in about half the cells, never near the kill line (0.6).
- H4 (ECE <= 0.10): **PASS** everywhere (<= 0.021).
- H5 (controls): var-/R2-sortnregress 0.61-0.67 AUROC, >= 0.3 below BASIL: PASS. Hetero stress reported above: AUROC -0.021, orientation +0.033.

## Other comparisons
- SCORE / NoGAM (d=10 n=1000 unif, 10 datasets): AUROC 0.858 / 0.895 vs BASIL 0.966, SPADE 0.980; orientation 0.80 / 0.85 vs 0.92 / 0.92.
- GAM-ridge (no DAG) has the best AUROC at d=50 (0.99 vs BASIL 0.96) and at d=20 n=2000 is slightly above BASIL (0.967 vs 0.951); its orientation is 0.65-0.79 in every cell. So DAG learners help orientation, not skeleton.

## Cost
BASIL: 2 s (d=10), 7-8 s (d=20), **~11-12 min per dataset at d=50** (subset enumeration, kmax=3). GAM-ridge at d=50: 3-9 min. SPADE: 7-40 s. Scaling beyond d=50 is not feasible as implemented.

## Caveats and diagnostics
1. d=50 vs SPADE is NOT a fair comparison: SPADE ran with its default lambda=0.01, which collapses at d=50 (curve slope 0.00, AUROC 0.87). The project's scaled variant (lambda=c/d + screening) was not included here. The d=50 numbers favour BASIL partly for that reason; GAM-ridge, with no tuning, beats BASIL on AUROC there.
2. d=50 has 6 datasets (n=500) and 1 dataset (n=2000): no conclusions on n=2000.
3. Post-hoc check (tie_check.py, not pre-registered): BASIL's AUROC loss is NOT due to ties among near-zero probabilities. Breaking ties by edge strength changes AUROC by <0.003. The loss comes from confidently missed or reversed edges (88% of off-diagonal pairs have P<1e-3).
4. Chain convergence: R-hat is meaningless when traces are constant (values like 47274 appeared); chain_gap (max abs difference in edge probabilities across chains) was 0.03-0.08 at d=10.
5. In an early validation run the true order scored 25-46 nats BELOW the order found by MCMC: the posterior favours wrong orders on some datasets, so errors are model errors (Gaussian likelihood, kmax=3, g-prior), not sampler errors.
6. Order-MCMC prior bias over orders is not corrected.
7. Prior-art overlap with DiBS, BayesDAG, GP-based order methods remains unresolved (04_bayes_formulation.md section 0).

## Deviations
- Grid cut for compute: planned d in {10,20,50} x n in {500,2000} with 30 datasets each; actual dataset counts above. SCORE/NoGAM run only on one full cell.
- Validation g-grid extended downward from {n, 5n} to include n/5, n/20 after observing missed weak edges at g=n (kept default g=n for all reported numbers).
- 1000-seed test pool reused across cells (same seed numbers, different d/n/noise), so cells are not independent draws of the same graphs.
