# Gap B: does held-out forecast loss select good causal graphs? (CausalRivers, netsim, synthetic SVAR)

Pre-registration: `PLAN.md` (commit 12cfdcc, committed before any code touched data; rules were not changed afterwards). Full tables: `results_test.md` (test seeds), `results_val.md` (validation seeds, runtime/degeneracy check only). Per-setting rows: `results_test_*.csv`; per-dataset statistics: `perdataset_test_*.csv`; necessity rows: `nec_test_*.csv`. No raw data in the repo (CausalRivers `product.zip`, netsim `.mat` files and caches live outside it).

Run order: `sample.py` (subgraphs, seeds 6000-6029 -> `subgraphs.json`), `run.py val|test [synth rivers netsim]`, `analyze.py val|test`. Library: `gapb.py`; `common.py`, `lib0.py`, `methods0.py` are unmodified copies of stage0 `common.py`, `lib.py`, `methods.py` (loading, aggregation, DYNOTEARS solver). Python 3.13, tigramite 5.2.10.1, lingam 1.13.0 (`--no-deps`), at most 2 processes, single-threaded BLAS.

## Design in one paragraph
Five model families with a regularisation or threshold knob and a native one-step forecast (lasso-VAR(1), thresholded-and-refit VAR(1), DYNOTEARS, PCMCI+ with a parent-restricted VAR, VAR-LiNGAM with a coefficient threshold), grids of 6 to 8 settings fixed in the plan. Chronological 70/30 split, everything fitted on the training part, selection by validation one-step MSE, ground truth used only for scoring. Per dataset: rho = Spearman(-validation MSE, directed AUROC) over non-empty grid settings; gain = AUROC(selected) - mean AUROC over the grid; controls: sparsity-only (edge count closest to d) and training BIC. Test data: CausalRivers 8 cells x 25 subgraphs (seeds 6005-6029; 12 h and 24 h; East Germany and Bavaria; d=5 and 10; 200 datasets, 100 clusters), netsim (21 sims with <=15 nodes, 20 subjects each, 420 datasets, clusters = sims), synthetic SVAR (50 datasets). Medians have 95% cluster-bootstrap CIs; gain tests are one-sided sign-flip tests on cluster means with Holm correction.

## Results (test seeds, run once)

### Gate S, synthetic linear SVAR (sanity): PASS
Median rho 0.74 (lasso), 0.69 (thresholded VAR), 0.51 (DYNOTEARS), 0.60 (PCMCI+), 0.76 (VAR-LiNGAM); all CIs above 0.4. Forecast loss tracks graph accuracy when the model is well specified and noise is non-Gaussian, so the pipeline is not broken. On this data H1 passes for all 5 families, but H2 only for 1 of 5 (thresholded VAR): BIC selects as well as forecast loss (gain FC 0.02 to 0.06, BIC 0.02 to 0.05, sparsity-only negative), and H3 fails for all 5.

### CausalRivers (primary): H1 NEGATIVE
| family | median rho [95% CI] | AUROC: forecast-selected / BIC / sparsity / grid mean / oracle | gain FC (Holm p) | gain SP | gain BIC |
|---|---|---|---|---|---|
| lasso-VAR | 0.00 [-0.18, 0.15] | 0.620 / 0.593 / 0.636 / 0.630 / 0.711 | -0.010 (1.0) | +0.006 | -0.038 |
| thresholded VAR | -0.08 [-0.24, 0.09] | 0.552 / 0.553 / 0.549 / 0.554 / 0.609 | -0.002 (1.0) | -0.005 | -0.002 |
| DYNOTEARS | -0.20 [-0.40, -0.10] | 0.565 / 0.545 / 0.619 / 0.601 / 0.706 | -0.037 (1.0) | +0.018 | -0.056 |
| PCMCI+ | 0.09 [-0.01, 0.12] | 0.678 / 0.693 / 0.667 / 0.676 / 0.767 | +0.001 (1.0) | -0.010 | +0.017 |
| VAR-LiNGAM | 0.25 [0.10, 0.41] | 0.607 / 0.611 / 0.584 / 0.600 / 0.646 | +0.007 (0.054) | -0.017 | +0.011 |

0 of 5 families pass H1 (needs median rho >= 0.3, CI above 0, Holm-significant gain); the median over families of the median rho is 0.0. The best family, VAR-LiNGAM, has median rho 0.25 (below 0.3) and a gain of +0.007 AUROC that misses significance after Holm (p = 0.054). Forecast-selected AUROC is within 0.01 of a random grid draw for four of five families, and regret to the oracle is 0.04 to 0.14. By cell, median rho exceeds 0.3 only in a few isolated cells for VAR-LiNGAM (e.g. East Germany d=10 12 h 0.61, Bavaria d=10 12 h 0.43); 12 h is better than 24 h for VAR-LiNGAM (0.43 vs 0.11) and not for the others. For DYNOTEARS the correlation is significantly negative: lower forecast loss goes with a worse graph.
H2 (forecast loss beats sparsity-only and BIC): 0 of 5. Forecast loss beats BIC for lasso and DYNOTEARS and beats sparsity-only for VAR-LiNGAM (each Holm-significant), but never beats both, and for lasso and DYNOTEARS the sparsity-only rule gives higher AUROC than forecast loss.
H3 (necessity score): 0 of 5. Median necessity AUROC over all pairs is 0.52 to 0.55 (CI lower bounds 0.49 to 0.52, only PCMCI+ and VAR-LiNGAM and thresholded VAR marginally above 0.5) and lower than the AUROC of the model's own |coefficient| score (0.55 to 0.68) for every family.

### netsim (secondary, no combined verdict by plan; the same rules give H1 negative, H2 0/5, H3 0/5)
Median rho: lasso 0.24 [0.11, 0.38], thresholded VAR -0.03, DYNOTEARS 0.10, PCMCI+ -0.17 [-0.32, -0.03], VAR-LiNGAM -0.06. Lasso is the only family with a significant gain (+0.012 AUROC, Holm p = 0.005) and still fails rho >= 0.3. PCMCI+ and VAR-LiNGAM selections are worse than a random draw on average (gain -0.011 and -0.015). Necessity AUROC 0.51 to 0.54.

## Verdicts against the pre-registered rules
- H1: not supported, negative (CausalRivers 0/5 families; netsim 0/5). Synthetic sanity passes, so the negative is a property of the real data, not a bug.
- H2: not supported on CausalRivers (0/5), netsim (0/5) and synthetic (1/5): forecast loss is not better than sparsity-only/BIC selection.
- H3: not supported anywhere (0/5 on all three data groups).

## Deviations (none changed a threshold or rule after seeing results)
1. netsim test uses the first 20 subjects of each sim (21 sims, 420 datasets) instead of all 50 to bound runtime; decided before the test run, after seeing that validation took 12 minutes for 300 datasets. The netsim validation run (6 sims, 300 datasets) used all 50 subjects. 27 sims have <=15 nodes; 6 (20%, chosen by `default_rng(6000)`) are validation, the other 21 are test.
2. tigramite 5.2.10.1 `run_pcmciplus` has no `alpha_level` argument; `pc_alpha` alone is the significance level, so "pc_alpha = alpha_level" in the plan reduces to `pc_alpha = a`.
3. The rivers validation run (seeds 6000-6004) was used only to check that the code runs, runtimes and degeneracy; no grid or rule was changed as a result. The synthetic and netsim validation runs likewise.
4. The H3 condition "AUROC_nec exceeds AUROC_S" is implemented as mean AUROC_nec > mean AUROC_S together with a Holm-adjusted sign-flip p < 0.05; the plan did not fix the exact statistic.
5. The H1 rule's undefined-rho clause (>30% NaN) was not triggered for rivers or synthetic; for netsim thresholded VAR (9.5%) and DYNOTEARS (9.5%) were below it.
6. 6 h resolution was not used (not needed for this question), and standardisation uses training statistics only (stage 0 standardised the whole window).

## Limitations and my reading
- Ground truth is the physical river graph. Strong regional co-movement makes many non-edges predictive, so a good forecaster legitimately keeps them; this is a plausible mechanism for the negative result, but I did not test it directly (that would be a new analysis, not part of the pre-registration).
- The forecast uses lag-1 information only, so contemporaneous edges (most river edges at 24 h per stage 0) are seen only through proxies. The necessity score is computed on a common lag-1 restricted-VAR surrogate and can therefore not recover contemporaneous orientation.
- The grid is small (6 to 8 settings) and per-dataset rho is noisy; the bootstrap CIs account for between-subgraph variation, not for grid choice. Only linear models were tested (no neural or KAN models), as in the plan.
- Sparsity-only used target d edges, a fixed prior. A variant with the true edge count (columns `*_spt` in `perdataset_*`) gives mean gain on rivers of +0.015 (DYNOTEARS), -0.003, -0.010, -0.005, -0.017 (others), i.e. it also does not reliably select good graphs; it beats FC only for lasso and DYNOTEARS.

## Is there a paper here?
A finding, not a method: "on real river benchmarks held-out forecast loss does not select good causal graphs (median Spearman about 0 for four of five linear model families, forecast-selected AUROC within 0.01 of a random grid draw, no advantage over BIC or a fixed sparsity target), whereas on well-specified synthetic SVAR it does (rho 0.5 to 0.76)" is clean, pre-registered and reproducible, and it is a useful warning for practitioners who tune causal discovery on predictive loss. It is thin as a standalone paper: linear models only, one real benchmark family plus simulated netsim, and no tested mechanism (hidden drivers, contemporaneous edges). It would become a credible workshop or benchmark-track paper if extended with nonlinear models (cMLP/NAVAR/KAN), CausalTime or other real benchmarks with ground truth, and an experiment that separates the hidden-driver and contemporaneous-edge explanations.
