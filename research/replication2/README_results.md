# Independent-DAG replication, four arms (plan: PLAN.md, committed before any run)

d=100, n=1000, iSCM mech m1; 20 datasets per cell, 60 in all. Seeds 3000-3019 (unif), 3100-3119 (laplace), 3200-3219 (hetero), so every cell has its own DAGs, mechanism assignments and noise, all new (earlier sets: 0-9, 1000-1004, 2000-2009). Methods: SSC, SSA, SSR_k1.5, BASIL-W (= BASILWs), same dataset, method seed 0, no parameter retuned. All 60 x 4 runs completed, 0 failures, 0 datasets with indirect do-NRMSE > 2 for any method. Full tables: results/analysis.md (all pairwise tests, effect sizes), results/pairwise.csv, raw results/all.csv. Code: bench.py (SSA branch added to the replication bench, per-method fit timing excluding metric and do-effect code), aggregate.py, run_all.sh. Statistics as pre-registered: paired exact Wilcoxon per cell, Holm over 3 cells x 3 pairs (BASIL-W/SSC, BASIL-W/SSA, SSA/SSC; separate Holm family for pairs involving SSR), 10000-resample bootstrap CIs over datasets.

## 1. Means per cell (20 datasets each)
| cell | method | NRMSE | slope | AUROC | AUPRC | orient | do direct | do indirect | median cost s |
|---|---|---|---|---|---|---|---|---|---|
| unif | SSC | 0.614 | 0.45 | 0.998 | 0.873 | 0.954 | 0.591 | 0.849 | 50 |
| unif | SSA | 0.238 | 0.91 | 0.998 | 0.885 | 0.954 | 0.200 | 0.378 | 83 |
| unif | SSR | 0.215 | 0.95 | 0.975 | 0.830 | 0.919 | 0.169 | 0.278 | 82 |
| unif | BASIL-W | **0.176** | 0.99 | 0.975 | 0.891 | 0.944 | 0.164 | 0.262 | 67 |
| laplace | SSC | 0.786 | 0.19 | 0.994 | 0.682 | 0.778 | 0.750 | 0.927 | 55 |
| laplace | SSA | 0.477 | 0.72 | 0.987 | 0.681 | 0.783 | 0.413 | 0.639 | 91 |
| laplace | SSR | 0.426 | 0.80 | 0.926 | 0.691 | 0.779 | 0.362 | 0.520 | 92 |
| laplace | BASIL-W | **0.253** | 0.98 | 0.964 | 0.862 | 0.922 | 0.205 | 0.307 | 74 |
| hetero | SSC | 0.770 | 0.22 | 0.996 | 0.854 | 0.895 | 0.723 | 0.933 | 33 |
| hetero | SSA | 0.379 | 0.87 | 0.995 | 0.877 | 0.902 | 0.281 | 0.535 | 55 |
| hetero | SSR | 0.338 | 0.92 | 0.963 | 0.845 | 0.863 | 0.260 | 0.512 | 54 |
| hetero | BASIL-W | **0.311** | 0.99 | 0.971 | 0.885 | 0.939 | 0.214 | 0.472 | 44 |

BASIL-W band coverage 0.789 / 0.761 / 0.757 (nominal 0.90, under-covers as before), ECE 0.0014 / 0.0021 / 0.0021.

## 2. Paired NRMSE differences (a minus b; negative favours a; Holm = adjusted over 9 primary tests)
| cell | pair | mean diff [95% bootstrap CI] | Holm p | dz | rank-biserial | datasets a lower |
|---|---|---|---|---|---|---|
| unif | BASIL-W - SSC | -0.439 [-0.450, -0.427] | 1.7e-5 | -15.6 | -1.00 | 20/20 |
| unif | BASIL-W - SSA | **-0.062 [-0.078, -0.046]** | 1.7e-5 | -1.65 | -0.99 | 19/20 |
| unif | SSA - SSC | -0.376 [-0.386, -0.367] | 1.7e-5 | -17.1 | -1.00 | 20/20 |
| laplace | BASIL-W - SSC | -0.533 [-0.546, -0.520] | 1.7e-5 | -16.7 | -1.00 | 20/20 |
| laplace | BASIL-W - SSA | **-0.224 [-0.239, -0.209]** | 1.7e-5 | -6.35 | -1.00 | 20/20 |
| laplace | SSA - SSC | -0.309 [-0.320, -0.297] | 1.7e-5 | -11.6 | -1.00 | 20/20 |
| hetero | BASIL-W - SSC | -0.458 [-0.469, -0.446] | 1.7e-5 | -16.9 | -1.00 | 20/20 |
| hetero | BASIL-W - SSA | **-0.068 [-0.082, -0.054]** | 1.7e-5 | -2.09 | -0.99 | 19/20 |
| hetero | SSA - SSC | -0.390 [-0.401, -0.379] | 1.7e-5 | -15.2 | -1.00 | 20/20 |

Secondary (separate Holm family, 9 tests, see analysis.md): BASIL-W - SSR: unif -0.039 [-0.057, -0.021] (Holm 0.003, BASIL-W lower in 16/20), laplace -0.173 (20/20), hetero -0.027 [-0.042, -0.012] (Holm 0.008, 14/20). SSA - SSR: unif +0.023 (SSA lower in 5/20), laplace +0.051 (0/20), hetero +0.041 (1/20), all Holm significant: the adaptive group lasso is consistently a little worse than the plain refit on edge NRMSE.

Per-dataset win counts (lowest NRMSE of the four methods): unif BASIL-W 15, SSR 4, SSA 1, SSC 0; laplace BASIL-W 20; hetero BASIL-W 14, SSR 6, SSA 0, SSC 0. Mean ranks BASIL-W 1.25 / 1.00 / 1.35, SSR 2.05 / 2.00 / 1.75, SSA 2.70 / 3.00 / 2.90, SSC 4 in all cells.

Pooled over the 60 datasets (exploratory; graphs are independent across cells here): BASIL-W - SSA -0.118 [-0.139, -0.098], lower in 58/60; BASIL-W - SSR -0.080 [-0.099, -0.061], 50/60; BASIL-W - SSC -0.477, 60/60.

## 3. Verdicts against PLAN.md
- **H1 (BASIL-W < SSC in >= 95% of datasets per cell): CONFIRMED**, 20/20 in all three cells. Ratio of mean NRMSE BASIL-W/SSC 0.29 (unif), 0.32 (laplace), 0.40 (hetero).
- **H2 (BASIL-W lower than SSA; predicted gap 0.05-0.08 unif/hetero, about 0.2 laplace; kill = no significant unif gap): CONFIRMED, kill not triggered.** Gaps (SSA minus BASIL-W) 0.062 / 0.224 / 0.068 sit inside the predicted ranges; Holm p 1.7e-5 in every cell; BASIL-W lower in 19/20, 20/20, 19/20. In unif the gap is small in absolute terms (0.06 NRMSE, about 26% of SSA's error) but consistent in sign (dz -1.65).
- **H3 (SSA AUROC within 0.005 of SSC; BASIL-W loses 0.01-0.03): PARTLY.** SSA minus SSC AUROC: +0.0001 (unif), -0.0005 (hetero) pass; **laplace -0.0067 fails the 0.005 bound** (Holm-significant, SSA lower in 20/20; small). BASIL-W minus SSC: -0.023, -0.030, -0.024: inside 0.01-0.03 (laplace right at the upper edge, -0.0302). BASIL-W AUROC is lower than SSC in 20/20, 20/20, 19/20 datasets. BASIL-W minus SSA AUROC is the same size (-0.023 to -0.024; laplace -0.023). The structure-versus-curve tradeoff predicted in research/adaptive is confirmed in unif and hetero; under Laplace noise SSA also gives up a little AUROC. BASIL-W has better orientation than SSC/SSA/SSR in laplace (+0.14) and hetero (+0.04), equal in unif.
- **H4 (do NRMSE BASIL-W <= SSA < SSC, direct and indirect): MET on cell means in all 3 cells for both pair types** (e.g. direct 0.164 / 0.200 / 0.591 in unif). Strength of the BASIL-W vs SSA part varies: direct, unif -0.037 [-0.074, +0.007], Holm p 0.07 (not significant; 14/20 datasets favour BASIL-W); indirect hetero -0.063 [-0.119, +0.006], Holm p 0.014 (CI includes 0). Elsewhere (laplace both, hetero direct, unif indirect) clearly significant (Holm <= 0.0024). SSA < SSC holds in every dataset and cell (20/20). So the ordering holds on means, but the BASIL-W advantage over SSA in do-effects is only a tie-like difference for direct effects in unif.
- **H5 (cost, descriptive).** Median cost relative to SSC: SSA 1.65 / 1.65 / 1.69, SSR 1.62 / 1.69 / 1.66, BASIL-W 1.34 / 1.35 / 1.35 (unif / laplace / hetero). BASIL-W is the cheapest repair, SSA is about as expensive as SSR. The machine was shared and loaded (SSC median ranged 33-55 s across cells), so ratios are +-30% estimates; ratios within a cell are more reliable than absolute times.

## 4. Comparison with the earlier seeds
Earlier numbers: research/replication (2000-2009 unif, 2000-2004 laplace/hetero) and research/adaptive. Descriptive, no test; new cell means with these differences:
| quantity | earlier | now |
|---|---|---|
| unif NRMSE SSC / SSA / SSR / BASIL-W | 0.612 / 0.241 / 0.213 / 0.169 | 0.614 / 0.238 / 0.215 / 0.176 |
| laplace NRMSE SSC / SSA / SSR / BASIL-W (5 earlier datasets) | 0.805 / 0.520 / 0.470 / 0.261 | 0.786 / 0.477 / 0.426 / 0.253 |
| hetero NRMSE (5 earlier datasets) | 0.766 / 0.375 / 0.349 / 0.294 | 0.770 / 0.379 / 0.338 / 0.311 |
| SSA minus BASIL-W unif / laplace / hetero | +0.072 / +0.258 / +0.082 | +0.062 / +0.224 / +0.068 |
| BASIL-W minus SSR unif / laplace / hetero | -0.043 / -0.218 / -0.065 | -0.039 / -0.173 / -0.027 |
| AUROC SSC / SSA / BASIL-W unif | 0.998 / 0.998 / 0.973 | 0.998 / 0.998 / 0.975 |
Everything reproduces qualitatively on independent DAGs: signs, orderings and the SSA = SSC structure result. The BASIL-W advantage is somewhat smaller on the new data: the SSA gap by 0.01-0.03 (still inside the pre-registered ranges), and the gap to SSR in hetero shrank from -0.065 to -0.027 (earlier value was from 5 datasets; still Holm-significant now, with SSR winning 6/20 individual datasets). The earlier BASIL-W-SSR unif difference (-0.043) is almost unchanged (-0.039) and is now significant after Holm (the earlier 10-dataset test was not). Do-effects (unif, earlier SSC 0.60/0.86, SSA 0.22/0.39, BASIL-W 0.15/0.23) are consistent with the new values (0.59/0.85, 0.20/0.38, 0.16/0.26). The earlier concern that the 7-cell replication reused the same DAGs is addressed for the three noise families here; the headline conclusions did not depend on those 10 graphs.

## 5. What can and cannot be concluded
Supported on this generator (iSCM, ER degree 2, d=100, n=1000, mech m1, noise unif/laplace/hetero), on independent DAGs: BASIL-W reduces edge-function error versus SPADE-scaled in every one of 60 datasets (by 56-68% in cell means) and versus the standard adaptive group lasso in 58/60 (by 26% unif, 47% laplace, 18% hetero in cell means); the adaptive group lasso keeps SPADE-scaled structure (AUROC within 0.0005 in unif/hetero, 0.0067 lower in laplace) and removes about 80% of the shrinkage gap to BASIL-W in unif/hetero (62% in laplace); BASIL-W costs 0.023-0.030 AUROC.
Not concluded: anything about real data, other generators, mech family m2, other n or d, Gaussian noise, or d >= 200; that SSA is the best possible adaptive baseline (gamma=1, eps=0.01 untuned, a tuned version might narrow the gap; this is the largest open threat to the BASIL-W-over-SSA claim); that BASIL-W beats SSR in a practically meaningful way in unif and hetero (differences 0.03-0.04 NRMSE, SSR wins 4/20 and 6/20 datasets); that direct do-effects of BASIL-W differ from SSA in unif or from SSR in unif (Holm p 0.07 and 0.43); that bands are calibrated (coverage 0.76-0.79 at nominal 0.90); that timing ratios transfer.

## 6. Caveats and deviations
- No deviations from PLAN.md. aggregate.py was fixed once during the run (crash when a cell had under 3 complete datasets, an error-handling bug; no change to the statistics) and used on partial data only for progress; interim commits contain results for completed cells.
- Same generator, same code family, same authors' hyperparameters as the earlier study: replication of the experiment, not an external validation. Noise cells share mechanism family and density.
- Three cells x 20 datasets: power is high for the effects above (dz > 1.6 for H2), but the number of noise families is three.
- Edge-function NRMSE and band coverage of BASIL-W use the model-averaged curves; the stabilised coefficient map enters only through do-effects (as in research/replication).
- Cost measured on a loaded shared machine; SSA and SSR costs include the shared SPADE-scaled fit time.
- Do-effect pairs skipped for tiny true effect, identical across methods (about 4 of 40 per dataset, as before).
- The Holm family for do-effects and AUROC is computed per metric over the same 9 comparisons (analysis.md); only the NRMSE family was named in the plan as the primary one.
