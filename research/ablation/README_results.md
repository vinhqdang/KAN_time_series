# BASIL-W ablations and band calibration: results (plan: PLAN.md, committed before the runs)

iSCM, d=100, n=1000, mech m1. Validation seeds 3000-3004 (band diagnosis and kappa only); test seeds 3005-3012 (8 datasets); Laplace and hetero noise seeds 3005-3009 (5 each). 1 CPU core. Code: `basil_s2.py` (copy of `spire/basil_s.py` that also exposes posterior draws; fitting logic unchanged except that `trace_gap` is now max-min of chain end scores over all chains), `run_ablation.py`, `run_all.sh`, `aggregate.py`. Raw: `results/*/*.csv`; full printout: `results/analysis.txt`. SPADE-scaled computed once per dataset and shared (mean 22.7 s). Times below are BASIL fit only (no metrics, no SPADE).

## Part A: ablations (mean over 8 test datasets)
| variant | NRMSE | slope | do direct | do indirect | AUROC | AUPRC | orient | band cover (kappa=1) | ECE | recall | end-score gap (nats) | fit time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base (BASIL-W stab.) | 0.177 | 0.993 | 0.173 | 0.248 | 0.971 | 0.880 | 0.943 | 0.798 | 0.001 | 1.000 | 8.8 | 10.8 |
| v1 R2 candidates | 0.176 | 0.993 | 0.173 | 0.251 | 0.972 | 0.871 | 0.943 | 0.798 | 0.001 | 1.000 | 17.8 | 10.6 |
| v2 cold start | 0.324 | 0.966 | 0.362 | 0.548 | 0.909 | 0.684 | 0.809 | 0.678 | 0.005 | 1.000 | 543 | 9.6 |
| v3 cold + R2 (BASIL-S) | 0.318 | 0.934 | 0.379 | 0.593 | 0.921 | 0.736 | 0.821 | 0.697 | 0.004 | 1.000 | 576 | 10.0 |
| v4a g=n/4 | 0.176 | 0.990 | 0.177 | 0.251 | 0.972 | 0.866 | 0.945 | 0.801 | 0.001 | 1.000 | 15.7 | 10.3 |
| v4b g=4n | 0.176 | 0.993 | 0.177 | 0.253 | 0.974 | 0.885 | 0.942 | 0.799 | 0.001 | 1.000 | 65.6 | 10.7 |
| v5a pi x0.5 | 0.177 | 0.993 | 0.173 | 0.248 | 0.972 | 0.865 | 0.943 | 0.798 | 0.001 | 1.000 | 7.5 | 10.0 |
| v5b pi x2 | 0.176 | 0.993 | 0.173 | 0.251 | 0.973 | 0.870 | 0.945 | 0.800 | 0.001 | 1.000 | 17.1 | 10.3 |
| v6 kmax=2 | 0.280 | 0.985 | 0.247 | 0.422 | 0.951 | 0.766 | 0.894 | 0.706 | 0.003 | 1.000 | 85 | 6.9 |
| v7 4 chains, 600 d iters | 0.173 | 0.992 | 0.165 | 0.227 | 0.974 | 0.887 | 0.934 | 0.807 | 0.001 | 1.000 | 30.9 | 23.5 |

Paired differences to base (mean, sd; number of datasets where worse) are in `results/analysis.txt`. Key ones: v2 NRMSE +0.147 (sd 0.040, worse 8/8), orientation -0.133 (8/8); v3 +0.141 / -0.121 (8/8); v6 NRMSE +0.103 (8/8), orientation -0.048 (7/8); v1, v4, v5: |dNRMSE| <= 0.002, |dorientation| <= 0.002; v7 dNRMSE -0.004 (better 7/8), dorientation -0.009 (lower in 4/8), d do-indirect -0.021.

## Verdicts against the pre-registered criteria
- A1 warm start matters: **YES, much more than predicted.** Removing it costs +0.147 NRMSE and -0.133 orientation (8/8 datasets; prediction was orientation -0.02..-0.05, NRMSE < 0.02: wrong). AUPRC drops 0.20. The cold chains do not converge in 200 d iterations (end-score gaps of hundreds of nats; 15-1400 per dataset). Caveat: this is "warm start at this chain length"; longer cold chains were not run, so we cannot say a cold start never works.
- A2 SPADE candidates matter: **NO at d=100.** Recall is 1.000 with either ranking, NRMSE -0.001. (As predicted.) v3 vs v2 (R2 candidates on cold start) is a tie, so BASIL-S's poor result is entirely mixing, not screening. Not tested for d >= 200 or weaker screening.
- A3 g: **robust** (|dNRMSE| 0.001 for n/4 and 4n). A4 pi: **robust** (dAUPRC -0.015 for x0.5, -0.010 for x2, within the 0.02 limit; dNRMSE <= 0.002). Differences of 0.01-0.02 AUPRC have sd 0.04-0.05 across datasets: ties.
- A5 kmax=2: **kill criterion hit** (NRMSE +0.103, prediction was < 0.02). With ER mean degree 2 a non-trivial share of nodes have >= 3 parents; capping at 2 forces misses there. kmax=3 is needed on this graph family; kmax=4 was not tried.
- A6 default converged: by the registered criteria **yes** (dNRMSE -0.004 < 0.01 in magnitude; orientation not improved). But: indirect do-effect NRMSE improves 0.248 -> 0.227, and the end-score gap does not shrink with 4 chains x 3x iterations (31 nats vs 8.8; four chains disagree more often than two). So the chains are not provably converged; the default is adequate for edge-function NRMSE and not demonstrably so for indirect effects. The end-score gap is dominated by a few runs (base: 0 in three datasets, 12-24 in four) and is not a reliable predictor of error across datasets.
- A7: the variants that keep the warm start all look alike (within 0.002 NRMSE); the structure and curves of BASIL-W here are largely determined by the SPADE-scaled warm start plus local refinement. Not tested what happens when SPADE-scaled's order is poor.

## Part B: credible bands

### Diagnosis (validation seeds 3000-3004, kappa=1; 517 true edges)
Overall pointwise coverage 0.802 (nominal 0.90). Curve-level: only 11% of curves lie entirely inside the band; 45% of curves have >= 90% of rows covered.
| posterior inclusion P | edges | pointwise cov | mean z | sd z | band width |
|---|---|---|---|---|---|
| <= 0.5 | 42 | 0.033 | -1.8 | 15.6 | 0.03 |
| 0.5-0.9 | 1 | 0.52 | | | 0.67 |
| 0.9-0.99 | 5 | 0.89 | | | |
| > 0.99 | 332 | 0.867 | -0.007 | 1.04 | 0.14 |
- Edges with P > 0.9 (92% of edges): coverage 0.870; edges with P <= 0.9 (8%): 0.044. **72% of the coverage shortfall comes from the 8% of true edges the model does not find (P <= 0.9)**: for them the band is a near-zero spike (width 0.03) around 0 while the truth is not 0. A band cannot cover an edge it excludes; this is a structure error showing up as a coverage error.
- For the P > 0.9 edges the remaining shortfall is small (0.87 vs 0.90). The z-scores are centred (mean z -0.004) with sd 1.03; the estimate of sigma^2 is not the problem (model s2 / actual residual variance of the true SEM: median 0.994, 87% of nodes within 5%; the mean 1.20 comes from nodes whose parent set is wrong). g/(1+g) shrinkage is 0.1% at g=n, so it cannot matter. The spline-basis approximation error is 0.2 posterior sd. Conditioning on edge inclusion (dropping model averaging) gives the same coverage (0.870), so model averaging is not the cause for P > 0.9 edges. Using the same data for sigma^2 matters little (dof n-3 versus n-p is a 2% effect). Heavy-tailed noise is a minor factor: P > 0.9 coverage 0.850 (Laplace) and 0.835 (hetero) vs 0.877 (uniform).
- Reading: the band under-covers mostly because of **missed edges (structure error)**, secondarily because of a small ~3 point mismatch for well-determined edges (bias of the same order as the posterior sd in the tails of x).

### Fix (the one pre-registered): posterior-variance inflation, band = 5/95 percentiles of m + kappa (D - m)
Validation coverage by kappa: 1.0 -> 0.802, 1.2 -> 0.858, 1.4 -> 0.890, 1.6 -> 0.907, 2.0 -> 0.919, plateau 0.923 (plateau because missed edges cannot be covered). **kappa\* = 1.55** (smallest grid value with validation coverage >= 0.90). Frozen, applied unchanged below.

| noise | datasets | kappa | pointwise cov (mean; sd over datasets) | cov P>0.9 | cov P<=0.9 | curve inside band (all rows) | curves with >=90% rows | mean width |
|---|---|---|---|---|---|---|---|---|
| uniform (test) | 8 | 1.00 | 0.798 (0.043) | 0.877 | 0.095 | 0.153 | 0.456 | 0.140 |
| uniform (test) | 8 | 1.55 | **0.894** (0.052) | 0.981 | 0.112 | 0.609 | 0.862 | 0.216 |
| Laplace | 5 | 1.00 | 0.796 (0.041) | 0.850 | 0.207 | 0.049 | 0.432 | 0.141 |
| Laplace | 5 | 1.55 | 0.904 (0.043) | 0.964 | 0.249 | 0.321 | 0.832 | 0.219 |
| hetero | 5 | 1.00 | 0.790 (0.044) | 0.835 | 0.304 | 0.028 | 0.379 | 0.173 |
| hetero | 5 | 1.55 | 0.908 (0.048) | 0.960 | 0.357 | 0.231 | 0.838 | 0.268 |

Verdict: **the pre-registered success criteria are met** (test coverage 0.894 in [0.85, 0.95]; width ratio 1.55 <= 2; Laplace 0.904 and hetero 0.908 in [0.80, 0.95], no retuning).

### What this does and does not show
- The target is hit **on average, partly for the wrong reason.** At kappa=1.55 edges the model found (P > 0.9) are over-covered (0.96-0.98 rather than 0.90) while the edges it missed stay uncovered (0.11-0.36). The average reaches 0.90 because the over-coverage of found edges offsets the misses. A global scalar is not conditional validity, and bands for well-determined edges are about 55% wider than necessary. The honest remedy for the missed edges is a better structure estimate, not wider bands.
- Per-dataset spread is large: at kappa=1.55 the uniform test datasets range 0.78-0.93 (validation: 0.86-0.93). The test share of edges with P <= 0.9 was 16% versus 8% on validation (seed 3012 and 3010 are the weak ones), so kappa* chosen on 5 validation datasets was a bit lucky; had it been chosen on test, the descriptive curve says about kappa=1.7-2.0 would be needed for 0.90 (not used).
- Curve-level (simultaneous) coverage is only 0.61 (uniform) with a pointwise band; a simultaneous band would need a sup-norm calibration, not tried.
- Alternatives (split-conformal on held-out rows, bootstrap over rows) were not implemented; the diagnosis suggests they would not repair missed edges either.
- 8 test datasets (5 per noise type), one generator, d=100, n=1000; no significance tests.

## Other failures and surprises
- Three registered predictions were wrong: warm-start effect size (larger), kmax=2 effect (large), and chain convergence (more chains did not reduce the end-score gap).
- Validation datasets are easier than the test ones (fewer missed edges), which affects the calibration of kappa.
- Heteroscedastic noise: NRMSE 0.284 and do-indirect 0.527 (versus 0.177 / 0.248 for uniform), so the coverage holding there says nothing about point accuracy.
- Laplace chains' end-score gap is large (mean 228 nats) without a visible change in NRMSE (0.214).
