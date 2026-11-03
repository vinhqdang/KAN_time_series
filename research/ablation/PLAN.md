# Ablation of BASIL-W and a principled fix for the credible bands (pre-registration)

Status: written and committed BEFORE any run in this directory. Code is written but unrun at commit time.

## Setting
iSCM generator `bench.gen_full`, d=100, n=1000, uniform noise, mechanism family m1. ER mean degree 2.
Validation seeds 3000-3004: used ONLY to choose the band constant kappa (Part B) and for diagnosing the band failure.
Test seeds 3005-3012 (8 datasets): all Part A tables and the Part B test coverage.
Robustness of bands: Laplace and heteroscedastic noise, seeds 3005-3009 (5 datasets each), final constants, no retuning.
Compute: 1 CPU core (OMP_NUM_THREADS=1, torch threads 1), sequential.

## Reference method (BASIL-W stabilised)
SPADE-scaled fit (lambda=0.2/d, tree screening K=15, ep=400, seed 0) computed ONCE per dataset and shared by all variants.
Candidates: top-20 parents per node by SPADE-scaled importance (ties by marginal R2); warm start at the order from source-peeling on SPADE-scaled importance;
g=n, pi=2/(d-1), kmax=3, 2 chains, 200*d iterations, burn 0.3, chain selection by end score; do-effects use the stabilised coefficients.

## Part A variants (each changes ONE thing)
| id | change |
|---|---|
| base | reference |
| v1 | candidates by marginal R2 only (cand_score=None), still warm-started |
| v2 | cold start (random initial order), SPADE-scaled candidates |
| v3 | cold start + R2 candidates (= BASIL-S at these settings) |
| v4a / v4b | g = n/4 / g = 4n |
| v5a / v5b | pi x0.5 / x2 |
| v6 | kmax = 2 |
| v7 | 4 chains, 600*d iterations |

Metrics per dataset: edge-function NRMSE (mean over true edges; missed edge = 1), calibration slope (median), do-effect NRMSE direct/indirect (stabilised coefficients, same pairs and truth for all variants), AUROC, AUPRC, orientation, 90% band coverage (pointwise, kappa=1), ECE, candidate recall (true parents inside the candidate set), chain end-score gap (max-min of the mean of the last 200 trace values over chains), BASIL fit time excluding metrics and excluding the shared SPADE-scaled fit (SPADE time reported separately).
Reporting: mean over 8 datasets and mean paired difference to base with sd and the number of datasets (of 8) where the variant is worse. No significance tests.

## Predictions and kill criteria (Part A)
- A1 warm start matters: v2 (and v3) worse than base by > 0.03 in NRMSE OR orientation. Prediction: orientation drops by about 0.02-0.05, NRMSE changes < 0.02 (so verdict probably "matters for orientation only" or "inconclusive"). If both differences are <= 0.03 the warm start is declared NOT needed at d=100 (it may still matter at larger d; not tested).
- A2 SPADE candidates matter: v1 recall lower than base by >= 0.01 or NRMSE worse by > 0.03. Prediction: recall differs by < 0.01 at d=100 (BASIL-S earlier reached 0.998), so candidates from SPADE do NOT matter here.
- A3 g: |NRMSE change| <= 0.03 for n/4 and 4n. Prediction: robust (changes < 0.02); kill: > 0.03 means g is a sensitive hyperparameter.
- A4 pi: |AUPRC change| <= 0.02, |NRMSE change| <= 0.03 for x0.5 and x2. Prediction: robust.
- A5 kmax=2: NRMSE worse by < 0.02 (few nodes with 3 parents). Kill: > 0.03.
- A6 default converged: v7 changes NRMSE by < 0.01 and orientation by < 0.02 and the end-score gap does not shrink to zero. If v7 improves orientation by > 0.02 or NRMSE by > 0.01 the default is declared under-converged.
- A7 (honest negative possibility): the structure and NRMSE of base may be dominated by the SPADE-scaled warm start; then every variant that keeps it will look alike.

## Part B: band calibration
Current band: 5/95 percentiles of 100 draws from the model-averaged posterior (model drawn from the top-60 models by weight, coefficients from the conditional g-prior posterior with s2 = n(1 - g/(1+g) R2)/(n-3)), curve = centred spline basis x coefficients; draws where the edge is absent give a curve of 0.

Diagnosis first (validation seeds 3000-3004, kappa=1), for each true edge: posterior inclusion probability P; pointwise coverage of the truth; whether the entire curve lies inside the band; coverage of a band built from ONLY the draws that include the edge (isolates model-averaging/zero-spike); pointwise z = (truth - mean)/sd_post (mean z shows bias, sd z shows variance misstatement); spline approximation bias (relative L2 error of the best least-squares projection of the truth on the edge basis, vs. posterior sd); ratio of the model's s2 to the actual residual variance of the true SEM (sigma^2 estimate); coverage split P>0.9 vs P<=0.9. g/(1+g) shrinkage is analytic (g=n: 0.1%).
Reported cause = the largest identified gap; no cause is asserted without these numbers.

Fix (ONE, fixed before running): posterior-variance inflation about the posterior mean.
band_kappa(x) = 5/95 percentiles over draws b of { m(x) + kappa * (D_b(x) - m(x)) }, m = model-averaged posterior mean curve, D_b = posterior draws (200 draws per edge in the new code). kappa is a single global constant.
Rule: kappa* = the smallest value on the grid {1.0, 1.05, ..., 3.0} such that the mean pointwise coverage over true edges (average over the 5 validation datasets of the per-dataset mean) is >= 0.90. Applied unchanged to test seeds, Laplace and hetero.
Success: test pointwise coverage (mean over 8 datasets) in [0.85, 0.95] AND mean band width <= 2x the kappa=1 width (i.e. kappa* <= 2). Also reported: curve-level coverage = fraction of true edges whose whole curve is inside the band (secondary; a pointwise 90% band is not expected to give 90% here), and fraction of edges with >=90% of rows covered. Robustness: Laplace/hetero coverage in [0.80, 0.95] counts as "holds"; lower is reported as a failure.
Fail conditions stated now: kappa* > 3 on validation, or test coverage < 0.85. A scalar kappa cannot repair edges that are missed or whose posterior is mis-centred (bias); if the diagnosis shows that most of the shortfall comes from low-P edges, the scalar fix is expected to widen bands of well-determined edges too much, and I will say so rather than claim success.
Prediction (kill): coverage at kappa=1 on validation about 0.75-0.80; I expect the shortfall to be mostly due to edges with P<0.9 (missing-edge zero-spike and wrong parent sets) plus spline approximation bias, not the sigma^2 estimate. If P>0.9 edges already cover >= 0.88 the shortfall is attributed to low-P edges.

## What will NOT be concluded
- 8 test datasets (5 per noise type for robustness), one generator, d=100, n=1000: no statements about d>=200, other n or real data; no significance claims; differences below about 0.02 NRMSE are treated as ties.
- Because the fix is tuned for average coverage on 5 validation datasets, conditional (per-edge) validity is not claimed.
- Dependence on SPADE-scaled when it is poor is not tested.
