# Interventional-effect error and robustness (plan: research/07_do_effect_robustness.md)

iSCM generator, ER mean degree 2, test seeds 1000-1004 (5 datasets/cell; d=500: 2 datasets). Methods: SSC = SPADE-scaled; SSR = SPADE-scaled + refit on its top 1.5 d edges (k frozen on validation); BASILW = BASIL-W as pre-registered. Raw: results/do/*.csv, results/do_all.csv; summary: results/do_analysis.txt; stabilised re-run: results/do_stab/, results/do_stab_analysis.txt. diag_do.py is the diagnostic that found the numerical problem described below.

## Edge-function NRMSE (lower = better), mean over datasets
| cell | SSC | SSR | BASIL-W | BASIL-W/SSC | BASIL-W minus SSR |
|---|---|---|---|---|---|
| d100 unif n1000 | 0.616 | 0.211 | 0.194 | 0.32 | -0.017 |
| d100 gauss | 0.690 | 0.254 | 0.174 | 0.25 | -0.080 |
| d100 laplace | 0.793 | **0.449** | 0.261 | 0.33 | **-0.188** |
| d100 hetero (ANM violated) | 0.766 | 0.339 | 0.305 | 0.40 | -0.034 |
| d100 n=500 | 0.613 | 0.229 | 0.225 | 0.37 | -0.005 |
| d100 n=2000 | 0.612 | 0.191 | 0.178 | 0.29 | -0.013 |
| d100 mechanism family 2 | 0.632 | 0.256 | 0.186 | 0.29 | -0.070 |
| d200 unif | 0.612 | 0.211 | 0.175 | 0.29 | -0.036 |
| d500 unif (2 datasets) | 0.690 | 0.229 | 0.162 | 0.23 | -0.067 |
Slope: SSC 0.17-0.45, SSR 0.78-0.97, BASIL-W 0.98-1.00. AUROC BASIL-W minus SSC: -0.012 to -0.034. ECE <= 0.002 everywhere. 90% band coverage 0.73-0.80 (below 0.80 in every cell; lowest 0.728 for mechanism family 2).

## Do-effect NRMSE (direct pairs / indirect pairs, mean over datasets)
| cell | SSC | SSR | BASIL-W as pre-registered | BASIL-W stabilised (post-hoc, re-run) |
|---|---|---|---|---|
| d100 unif | 0.58 / 0.88 | 0.20 / 0.27 | 0.24 / 0.39 | not re-run |
| d100 n=2000 | 0.57 / 0.87 | 0.14 / 0.28 | 0.18 / 0.38 | not re-run |
| d100 gauss | 0.64 / 0.89 | 0.19 / 0.30 | 0.17 / 0.28 | not re-run |
| d100 laplace | 0.75 / 0.93 | 0.39 / 0.53 | 0.20 / 0.39 | not re-run |
| d100 mech 2 | 0.61 / 0.83 | 0.18 / 0.27 | 0.25 / 0.51 | not re-run |
| d100 n=500 | 0.57 / 0.87 | 0.19 / 0.31 | 0.24 / **7.84** | 0.21 / 0.32 |
| d100 hetero | 0.72 / 0.93 | 0.27 / 0.52 | 0.22 / **1.55** | 0.22 / 0.47 |
| d200 unif | 0.62 / 0.82 | 0.21 / 0.26 | 0.20 / **37.2** | 0.19 / 0.24 |
| d500 unif (2) | 0.68 / 0.90 | 0.19 / 0.28 | 0.13 / **0.76** | 0.13 / 0.18 |

## What went wrong in the pre-registered BASIL-W and why (diagnosed, not guessed)
Four BASIL-W datasets gave indirect-effect NRMSE far above 1 (d=200 seed 1002: 185; n=500 seed 1001: 38; hetero seed 1003: 5.8; d=500 seed 1001: 1.33). diag_do.py on the n=500 case found coefficients up to 1.4e7 in the model-averaged coefficient tensor: the whitening divides by sqrt(eigenvalue + 1e-9) along spline directions that the data barely excite (tails), so unidentified directions get unit prior scale; in-sample predictions are fine (edge NRMSE and structure unaffected) but propagated interventions land in sparse regions of the basis and the outputs explode (one node reached 380 when the data lie in +-3.5). A ridge-stabilised map from whitened to spline coefficients (lambda = 1e-3 x the variable's largest basis eigenvalue; equals ridge along weak directions) removes this: indirect NRMSE becomes 0.32 / 0.47 / 0.24 / 0.18 in those four cells, with edge NRMSE, AUROC and coverage unchanged by construction (only the point-estimate extraction changed). The stabilisation was chosen after seeing these test results, with a single value of lambda, and was re-run only on the cells that blew up (15 datasets + 1 at d=500). The other cells report the pre-registered numbers (no explosion there).

## Verdicts (pre-registered R1-R5)
- R1 (do-effect, uniform d=100 and d=200; ratio to SSC <= 0.70 for direct and indirect pairs): **direct PASS** (ratios 0.20-0.42). **Indirect: pre-registered BASIL-W FAILS at d=200** (mean 37 vs SSC 0.82, one dataset at 185), passes at d=100 (0.39/0.88 = 0.44). The stabilised version passes at d=200 (0.24/0.82 = 0.29). BASIL-W minus SSR (tie band 0.03): direct about +0.04 at d=100 (SSR slightly better), -0.014 at d=200; indirect +0.12 at d=100 (SSR better), -0.016 at d=200 stabilised.
- R2 (robustness): **PASS** on the pre-registered edge-function metric: BASIL-W/SSC <= 0.40 in every cell and BASIL-W <= SSR + 0.03 in all cells (better than SSR in every cell). The plain refit is fragile under heavy-tailed noise (Laplace: NRMSE 0.45, AUROC 0.907 vs SSC 0.994) while BASIL-W stays at 0.26.
- R3 (d=500): **PASS** on the 2 datasets: completes, ratio 0.23 to SSC. The time column in these CSVs includes the do-effect evaluation and metric code, so it is not a valid cost comparison; cost evidence is the earlier grid (README_focus.md: BASIL-W about 1.4-1.5x SPADE-scaled at d=100/200).
- R4 (AUROC >= SSC - 0.03 everywhere, kill > 0.05): **missed target in one cell** (mechanism family 2: -0.034), borderline at n=2000 (-0.030), kill line never reached.
- R5 (coverage >= 0.80): **fails the target in every cell** (0.73-0.80; kill 0.60 not reached); ECE <= 0.10 passes (0.002).

## Honest summary
1. The edge-function result is robust across noise family, sample size, mechanism family and d up to 500 (5 datasets/cell, d=500: 2). BASIL-W beats SPADE-scaled by 3-4x in NRMSE everywhere and beats the plain refit in every cell, hugely under Laplace noise.
2. For do-effects, BASIL-W as specified had a numerical instability; stabilised, it matches the refit (indirect effects within about 0.05, not better) and is much better than SPADE-scaled (0.2-0.5 vs 0.8-0.9).
3. SPADE-scaled's structure is still best (AUROC 0.994-0.998) but its curves carry 17-45% of the true amplitude, so its do-effect curves are wrong even for direct pairs (NRMSE 0.6-0.75).
4. Credible bands under-cover (0.73-0.80 against a nominal 0.90); the claim "calibrated uncertainty" holds for edge probabilities (ECE 0.002) but not for the curve bands.
5. Not done: interventions beyond single-node hard do on the grid used; real data; non-additive mechanisms; datasets beyond 5/cell; formal significance tests; stabilised BASIL-W on the cells that did not blow up.
