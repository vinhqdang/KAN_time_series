# Edge-function focus: BASIL-W vs SPADE-scaled and refit (plan: research/06_focus_edge_functions.md)

Setup as pre-registered: iSCM, ER mean degree 2, uniform noise, n=1000, d in {100, 200}, 5 test datasets each (seeds 1000-1004). Validation seeds 0-2 (d=100) chose refit k=1.5 (NRMSE 0.207 vs 0.237 at k=2 and 0.251 at k=1); k=1 was dropped from the test run, k=2 kept as a reference. Raw: results/scale/F_d*_s*.csv, results/focus_all.csv; summary: results/focus_analysis.txt (aggregate_focus.py). Times: SSC = SPADE-scaled fit only; SSR/BASILW include the SPADE-scaled fit they depend on.

## Means over 5 datasets
| d | method | NRMSE (lower better) | slope | curve R2 | AUROC | AUPRC | orientation | 90% band cover | ECE | time |
|---|---|---|---|---|---|---|---|---|---|---|
| 100 | SPADE-scaled (SSC) | 0.616 | 0.44 | 0.58 | 0.997 | 0.878 | 0.958 | - | - | 47 s |
| 100 | SSC + refit k=1.5 | 0.211 | 0.96 | 0.88 | 0.966 | 0.853 | 0.922 | - | - | 80 s |
| 100 | SSC + refit k=2 | 0.256 | 0.92 | 0.85 | 0.975 | 0.817 | 0.895 | - | - | 79 s |
| 100 | BASIL-W | **0.194** | 0.99 | 0.88 | 0.973 | 0.870 | 0.940 | 0.78 | 0.002 | 71 s |
| 200 | SPADE-scaled (SSC) | 0.612 | 0.45 | 0.58 | 0.997 | 0.855 | 0.942 | - | - | 122 s |
| 200 | SSC + refit k=1.5 | 0.211 | 0.96 | 0.88 | 0.967 | 0.828 | 0.920 | - | - | 200 s |
| 200 | SSC + refit k=2 | 0.259 | 0.91 | 0.85 | 0.980 | 0.806 | 0.893 | - | - | 205 s |
| 200 | BASIL-W | **0.175** | 0.995 | 0.90 | 0.979 | 0.890 | 0.953 | 0.79 | 0.001 | 169 s |

## Verdicts (pre-registered F1-F6)
- F1 (NRMSE <= 0.70x SPADE-scaled): **PASS**. Ratio 0.32 (d=100) and 0.29 (d=200); every dataset in 0.25-0.34.
- F2 (BASIL-W <= refit + 0.03): **PASS, but the simple refit gets most of the gain.** BASIL-W minus refit(k=1.5): -0.017 at d=100 (range -0.046..+0.024, i.e. a tie) and -0.036 at d=200 (all 5 datasets negative). Reading: the cheap refit already removes ~65% of the NRMSE; BASIL-W adds a small, at d=200 consistent, improvement plus uncertainty.
- F3 (AUROC >= SSC - 0.01; orientation >= SSC - 0.03): AUROC **misses the target** (-0.024 at d=100, -0.018 at d=200; kill line -0.03 not reached). Orientation passes (-0.018, +0.011).
- F4 (time <= 2x SSC at d=200): **PASS** (1.4x; 1.5x at d=100). d=500 not run.
- F5 (coverage >= 0.80; ECE <= 0.10): ECE **PASS** (0.001-0.002). Coverage 0.78-0.79: **marginally below** the 0.80 target; kill line (0.60) not reached.
- F6 (control var-sortnregress >= 0.15 below): not re-run in this grid; earlier runs on the same generator at d=100/200 gave AUROC 0.68-0.69 (README_scale.md), i.e. about 0.3 below.

## What this means / does not mean
- Established on this generator: SPADE-scaled finds the edges but returns curves with 44% of the true amplitude (NRMSE 0.61); two different post-processing routes cut the error by roughly two thirds (refit) or a bit more (BASIL-W), at about 1.4-1.7x the cost, with a small AUROC loss (0.02) relative to SPADE-scaled.
- BASIL-W's distinctive additions over refit are calibrated edge probabilities (ECE 0.002) and credible bands (coverage 0.78-0.79, slightly under nominal 0.90 pointwise by the preregistered 0.80 criterion); the point-accuracy edge over refit is 0.02 (tie) at d=100 and 0.04 at d=200.
- Not tested: d=500; n other than 1000; non-uniform noise; other mechanism families; real data; whether BASIL-W's structure inherits SPADE-scaled's errors when SPADE-scaled is poor (the candidate set and the warm start both come from it); interventional effect error (the quantity users actually act on).
- Statistical power: 5 datasets per cell, differences across methods are consistent in sign across datasets for F1/F2(d=200) but no formal tests; F2 at d=100 is a tie.
- refit k was chosen on 3 validation datasets at d=100 only; the oracle-free support rule (top 1.5 d edges) uses the fact that the graph has mean degree 2, which a user may not know.
