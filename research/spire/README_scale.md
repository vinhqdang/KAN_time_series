# BASIL-S: scaling results (pre-registered in research/05_bayes_scale.md)

Generator: iSCM, ER mean degree 2, uniform noise, n=1000. Test seeds 1000+ (5 datasets per cell, d=500 only 1). Baseline: SPADE-scaled = project's lambda=0.2/d + ExtraTrees K=15 screening (settings frozen earlier on the project's generator, NOT retuned here). Raw: results/scale/*.csv.

## Table (means over datasets)
| d | method | AUROC | AUPRC | orientation | curve slope | time/dataset |
|---|---|---|---|---|---|---|
| 100 | SPADE-scaled | 0.997 | 0.878 | 0.958 | **0.44** (4 datasets, curve R2 0.57) | 42 s |
| 100 | BASIL-S (orig: 600 d iters, random init) | 0.977 | 0.885 | 0.925 | 0.98 | 30 s |
| 100 | BASIL-S (amended: 1800 d iters, greedy init, chain select) | 0.965 | 0.860 | 0.934 | 0.99 | 55 s |
| 200 | SPADE-scaled | 0.997 | 0.855 | 0.942 | n/a | 133 s |
| 200 | BASIL-S orig | 0.966 | 0.785 | 0.830 | 0.75 | 60 s |
| 200 | BASIL-S amended | 0.973 | 0.875 | 0.929 | 0.99 | 114 s |
| 500 | SPADE-scaled (1 dataset) | 0.999 | 0.785 | 0.920 | n/a | 523 s |
| 500 | BASIL-S orig (1 dataset) | 0.926 | 0.617 | 0.714 | 0.51 | 181 s |
var-sortnregress control: AUROC 0.68-0.69 (d=100,200). BASIL-S screening recall 0.998-1.000 at every d. ECE <= 0.002. BASIL at d=50 full vs BASIL-S (6 datasets): AUROC +0.012, 57x faster (11.5 s vs ~660 s).

## Verdicts
- S1 (faithful to full BASIL at d=50, >=20x faster): **PASS** (gap +0.012, 57x).
- S2 (AUROC>=0.85, <=10 min at d=200, recall >=0.85): PASS at d=100/200 for both configs. d=500: AUROC 0.926 but orientation 0.71 and slope 0.51 (1 dataset): not a pass in any useful sense.
- S3 (AUROC >= SPADE-scaled - 0.02): **FAIL** at d=100 (-0.020 orig / -0.032 amended) and d=200 (-0.031 / -0.024). AUPRC within 0.02 or better at d=100/200 (amended).
- S4 (orientation >= SPADE-scaled - 0.03): orig config **FAILS and hits the kill line** at d=200 (0.830 vs 0.942, gap -0.112 < -0.08) and d=500; amended config passes (-0.024 d=100, -0.013 d=200). Slope 0.99, ECE<=0.002 pass.
- S5 (control): PASS.

## What changed after seeing test diagnostics (amendment, stated openly)
Chain end-scores of the two chains differed by hundreds to thousands of nats at d>=100 (not converged), explaining the orientation collapse at d=200. Added greedy sink-peeling initialisation, 3x longer chains, and chain selection by end score. These were tried on validation seeds 0-2 (d=100/200; 3 variants) and then run on test seeds 1000-1004; both configs are reported above. Validation was noisy (3 seeds); the amended config is not clearly better in AUROC (slightly worse), clearly better in orientation at d=200. At d=500 only the original config was run and it did not converge.

## Main findings
1. **SPADE-scaled (lambda/d + tree screening) already scales**: AUROC 0.997-0.999 at d=100/200/500, 0.7-9 min per dataset. BASIL-S does not beat it on structure at any d, and after the mixing fix is not faster at d=200 (114 s vs 133 s). Scalability is not a differentiator for BASIL.
2. **What SPADE-scaled lacks is calibrated function estimates**: curve slope 0.44 at d=100 (curve R2 0.57), i.e. edges are found but their amplitudes are shrunk by more than half, consistent with the G1 shrinkage finding at d=10-20 and worse at scale. BASIL-S gives slope 0.99 and calibrated edge probabilities (ECE<=0.002) at d=100/200.
3. BASIL-S mixing is the bottleneck: random-init chains fail at d>=200 and d=500 needs more work (1 dataset only).

## Caveats
5 datasets per cell, 1 at d=500; CIs not computed (too few). SPADE-scaled constants not tuned on iSCM (if anything a handicap). SPADE-scaled curve slope measured at d=100 only (4 datasets) and for d=200/500 not measured. n=1000 only; uniform noise only. Order-MCMC prior bias, kmax=3, Gaussian likelihood limitations as in README_basil.md. Novelty vs DiBS/BayesDAG unresolved.
