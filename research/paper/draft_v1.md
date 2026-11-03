# Draft v1 (supersedes draft_v0)

**Working title:** *Finding the edges is not enough: how much do sparse additive DAG learners shrink their edge functions, and what fixes it?*

Status: every number below is traceable to a CSV under research/. Related Work text: research/paper/related_work_v1.md (verified refs only). Not submittable: synthetic data except Sachs, 10 datasets per cell, novelty of the repair methods is limited (see 1.3).

## Abstract (draft)
Nonlinear additive-noise DAG learners are valued because each edge is an inspectable function, yet benchmarks score only whether edges are found. We evaluate recovered edge functions against ground truth. A screened, group-lasso spline learner (SPADE-scaled) recovers structure almost perfectly on iSCM data (AUROC 0.997-0.999, d = 100-500) while its edge curves keep a median 44% of the true amplitude (edge-function NRMSE 0.61-0.80) and interventional-effect curves derived from them are wrong even for direct parents (NRMSE 0.57-0.75). Three repairs are compared on fresh seeds (10 datasets per cell, 7 cells at d = 100 plus d = 200): an unpenalised refit on the selected support (NRMSE 0.20-0.47), the standard adaptive group lasso (0.24-0.52 at d = 100, structure unchanged, AUROC 0.998) and BASIL-W, a Bayesian model-averaged estimator warm-started from the learner's order (0.15-0.29; lower than SPADE-scaled in all 75 datasets). BASIL-W is clearly better than the refit under heavy-tailed and heteroscedastic noise and supplies calibrated edge probabilities (ECE <= 0.003), but costs 0.01-0.03 AUROC and, on the real Sachs data, no repair improves structure and the Bayesian and refit versions are less stable than SPADE-scaled because edges flip in and out of the selected set.

## 1. Introduction (outline) and honest contributions
1.1 Setting and gap: edge-function accuracy is not evaluated; our searches (research/lit_verify) found no ground-truth benchmark of it, but this is absence of evidence.
1.2 Contributions that survive the evidence: (i) metrics: edge-function NRMSE, do-effect NRMSE (direct/indirect), calibration slope; (ii) a measurement: structure-good learners return functions with less than half the amplitude at d up to 500, across noise families, n, a second mechanism family; (iii) a comparison of three repairs including the standard adaptive group lasso, with a protocol pre-registered by commit; (iv) negative results: SPIRE (group-MCP + independence term), BASIL-S cold start, full BASIL scaling, real-data instability.
1.3 What is NOT novel: refit = post-lasso (Belloni & Chernozhukov); adaptive group lasso for additive models (Huang, Horowitz & Wei 2010); the BASIL ingredients (Friedman & Koller 2003, Ni et al. 2015, Toth et al. 2025); interventional evaluation (Gentzel et al. 2019); the independence-plus-adaptive-group-sparsity idea (an ICML 2026 paper, read via abstract only). The exact combination BASIL-W and the edge-function NRMSE metric were not found in prior work.

## 2. Related work: see related_work_v1.md.
## 3. Metrics: edge-function NRMSE over TRUE edges (missed edge = 1), slope, curve R², do-effect NRMSE (propagation through the learned SEM in the learned order, 15-point grid, 20 direct + 20 indirect pairs, Monte-Carlo truth), ECE, band coverage.
## 4. Methods: SPADE-scaled (screen K = 15, lambda = 0.2/d); refit (top 1.5 d edges, lambda = 0); SSA (adaptive group lasso, w = (||theta||+0.01)^-1, same lambda); BASIL-W (g-prior marginal likelihood per (node, parent set), candidates = top 20 by SPADE-scaled importance, order MCMC warm-started at SPADE-scaled's order, 2 chains, 200 d iterations, ridge-stabilised coefficient map for propagation).

## 5. Results
### 5.1 Confirmatory replication (fresh seeds 2000-2009; research/replication)
Edge-function NRMSE, SPADE-scaled / refit / BASIL-W (ratio BASIL-W/SSC):
| cell | SSC | refit | BASIL-W | ratio |
|---|---|---|---|---|
| d100 unif | 0.612 | 0.213 | 0.169 | 0.28 |
| d100 Laplace | 0.803 | 0.470 | 0.252 | 0.31 |
| d100 hetero | 0.768 | 0.359 | 0.294 | 0.38 |
| d100 Gaussian | 0.692 | 0.278 | 0.150 | 0.22 |
| d100 n=500 | 0.617 | 0.231 | 0.201 | 0.33 |
| d100 n=2000 | 0.611 | 0.199 | 0.152 | 0.25 |
| d100 mech 2 | 0.625 | 0.270 | 0.155 | 0.25 |
| d200 (5 datasets) | 0.612 | 0.203 | 0.173 | 0.28 |
- BASIL-W < SPADE-scaled in 75/75 datasets (Wilcoxon p = 0.002 per d=100 cell, Holm 0.014).
- BASIL-W vs refit: ahead in 7/7 cells, significant after Holm only under Laplace, hetero, Gaussian and mech 2; ties for uniform, n=500, n=2000 (differences 0.03-0.05).
- Do-effect NRMSE ratio BASIL-W/SSC: direct 0.19-0.30, indirect 0.24-0.52; 0/75 datasets with indirect NRMSE > 2.
- AUROC gap BASIL-W - SSC: -0.006 to -0.032 (two cells just beyond the -0.03 target). Cost: BASIL-W 1.2-1.4x SPADE-scaled's time.
### 5.2 The standard fix (research/adaptive)
SSA at d=100 uniform: NRMSE 0.241, slope 0.90, AUROC 0.998 (unchanged), orientation 0.950; worse than BASIL-W by 0.072 (10/10 datasets, p = 0.002), worse than the refit by 0.029 (8/10, p = 0.037); d=200: 0.230 vs BASIL-W 0.176; Laplace: 0.52 vs 0.26. Reading: adaptive group lasso closes about 85% of the SPADE-scaled-to-BASIL-W gap at no structural cost; BASIL-W's remaining edge is small under uniform noise and large under heavy tails.
### 5.3 Ablations and bands (research/ablation, 8 datasets)
Warm start from the learner's order is essential (cold start: NRMSE 0.324 vs 0.177, orientation 0.809 vs 0.943); candidate source irrelevant at d=100 (recall 1.0); g, pi robust; kmax=2 harmful (+0.10); 4 chains x 600 d iterations changes edge NRMSE by -0.004 but chain end-scores still differ (not provably converged). Bands: 72% of the coverage shortfall comes from the 8% of true edges the model misses; a single inflation factor kappa = 1.55 (validation) gives test coverage 0.89-0.91 across uniform, Laplace and hetero noise, but over-covers found edges (0.96-0.98) and still misses missed ones (0.11-0.36): average, not conditional, validity.
### 5.4 Real data (Sachs, research/real_data): a negative result
10 subsamples of 2000: AUROC SPADE-scaled 0.613, refit 0.595, BASIL-W 0.563, full BASIL 0.544; orientation about 0.5 for every method. Function stability on a common edge set: SPADE-scaled 0.98, BASIL-W 0.54, refit 0.49 (edge flips: refit and BASIL-W return a zero curve in about 35% of edge-subsample cells; kept edges are stable, R² 0.96-0.98). Amplitude ratios relative to SPADE-scaled: refit 1.60, BASIL-W 1.41, i.e. shrinkage is present but smaller than in simulation. Caveats: d = 11, 18 contested consensus edges, pooled interventional data, overlapping subsamples.
### 5.5 Negative results kept in the paper
SPIRE (group-MCP + independence term): structure -0.02 AUROC vs the heuristic pipeline, slope 0.75; HSIC adds at most +0.05 orientation. BASIL-S cold start: chains disagree by hundreds of nats at d >= 100. Full BASIL: 11 min/dataset at d=50. SCORE/NoGAM: 0.07-0.11 AUROC below at d=10, 5-10x slower.

## 6. Limitations (stay in the paper)
Simulation is the only place function ground truth exists; real-data evidence is negative/ambiguous; 10 datasets per cell and C1-C6 share the same 10 DAGs, so pooled p-values overstate evidence; one generator family (iSCM, two mechanism families, four noise types); d=500 only 2 datasets and only the pre-registered BASIL-W (before stabilisation) on the first grid; the ridge stabilisation was chosen after seeing test results and later confirmed on fresh seeds (0/75 explosions); adaptive group lasso gamma/eps untuned; novelty limited as in 1.3; ICML 2026 related paper read via abstract only; calibrated bands are average-coverage only; BASIL-W inherits candidates and starting order from SPADE-scaled; BASIL-W's AUROC is below SPADE-scaled in every dataset.

## 7. Where the story stands (my honest read)
Strongest: (a) the measurement that structure-good learners return amplitude-poor functions, at scale, with ground truth; (b) the paired comparison showing the standard adaptive group lasso removes most of it without hurting structure; (c) BASIL-W's extra value is heavy-tail robustness and edge probabilities at a structure cost. Weakest: real-data evidence, novelty of the repairs, 10 datasets per cell. A paper that frames itself as an evaluation study ("how bad is the shrinkage and which standard fix works") fits the evidence better than one that sells BASIL-W.

## 8. TODO
1. Real data with function ground truth is unavailable; consider a semi-synthetic benchmark (fit on real data, simulate from the fitted SEM). 2. More datasets per cell and independent DAGs per cell. 3. Adaptive arm at d=500, with gamma/eps sensitivity. 4. Read the ICML 2026 paper in full when accessible. 5. Decide target venue; keep manuscript/ untouched.
