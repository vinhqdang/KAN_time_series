# Draft v0 — working title
**Finding the edges is not enough: measuring and repairing edge-function shrinkage in sparse additive DAG learners at scale**

Status: skeleton with real numbers from research/spire/README_*.md. Every number below is traceable to a CSV in research/spire/results/. Items marked [TODO] are not done. Not for submission: prior-art check is incomplete, no real-data results, 5 datasets per cell.

## Abstract (draft)
Additive-noise DAG learners with spline or neural edges are valued because each learned edge is an inspectable function. We show that this value is not guaranteed: on iSCM benchmarks a screened, sparsity-penalised spline learner (SPADE-scaled) recovers structure almost perfectly (AUROC 0.997-0.999 for d = 100-500) while the edge curves keep only 17-45% of the true amplitude (median calibration slope 0.44; edge-function NRMSE 0.61-0.79), and interventional effects computed from them are wrong even for direct parents (NRMSE 0.57-0.75). We introduce edge-function NRMSE and do-effect NRMSE as evaluation metrics and compare two repairs: a plain unpenalised refit on the selected support and BASIL-W, a Bayesian model-averaged estimator (closed-form g-prior marginal likelihood, order MCMC warm-started from the learner's own order). Both cut function error by roughly 3-4x across noise families, sample sizes, a second mechanism family and d up to 500. BASIL-W beats the refit in every cell (by 0.19 NRMSE under Laplace noise where the refit degrades) and supplies calibrated edge probabilities (ECE 0.002); its credible bands under-cover (0.73-0.80 at nominal 0.90), and an unregularised version exploded on indirect effects until ridge-stabilised.

## 1. Introduction (outline)
1. Interpretable edge functions are the selling point of KAN/spline/NAM-style causal learners; evaluation in the literature scores edge presence only (our searches found no benchmark that scores recovered edge functions).
2. Group-lasso-type penalties shrink amplitudes; the effect is known for lasso, ignored in DAG evaluation. We measure it.
3. Contributions: (i) metrics (edge-function NRMSE, do-effect NRMSE, calibration slope); (ii) evidence that structure-good learners return amplitude-poor functions at scale; (iii) two repairs, one simple, one Bayesian, with an honest comparison; (iv) a record of what did not work (SPIRE: group-MCP + HSIC, BASIL-S cold-start) so others do not repeat it.

## 2. Related work [TODO: verify every reference; see research/01_literature_review.md]
Identifiability (Hoyer 2008, Peters 2014); continuous DAG learning (NOTEARS, DAGMA, GOLEM, NOTEARS-MLP, GraN-DAG); score-based ordering (CAM, SCORE, NoGAM, DAS); additive and KAN models (SpAM, NAM, KAN and critiques); sparsity bias (adaptive lasso, MCP/SCAD, Xu et al. 2022; an ICML 2026 paper on independence + adaptive group sparsity not yet read); Bayesian structure learning (order MCMC, DiBS, BayesDAG, GP-based order methods, Wang-Kolar-Drton confidence sets); benchmark validity (varsortability, R²-sortability, iSCM). Prior art on Bayesian additive DAG learning is substantial; novelty of the estimator is NOT claimed.

## 3. Problem and metrics
Standardised x, x_i = Σ f_ij(x_j) + ε_i. For each TRUE edge: edge-function NRMSE = ||φ̂−f||/||f|| on the data points (centered, truth divided by the effect's sd; a missed edge scores 1); calibration slope (median of regression of φ̂ on f); curve R². Do-effect NRMSE: propagate do(x_c = v) through the learned SEM in the learned causal order (training residuals reused), compare with Monte-Carlo truth (1000 draws), 15 grid points, 20 direct and 20 indirect pairs per dataset. Edge probabilities: ECE; band coverage for credible bands.

## 4. Methods
- SPADE-scaled: per-node ExtraTrees screening (K=15), spline edges (G=8), group-lasso lambda = 0.2/d, NOTEARS acyclicity by augmented Lagrangian.
- Refit: keep SPADE-scaled's top 1.5 d edges, refit unpenalised under the same acyclicity constraint. k chosen on 3 validation datasets.
- BASIL-W: candidate parents (top 20 by SPADE-scaled importance), per-(node, parent set) log marginal likelihood under a g-prior (g = n) on whitened spline features, structure prior π = 2/(d-1), |pa| ≤ 3, order MCMC (2 chains, 200 d iterations, chain selection by end score) warm-started at the order implied by SPADE-scaled. Outputs: P(edge), model-averaged coefficients, 90% bands. Post-hoc: ridge-stabilised coefficient map (λ = 1e-3 x largest basis eigenvalue) for propagation.

## 5. Experiments and results
Generator: iSCM (standardise each node right after generation), ER mean degree 2, mechanism families {tanh, sin, quadratic, softsign} and {linear, tanh, quadratic, sigmoid, sin}, noise {uniform, Gaussian, Laplace, heteroscedastic}. Test seeds 1000-1004; validation seeds 0-2/0-9; all hypotheses pre-registered by commit before runs (research/03-07).
### 5.1 Structure vs function at scale (d=100/200, n=1000)
| d | method | NRMSE | slope | AUROC | orientation | AUPRC |
|---|---|---|---|---|---|---|
| 100 | SPADE-scaled | 0.616 | 0.44 | 0.997 | 0.958 | 0.878 |
| 100 | refit | 0.211 | 0.96 | 0.966 | 0.922 | 0.853 |
| 100 | BASIL-W | 0.194 | 0.99 | 0.973 | 0.940 | 0.870 |
| 200 | SPADE-scaled | 0.612 | 0.45 | 0.997 | 0.942 | 0.855 |
| 200 | refit | 0.211 | 0.96 | 0.967 | 0.920 | 0.828 |
| 200 | BASIL-W | 0.175 | 0.995 | 0.979 | 0.953 | 0.890 |
### 5.2 Robustness (d=100 unless noted): see research/spire/README_do.md table (BASIL-W/SPADE-scaled NRMSE ratio 0.23-0.40; BASIL-W below refit in all 9 cells; Laplace: refit 0.449 vs BASIL-W 0.261; d=500, 2 datasets: 0.69 / 0.23 / 0.16).
### 5.3 Interventional effects: direct pairs 0.13-0.25 (BASIL-W) vs 0.57-0.75 (SPADE-scaled); indirect: pre-registered BASIL-W unstable in 4 of 45 dataset-cells (mean up to 37), stabilised 0.18-0.47, refit 0.26-0.53, SPADE-scaled 0.82-0.93.
### 5.4 Negative results (include, they are findings)
SPIRE (group-MCP + independence term): worse than the heuristic pipeline on structure (AUROC -0.021, slope 0.75), HSIC term +0.05 orientation at best. BASIL-S cold start: order MCMC chains disagree by hundreds of nats at d ≥ 100, orientation collapses at d=200 (0.83 vs 0.94). Full BASIL does not scale (11 min/dataset at d=50). SCORE/NoGAM are 5-10x slower and 0.07-0.11 AUROC below at d=10.

## 6. Limitations (must stay in the paper)
Synthetic data only; ≤5 datasets per cell, d=500 only 2; no formal significance tests; stabilisation chosen after seeing test results and re-run only on the cells that exploded; credible bands under-cover; BASIL-W inherits candidates and starting order from SPADE-scaled; refit's support rule uses the mean degree; AUROC of BASIL-W is 0.01-0.03 below SPADE-scaled; novelty of the Bayesian estimator unproven; truth for NRMSE exists only in simulation.

## 7. TODO before any submission
1. Re-run the full grid with the stabilised BASIL-W so all rows share one version. 2. Real data: Sachs (structure and stability only; no function ground truth). 3. ≥ 20 datasets per cell and significance tests. 4. Complete literature verification, incl. the ICML 2026 independence + adaptive group sparsity paper. 5. Ablations: candidates from SPADE-scaled vs marginal-R², warm start vs cold start, g and π sensitivity at scale. 6. Fix band coverage (e.g. posterior with variance inflation or a conformal band). 7. Decide venue and what stays from published SPADE (do not modify manuscript/).
