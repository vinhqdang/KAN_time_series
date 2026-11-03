# Independent-DAG replication with all four arms (SSC, SSA, SSR_k1.5, BASIL-W)

Status: committed BEFORE any run. Nothing here changes after results exist; deviations go to README_results.md under "Deviations".

## Why
Two weaknesses of the evidence in paper/draft_v1.md: (a) only 10 datasets per cell (5 in some), (b) cells C1-C6 of research/replication reuse the same 10 DAGs, so they are not independent replications. SSA (adaptive group lasso) was run in research/adaptive only on seeds 2000-2009/2000-2004 and with the old timing code.

## Design
- Generator: bench.py gen_full (iSCM, ER mean degree 2, mech m1), unchanged copy. d=100, n=1000.
- Cells (20 datasets each, 60 total). Seeds differ across cells so DAGs, mechanism assignment and noise are all new and different per cell, and differ from every earlier seed set (0-9, 1000-1004, 2000-2009):
  - unif: seeds 3000-3019; laplace: 3100-3119; hetero (ANM violated): 3200-3219.
- Methods per dataset, in this order in one process: SSC (fills SS_CACHE), SSA, SSR_k1.5, BASILWs. Method seed 0. No parameter retuned (screen K=15, lambda=0.2/d, 400 epochs; SSA gamma=1, eps=0.01; SSR k=1.5; BASIL-W g=n, pi=2/(d-1), m=20, 200d iterations, 2 chains, stabilisation 1e-3).
- "BASIL-W" = BASILWs (ridge-stabilised coefficient map used for do-effects; edge-function NRMSE from model-averaged curves, as in research/replication).
- bench.py: copy of research/replication/bench.py with the SSA branch from research/adaptive/bench.py added. Timing: per-method t_cost = t_ss (shared SPADE-scaled fit, measured once per dataset) + method-specific fit time (SSA/SSR: t_refit; BASIL-W: t_mcmc + t_extract; SSC: t_ss). Metric code and do-effect code are excluded. For SSA, t_refit covers the weight computation, the weighted fit and the importance computation.
- Compute: at most 2 cores (xargs -P2, OMP_NUM_THREADS=1, NJOBS=1). Job order: all unif seeds first, then laplace, then hetero, so a full unif cell completes first.
- Primary metric: edge-function NRMSE (mean over true edges per dataset, bench.py edge_curve_metrics), the dataset being the unit of analysis. Others: AUROC, AUPRC, orientation, do-effect NRMSE direct and indirect (docalc), calibration, coverage, cost.

## Pre-registered hypotheses
- H1: per cell, BASIL-W NRMSE < SSC NRMSE in at least 95% of datasets (i.e. >= 19 of 20). Kill: < 90% in any cell.
- H2: BASIL-W has lower NRMSE than SSA. Report paired mean difference (BASIL-W minus SSA), bootstrap CI, paired Wilcoxon, matched-pair rank-biserial and Cohen dz, wins. Predicted gap SSA minus BASIL-W: 0.05-0.08 for unif and hetero, about 0.2 for laplace. Kill: no significant gap (Holm-adjusted p >= 0.05) in unif.
- H3: SSA AUROC within 0.005 of SSC (|diff of cell means| <= 0.005) while BASIL-W AUROC is 0.01-0.03 below SSC.
- H4: do-effect NRMSE ordering BASIL-W <= SSA < SSC (cell means), separately for direct and indirect pairs. Counted as met if both inequalities hold in the cell mean (BASIL-W <= SSA is allowed to be a tie up to +0.01).
- H5 (descriptive): cost ratio vs SSC from median t_cost for SSA, SSR, BASIL-W per cell. Earlier estimates: SSR 1.5-1.7x, BASIL-W 1.3-1.45x.
Secondary (reported, not hypotheses): BASIL-W vs SSR; SSA vs SSR; SSA vs SSC; coverage and ECE of BASIL-W.

## Statistics
- Per cell, paired by dataset (seed): exact two-sided Wilcoxon signed-rank (zeros dropped), paired mean difference with 95% percentile bootstrap CI (10000 resamples of datasets, fixed RNG), Cohen dz, matched-pair rank-biserial, count of datasets won by each side.
- Multiplicity: Holm across the family of 3 cells x 3 pairwise comparisons on NRMSE (BASIL-W vs SSC, BASIL-W vs SSA, SSA vs SSC) = 9 tests; a second Holm family for the same 9 comparisons on do-effect (direct) and on AUROC is reported separately. Secondary comparisons involving SSR get their own Holm family (3 cells x 3 comparisons).
- Pooled-over-cells tests are exploratory only, but here, unlike research/replication, the three cells use independent graphs, so a pooled 60-dataset Wilcoxon is valid as a statement about this generator family; it is reported as secondary.
- Comparison with earlier seeds (1000-1004, 2000-2009) is descriptive: new cell means vs earlier means with bootstrap CIs; no test.
- Aggregation script: aggregate.py (written before results are inspected beyond progress counts), output results/analysis.md.

## What will NOT be concluded
Anything about real data, other generators, mechanism family m2, other n, d, density, Gaussian noise, d >= 200; significance from fewer than 20 datasets; that curve bands are calibrated; that timing ratios transfer to other machines (shared loaded machine, +-30%); that SSA hyperparameters (gamma, eps) are optimal, since they were not tuned and a tuned adaptive lasso could narrow the BASIL-W gap; that BASIL-W's win over SSA matters in practice if the gap is below about 0.03 even if significant; novelty. Three cells give three independent noise families, not replications of one cell.
