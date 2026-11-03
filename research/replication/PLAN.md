# Confirmatory replication of the main results: fresh seeds, final method version, significance tests

Status: committed BEFORE any replication run. Thresholds are copied from research/06 (F1-F5) and research/07 (R1-R5) and tightened as stated below. Nothing in this plan may be changed after results exist; deviations go into README_results.md under "Deviations".

## 1. What is replicated
The claims of research/paper/draft_v0.md section 5: SPADE-scaled keeps the edges but shrinks amplitudes; the plain refit and BASIL-W repair the edge functions; BASIL-W is more robust than the refit; do-effects follow. The earlier results used seeds 1000-1004 and, for BASIL-W, the pre-registered coefficient map in most cells and a post-hoc ridge-stabilised map in 4 cells. Here every cell uses the FINAL version (BASILWs = BASIL-W with the ridge-stabilised coefficient map; the stabilisation was chosen after seeing the seeds 1000-1004 test results, so this is its first clean test). Seeds 2000-2009 / 2000-2004 were never used for any decision (validation used seeds 0-9, earlier tests 1000-1004). No parameter (k=1.5, lambda=0.2/d, g=n, pi=2/(d-1), m=20, 200 d iterations, 2 chains, stabilisation lambda=1e-3) is retuned.

"BASIL-W" in all tables of this replication means the BASILWs run (the original-version BASILW is not run).

## 2. Design
Generator: iSCM, ER mean degree 2, bench.py gen_full (copied unchanged). Methods per dataset: SSC (SPADE-scaled), SSR_k1.5 (top 1.5 d edges, unpenalised refit), BASILWs. Same dataset for all three; method seed 0.

| cell | d | n | noise | mech | datasets (seeds) |
|---|---|---|---|---|---|
| C1 | 100 | 1000 | unif | m1 | 10 (2000-2009) |
| C2 | 100 | 1000 | laplace | m1 | 10 |
| C3 | 100 | 1000 | hetero (ANM violated) | m1 | 10 |
| C4 | 100 | 1000 | gauss | m1 | 10 |
| C5 | 100 | 500 | unif | m1 | 10 |
| C6 | 100 | 2000 | unif | m1 | 10 |
| C7 | 100 | 1000 | unif | m2 | 10 |
| C8 | 200 | 1000 | unif | m1 | 5 (2000-2004) |

Run order (so the most important cells finish first): C1, C2, C3, C4, C5, C6, C7, C8. 2 CPU cores max (OMP_NUM_THREADS=1, NJOBS=1, xargs -P2).
Caveat on independence: for a given seed the DAG and mechanism assignment are identical across cells that differ only in noise or n (the graph is drawn first from the seed). Cells are therefore NOT independent samples of graphs; pooled analyses below overstate the effective sample size and are labelled accordingly.

## 3. Metrics (definitions fixed, unchanged from 06/07)
Edge-function NRMSE over true edges (per dataset mean over its true edges; missed edge = about 1); AUROC (directed, from edge scores); orientation accuracy; slope (median calibration slope); 90% band coverage and ECE (BASIL-W only); do-effect NRMSE for up to 20 direct and 20 indirect true pairs per dataset (skipped pairs with std(true curve)<0.05 are reported). Edge-function NRMSE for BASIL-W uses the model-averaged curve of the pre-stabilisation map exactly as in the earlier runs; do-effects use the stabilised coefficients.
Cost (fix of a known problem): the old `time` column included do-effect and metric code. Now recorded separately and excluding metric / do-effect code: t_ss (SPADE-scaled screening+fit+importance), t_refit (unpenalised refit + importance), t_mcmc (fit_basil_s: features, score tables, order MCMC), t_extract (coefficient extraction). Cost of SSC = t_ss; SSR = t_ss + t_refit; BASIL-W = t_ss + t_mcmc + t_extract. Edge-curve extraction used by the NRMSE metric (mean_curve) is part of the metric code and is not timed. Timings are wall-clock with 2 jobs sharing the machine with other agents, so they are noisy; report medians.

## 4. Hypotheses, thresholds, verdict rules
Cell-level quantities are means over the cell's datasets unless stated. "Ratio" = (mean over datasets of BASIL-W) / (mean over datasets of SSC). Uncertainty on every ratio and difference: bootstrap 95% CI over datasets (10,000 resamples, seed 0, percentile).

- **H1 (edge-function accuracy vs SPADE-scaled).** BASIL-W/SSC NRMSE ratio <= 0.70 in EVERY cell C1-C8. Kill: ratio >= 0.90 in any cell. Tightening: a cell counts as PASS only if the point ratio is <= 0.70 AND the upper bootstrap 95% limit of the ratio is < 0.90; point ratio in (0.70, 0.90) or upper limit >= 0.90 is "not confirmed"; ratio >= 0.90 is "killed". H1 is confirmed only if all 8 cells pass.
- **H2 (vs the plain refit).** BASIL-W mean NRMSE <= SSR mean + 0.03 in at least 5 of the 7 d=100 cells (C1-C7). Kill: BASIL-W worse than SSR by more than 0.06 in 3 or more of the 7 cells. (Superiority over SSR is NOT part of H2; it is tested as a secondary question in section 5.)
- **H3 (do-effects).** Do-effect NRMSE ratio BASIL-W/SSC <= 0.70 for direct pairs and, separately, for indirect pairs, in every cell C1-C8, where the ratio uses the cell mean of per-dataset mean NRMSE_do. Kill: ratio >= 0.90 for direct or indirect in any cell. Stability criterion: no dataset with BASIL-W indirect NRMSE_do > 2 (the count over all 75 datasets is reported; H3 stability passes only if the count is 0; the count for SSC and SSR is reported for context). Medians over datasets are reported next to means because indirect errors are heavy-tailed.
- **H4 (structure not sacrificed).** AUROC BASIL-W - SSC >= -0.03 in every cell (cell mean of the paired difference). Kill: any cell < -0.05.
- **H5 (uncertainty).** 90% band coverage >= 0.80 (kill < 0.60) and ECE <= 0.10, in every cell. The earlier runs gave 0.73-0.80 coverage, so a miss on coverage is the expected outcome and will be reported as such.
- **S1 (cost, descriptive, not a hypothesis).** Median cost of BASIL-W / SSC at d=100 and d=200 (earlier estimate 1.4-1.5x; earlier criterion <= 2x, kill > 3x).

## 5. Statistics
Primary unit: the dataset (paired across methods). Per cell:
1. Paired two-sided Wilcoxon signed-rank test on dataset-level edge NRMSE, BASIL-W vs SSR and BASIL-W vs SSC (scipy exact where possible, zero differences dropped by the default rule).
2. Bootstrap 95% CI of the mean paired difference (BASIL-W minus comparator), and of the median paired difference.
3. Effect sizes: matched-pairs rank-biserial correlation, Cohen's dz (mean/sd of the paired differences), and the number of datasets in which BASIL-W is better.
4. Holm correction across the 7 d=100 cells, separately for each of the two comparisons (family size 7). C8 has 5 datasets: the smallest attainable two-sided exact Wilcoxon p is 2/32 = 0.0625, so C8 cannot reach significance at 0.05 by construction; it is reported descriptively (sign counts, CI) and excluded from the Holm family. With 10 datasets the smallest attainable p is 2/1024 = 0.00195, which survives Holm over 7 only when essentially all 10 differences share a sign.
Pooled analysis across cells (labelled as overstating evidence because graphs repeat across cells):
a. Per-cell mean paired difference BASIL-W - SSR and BASIL-W - SSC for C1-C7; their median across cells; exact two-sided sign test across the 7 cells (smallest attainable p = 0.0156).
b. Equal-weight mean of the 7 cell means of the paired difference with a cluster bootstrap 95% CI (resample datasets within each cell, cells fixed; and a second version resampling seeds jointly across cells, which respects the shared graphs).
c. All 70 dataset-level differences pooled in one Wilcoxon test, reported as exploratory only.
The same paired tests (Wilcoxon, Holm, CI) are run for do-effect NRMSE (direct and indirect) BASIL-W vs SSC and vs SSR, and for AUROC BASIL-W vs SSC, as secondary results.
Multiplicity: only the H1-H5 decision rules above are confirmatory; everything else is exploratory.

## 6. Power and what will NOT be concluded
Ten datasets per cell give low power: a paired Wilcoxon test with 10 pairs detects an effect around dz = 1 (a mean difference about as large as the sd of the dataset-to-dataset differences) with roughly 80% power at alpha 0.05, and after Holm over 7 cells much less; smaller differences such as the earlier -0.017 vs SSR at d=100 will not be resolved. Five datasets (C8) give no testable significance.
We will NOT conclude: that BASIL-W is better or worse than the refit in a cell where the Holm-adjusted test is not significant and the CI contains 0 (a tie within +-0.03 is the most that can be said); anything about real data, other generators, other graph densities, d > 200, or non-iSCM data; that band coverage is calibrated when it is below 0.80; that cells are independent replications (shared graphs); that cost ratios generalise beyond this machine; that a pass in this synthetic setting establishes novelty (prior-art check is incomplete, per draft_v0).
Consistency with the earlier seeds 1000-1004 is assessed descriptively: replication-cell means vs earlier means, and whether the earlier mean lies inside the new bootstrap CI. It is not a test.

## 7. Failure handling
Crashes are rerun (resume-safe CSVs, one file per dataset); a dataset that fails twice is reported as missing and counts against the hypotheses it would have informed (never silently dropped). Any numerical blow-up (do-effect NRMSE > 2, NaN) is reported with its dataset id. Code changes relative to research/spire (only bench.py: timing columns; paths) are listed in README_results.md.
