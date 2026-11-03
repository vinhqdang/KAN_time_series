# Confirmatory replication: results (plan: PLAN.md, committed before any run)

Fresh seeds 2000-2009 (C1-C7, 10 datasets each) and 2000-2004 (C8, 5 datasets); methods SSC (SPADE-scaled), SSR_k1.5 (refit), and BASIL-W = the FINAL version (BASILWs, ridge-stabilised coefficient map). 75 of 75 datasets finished, no crashes, no retries, no NaNs, no dataset dropped. Raw: results/d*_s*.csv (one file per dataset), results/all.csv; every table and test below is produced by `python3.13 aggregate.py` into results/analysis.md (full tables there). Jobs: jobs.txt, run_all.sh (2 parallel jobs, OMP_NUM_THREADS=1, NJOBS=1).

Cells: C1 d100 n1000 uniform; C2 laplace; C3 heteroscedastic (ANM violated); C4 gaussian; C5 n=500; C6 n=2000; C7 mechanism family 2 (uniform, n=1000); C8 d=200 uniform n=1000 (5 datasets).

## 1. Means over datasets (edge-function NRMSE, lower is better; sd = across datasets)
| cell | SSC | SSR | BASIL-W (sd) | BASIL-W/SSC [95% CI] | BASIL-W - SSR [95% CI] | datasets BASIL-W < SSR | Holm p (Wilcoxon, C1-C7) |
|---|---|---|---|---|---|---|---|
| C1 d100 unif | 0.612 | 0.213 | 0.169 (0.028) | 0.277 [0.253, 0.301] | -0.043 [-0.070, -0.017] | 8/10 | 0.082 |
| C2 laplace | 0.803 | 0.470 | 0.252 (0.052) | 0.314 [0.278, 0.347] | -0.218 [-0.250, -0.188] | 10/10 | 0.014 |
| C3 hetero | 0.768 | 0.359 | 0.294 (0.025) | 0.383 [0.365, 0.400] | -0.065 [-0.087, -0.046] | 10/10 | 0.014 |
| C4 gauss | 0.692 | 0.278 | 0.150 (0.029) | 0.216 [0.193, 0.240] | -0.128 [-0.155, -0.099] | 10/10 | 0.014 |
| C5 n=500 | 0.617 | 0.231 | 0.201 (0.031) | 0.326 [0.299, 0.350] | -0.030 [-0.056, -0.004] | 6/10 | 0.131 |
| C6 n=2000 | 0.611 | 0.199 | 0.152 (0.028) | 0.248 [0.222, 0.274] | -0.047 [-0.078, -0.017] | 8/10 | 0.082 |
| C7 mech 2 | 0.625 | 0.270 | 0.155 (0.033) | 0.247 [0.215, 0.282] | -0.115 [-0.143, -0.089] | 10/10 | 0.014 |
| C8 d200 (5) | 0.612 | 0.203 | 0.173 (0.012) | 0.282 [0.265, 0.305] | -0.031 [-0.049, -0.011] | 4/5 | not testable (min p = 0.0625; p = 0.125) |

BASIL-W vs SSC: BASIL-W lower in every dataset of every cell (75/75), Wilcoxon p = 0.002 per d=100 cell (the minimum attainable with 10 pairs), Holm 0.0137; C8 p = 0.0625 (minimum attainable with 5). Per-dataset ratio ranges 0.15-0.42.

Other means (SSC / SSR / BASIL-W): slope 0.14-0.46 / 0.74-0.96 / 0.98-1.00; AUROC and the rest are in section 3 and results/analysis.md.

## 2. Verdicts against PLAN.md
- **H1 (NRMSE ratio BASIL-W/SSC <= 0.70 in every cell; kill >= 0.90): CONFIRMED.** 8/8 cells pass, ratios 0.22-0.38, upper CI limits <= 0.40.
- **H2 (BASIL-W <= SSR + 0.03 in >= 5 of 7 d=100 cells): CONFIRMED, 7 of 7**, and BASIL-W is lower than SSR in all 7 cell means (kill rule not triggered). Superiority is only partly established: Holm-significant (p = 0.014) in C2, C3, C4, C7; NOT significant after Holm in C1, C5, C6 (differences -0.043, -0.030, -0.047, CIs exclude 0 unadjusted, dz about -0.7 to -0.95, 6-8 of 10 datasets favourable). In those three cells the supported statement is "tie within about 0.05, BASIL-W probably slightly better", not "better". Pooled over C1-C7: median per-cell difference -0.065, 7/7 cells favour BASIL-W (exact sign test p = 0.0156), equal-weight mean -0.092 (cluster-bootstrap CI [-0.103, -0.082]; seed-joint [-0.111, -0.073]); pooled 70-dataset Wilcoxon p = 5e-12 is exploratory and overstated because graphs repeat across cells. Largest per-dataset deficit against SSR anywhere: +0.025 (inside the 0.03 tie band; no dataset exceeds 0.03).
- **H3 (do-effect ratio BASIL-W/SSC <= 0.70, direct and indirect, every cell; no indirect NRMSE_do > 2): CONFIRMED.** Direct ratios 0.19-0.30 (upper CI <= 0.35); indirect 0.24-0.52 (upper CI <= 0.59; hetero is the weakest at 0.516). **Datasets with BASIL-W indirect NRMSE_do > 2: 0 of 75** (maximum 0.69; also 0 of 75 for SSC and SSR). The earlier explosions (37, 7.8, 1.5) do not recur with the stabilised map. Against SSR (secondary): BASIL-W direct is lower in every cell mean but Holm-significant only in C2, C3, C4, C7 (p 0.014-0.039); indirect only in C4 (Holm 0.041; C2 0.059); in C1, C5, C6, C8 do-effects of BASIL-W and SSR are statistically indistinguishable (indirect differences -0.03 to -0.06, CIs include 0).
- **H4 (AUROC BASIL-W >= SSC - 0.03 in every cell; kill < -0.05): NOT CONFIRMED as stated, kill not reached.** Cell differences: -0.025, -0.032 (C2), -0.022, -0.006, -0.020, -0.031 (C6), -0.029, -0.019 (C8). C2 and C6 miss the target by 0.002 and 0.001 (6 of 8 cells pass). The loss is real, not noise: BASIL-W AUROC is lower than SSC in 9/10 or 10/10 datasets in every d=100 cell (Holm p 0.014), C8 5/5. Size 0.006-0.032.
- **H5 (coverage >= 0.80, ECE <= 0.10): ECE CONFIRMED (<= 0.0024 everywhere); coverage NOT CONFIRMED** in 7 of 8 cells: 0.793, 0.753, 0.781, **0.808**, 0.798, 0.774, 0.742 (lowest, mechanism family 2), 0.788 (C8). Kill line (0.60) not reached. As the earlier runs predicted, bands under-cover the nominal 0.90 pointwise.
- **S1 (cost, descriptive).** Median wall-clock excluding metric and do-effect code: BASIL-W 1.23-1.44x SSC (d100: 1.31-1.44; d200: 1.23 = 163 s vs 137 s); SSR 1.47-1.70x SSC. At d100 n1000, SSC 27-31 s, SSR 43-48 s, BASIL-W 36-43 s (SPADE-scaled fit 27-31 s + MCMC 10-11 s + coefficient extraction under 0.03 s). Because BASIL-W reuses the SPADE-scaled fit, its marginal cost over SSC is the MCMC (10-20 s at d100-200), cheaper than the refit (8-55 s). Timings were taken with other jobs on the machine (load 3-6 on 4 cores), so treat them as +-30% estimates. The earlier 1.4-1.5x estimate stands.

## 3. Structure and calibration (means; BASIL-W minus SSC)
| cell | AUROC SSC | AUROC SSR | AUROC BASIL-W | orientation SSC / SSR / BASIL-W | do direct SSC / SSR / BASIL-W | do indirect SSC / SSR / BASIL-W (median) | coverage | ECE |
|---|---|---|---|---|---|---|---|---|
| C1 | 0.998 | 0.982 | 0.973 | 0.952 / 0.925 / 0.943 | 0.603 / 0.188 / 0.148 | 0.855 / 0.258 / 0.225 (0.182) | 0.793 | 0.0014 |
| C2 | 0.991 | 0.908 | 0.959 | 0.745 / 0.747 / 0.919 | 0.760 / 0.406 / 0.182 | 0.920 / 0.501 / 0.328 (0.328) | 0.753 | 0.0024 |
| C3 | 0.998 | 0.968 | 0.976 | 0.892 / 0.843 / 0.947 | 0.735 / 0.317 / 0.199 | 0.926 / 0.540 / 0.478 (0.492) | 0.781 | 0.0016 |
| C4 | 0.997 | 0.972 | 0.992 | 0.897 / 0.884 / 0.984 | 0.660 / 0.221 / 0.126 | 0.868 / 0.343 / 0.231 (0.209) | 0.808 | 0.0008 |
| C5 | 0.998 | 0.980 | 0.978 | 0.947 / 0.918 / 0.953 | 0.601 / 0.213 / 0.180 | 0.861 / 0.300 / 0.266 (0.213) | 0.798 | 0.0012 |
| C6 | 0.999 | 0.981 | 0.967 | 0.954 / 0.920 / 0.930 | 0.597 / 0.162 / 0.149 | 0.855 / 0.235 / 0.207 (0.151) | 0.774 | 0.0016 |
| C7 | 0.998 | 0.959 | 0.969 | 0.911 / 0.902 / 0.935 | 0.592 / 0.221 / 0.155 | 0.817 / 0.310 / 0.272 (0.306) | 0.742 | 0.0015 |
| C8 | 0.997 | 0.967 | 0.979 | 0.946 / 0.930 / 0.950 | 0.582 / 0.161 / 0.144 | 0.855 / 0.271 / 0.211 (0.199) | 0.788 | 0.0006 |
Notes: SSC finds structure best (AUROC 0.99-1.00) but returns curves with 14-46% of true amplitude (median slope), and its do-effects are wrong even for direct pairs (NRMSE 0.58-0.76). The refit collapses under Laplace noise (AUROC 0.908; edge NRMSE 0.47) while BASIL-W stays at 0.25, AUROC 0.959. Share of do-effect pairs skipped for tiny true effect: about 3.8 of about 40 per dataset, identical across methods.

## 4. Consistency with the earlier seeds 1000-1004 (descriptive, not a test)
- SSC and SSR edge-NRMSE means: the earlier mean lies inside the new bootstrap CI in all 16 cases (differences <= 0.025). Qualitative results reproduce: SSC about 0.61-0.80, SSR 0.20-0.47, Laplace is the cell where the refit is fragile (0.449 earlier, 0.470 now).
- BASIL-W edge NRMSE is LOWER on the new seeds in 7 of 7 d=100 cells (by 0.009 to 0.031; e.g. C1 0.194 -> 0.169, C4 0.174 -> 0.150, C7 0.186 -> 0.155); the earlier mean falls outside the new CI in 5 of 7 d=100 cells (C1, C4, C5, C6, C7); C8 equal (0.175 vs 0.173). We checked that this is not a code change: re-running seeds 1000 and 1001 (C1) with the copied code gives bit-identical NRMSE, AUROC and coverage to the earlier files (results_repro/, not used in any analysis). With 5 earlier datasets (sd of the mean about 0.012) the shift is about 2 sd in one direction; the seeds 1000-1004 were simply somewhat harder for BASIL-W. The conclusion is that the earlier BASIL-W-minus-SSR gaps were if anything underestimates: earlier -0.017/-0.080/-0.188/-0.034/-0.005/-0.013/-0.070 for C1/C4/C2/C3/C5/C6/C7 vs now -0.043/-0.128/-0.218/-0.065/-0.030/-0.047/-0.115 (same ordering, same sign, all larger in magnitude). The earlier near-ties (C1, C5, C6) remain near-ties.
- Do-effects: the earlier non-stabilised BASIL-W numbers (e.g. C1 0.24 / 0.39; C5 0.24 / 7.84; C8 0.20 / 37) are not comparable with the new stabilised ones (C1 0.15 / 0.23; C5 0.18 / 0.27; C8 0.14 / 0.21). The earlier stabilised re-runs on the 4 blow-up cells (indirect 0.32, 0.47, 0.24) are similar to the new indirect values for the same cells (0.27 for n=500, 0.48 for hetero, 0.21 for d=200). The earlier "stabilised matches the refit, not better" statement also holds: indirect BASIL-W - SSR is -0.03 to -0.06 in the unaffected cells, not significant.
- Coverage: earlier 0.73-0.79, now 0.74-0.81. AUROC gap BASIL-W - SSC: earlier -0.012 to -0.034, now -0.006 to -0.032. Consistent.

## 5. What can and cannot be concluded
Established on this generator (iSCM, ER degree 2, d 100-200, n 500-2000, four noise families, two mechanism families): (i) BASIL-W cuts edge-function error by 62-78% relative to SPADE-scaled in every dataset (75/75), with the do-effect error cut by 70-81% (direct) and 48-76% (indirect); (ii) BASIL-W is never worse than the unpenalised refit by more than the 0.03 tie band, and is clearly better only under non-Gaussian-uniform conditions (Laplace -0.22, Gaussian -0.13, mechanism family 2 -0.115, heteroscedastic -0.065) with Holm p = 0.014; (iii) the stabilised version has no numerical blow-up in 75 datasets (0 with indirect NRMSE_do > 2; the earlier rate was 4 of 45); (iv) a small, consistent AUROC loss vs SPADE-scaled (0.006-0.032); (v) bands under-cover (0.74-0.81 at nominal 0.90), edge probabilities are well calibrated (ECE <= 0.0024).
NOT concluded: that BASIL-W beats the refit in C1, C5, C6 or C8 (not significant after Holm / not testable; a tie within 0.05 is the supported claim); anything about real data, other generators, other densities, d > 200 (d=500 was not re-run), non-additive mechanisms; that curve bands are calibrated; that the 7 cells are independent replications (the same 10 graphs underlie C1-C6 and the same graph structures C7; pooled p-values are therefore indicative only); that timing ratios transfer to other machines; novelty.
Power: 10 datasets per cell resolve differences of roughly one sd of the paired differences (here about 0.02-0.05 for NRMSE); that is why the small gaps to SSR in C1/C5/C6 stay unresolved while the large effects (vs SSC everywhere, vs SSR in C2/C3/C4/C7) are decisive.

## 6. Caveats, failures, deviations
- All pre-registered hypotheses except H4 and the coverage half of H5 were met; those two missed their TARGETS (not the kill lines): H4 by 0.002 and 0.001 in two cells, H5 coverage in 7 of 8 cells.
- The stabilisation was chosen after seeing seeds 1000-1004; this replication is its first clean test, but the stabilisation lambda (1e-3) was not re-tuned and only 1 value was ever used.
- BASIL-W's edge-function NRMSE and band coverage use the model-averaged curves of the unstabilised map (as in the earlier runs); the stabilised map is used for the do-effect propagation only. Edge-function results are therefore identical to the pre-registered BASIL-W (confirmed by bit-identical re-run of two earlier datasets).
- Code changes relative to research/spire (bench.py copy only): timing columns t_ss, t_refit, t_mcmc, t_extract, t_cost (cost excludes metric and do-effect code; the old `time` column is removed); BASIL-W is BASILWs throughout. t_mcmc contains feature whitening, score tables and the order MCMC; t_extract is only the coefficient-tensor extraction (under 0.03 s); the mean-curve extraction used by the NRMSE metric is metric code and untimed. Cost of SSR and BASIL-W includes the shared SPADE-scaled time t_ss (cached, measured once per dataset).
- Extra, outside the plan: results_repro/ (2 datasets, SSC + BASILWs) as a code-reproducibility check; not included in any table.
- Cells share graphs across seeds (C1-C6 use the same 10 DAGs and mechanism assignments, C7 the same DAGs): see section 5.
- Cost measurements were made on a shared, loaded machine.
- d=500 and Sachs are not part of this replication.
