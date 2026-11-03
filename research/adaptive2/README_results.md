# SSA at d=500 and (gamma, eps) sensitivity (plan: PLAN.md, committed before any run)

Code: bench.py (copy of research/replication/bench.py plus parameterised SSA), run_all.sh, jobs.txt; raw CSVs in results/; tables in analysis.txt (aggregate.py). One CPU core, sequential. No file outside research/adaptive2/ was changed. Everything below is 2 datasets (Part A) or 5 datasets (Part B): descriptive, no significance claims.

## Part A: d=500, n=1000, uniform, m1, seeds 1000 and 1001 (mean of 2 datasets; per-dataset values in analysis.txt)
| method | NRMSE | slope | do-direct | do-indirect | AUROC | orientation | fit time (s) |
|---|---|---|---|---|---|---|---|
| SSC (SPADE-scaled) | 0.690 | 0.33 | 0.68 | 0.90 | 0.9983 | 0.919 | 472 |
| SSA (gamma 1, eps 0.01) | **0.295** | 0.86 | 0.19 | 0.42 | 0.9975 | 0.919 | 472 + 250 = 721 |
| SSR (top 1.5 d refit) | 0.229 | 0.93 | 0.19 | 0.28 | 0.9534 | 0.912 | 472 + 225..259 = 697..730 |
| BASIL-W (stabilised, final) | 0.162 | 0.995 | 0.126 | 0.174 | 0.9773 | 0.952 | 472 + 69 = 540 (seed 1000 only) |

Per seed SSA NRMSE 0.308 / 0.282 (BASIL-W 0.166 / 0.157, SSR 0.229 / 0.228). Re-run SSC and SSR reproduce the old CSVs exactly (same NRMSE, slope, do-effects, AUROC), so the comparison is on identical models. BASILWs on seed 1000 was run here (NRMSE 0.1664 unchanged from the pre-stabilisation run, as designed; do-indirect 0.173 vs 0.196 before; the old 1.33 on seed 1001 became 0.176 under stabilisation); seed 1001 BASILWs values are from research/spire/results/do_stab (its time column is not a clean fit time, so BASIL-W time is seed 1000 only). Fit time = shared SPADE-scaled fit + method-specific fit, metric and do-effect code excluded; wall clock on a shared machine (load about 5 on 4 cores), noisy.

Reading against the pre-registered predictions:
- P1 (SSA NRMSE 0.22-0.30, slope 0.85-0.95, next to SSR, worse than BASIL-W): inside the range, but at the top of it. SSA is **0.066 worse than SSR and 0.133 worse than BASIL-W** at d=500, a larger gap than at d=100/200 (+0.072 / +0.055 vs BASIL-W). SSA is therefore not "next to SSR" at d=500.
- P2 (AUROC within 0.005 of SSC): met (0.9975 vs 0.9983). SSA keeps SSC's structure; SSR and BASIL-W lose 0.03 and 0.02 AUROC.
- P3 (SSA do-effects between SSR and BASIL-W): **not met**. Direct is a tie with SSR (0.19 vs 0.19), indirect is clearly worse (0.42 vs 0.28 SSR, 0.17 BASIL-W); seed 1000 SSA indirect 0.53. Still far better than SSC (0.90).
- P4 (fit time near SSR): met (SSA refit 250 s vs SSR 225-259 s); BASIL-W's extra cost over SPADE-scaled is smaller (69 s, seed 1000) than either refit at d=500.
Orientation: SSA equals SSC (0.919); BASIL-W is higher (0.952) on these two datasets.
Verdict: at d=500 the standard adaptive group lasso removes about 2/3 of the SPADE-scaled NRMSE gap to BASIL-W (0.69 to 0.30 against 0.16), keeps AUROC, but leaves a larger residual gap than at d=100 and is worse than the plain refit on edge NRMSE and indirect do-effects. Two datasets only.

## Part B: sensitivity grid, d=100, n=1000, uniform, seeds 2000-2004 (mean over 5 datasets; one SPADE-scaled first fit reused by the 9 refits)
Reference on the same 5 seeds (research/replication/results/all.csv): SSC 0.610 / slope 0.44 / AUROC 0.9987; SSR 0.216 / 0.94 / 0.9852; BASIL-W 0.159 / 0.99 / 0.9759.

NRMSE (slope; AUROC):
| gamma \ eps | 0.001 | 0.01 | 0.1 |
|---|---|---|---|
| 0.5 | 0.228 (0.89; 0.9985) | 0.318 (0.79; 0.9982) | 0.359 (0.77; 0.9985) |
| 1 | **0.199** (0.92; 0.9988) | **0.234 default** (0.89; 0.9984) | 0.336 (0.78; 0.9979) |
| 2 | 0.215 (0.90; 0.9988) | **0.204** (0.92; 0.9986) | 0.245 (0.90; 0.9978) |

Per-dataset NRMSE for the default: 0.210, 0.191, 0.211, 0.297, 0.260 (this reproduces the 0.241 reported earlier on 10 seeds, the 5-seed mean is 0.234).
- Range over the grid: 0.199 to 0.359 (spread 0.16). **Q1 (spread < 0.05) failed**: eps = 0.1 is clearly worse (0.25-0.36) in every gamma; a large eps flattens the weights toward plain group lasso (Q2 first half confirmed: worst cell gamma 0.5, eps 0.1, slope 0.77). The (0.5, 0.01) cell (0.318) is worse than both neighbours in eps for gamma 0.5 (0.228 at 0.001) and the pattern is not smooth in gamma; with 5 datasets that irregularity is not resolved.
- AUROC is 0.9978-0.9988 in all 9 settings (SSC 0.9987): structure is insensitive. Q2's second half (gamma 2, small eps loses AUROC) failed: it has the highest AUROC (0.9988).
- **No setting reaches BASIL-W** (0.159; rule: <= 0.169). Best grid mean 0.199 (gamma 1, eps 0.001), still 0.040 above. Per dataset, the best settings come near BASIL-W's per-dataset value on one of the five (counts in analysis.txt). Q3 confirmed.
- Settings better than the default by more than 0.03 on the mean and on 5/5 datasets: (gamma 1, eps 0.001): -0.035; (gamma 2, eps 0.01): -0.030. By the pre-registered rule these are "clearly better" in this sample, but they were examined on the very seeds used for sensitivity, and SSA's default had already been run on seeds 2000-2004 earlier. **They are not adopted.** The default (1, 0.01) remains the SSA configuration for all headline numbers (this report, research/adaptive, the paper draft). Recommendation: if an improved setting is wanted, validate (1, 0.001) and (2, 0.01) against the default on fresh seeds not used in any of these studies, with the choice and threshold written down first.

## Implications for the paper
1. The statement "gamma and eps untuned" can now be accompanied by a sensitivity result: conclusions about structure (AUROC) do not depend on them; NRMSE does (0.20-0.36), mainly through eps; the default sits mid-range, and smaller eps (or gamma 2) would likely shrink SSA's gap to BASIL-W by roughly a quarter to a third at d=100 (0.234 to about 0.20 against 0.159), but not close it. The claim "BASIL-W is still better than the standard adaptive group lasso" holds for every grid setting (mean) at d=100.
2. At d=500 the SSA gap to BASIL-W is larger (0.13) than at d=100 and the paper's sentence "closes about 85% of the gap" is specific to d=100 (2 datasets at d=500: about 76% of the gap to BASIL-W: (0.690-0.295)/(0.690-0.162)=0.75).

## Limitations
2 datasets at d=500 (BASIL-W time from one), 5 for the grid, no significance claims; one generator family, uniform noise only, mech m1; the grid seeds are a subset of those on which the default was already evaluated, so the grid is sensitivity only and cannot select a setting; BASIL-W seed 1001 values come from an earlier independent run on the same data; wall-clock times on a loaded shared machine.
