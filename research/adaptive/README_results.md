# Adaptive group lasso arm (plan: PLAN.md, committed before running)

SSA = SPADE-scaled, then refit with adaptive group-lasso weights w = (||theta|| + 0.01)^-1 (normalised over screened edges), same lambda = 0.2/d. gamma and eps were not tuned. Seeds 2000-2009 (unif d=100, 10 datasets), 2000-2004 (laplace and hetero d=100, 5 each), 2000-2002 (d=200 unif, 3 datasets). SSA and SSC were run here; SSR and BASIL-W numbers are from research/replication/results/ (same generator and seeds, independent runs, deduplicated by cell-seed-method). Raw: results/*.csv; paired table: results_compare.txt.

## Means (SSC = SPADE-scaled)
| cell | method | NRMSE | slope | AUROC | AUPRC | orientation | do-direct | do-indirect |
|---|---|---|---|---|---|---|---|---|
| d100 unif (10) | SSC | 0.612 | 0.44 | 0.998 | 0.878 | 0.952 | 0.60 | 0.86 |
| d100 unif | **SSA** | 0.241 | 0.90 | 0.998 | 0.889 | 0.950 | 0.22 | 0.39 |
| d100 unif | SSR (replication) | 0.213 | - | - | - | - | - | - |
| d100 unif | BASIL-W (replication) | 0.169 | - | - | - | - | - | - |
| d100 laplace (5) | SSA | 0.520 | 0.66 | 0.984 | 0.615 | 0.742 | 0.42 | 0.57 |
| d100 laplace | SSC / SSR / BASIL-W | 0.805 / 0.470 / 0.261 | | | | | | |
| d100 hetero (5) | SSA | 0.375 | 0.87 | 0.996 | 0.862 | 0.910 | 0.26 | 0.55 |
| d100 hetero | SSC / SSR / BASIL-W | 0.766 / 0.349 / 0.294 | | | | | | |
| d200 unif (3) | SSA | 0.230 | 0.92 | 0.997 | 0.872 | 0.949 | 0.20 | 0.32 |
| d200 unif | SSC / SSR / BASIL-W | 0.606 / 0.193 / 0.176 | | | | | | |

## Paired differences in NRMSE (SSA minus other; positive = SSA worse)
- vs SSC: -0.37 (unif, 10/10 datasets better, Wilcoxon p=0.002), -0.29 laplace, -0.39 hetero, -0.38 d200.
- vs SSR: +0.029 unif (SSA worse on 8/10, p=0.037), +0.049 laplace, +0.027 hetero, +0.037 d200.
- vs BASIL-W: **+0.072 unif (SSA worse on 10/10, p=0.002)**, +0.082 hetero (5/5), +0.258 laplace (5/5), +0.055 d200 (3/3). With 5 datasets the smallest attainable Wilcoxon p is 0.0625.

## Reading against the pre-registered predictions
- A1 predicted SSA between SSC and SSR with NRMSE 0.3-0.5 and slope 0.7-0.9: **SSA did better than predicted** on uniform noise (0.24, slope 0.90) and is slightly worse than the plain refit (+0.03), i.e. it behaves like the refit. It is not between SSC and SSR in the sense predicted (it is next to SSR).
- A2 (structure kept): **confirmed**: AUROC equals SSC (0.998, 0.997); AUPRC 0.889 vs 0.878; orientation 0.950 vs 0.952. This is an advantage over the refit and BASIL-W, which lose 0.01-0.03 AUROC.
- Decision rule: SSA NRMSE minus BASIL-W is +0.072 (d100 unif) and +0.055 (d200): between the pre-registered 0.05 and 0.10 lines. Honest reading: **the standard adaptive group lasso closes about 85% of the NRMSE gap between SPADE-scaled and BASIL-W** at no structural cost; BASIL-W is still better in every dataset (consistent sign), by a small margin under uniform noise and a large one under Laplace noise (0.52 vs 0.26).

## Implications for the paper
1. The headline cannot be "our estimator fixes shrinkage"; the standard adaptive group lasso already removes most of it. The defensible claims: (a) the shrinkage exists and is large at scale in the SPADE-scaled configuration (slope 0.44); (b) the standard fix (adaptive group lasso) recovers most of it; (c) BASIL-W recovers somewhat more, is more robust to heavy tails, and adds edge probabilities, at a structural cost (AUROC -0.01 to -0.03) and a more complex pipeline.
2. Structure-preserving: SSA keeps AUROC 0.997-0.998, BASIL-W 0.973-0.979. A practitioner who wants accurate curves AND the best structure should use SSA, not BASIL-W, unless heavy-tailed noise or edge probabilities matter.

## Limitations
5-10 datasets (d=200: 3), one generator family, gamma and eps untuned (single setting), SSR/BASIL-W comparison numbers come from a separate agent's run on the same seeds (replication still in progress at the time of writing: only the files present were used), no do-effect comparison against SSR/BASIL-W reported here (columns omitted where the replication CSVs were not paired).
