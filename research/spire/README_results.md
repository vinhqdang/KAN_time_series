# SPIRE: first pre-registered test (d=10, n=1000, uniform noise, iSCM, ER graphs)

Protocol: research/03_formulation.md (hypotheses committed in 5e971a5 before any code ran).
Tuning (validation seeds 0-9): c_lam in {0.5,1,2} -> c=2; mu in {0,0.3,1,3} at c=2. Test seeds 1000-1029, never used for any choice.
Debug seeds 5000-5002 were used while fixing the solver (not validation, not test).
Files: results/valA..D (validation), results/test_d10_n1000_part*.csv, results/test_d10_analysis.txt, analyze.py.

## Result (30 test datasets, means)
| method | AUROC | skeleton AUROC | orientation acc | curve slope (1=ideal) | curve R2 |
|---|---|---|---|---|---|
| SPADE | 0.989 | 0.994 | 0.948 | 0.68 | 0.81 |
| SPADE-adaptive (old heuristic) | 0.986 | 0.995 | 0.948 | 0.93 | 0.92 |
| SPIRE mu=0 | 0.968 | 0.992 | 0.780 | 0.75 | 0.72 |
| SPIRE mu=0.3 | 0.970 | 0.992 | 0.829 | 0.83 | 0.73 |
| GAM-ridge (no DAG) | 0.945 | 0.986 | 0.654 | 0.90 | 0.94 |
| var-sortnregress | 0.667 | 0.854 | 0.550 | - | - |
| R2-sortnregress | 0.641 | 0.854 | 0.505 | - | - |

Paired differences vs SPADE (AUROC): SPIRE mu=0 -0.021 [-0.031,-0.010]; SPADE-adaptive -0.003 [-0.006,0.000].
Orientation, mu=0.3 minus mu=0: +0.048 [0.018,0.081], Wilcoxon p=0.015 (uncorrected).

## Verdicts on pre-registered hypotheses
- H1 (bias fix, no ranking loss): FAIL. SPIRE slope 0.75 (<0.8 kill line), AUROC loss beyond the -0.02 margin. The old adaptive heuristic reaches slope 0.93 with no AUROC loss.
- H2 (independence term helps orientation): WEAK/INCONCLUSIVE. +0.05 orientation at the noise-floor scale; not significant on validation (+0.008, 10 datasets); larger mu hurts.
- H3 (beats GAM-ridge on orientation): SPIRE yes (0.83 vs 0.65), but SPADE already does far better (0.95). Side finding: skeleton AUROC is similar for SPADE and GAM-ridge (0.994 vs 0.986) while orientation differs by 0.29, so the DAG constraint matters for orientation; the earlier G2 "GAM-ridge ~ SPADE" was a skeleton-level statement.
- H4 (not a generator artifact): PASS on this generator. Sort baselines are at 0.64-0.67.

## Deviations from 03_formulation.md (not yet written into that file)
1. nb=128 (doc: 256). 2. HSIC gradient restricted to in-support edges (gate_grad=True). 3. rho schedule: double only if h did not fall by 4x since last check, cap 1e4. 4. Ranking score = W + 1e-9*KKT-margin to break ties among exact zeros. 5. sigma fixed to 1 (data standardized); doc suggested a pilot estimate. 6. c_lam grid {0.5,1,2}; an early validation run with c=3 was overwritten and its numbers are not kept (it was worse than c=2).
## Not yet done
d=20/50, n=500/2000, Gaussian/Laplace noise, CAM/SCORE/NoGAM/DAGMA baselines, Sachs, hetero-noise stress.
## Suspected reason
Exact zeros from the prox discard ranking information that SPADE's dense soft fit keeps; hard selection errors hit AUROC and orientation. A convergence budget test (T=1500) did not change this.
