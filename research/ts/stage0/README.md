# Stage 0: aggregated CausalRivers as a test of contemporaneous-edge discovery

Plan: `research/ts/02_plan_stage0.md` (commit 3cd4581). Gates: `gates.md` (committed before any method ran). Validation: `validation.md`. Results: `results.md` (written after the test run). Raw data (CausalRivers `product.zip`, Stein et al., ICLR 2025, licence dl-de/by-2-0) and the cached 15-min arrays are kept outside the repository.

## Pipeline (run in this order, all in this folder)
`load_raw.py` (CSV to float32 arrays), `prep_pool.py` (aggregation, pool statistics), `prep.py` (subgraph sampling, tau_e, labels -> `subgraphs.json`, `edges.csv`), `gates.py` (-> `gates.md`), `run_methods.py val|val2|test`, `select_hparams.py`, `analyze.py` (-> `results.md`). Library code: `common.py`, `lib.py`, `methods.py`. Use `/usr/bin/python3` (3.13) with numpy 1.26.4, scipy 1.16.3, pandas 3.0.5, scikit-learn 1.9.1, statsmodels 0.15.0, tigramite 5.2.10.1, lingam 1.13.0, networkx 3.7. At most 2 processes were used.

## Software notes
- tigramite installed with pip without trouble. lingam would not install with its pinned scipy build; `pip install --no-deps lingam` plus graphviz, semopy, psy, autograd worked. lingam's `VARLiNGAM(lags=p, prune=...)` is used as is.
- DYNOTEARS: no maintained package installs (causalnex is retired); own implementation (`methods._solve`): Pamfil et al. objective (least squares, L1 on W and lag matrices, trace-exponential acyclicity), augmented Lagrangian, L-BFGS-B on split positive/negative parts, h tolerance 1e-8. A synthetic check (4 nodes, Laplace noise) shows it recovers the skeleton but reverses contemporaneous orientation on standardised data, as expected from a Gaussian-likelihood method without a non-Gaussianity step; it was not tuned on that check.
- NOTEARS-style SVAR reference (`score_notears_svar2`): OLS VAR, then linear NOTEARS (same solver, no lags) on the reduced-form residuals, lagged structural matrices A_k = A_red_k (I - W). Distinct from DYNOTEARS, which estimates W and A jointly.
- PCMCI+: tigramite, ParCorr, tau_min=0, tau_max=p. Score: 2 * [link i->j in the PCMCI+ graph; undirected contemporaneous links count in both directions] + max_tau |val_matrix_ij|. Graph links therefore outrank non-links; ties within tiers are broken by |partial correlation|. Undirected 'o-o'/'x-x' links give a tie in orientation (0.5).
- VAR: OLS with intercept, score = max over lags of |coefficient|. Correlation: |Pearson r| (symmetric).
- Controls: var-sortnregress / R2-sortnregress (order by increasing marginal variance / by R^2 on all other nodes at the same time step, random tie-break, OLS on predecessors, score |beta|), random uniform scores. The pickled river graphs were inspected with pickletools before loading (only networkx classes).

## Hyperparameters (fixed on validation seeds 5000..5004, all four cells = 20 subgraphs; criterion mean directed AUROC; table in `validation.md`)
| method | 6 h | 12 h | 24 h |
|---|---|---|---|
| PCMCI+ pc_alpha | 0.01 | 0.01 | 0.01 |
| VAR-LiNGAM prune | False | True | True |
| DYNOTEARS lambda | 0.1 | 0.1 | 0.1 |
| NOTEARS-SVAR lambda | 0.1 | 0.1 | 0.1 |
Grids: pc_alpha {0.01, 0.05}; prune {True, False}; DYNOTEARS lambda {0.001, 0.01, 0.1}, extended after the first validation run with {0.3, 1.0} because the best value sat at the edge of the grid (run `val2`; 0.1 stayed best); NOTEARS-SVAR lambda {0.01, 0.1, 0.3, 1.0}. Lag order p=1 primary; the p=2 sensitivity run reuses the p=1 hyperparameters (not re-tuned). Validation subgraphs are excluded from every reported number.

## Decision rules as operationalised before the test run
See the last section of `gates.md` (committed before any method ran): verdict methods PCMCI+, VAR-LiNGAM, DYNOTEARS; NOTEARS-SVAR is the plan's reference and excluded from the verdicts (shown as sensitivity); Holm family of 9 (12 with the reference) VAR-vs-capable pairs; negative rule (ii) evaluated at 24 h.

## Deviations from the plan
1. Subgraph pool and windows (D1 in `gates.md`): the plan's rule would leave no d=10 East Germany subgraphs. A contiguous common window of >= 365 days per subgraph is used instead. Decided before any label, gate or method result existed.
2. Aggregation requires >= 50% of the 15-min values in a step (D2); the plan did not specify.
3. DYNOTEARS lambda grid extended after validation (above). Chosen on validation seeds only.
4. G1 passes overall (0.797) but not in East Germany alone (0.624, 37% of edges have peak lag 0 at 15 min). The plan's gate is pooled, so it passes; the regional failure is reported here and in `gates.md`.
5. The G3 R2-sortnregress control reaches 0.73 to 0.74 on standardised contemporaneous edges, below the pre-registered 0.8 flag but far above 0.5.

## Results (full tables in `results.md`; test seeds 5005..5029, 100 subgraphs, standardised, p=1)
Gates: G1 0.797 (pass; Bavaria 0.970, East Germany 0.624), G2 pass at 12 h and 24 h (24 h: 79.5% contemporaneous; 6 h fails the 100-edge condition in Bavaria, 84), G3 no flag (R2-sortnregress 0.72 to 0.74 on standardised contemporaneous edges, var-sortnregress 0.50 by construction; raw scale 0.92 to 0.94).

| 24 h | directed AUROC | orientation acc (contemporaneous, n=533 edges in 99 subgraphs) |
|---|---|---|
| VAR (lag-only) | 0.531 | 0.5 by definition |
| |corr| | 0.684 | 0.5 by definition |
| PCMCI+ | 0.689 | 0.532 [0.498, 0.565] |
| VAR-LiNGAM | 0.614 | 0.598 [0.546, 0.646] |
| DYNOTEARS (own) | 0.648 | 0.462 [0.423, 0.500] |
| NOTEARS-SVAR (reference) | 0.541 | 0.417 [0.375, 0.457] |
| R2-sortnregress (control) | 0.654 | 0.720 [0.690, 0.750] |

Verdicts by the pre-fixed rules:
- H1: does not hold. At 12 h VAR-LiNGAM meets every condition (accuracy 0.611, CI [0.565, 0.660], AUROC +0.066 over VAR, Holm p 7.7e-06). At 24 h no method does: VAR-LiNGAM accuracy 0.598 (threshold 0.60, CI lower bound 0.546) with AUROC +0.083; PCMCI+ and DYNOTEARS accuracies are 0.532 and 0.462.
- Negative (i) (every capable method within 2 SE of 0.5 at 24 h): not met (VAR-LiNGAM is clearly above; PCMCI+ at 0.532 is inside 2 SE, DYNOTEARS is below 0.5).
- Negative (ii) (VAR AUROC >= every capable method at 24 h): not met; VAR is the worst or near-worst, every capable method beats it at every resolution by 0.016 to 0.158 (all Holm p < 0.05 except VAR-LiNGAM at 6 h).
- Overall: ambiguous.

What the numbers say beyond the verdicts (my reading, not a pre-registered claim):
- The AUROC gain over VAR is not evidence for contemporaneous machinery. The symmetric |correlation| score, which cannot orient anything, reaches 0.684 at 24 h and beats VAR-LiNGAM and DYNOTEARS and ties PCMCI+. Lag-only VAR is simply a poor skeleton detector at coarse resolution. Directed AUROC mixes skeleton detection with orientation.
- Orientation of contemporaneous edges is weak: best 0.60 (VAR-LiNGAM), PCMCI+ 0.53 to 0.57, DYNOTEARS and the NOTEARS reference at or below chance on standardised data. The label-rule sensitivity (Delta/4, Delta) does not change this picture (VAR-LiNGAM 0.575 to 0.620; PCMCI+ 0.51 to 0.61).
- A control that uses no causal mechanism, R2-sortnregress, orients contemporaneous edges at 0.72 to 0.74, better than every method; on raw scale var-sortnregress reaches 0.92 to 0.94 and the NOTEARS reference 0.87. Orientation by "downstream station is better explained by the others" and by magnitude is available in this data. That did not trip the pre-registered 0.8 flag, but it means orientation accuracy on this benchmark cannot be credited to modelling the contemporaneous mechanism.
- Hidden-driver diagnostic (gates.md): at 24 h the VAR(1) residual correlation among sibling stations is 0.51, about the same as among unrelated non-adjacent stations in the same subgraph (0.51) and below adjacent pairs (0.65). Shared regional forcing is large and the contemporaneous dependence is not specific to the river edges.
- p=2 does not change conclusions. The raw-scale DYNOTEARS/NOTEARS solves use the same lambda as the standardised ones (not re-tuned for scale).

## Novelty caveat: does it hold? (CausalRivers paper, arXiv 2503.17452v1, full text read)
Read: abstract, Sections 3.3 (baselines), 3.x challenges (causal lag, sampling rate), 4 (experiments 1 to 3), appendix A.2 (hyperparameters), Tables 3, 7, 8 and Figure 3. Findings:
- The paper evaluates resolutions 15 min, 1 h, 6 h, 12 h, 24 h, but only as a hyperparameter: the reported number per method is the best hyperparameter combination (resolution, normalisation, max lag 3 or 5) averaged over samples. No per-resolution results are shown in the main tables (the one disclosed pick is a GRU Causal Pretraining network at 12 h without normalisation).
- It mentions "misalignment of causal lag and sampling rate" and cites Gong et al. on subsampling as a challenge, and says the data allow exploring aggregation rates. It never mentions instantaneous or contemporaneous effects, never labels edges by travel time versus resolution, and never scores contemporaneous edges separately.
- It uses maximum lags 3 and 5 for VAR, PCMCI, DYNOTEARS and VAR-LiNGAM, ignores autoregressive links, and its naive baselines include a magnitude-based orientation (RP, 0.62 to 0.80 AUROC on its graph sets) and cross-correlation orientation. So scale-based orientation leakage was already visible there; our R2/scale-leakage result is consistent with it.
So the plan's caveat holds in this form: the paper already covers coarse resolutions but only as a tuning knob, and contemporaneous-edge scoring with travel-time labels is not in it. The novelty is narrow (the labelling and scoring protocol), not the use of aggregated CausalRivers. I did not search other literature for the same protocol in this stage.

## Limitations
One data cut (2019 to 2023 windows of 377 to 1826 days per subgraph, selection toward well-measured stations, see D1); linear methods only; tau_e is a single static estimate while the true lag varies with discharge; DYNOTEARS is my implementation and its poor orientation on standardised data may partly reflect the Gaussian objective rather than the implementation (checked only on one synthetic example); the 100 test subgraphs contain overlapping river sections, so the bootstrap over subgraphs may be somewhat optimistic.
