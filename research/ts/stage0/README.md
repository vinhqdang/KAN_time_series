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
