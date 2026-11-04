# Theory 2, part 2: finite-sample relatives estimation under a hidden common driver, conditioning remedy, Lean 4 verification

Pre-registered by commit before any new derivation, simulation or data access. Builds on research/theory2/THEORY.md (Phase 1, commit e92696e) and the failed real-data test (commit de9ea6b: saturation, about 98% of station pairs dependent).

## F. Finite-sample theory (to prove)

Setting: stable linear Gaussian SVAR on a summary graph with self-loops, observed as a long stationary series (and, as a contrast, as i.i.d. snapshots), possibly with a latent driver L entering every node (or a subset), and an observed proxy Z of L with reliability r = corr(Z, L)^2.

- F1 (recovery). For the estimator "u, v are relatives iff the Fisher-z test of marginal correlation rejects at level alpha (Bonferroni over pairs)", give a lower bound on the probability that the estimated relatives sets equal the true ones, as a function of n, the smallest nonzero |rho| among true relatives, and an effective sample size that accounts for autocorrelation (Bartlett-type variance of the sample correlation). State the resulting probability that the estimated rel-sortability equals the population value.
- F2 (saturation). With a latent driver reaching every node of a weak component, every pair in it becomes a relative; the estimated rel-sortability converges to 1/2 as n grows (consistent with Cor 1.10). Give the rate and the sample size at which saturation sets in as a function of the driver's loading.
- F3 (conditioning remedy). Define conditional relatives given Z (partial correlation given Z). If Z = L exactly, conditional relatives recover the summary relatives without the driver and the stationary value of Thm 1.9 is restored. If r < 1, bound the residual partial correlation between non-relatives in terms of (1 - r) and the loadings, and derive the critical n beyond which spurious conditional relatives appear.
- Every result is labelled proved, sketched or conjectured, and checked by simulation (fixed seeds) including counterexample search.

## L. Lean 4 verification

Formalise in Lean 4 with Mathlib the deterministic results of THEORY.md: at minimum Thm 1.3 (per-edge value in G_T for every T), the exact finite-T value of s_rho(G_T) and its limit (Thm 1.5, with Filter.Tendsto), Cor 1.6 (iff condition), and the stationary formula Thm 1.9 / Cor 1.10; then any deterministic or algebraic lemma used in F1-F3 that is feasible. Probabilistic statements that cannot be formalised with reasonable effort are listed as not formalised, with the reason. No `sorry`, `admit` or new `axiom` may remain in anything reported as verified; the build log is committed.

## Pre-registered empirical predictions (tested only after F1-F3 are written and committed)

CausalRivers, the same loaders as research/ts/stage0, new seeds 8000-8029, d = 5 and 10, East Germany and Bavaria, 12 h and 24 h. Proxy Z: the mean of all standardised stations in the same region that are NOT in the subgraph and not within 2 river hops of any subgraph station (leave-out regional mean).

- P4: after conditioning on Z, the fraction of non-relative pairs that test as dependent falls below 0.5 at both resolutions (before conditioning it was about 0.98).
- P5: conditional relatives-sort orients edges on confluence subgraphs with accuracy above 0.5, bootstrap 95% CI excluding 0.5, at least at one resolution.
- P6: Spearman between the theorem-predicted (stationary) and estimated per-subgraph rel-sortability is at least 0.3 at least at one resolution.
- Kill rule: if P4 fails at both resolutions, the conditioning remedy is reported as unable to remove the hidden driver on this data, and P5/P6 are reported but not interpreted as support.

No threshold changes after results are seen; deviations are listed.
