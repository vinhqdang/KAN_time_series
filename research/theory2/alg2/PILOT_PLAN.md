# DRAFT pilot plan: frontier structure learning on CausalRivers

**Status: DRAFT. Not run. Nothing below has been computed on real data.** The plan becomes binding when the author approves it and it is pushed; the push time is the time stamp. Until then every number in it may be changed.

Date of draft: 2026-11-10. Method and theorems: `RECOMMENDATION.md`. Prior evidence: `CANDIDATES.md`, Section 1, and `sanity/OUTPUT.txt`.

## 1. Question

Does the frontier LP recover river edges better than (a) the trivial physical baseline and (b) the standard time-series discovery methods, on real discharge data with ground truth?

The sanity check on synthetic data answered (b) yes and (a) no. The pilot is therefore built around (a).

## 2. Data

- CausalRivers `product.zip`, East Germany and Bavaria, as in `../DATA.md`, Section 1 (SHA-256 4f779fa9…). Read through `../datapaths.py`.
- **Raw discharge**, not standardised: the model is stated in physical units. Baselines receive raw and standardised input, and the better of the two is reported for each.
- Resolution: 6 h block means (stage0 cache `<region>_agg_6h.npy`, missing-value rules of stage0 unchanged). Lags L = 4, so kernels cover 0 to 24 h.
- Subgraphs: the stage0 sampler as copied in `../empirical/sample.py`, d = 5 and d = 10, both regions.
  - Development seeds 9000 to 9004 (20 subgraphs): used only to check that the code runs and to time it. No hyperparameter is chosen on them, because none is free (Section 3).
  - Test seeds 9005 to 9029 (100 subgraphs). Run once.
- Seeds 5000 to 8029 were used by earlier studies and are not reused.

## 3. Methods (all settings fixed here)

- **F-LP (primary).** Per target, the linear programme of `RECOMMENDATION.md`, Section 4, with the pinball loss at τ = 0.01 instead of hard constraints, K ≥ 0, Σ_ℓ K_ij[ℓ] ≤ 1, solver HiGHS through `scipy.optimize.linprog`. Score s_ij = (Σ_ℓ k̂_ij[ℓ]) X̄_i / X̄_j.
- **F-LP-hard (secondary).** The same with hard constraints (τ → 0).
- No other variant is run in the pilot. The stochastic-frontier likelihood and the coupled programme are left for later.

Baselines, with the hyperparameters of `../../ts/stage0/README.md` at 6 h:

1. |corr| at the best lag in 0..L (symmetric).
2. **MOC**: |corr| at the best lag, times 1[X̄_i < X̄_j]. This is the trivial physical baseline.
3. NNLS-VAR: nonnegative least squares of X_j on the same regressors with an intercept, score as for F-LP.
4. VAR(1), PCMCI+ (pc_alpha 0.01), VAR-LiNGAM (prune False), DYNOTEARS (λ 0.1): stage0 code, unchanged.
5. The naive physical baseline of the CausalRivers benchmark, transcribed from the benchmark repository before the run. If it cannot be transcribed exactly, it is left out and this is reported.

## 4. Metrics

- **Primary:** directed AUROC per subgraph (all ordered pairs i ≠ j; positive = edge of the induced river graph), averaged over the 100 test subgraphs.
- **Secondary:** undirected (skeleton) AUROC with score max(s_ij, s_ji); orientation accuracy on true edges; all three per region and per d.
- Uncertainty: paired bootstrap over subgraphs, 10,000 resamples, 95 % percentile interval for each difference.

## 5. Stage 0 gate (synthetic, before any real data)

Generator: `sanity/sanity_flow_lib.py` changed in one respect so that assumption A2 holds by construction: the local input has no constant base flow and is exactly zero in at least 5 % of the time steps of each node, independently of the regional driver. 20 seeds, p = 8, T = 1500, no measurement noise, F-LP-hard.

- **Gate G0:** mean directed AUROC ≥ 0.98. If not, the estimator or Theorem 1 is wrong as implemented; stop and find out why. No real data are touched before G0 passes.
- G0 is a check of the theory, not evidence for the method.

## 6. Kill criterion (fixed in advance)

On the 100 test subgraphs, with F-LP:

- **K1 (trivial baseline):** directed AUROC(F-LP) − directed AUROC(MOC) ≥ 0.05, and the lower end of the 95 % interval of the difference is above 0.
- **K2 (skeleton):** skeleton AUROC(F-LP) − skeleton AUROC(|corr|) ≥ 0.05, lower end above 0. Orientation is given by size on these data, so a gain must show in the skeleton.
- **K3 (standard methods):** directed AUROC(F-LP) − max over {VAR, PCMCI+, VAR-LiNGAM, DYNOTEARS, |corr|} ≥ 0.10.

**The direction is killed if K1 or K2 fails.** K3 alone is not enough: the sanity check suggests that K3 can pass because of the size order only.

If K1 and K2 pass and K3 fails, the result is reported as inconclusive and the direction is not pursued without a new plan.

No rescue analyses: no other τ, L, resolution or score is tried after the test run. F-LP-hard and NNLS-VAR are reported as ablations and do not enter the criterion.

## 7. Diagnostics reported whatever the outcome

- Share of time steps with a negative residual under the true parents with kernels fitted by NNLS (how far the data are from the model).
- For each target, min_t X_j(t) / X̄_j (base-flow share; Proposition 3 predicts more spurious edges where it is large).
- AUROC by local-input share E U_j / E X_j estimated from the true graph (post hoc, labelled as such).

## 8. Cost

- F-LP: at most 10 targets per subgraph, each an LP with at most 45 kernel variables and about 7,300 rows; about 0.1 s each in the sanity runs (21 variables, 1,500 rows); a few seconds each are expected at this size. Estimated under 30 minutes for 120 subgraphs.
- Baselines: PCMCI+ and DYNOTEARS dominate; about 1 to 2 hours on one core, from the stage0 timings.
- Total under 3 CPU-hours, single core, Python 3.13 with numpy, scipy, tigramite, lingam as in `../requirements.txt`.

## 9. What a pass would and would not show

- A pass shows headroom on one real benchmark against a baseline that already uses the physics. It does not show generality; a second domain is needed for a paper.
- A fail closes C5. Together with `CANDIDATES.md` it means that this scan found no method direction that meets the author's requirements.

## 10. Prior expectation

About 0.3 that K1 and K2 both pass. Reasons against: MOC matched the method on synthetic data; base flow violates A2 on every river. Reasons for: on real data correlation is a poor skeleton score (0.684 directed at 24 h, where a perfect symmetric score would reach about 0.88 on a 5-node tree and 0.94 on a 10-node tree), while the frontier uses the timing and the magnitude of flood waves.
