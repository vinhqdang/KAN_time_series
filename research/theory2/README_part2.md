# Theory 2, part 2: finite-sample relatives, hidden-driver saturation and proxy conditioning (Phases F and E)

Plan: `PLAN2.md` (commit 6cbecc3). Phase F theory, proofs, simulations and the Phase E operationalisation are in `finite/FINITE.md`. Commit 11cb220 pushed them before any data for seeds 8000-8029 were sampled or loaded. The Phase E code and derived outputs are in `empirical2/`. The Lean part (L) is done by another agent in `lean/` and is not covered here. No raw data are in the repository.

## Phase F: what survived (full statements and proofs in finite/FINITE.md)

Setting: a stationary Gaussian process (e.g. a stable VAR on a summary graph with self-loops). The estimator declares two nodes relatives when a Fisher z test of their sample correlation rejects at level α, Bonferroni-corrected over pairs. Constants: B_ab = Σ_h |γ_aa(h)γ_bb(h) + γ_ab(h)γ_ab(−h)|, the Bartlett-type sum, which for independent series is Σ_h ρ_a(h)ρ_b(h); D = 2 max Σ_h γ_aa(h)²; κ_1 and κ_2 are absolute autocovariance sums.

- **F1, Thm 1.5 (proved).** Let ε_ab = √(2B_ab x/n) + κ_2 x/n + 2κ_1 y/n and ε_d = √(2Dy/n) + 4κ_1 y/n. Let δ_ab = (ε_ab + |ρ_ab|ε_d)/(1 − ε_d). If δ_ab ≤ τ_ab on true non-relatives and |ρ_ab| − δ_ab > τ_ab on true relatives, then P(R̂ = R) ≥ 1 − d(d−1)e^{−x} − 4de^{−y}. The same holds for any target set. Consequently P(estimated rel-sortability = population value) is at least that bound (Cor 1.6). For a VAR, the population value is the stationary value of THEORY.md Thm 1.9. The exponent is n·min(τ, ρ_min − τ)²/(2(√B + ρ_max√D)²), so n/B is the effective sample size. The ingredients are a self-contained Gaussian-chaos tail bound (Lemma 1.1; the Gaussian case of Hsu-Kakade-Zhang 2012, Prop 1, which was checked in the source), an exact finite-n variance identity (Lemma 1.2) and a deterministic correlation-perturbation lemma (Lemma 1.4).
- **F1, Prop 1.7 / Cor 1.8 (sketched: they need a CLT that is not proved here).** The asymptotic variance of r̂ is a delta-method Bartlett formula W_ab, and n_eff = n(1−ρ²)²/W. The nominal-n Fisher test is inconsistent under autocorrelation: its per-null-pair false-positive rate tends to 2Φ̄(z_c/√B_ab), not to α/m.
- **F2, Prop 2.1 and Thm 2.2 (proved).** An additive latent driver with loadings ℓ_v (its share of variance) makes every pair a population relative, with κ_uv = √(ℓ_uℓ_v), so the population rel-sortability is 1/2. P(ŝ ≠ 1/2) ≤ (d² + 3d)e^{−cn} with c ≈ κ_min²/(18(√B + √D)²).
- **F2, Prop 2.3 (sketched).** Saturation onset: the nominal test crosses 50% detection at n_1/2 = 3 + (z_c/atanh κ)², which is about z_c²/ℓ² for equal shares and does not depend on autocorrelation. The effective test crosses at the same value of n_eff.
- **F3, Props 3.1 and 3.3, Cor 3.2 (proved).** Take a proxy Z = √r C + noise. The partial correlation is π_uv = (ρ^Y_uv√((1−ℓ_u)(1−ℓ_v)) + (1−r)√(ℓ_uℓ_v))/√((1−rℓ_u)(1−rℓ_v)). At r = 1 it equals the driver-free correlation, so the conditional relatives are exactly the river relatives and Thm 1.9 is restored. For river non-relatives, (1−r)√(ℓ_uℓ_v) ≤ |π| ≤ (1−r)√(ℓ_uℓ_v)/(1−ℓ_max), and |π| decreases strictly in r.
- **F3, Thm 3.5 (bracket proved, location sketched).** Spurious conditional relatives appear for every r < 1 once n is large. The critical n is n_crit = 4 + (z_c/atanh|π|)², equivalently a required reliability r_crit = 1 − τ'(1−ℓ)/(ℓ(1−τ')). The bracket comes from Lemma 3.4, a Lipschitz bound for the partial correlation that is also proved.
- **F3, Props 3.6-3.8 (proved).** A scalar proxy removes the driver only if the driver has rank one. With node-specific travel times (multi-filter driver), a residual remains even at r = 1, and an explicit AR(1) formula gives it. A proxy that contains river relatives of the subgraph opens collider paths (an explicit bound is given). The reliability is identified by the triad ρ_uZρ_vZ/ρ_uv = r.
- **Lean candidates.** FINITE.md Section 5 lists 10 deterministic lemmas: Lemma 1.4, the log-series inequality, the Chernoff algebra, the decision lemma, equal-sets ⇒ equal-sortability, the π identities, the triad identity, the Lipschitz bound, the AR(1) residual and the trace identity. Not formalisable with reasonable effort: the Gaussian MGF and chaos bound, Isserlis and the CLT.

**Simulation checks (fixed seeds; `finite/*_results.txt`).**
- In all 105 F1 cells (5 processes, including a λ = 0.97 near-unit-root break attempt, × 7 values of n × 3 thresholds), the bounds are never violated. The same holds in all F2 cells and in all F3 cells, including the multi-filter and contaminated-proxy break attempts.
- Bartlett/delta-method variance: the empirical n·Var(r̂)/W lies in [0.93, 1.11].
- Asymptotic predictions agree within about 0.03 everywhere:
  - Cor 1.8 false-positive rates, e.g. 0.33 predicted vs 0.32-0.33 simulated;
  - F2 detection curves and half-points (e.g. ℓ = 0.1: n_1/2 = 1079, simulated 0.479 at n = 1000);
  - F3 n_crit (e.g. 472 → 0.495; 1622 → 0.496).
- Tightness: the nonasymptotic bounds are valid but loose. They reach 0.9 at roughly 3 to 13 times the n at which the simulated probability does (fixed threshold), and at about 2,000 times that n with Bonferroni thresholds, where Chernoff loses the Mills-ratio factor. With q' ≈ 0.8 the F3 bracket is vacuous at every simulated n.
- Informativeness: the F1 bound needs n_eff ≈ 1,500 to 17,000 to reach 0.9.

## Phase E: pre-registered predictions P4-P6 on CausalRivers

Design (FINITE.md Section 7, fixed pre-data):
- Sampler: the stage0 sampler, copied unchanged with seeds 8000-8029. East Germany and Bavaria, d = 5 and 10: 120 subgraphs, 117 with a confluence and 115 with at least one river non-relative pair (1,376 non-relative pairs).
- Proxy: the leave-out regional mean. It excludes the subgraph and all stations within 2 undirected river hops, and uses stations with ≥ 90% valid values. Between 429 and 615 candidate stations were used. No subgraph was excluded and no time step was dropped.
- Primary test: Fisher z on the partial correlation given Z, with a residual Bartlett n_eff (K = 100 steps at 12 h, 50 at 24 h), α = 0.01 Bonferroni.
- Secondary test: nominal n, the Phase 2 estimator.
- CIs: cluster bootstrap over subgraphs (2,000 resamples, seed 0).

Sources: `empirical2/results2.json`, `empirical2/test2.log`.

| | 12 h | 24 h |
|---|---|---|
| n (time steps), median [range] | 2110 [754, 3652] | 1055 [377, 1826] |
| observed n_eff, raw pairs / residual pairs (median) | 139 / 231 | 113 / 211 |
| fraction of non-relative pairs dependent, unconditioned, nominal | 0.980 [0.965, 0.992] | 0.975 [0.956, 0.989] |
| same, unconditioned, n_eff | 0.866 [0.807, 0.915] | 0.866 [0.807, 0.914] |
| **P4**: same after conditioning on Z, primary (n_eff) | **0.741 [0.678, 0.802]** | **0.735 [0.672, 0.797]** |
| P4 secondary (nominal) | 0.896 [0.860, 0.931] | 0.869 [0.827, 0.909] |
| **P5**: conditional relatives-sort accuracy, confluence subgraphs (780 edges), primary | **0.539 [0.513, 0.565]** | **0.540 [0.514, 0.567]** |
| P5 secondary (nominal) | 0.517 [0.499, 0.535] | 0.523 [0.504, 0.540] |
| (Phase 2-style marginal accuracy, nominal) | 0.508 [0.501, 0.516] | 0.511 [0.501, 0.522] |
| **P6**: Spearman(predicted s̃, conditional accuracy), n = 120, primary | **−0.113 (p = 0.22)** | **−0.121 (p = 0.19)** |

Verdicts:
- **P4 fails at both resolutions** (0.741 and 0.735, both CIs entirely above 0.5). The secondary test fails too.
- **Kill rule: triggered.** The conditioning remedy, as pre-registered with the leave-out regional mean, is unable to remove the hidden driver on these data.
- **P5 passes as worded** at both resolutions (CI lower bounds 0.513 and 0.514). Under the kill rule it is reported, **not interpreted as support**. The effect is small: +0.03 over the Phase 2-style estimator.
- **P6 fails** at both resolutions. The correlation is slightly negative.

**Does the F-theory predict the outcome? Yes, quantitatively, but the check is partly circular.**
- The proxy is poor. The triad estimate of reliability (Prop 3.8) has median r̂ = 0.62 at 12 h (quartiles 0.47 to 0.73) and 0.63 at 24 h. Unconditioned non-relative correlations are κ ≈ 0.71 to 0.74, so ℓ ≈ 0.7.
- At the observed n_eff, Thm 3.5 requires r_crit ≈ 0.87 (median over non-relative pairs). Only 26% (12 h) and 27% (24 h) of non-relative pairs have r̂ > r_crit.
- The predicted median n_crit is about 52 to 57 time steps, against actual n ≥ 377.
- The plug-in prediction of the fraction of non-relative pairs declared after conditioning (Prop 3.3 with r̂ and ℓ̂, asymptotic detection probability) is 0.743 (primary) and 0.889 (nominal) at 12 h, against observed 0.741 and 0.896. At 24 h it is 0.736 and 0.863, against observed 0.735 and 0.869.
- The predicted median |π| is 0.46 and 0.48, against observed 0.50 and 0.52.
- Caveat: r̂ and ℓ̂ are estimated from the same correlations that are tested. The agreement shows that a one-factor driver with reliability about 0.6 is a consistent description. It is not an out-of-sample test.
- Within subgraphs, a higher r̂ goes with fewer spurious conditional relatives (Spearman −0.33, p < 0.001, at both resolutions). In the 21 to 23 subgraphs with r̂ > 0.8, the non-relative dependent fraction is 0.27 to 0.29 and the confluence accuracy 0.58 (post hoc split).
- Phase 2, re-read with F2 (post hoc): with κ ≈ 0.64 to 0.74, n_1/2 = 3 + (z_c/atanh κ)² ≈ 20 to 27. Phase 2 ran at n of 700 to 3,600, deep in the saturated regime, which explains its ŝ = 0.50 exactly.

**Why r̂ is low (exploratory, post hoc, `empirical2/explore2.py`).** Prop 3.6 says one scalar cannot remove a driver of rank greater than one (spatially heterogeneous rain, different travel times). The test conditioned on the first k principal components of the same leave-out stations instead of their mean, using the same primary test. The non-relative dependent fraction was:

| k | 12 h | 24 h |
|---|---|---|
| 1 | 0.745 | 0.737 |
| 3 | 0.605 | 0.593 |
| 5 | 0.576 | 0.568 |
| 10 | 0.491 | 0.476 |

At k = 10, confluence accuracy rose to 0.580 [0.552, 0.608] and 0.578 [0.550, 0.607], and the Spearman correlation with s̃ to 0.16 and 0.21, still below 0.3. This is consistent with a multi-dimensional driver, but it was chosen after seeing the results, and it does not rescue P4 or P6 as pre-registered.

**Are the bounds informative at CausalRivers sizes?**
- No for the nonasymptotic F1 and F3 bounds. The observed n_eff of 110 to 230 is far below the roughly 1,500 or more they need.
- The F2 saturation statement is decisive even in its asymptotic form (n ≫ n_1/2).
- The asymptotic F1-F3 formulas are what predicted the observed fractions.
- The proxy was contaminated: a median of 2.6% of proxy stations were river relatives of a subgraph station, with a maximum of 72% in one subgraph (Prop 3.7).

## Deviations and clarifications
1. The plan did not say which test "tests as dependent" in P4. Before the data were seen (FINITE.md §7), the primary test was set to the F1 estimator with a Bartlett n_eff (residual ACFs, K = 100 / 50), because Cor 1.8 shows the nominal test is inconsistent under autocorrelation. The nominal (Phase 2) test is secondary. Both fail P4, so the verdict does not depend on this choice. For P5, the secondary test fails at 12 h and passes at 24 h. The primary test passes at both.
2. Proxy details not in the plan were fixed in FINITE.md §7 before the data were seen: ≥ 90% valid values, per-station standardisation, and the nanmean. Candidate stations that are constant over the window give NaN after standardisation and are thereby ignored; this was not specified beforehand (numpy warning; `explore2.py` excludes them explicitly).
3. The P4 verdict uses the point estimate, as worded ("falls below 0.5"). The CIs are reported, and both lie entirely above 0.5.
4. `explore2.py` (principal-component proxies) and the r̂ > 0.8 split are post hoc and carry no verdict.
5. Of the 117 confluence subgraphs, 115 have a predicted s̃ > 1/2 (mean 0.838). The 2 exceptions are split-and-rejoin channels, as in Phase 2.

## Honest assessment
- Phase F adds a correct, self-contained finite-sample layer to Theory 2: an explicit recovery bound with an effective sample size, an exponential saturation rate, and an exact conditioning formula with a critical-n and required-reliability criterion. The tools are textbook ones (a Gaussian chaos tail, Isserlis, the delta method). The bounds are valid but loose by factors of 3 to 13 in n, and uninformative at CausalRivers sizes. The asymptotic versions are accurate.
- The most useful practical results are three:
  - (i) the nominal-n test cannot be consistent under autocorrelation (Cor 1.8);
  - (ii) saturation sets in at n ≈ z_c²/ℓ², so any hidden driver with a non-negligible share saturates long records;
  - (iii) proxy conditioning needs reliability above r_crit(n, ℓ), which is close to 1 for long series, and a scalar proxy can only work for a rank-one driver.
- Empirically the remedy failed as pre-registered (kill rule triggered). P5's small pass is not support, and P6 failed.
- For the paper, this turns the Phase 2 negative result from "it did not work" into a quantitatively explained negative result: the theory predicted the saturation level and the post-conditioning dependence rate. That makes it a stronger cautionary section on relatives-based sortability for real time series. It does not supply a positive real-data demonstration.
- I would rate the overall gain as modest: a solid appendix and a sharper discussion, not a new headline result.
