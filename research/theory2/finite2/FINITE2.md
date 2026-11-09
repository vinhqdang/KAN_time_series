# Finite-sample theory, part 3: plug-in effective sample size, the scope of F1, non-Gaussian innovations

This answers B'4 and R-N11, the open finite-sample gaps raised by Reviewer 1 (`icml_paper/review/03_rereview_reports.md`). Earlier material: `../finite/FINITE.md` (F1-F3), and the paper's Sec. "Finite-Sample Theory" with appendix `app:finite`. Code and outputs are in this folder:
- `sim_finite2.py` → `sim_finite2_results.txt`: fixed-seed checks of every new claim, parts A-E.
- `k_sensitivity_b3.py` → `k_sensitivity_b3_results.txt`: post hoc and descriptive. It shows how much the truncation lag matters on the B3 data.
- `er_c1_conv.py` → `er_c1_conv_results.txt`: convergence in p for the ER table row at c = 1.

LaTeX ready to paste (appendix proposition and proofs, main-text sentences, ER sentence): `latex_snippets.tex`.

Labels follow FINITE.md:
- **Proved**: the proof is complete here.
- **Proved given BM**: the proof is complete except for one step, the multivariate Breuer-Major CLT. The paper already uses that theorem for Prop. `nominal`, citing Arcones 1994 in the form of Nourdin-Peccati-Podolskij 2011, Thm 1.1.
- **Sketched**: the argument relies on a classical result that was not opened here.
- **Conjectured**: not proved.

"Checked" means compared with a simulation. A simulation never replaces a proof.

## 0. The estimator actually used

`realdata/test3.py` (`acf`, `neff_matrix`, `declare`), `icml_exp/test4.py` (`acf`, `unit_stats`, `tstat`; L = 0 for E1) and `b3/test_b3.py` (`unit_stats`, `declare`) all implement the same estimator. test_b3.py says it is copied unchanged from test4.py. For a unit X(1..n) with d series it computes:

- the ACF ρ̂_a(h) = Σ_{t=1}^{n−h} X̃_a(t+h) X̃_a(t) / Σ_{t=1}^{n} X̃_a(t)², for h = 1..K, where X̃ = X − X̄. This equals γ̂_a(h)/γ̂_a(0) with γ̂_a(h) = n^{-1} Σ_{t≤n−h} X̃_a(t+h) X̃_a(t).
- the plug-in Bartlett factor B̂_ab = max(1, Ŵ_ab), where Ŵ_ab = 1 + 2 Σ_{h=1}^{K} ρ̂_a(h) ρ̂_b(h).
- n̂_eff,ab = n / B̂_ab.
- the decision: declare {a, b} iff T_ab = |atanh r̂_ab| · √(max(n̂_eff,ab − 3 − k, 10⁻⁹)) > z_c, with z_c = Φ^{-1}(1 − α/(2m)), m = d(d−1)/2 and α = 0.01. Here k = 0 for marginal tests and k = 1 for the proxy-conditional tests of study 2. In the conditional case the ACFs are those of the residuals.

The truncation lag K is **fixed**:
- K = 50 for the time series of studies 4 and B3, with units of 3,000 to 10,000 rows;
- K = 20 for snapshots and for the primary units of study 3;
- K = 100 for the study-3 levels;
- K = 100 or 50 for the rivers (FINITE.md Sec. 7).

The targets of the factor are two different quantities, and they must be kept apart:
- W°_ab := Σ_{h∈ℤ} ρ_a(h) ρ_b(h) = 1 + 2 Σ_{h≥1} ρ_a(h) ρ_b(h), the *null Bartlett sum*. Ŵ estimates this.
- W_ab, the delta-method asymptotic variance of √n r̂_ab (Prop. 1.7 of FINITE.md). W_ab = W°_ab for an independent pair. For a dependent pair W_ab ≠ W°_ab in general.

So the plug-in n̂_eff is exactly right for the null pairs, which is where size matters. For relatives it is only a scale factor, which matters for power and not for size.

Assumption (G). X is a zero-mean, strictly stationary, jointly Gaussian ℝ^d-valued process. It is standardised (γ_aa(0) = 1) and has absolutely summable auto- and cross-covariances. Let κ_1 = max_a Σ_h |γ_aa(h)|. A pair (a, b) is *independent* if γ_ab(h) = 0 for all h. Under joint Gaussianity this is equivalent to independence of the two coordinate processes. In a Gaussian VAR on a summary graph (assumptions (A) of THEORY.md), summary non-relatives are independent pairs: they are measurable functions of disjoint sets of mutually independent innovation coordinates.

## 1. Consistency of the plug-in Bartlett factor and the size of the plug-in test

**Lemma 1 (uniform L² bound for sample autocovariances; proved).** Under (G), for each a and every 0 ≤ h < n,
E[(γ̂_a(h) − γ_a(h))²] ≤ c_0/n + 2 (h/n)² γ_a(h)²,  with c_0 = 8κ_1 + 108κ_1².

*Proof.* Drop the index a.

Step 1 (uncentred estimate). Let γ̃(h) = n^{-1} Σ_{t=1}^{n−h} X(t+h) X(t). Then E γ̃(h) = (1 − h/n) γ(h). By Isserlis, Cov(X(s+h)X(s), X(t+h)X(t)) = γ(s−t)² + γ(s−t+h) γ(s−t−h). Hence
Var γ̃(h) = n^{-2} Σ_{s,t=1}^{n−h} [γ(s−t)² + γ(s−t+h) γ(s−t−h)].
Fix s. Then Σ_t γ(s−t)² ≤ Σ_k γ(k)² ≤ max|γ| · Σ|γ| ≤ κ_1, because |γ(k)| ≤ γ(0) = 1. Also Σ_t |γ(s−t+h) γ(s−t−h)| ≤ Σ_k |γ(k+h)| ≤ κ_1. So Var γ̃(h) ≤ 2κ_1/n.

Step 2 (centring). With S_1 = Σ_{t≤n−h} X(t+h) and S_2 = Σ_{t≤n−h} X(t),
R := γ̂(h) − γ̃(h) = −X̄ S_1/n − X̄ S_2/n + ((n−h)/n) X̄².
The variables X̄, S_1/n and S_2/n are centred Gaussians, each with variance at most κ_1/n (for example Var(S_1/n) = n^{-2} Σ_{s,t≤n−h} γ(s−t) ≤ κ_1/n). For centred Gaussians U and V with variances at most σ², E(UV)² ≤ (EU⁴ EV⁴)^{1/2} = 3σ⁴. Hence E R² ≤ 3 · 3 · 3κ_1²/n² = 27κ_1²/n².

Step 3. γ̂ − γ = (γ̃ − Eγ̃) + R − (h/n)γ(h). Therefore E(γ̂ − γ)² ≤ 4 Var γ̃ + 4 E R² + 2 (hγ(h)/n)² ≤ 8κ_1/n + 108κ_1²/n² + 2(h/n)² γ(h)². ∎

**Lemma 2 (consistency of the plug-in factor; proved).** Assume (G) for the two coordinates a and b. No assumption is made on their cross-dependence. Let M_n → ∞ with M_n/n → 0, and let Ŵ_n = 1 + 2 Σ_{h=1}^{M_n} ρ̂_a(h) ρ̂_b(h). Then

|Ŵ_n − W°_ab| = O_P(n^{-1/2} + M_n/n) + 2 Σ_{h>M_n} |ρ_a(h) ρ_b(h)|,

so Ŵ_n → W°_ab in probability, and B̂ = max(1, Ŵ_n) → max(1, W°_ab) in probability. For a fixed K (the code), Ŵ → W°_K := 1 + 2 Σ_{h=1}^{K} ρ_a(h) ρ_b(h) in probability, at rate O_P(n^{-1/2}).

*Proof.*

Step 1 (good event). By Lemma 1 at h = 0 and Chebyshev, the event E_n = {γ̂_a(0) ≥ ½, γ̂_b(0) ≥ ½} has probability tending to 1, and |γ̂_a(0) − 1| = O_P(n^{-1/2}).

Step 2 (error in each ACF). Let Δ_a(h) = ρ̂_a(h) − ρ_a(h). Since ρ_a(h) = γ_a(h), on E_n
Δ_a(h) = (γ̂_a(h) − γ_a(h))/γ̂_a(0) + γ_a(h)(1 − γ̂_a(0))/γ̂_a(0),
so |Δ_a(h)| ≤ 2|γ̂_a(h) − γ_a(h)| + 2|γ_a(h)||γ̂_a(0) − 1|.

Step 3 (decomposition). Ŵ − W° = 2 Σ_{h≤M} (Δ_a ρ_b + ρ_a Δ_b + Δ_a Δ_b) − 2 Σ_{h>M} ρ_a ρ_b. The three sums are bounded in turn.

Step 4 (linear terms). T_1 = Σ_{h≤M} |Δ_a(h)||ρ_b(h)|. By Step 2,
T_1 ≤ 2 Σ_h |γ̂_a(h) − γ_a(h)||ρ_b(h)| + 2|γ̂_a(0) − 1| Σ_h |γ_a(h) ρ_b(h)|.
By Lemma 1 and √(u+v) ≤ √u + √v, the expectation of the first sum is at most
2 Σ_h |ρ_b(h)| (√(c_0/n) + √2 (h/n)|γ_a(h)|) ≤ 2κ_1 √(c_0/n) + 2√2 κ_1 M/n.
This uses h ≤ M and Σ_h |ρ_b(h)||γ_a(h)| ≤ κ_1. The second sum is O_P(n^{-1/2}) · κ_1. So T_1 = O_P(n^{-1/2} + M/n), and the same holds for T_2 (a and b swapped).

Step 5 (product term). T_3 = Σ_{h≤M} |Δ_a Δ_b| ≤ (Σ Δ_a²)^{1/2} (Σ Δ_b²)^{1/2} by Cauchy-Schwarz. By Step 2,
Σ_{h≤M} Δ_a² ≤ 8 Σ_{h≤M} (γ̂_a − γ_a)² + 8(γ̂_a(0) − 1)² Σ_h γ_a(h)².
By Lemma 1 the expectation of the first sum is at most M c_0/n + 2(M/n)² κ_1. So Σ Δ_a² = O_P(M/n + 1/n), and T_3 = O_P(M/n).

Step 6 (tail). |Σ_{h>M} ρ_a ρ_b| ≤ Σ_{h>M} |ρ_a(h)| → 0, because |ρ_b| ≤ 1.

The map max(1, ·) is continuous. For fixed K, the same steps apply without the tail. ∎

*Remark (why M = o(n) suffices).* Generic HAC results, which estimate a long-run variance Σ_h γ(h) by a kernel-weighted sum of sample autocovariances, are stated under bandwidth conditions such as M_n → ∞ with M_n²/n → 0 or similar (Newey-West 1987; Andrews 1991, Econometrica 59(3), 817-858). Only the bibliographic record of Andrews 1991 was verified. The text was not opened, so its exact bandwidth condition is not quoted here, and nothing here depends on it. The target W° is a sum of *products* of two absolutely summable sequences. Its estimation error therefore splits into linear terms of order n^{-1/2}, weighted by the summable ρ_b, plus one product term that Cauchy-Schwarz controls at O(M/n). That is why M = o(n) is enough. Every step above is self-contained.

**Theorem 3 (size of the plug-in test; proved given BM).** Assume (G), and let (a, b) be an independent pair. Write W = W°_ab = W_ab ≥ 0. Fix z_c and k.
(i) Let B̂ = max(1, Ŵ_n) with M_n → ∞ and M_n = o(n). If W > 0, then
 P(T_ab > z_c) → 2Φ̄(z_c √(max(1,W)/W)).
 This equals α/m when W ≥ 1 and is smaller than α/m when W < 1, so the floor makes the test conservative. If W = 0, then P(T_ab > z_c) → 0. Without the floor, the limit is exactly α/m whenever W > 0.
(ii) With a fixed truncation K (the code),
 P(T_ab > z_c) → 2Φ̄(z_c √(max(1,W°_K)/W)).
 This exceeds α/m when W°_K < W and W > 1. For two AR(1) series with coefficients λ_a and λ_b, q = λ_aλ_b, we have W = (1+q)/(1−q) and W°_K = W − 2q^{K+1}/(1−q).
(iii) For the nominal test (B̂ ≡ 1) the limit is 2Φ̄(z_c/√W). This is Prop. `nominal`.

*Proof.*
- Prop. `nominal` (appendix proof, which is where the Breuer-Major theorem enters) gives √n atanh r̂_ab ⇒ N(0, W) when W > 0.
- Write T_ab = |√n atanh r̂_ab| · √(max(1/B̂ − (3+k)/n, 10⁻⁹/n)). By Lemma 2, B̂ → b in probability, where b = max(1, W) in case (i) and b = max(1, W°_K) in case (ii). In both cases b ∈ [1, ∞), so the second factor tends to b^{-1/2} in probability.
- By Slutsky, T_ab ⇒ |N(0, W/b)|. This law has no atom at z_c, so P(T_ab > z_c) → P(|N(0,1)| > z_c √(b/W)).
- If W = 0, the exact finite-n variance identity of Lemma 1.2 (FINITE.md) gives n Var γ̃_ab = Σ_{|h|<n} (1 − |h|/n) γ_aa(h) γ_bb(h). Its Cesàro limit is W = 0. So √n r̂_ab → 0 in probability, and T_ab → 0 in probability. ∎

*Remark (W ≥ 0, and when W < 1).* f_a(λ) = (2π)^{-1} Σ_h ρ_a(h) e^{−ihλ} is a continuous, nonnegative spectral density, because the series is absolutely summable. Parseval gives W° = 2π ∫_{−π}^{π} f_a f_b dλ ≥ 0. W° < 1 happens when the two spectra concentrate in different frequency bands, for example two AR(1) series with coefficients of opposite sign (0.8 and −0.8 give W° = 0.22).

## 2. The scope of F1, and the right statement for the Fisher-z estimator

**Proposition 4 (F1 certifies recovery only for thresholds with nτ_n² → ∞; proved).** Let N be the set of pairs with ρ_ab = 0, assumed non-empty. Use deterministic thresholds with n τ_{n,ab}² → t_ab² ∈ (0, ∞) for ab ∈ N. Let F_n be the smallest failure probability that Theorem 1.5 (F1, `thm:f1`) can certify at n, that is, inf {d(d−1)e^{−x} + 4d e^{−y}} over the (x, y) that satisfy its separation conditions. Then
 liminf_n F_n ≥ d(d−1) exp(−min_{ab∈N} t_ab²/(2B_ab)).
Assume also that relatives satisfy |ρ_ab| ≥ ρ_min > 0 and that their thresholds tend to 0. Then the limit exists and equals the right-hand side.

Two cases in particular:
- The nominal Bonferroni Fisher-z threshold τ_n = tanh(z_c/√(n−3)) gives t² = z_c².
- The oracle effective threshold with n' = n/b_ab gives t² = z_c² b_ab.

In both cases the best certified failure probability is bounded away from 0.

With B_ab = 1 (i.i.d. snapshots, nominal test) the limit is d(d−1) e^{−z_c²/2}. Mills' inequality Φ̄(z) ≤ φ(z)/z gives d(d−1) e^{−z_c²/2} ≥ √(2π) z_c α. So the certified recovery probability is at most 1 − √(2π) z_c α (1 + o(1)) < 1 − α. For d = 5 and α = 0.01 the limit is 1 − 20 e^{−5.414} = 0.911, against the asymptotic truth ≥ 0.99. For effective thresholds the ratio b_ab/B_ab ≤ 1 makes the limit worse.

*Proof.*
- Lower bound. For ab ∈ N the separation condition reads δ_ab = ε_ab/(1 − ε_d) ≤ τ_{n,ab}, and ε_ab ≥ √(2B_ab x/n). Hence any admissible x satisfies x ≤ n τ_{n,ab}²/(2B_ab) for every ab ∈ N. Therefore F_n ≥ d(d−1) exp(−min_N n τ_{n,ab}²/(2B_ab)), which tends to the stated value.
- Attainment. Fix η ∈ (0,1) and take x_n = (1 − η) min_N n τ_{n,ab}²/(2B_ab) and y_n = log n. Then ε_d → 0 and κ_2 x_n/n + 2κ_1 y_n/n = o(n^{-1/2}). So for ab ∈ N, δ_ab ≤ √(1 − η) τ_{n,ab}(1 + o(1)) ≤ τ_{n,ab} eventually. For relatives, δ_ab → 0 and τ_{n,ab} → 0, so |ρ_ab| − δ_ab > τ_{n,ab} eventually. Hence F_n ≤ d(d−1) e^{−x_n} + 4d/n, and then let η ↓ 0. ∎

*Plug-in thresholds.* With n' = n/B̂, the threshold is random and depends on the same data, so Theorem 1.5 does not apply directly. Intersecting with an event {|B̂ − b| ≤ η} would need a non-asymptotic concentration bound for B̂. B̂ is a ratio of fourth-order polynomials of the Gaussian vector, not a second-order chaos, so Lemma 1.1 does not give such a bound, and none is proved here. A non-asymptotic recovery statement for the plug-in estimator is therefore **not available**. The correct statement is asymptotic:

**Theorem 5 (asymptotic recovery for the estimator used; (a) proved, (b)-(c) proved given BM).** Assume (G) for X = (X_1, ..., X_d), with d, α and z_c fixed. Let N be the set of independent pairs and R the set of pairs with ρ_ab ≠ 0, and assume N ∪ R contains every pair. For a stable Gaussian VAR on a summary graph, N is the set of summary non-relatives. N ∪ R is every pair except on the Lebesgue-null parameter set of THEORY.md Prop. 2.8, where a summary relative has ρ_ab = 0. Let B̂ be the code's factor, with truncation M_n satisfying 1 ≤ M_n = o(n), where M_n may be constant.
(a) Power. For every ab ∈ R, P(ab declared) → 1.
(b) Size. If M_n → ∞, then limsup P(some ab ∈ N is declared) ≤ Σ_{ab∈N} 2Φ̄(z_c √(max(1,W_ab)/W_ab)) ≤ |N| α/m ≤ α. For fixed K, replace max(1, W_ab) by max(1, W°_{K,ab}). For the nominal test the sum is Σ_N 2Φ̄(z_c/√W_ab), which can be close to |N|.
(c) Hence, if M_n → ∞,
 liminf_n P(R̂ = R) ≥ 1 − |N|α/m ≥ 1 − α,
and the same bound holds for P(ŝ = s_pop), by Cor. 1.6 of FINITE.md. For a VAR, s_pop is the stationary rel-sortability of Thm `thm:stat` on the non-self-loop edges.

*Proof.*
(a) By Lemma 1.2 of FINITE.md, Var γ̃_ab ≤ B_ab/n and Var γ̃_aa ≤ D/(2n). The centring terms are O_P(1/n). So r̂_ab → ρ_ab in probability, and |atanh r̂_ab| → atanh|ρ_ab| > 0 in probability. By Cauchy-Schwarz |ρ̂_a(h)| ≤ 1, so B̂_ab ≤ 1 + 2M_n deterministically, and n̂_eff ≥ n/(1 + 2M_n) → ∞. Hence T_ab → ∞ in probability. No consistency of B̂ and no CLT is needed for this part.
(b) Apply Theorem 3 to each ab ∈ N and take a union bound.
(c) {R̂ ≠ R} ⊆ {some ab ∈ N declared} ∪ {some ab ∈ R not declared}. Apply (a) and (b). ∎

**Corollary 6 (a correct non-asymptotic use of F1; proved).** F1 does certify consistent recovery, with an explicit rate, for vanishing-level tests. Take deterministic Fisher-z thresholds (nominal, or oracle effective with n' = n/b_ab) at level α_n → 0 with log(1/α_n) = o(n), so that z_{c,n} → ∞ and z_{c,n}/√n → 0. Then, by Theorem 1.5 with x_n = (1 − η) min_N n τ_{n,ab}²/(2B_ab) and y_n = log n,
 P(R̂ ≠ R) ≤ d(d−1) exp(−(1 − η) z_{c,n}² r/2) + 4d/n  for all large n,
where r = min_N (n'_ab/n)^{-1}/B_ab: r = 1/B_ab for the nominal test and r = b_ab/B_ab for the oracle effective test. This tends to 0. *Proof:* as the attainment part of Prop. 4. ∎

**Summary of the scope correction.**
- Theorem 1.5 (F1) is a correct non-asymptotic bound. For a fixed threshold τ ∈ (0, ρ_min) the failure probability decays exponentially in n/B.
- For the Bonferroni Fisher-z threshold actually used, which shrinks like z_c/√n', F1's guarantee does not tend to 0 (Prop. 4). F1 does not certify recovery for that estimator.
- What holds for the estimator used is Theorem 5: asymptotic recovery probability ≥ 1 − |N|α/m ≥ 1 − α, provided M_n → ∞. With the fixed K of the code, size is controlled only if the truncated Bartlett sum has captured the null Bartlett sum (W°_K ≥ W).

## 3. Non-Gaussian innovations: which fourth-cumulant terms enter

Setting (L). X(t) = Σ_{j∈ℤ} Ψ_j ε(t−j), with Σ_j ‖Ψ_j‖ < ∞. Here ε(t) ∈ ℝ^q is i.i.d. over t, has independent coordinates with E ε_i = 0 and E ε_i² = 1, and E ε_i⁴ < ∞. Let κ_i = E ε_i⁴ − 3 be the fourth cumulant, with κ_i ≥ −2. A stable VAR with i.i.d. innovations with independent components is a special case, with Ψ_j = A^j for j ≥ 0. As before, X is standardised. Define the contribution of innovation i to the lag-0 covariance:
 G_i(a,b) = Σ_j Ψ_j[a,i] Ψ_j[b,i], so that Σ_i G_i(a,b) = γ_ab(0).

**Proposition 7 (covariance limit with fourth cumulants; proved).** Under (L),
 n Cov(γ̃_ab, γ̃_cd) → V(ab,cd) + Σ_i κ_i G_i(a,b) G_i(c,d),
where V is the Gaussian (Isserlis) term of Prop. 1.7 of FINITE.md. By the delta method, the asymptotic variance of √n r̂_ab is
 W_ab^{ng} = W_ab + Σ_i κ_i g_i²,  g_i = G_i(a,b) − (ρ_ab/2)(G_i(a,a) + G_i(b,b)).

*Proof.*
- The moment-cumulant formula for zero-mean variables gives E[X_a(s)X_b(s)X_c(t)X_d(t)] − γ_ab(0)γ_cd(0) = γ_ac(t−s)γ_bd(t−s) + γ_ad(t−s)γ_bc(t−s) + cum(X_a(s), X_b(s), X_c(t), X_d(t)).
- By multilinearity of cumulants and independence over (i, time), only terms where all four factors use the same innovation ε_i(u) survive:
 cum = Σ_i κ_i Σ_u Ψ_{s−u}[a,i] Ψ_{s−u}[b,i] Ψ_{t−u}[c,i] Ψ_{t−u}[d,i].
- Sum (1/n) Σ_{s,t=1}^{n}. Index by k = t − s with Fejér weights (1 − |k|/n). The double sum over (k, u) is absolutely convergent, because it is bounded by (Σ_u |Ψ_u[a,i]Ψ_u[b,i]|)(Σ_v |Ψ_v[c,i]Ψ_v[d,i]|) < ∞. Dominated convergence then gives the limit Σ_i κ_i G_i(a,b) G_i(c,d).
- The Gaussian part converges to V as in Lemma 1.2.
- The extra term is a sum of rank-one matrices κ_i u_i u_iᵀ, with u_i = (G_i(a,a), G_i(b,b), G_i(a,b)). Contracting with the gradient (−ρ/2, −ρ/2, 1) gives κ_i g_i². ∎

Consequences:
1. **Independent pairs: no fourth-cumulant term (proved).** If a and b share no innovation coordinate (summary non-relatives in a VAR), then for every i either G_i(a,·) = 0 or G_i(b,·) = 0. So G_i(a,b) = 0, ρ_ab = 0 and g_i = 0. Thus W^{ng} = W = Σ_h ρ_a(h)ρ_b(h). Direct check: E[X_a(s)X_b(s)X_a(t)X_b(t)] = γ_aa(t−s)γ_bb(t−s) by independence. So the null variance, the size limit of Prop. `nominal`, and Theorems 3 and 5(b) are unchanged by non-Gaussian innovations.
   - The variance step is proved. Lemma 1 holds under (L) with a larger constant. The extra cumulant term in Var γ̃(h) is at most |κ| (Σ_j |ψ_j|)⁴/n uniformly in h, and E X̄⁴ ≤ 3(κ_1/n)² + |κ|(Σ|ψ|)⁴/n³. Hence Lemma 2 holds under (L).
   - What is not proved here is the CLT. Asymptotic normality of the sample cross-correlation of two independent linear processes is classical: Brockwell and Davis, *Time Series: Theory and Methods*, Thm 11.2.2. That textbook was **not opened**. The attribution is taken from a secondary source (arXiv 2310.08575), so its hypotheses are not verified. So in the non-Gaussian case Theorems 3 and 5(b) are **sketched**.
2. **Relatives: the variance changes (proved for the covariance limit).** W_ab^{ng} − W_ab = Σ_i κ_i g_i² can have either sign, because κ_i ∈ [−2, ∞). This changes three things: the oracle n_eff of relatives, the power, and the *width* of the F2 and F3 transitions. It does not change the limits. Population correlations, saturation (Prop. `f2`, population part), the location n_{1/2} = 3 + (z_c/atanh κ)² of the nominal-test transition (which depends only on τ_n = |κ|), r_crit, and the identities of F3 are all second-moment statements. Theorem 5(a) (power → 1) needs only r̂ → ρ in probability, which holds under (L) by the same variance bound.
   - The univariate special case (a = b = c = d, lags p and q) reduces to the familiar term (η−3)γ(p)γ(q) of Bartlett's formula for linear processes, with η = E ε⁴ (Brockwell-Davis Prop. 7.3.1, recalled from memory, not opened). Our derivation does not depend on that source.
3. **The non-asymptotic F1-F3 bounds do not carry over as stated.** They use the Gaussian chaos tail (Lemma 1.1). For sub-Gaussian innovations, the Hanson-Wright inequality (Rudelson and Vershynin, arXiv 1306.2872, Thm 1.1, opened) gives P(|ξᵀAξ − EξᵀAξ| > t) ≤ 2 exp(−c min(t²/(K⁴‖A‖_HS²), t/(K²‖A‖))) for ξ with independent centred coordinates with ‖ξ_i‖_ψ2 ≤ K. This has the same shape as Lemma 1.1, but with an unspecified absolute constant c. Applying it to a finite-order moving average, or to a truncated MA(∞) plus a remainder, would give an F1-type bound with the same n/B scaling and non-explicit constants. That is **sketched only**. For heavy-tailed innovations without a fourth moment, the √n calibration and Fisher z fail altogether. r̂ is still consistent with a finite variance (ergodic theorem), so the qualitative saturation statement (P(declared) → 1 for every pair with ρ ≠ 0) survives.

## 4. B_ab versus W_ab (main text l. 528-529 and Prop. `nominal`; proved)

For an independent pair:
 B_ab = Σ_h |ρ_a(h) ρ_b(h)|  ≥  W_ab = W°_ab = Σ_h ρ_a(h) ρ_b(h)  ≥ 0,
with equality on the left iff ρ_a(h)ρ_b(h) ≥ 0 for every h. The h = 0 term is 1 > 0, so equality of |Σ| and Σ|·| forces all terms to be nonnegative. The right inequality is the Parseval remark after Theorem 3.

The two constants play different roles:
- B_ab is a valid *upper bound* on the finite-n variance n Var γ̃_ab (Lemma 1.2, triangle inequality), so F1 is correct as stated with B_ab.
- W_ab is the asymptotic null variance. It is the quantity in Prop. `nominal` and the quantity the plug-in Ŵ estimates.

Example (checked, part E): two AR(1) series with coefficients 0.8 and −0.8 have B = 4.556 and W = 0.220. With 0.9 and −0.5, B = 2.636 and W = 0.379. With equal signs they coincide.

The main text currently says "(for an independent pair, the Bartlett sum Σ_h ρ_a(h)ρ_b(h))". That is exact only when all products are nonnegative. FINITE.md Sec. 0 has the same imprecision.

## 5. Simulation checks

All numbers are from `sim_finite2_results.txt`, seeds 20261109-20261112 (part A to part D). The ACF is computed by FFT, and it equals the code's loop to 3e-15.

**A. Lemma 2 (consistency of B̂).** Independent AR(1) pairs, 200 reps (60 at n = 64,000). The table gives the mean (sd) of the unfloored Ŵ for M_n = ⌈√n⌉ and for K = 50.

| (λ_a, λ_b) | W | W°_50 | n = 1,000 | n = 4,000 | n = 16,000 | n = 64,000 (M = 253) |
|---|---|---|---|---|---|---|
| 0.9, 0.9 | 9.53 | 9.53 | 8.99 (1.24) | 9.39 (0.60) | 9.46 (0.36) | 9.50 (0.16) |
| 0.8, −0.8 | 0.220 | 0.220 | 0.222 (0.022) | 0.222 (0.010) | 0.219 (0.005) | 0.219 (0.003) |
| 0.97, 0.97 | 32.84 | 31.33 | 25.1 / K50 26.7 | 30.9 / 30.2 | 32.6 / 31.2 | 32.7 (1.0) / K50 31.25 (0.76) |
| 0.99, 0.99 | 99.50 | 63.45 | 40.6 / 50.0 | 66.3 / 59.1 | 89.3 / 62.6 | 98.9 (4.8) / K50 63.40 (0.96) |

With M_n → ∞ the estimate converges to W. With fixed K = 50 it converges to W°_50, exactly as Lemma 2 says. The fixed-K bias matters only when the persistence is near a unit root (λ = 0.99: 63.4 vs 99.5).

**B. Theorem 3 (size).** Independent AR(1) pairs, 4,000 reps (2,000 at n = 64,000). Each entry is the empirical rejection rate, with the predicted limit in brackets.

| (λ, n) | per-pair level | nominal | plug-in, M_n | plug-in, K = 50 | oracle |
|---|---|---|---|---|---|
| 0.9, 16,000 | 0.05 | 0.530 [0.525] | 0.0498 [0.05] | 0.0498 [0.05] | 0.0495 |
| 0.97, 16,000 | 0.05 | 0.728 [0.732] | 0.0517 [0.05] | 0.0563 [0.0556] | 0.0500 |
| 0.99, 64,000 (M_n = n^0.6 = 766) | 0.05 | 0.853 [0.844] | 0.0495 [0.05] | **0.113 [0.118]** | 0.0475 |
| 0.99, 64,000 | 0.001 | 0.732 [0.742] | 0.0005 [0.001] | **0.0090 [0.0086]** | 0.0005 |
| (0.8, −0.8), 4,000 | 0.05 | 0.000 [0.000] | 0.000 [0.000] | 0.000 | 0.000 |

At n = 4,000 with λ = 0.9 the rates are 0.055 (level 0.05) and 0.0022 (level 0.001). The oracle shows the same excess (0.0517 and 0.0025), so it is a finite-n CLT effect in the far tail, not a plug-in effect. The fixed-K distortion predicted by Theorem 3(ii) is reproduced: at λ = 0.99 and K = 50 the per-pair size is 9 times nominal. The floor makes the opposite-sign pair conservative, as predicted (size 0).

**C. Proposition 4 and Theorem 5 (recovery).** d = 5 in-forest (0→2←1, 2→3, node 4 isolated), with 5 null pairs out of 10; 400 reps per cell. The table gives P(R̂ = R).

| case | α | n | nominal | plug-in M_n | plug-in K = 50 | oracle | best F1 guarantee (oracle threshold) |
|---|---|---|---|---|---|---|---|
| λ = 0.9, w = 0.3 (ρ_min = 0.25) | 0.05 | 8,000 | 0.152 | 0.975 | 0.975 | 0.975 | 0 |
| | 0.05 | 32,000 | 0.152 | 0.990 | 0.990 | 0.990 | 0 |
| | 0.2 | 32,000 | 0.105 | 0.895 | 0.895 | 0.892 | 0 |
| λ = 0.5, w = 0.5 (ρ_min = 0.072) | 0.05 | 32,000 | 0.835 | 0.988 | 0.988 | 0.988 | 0 |
| | 0.2 | 32,000 | 0.655 | 0.897 | 0.897 | 0.895 | 0 |

- Theorem 5 predicts liminf ≥ 1 − |N|α/m: 0.975 for α = 0.05 and 0.900 for α = 0.2. Observed: 0.988 to 0.990, and 0.895 to 0.897 (MC se ≈ 0.015).
- The nominal test plateaus far below. This is Cor. 1.8 again.
- The best F1 guarantee for the same oracle threshold is 0 at all simulated n. Prop. 4 says it converges to 1 − d(d−1)exp(−z_c² min_N(b/B)/2): 0.611 for α = 0.05 (it is 0.46 to 0.60 at n = 10⁶ to 10⁸) and a vacuous −0.34 for α = 0.2 (computed 0 at n = 10⁸). Both are below the true asymptotic recovery probability.

**D. Proposition 7 (non-Gaussian innovations).**

| setting | n Var(r̂) (simulated) | Gaussian W | W^{ng} |
|---|---|---|---|
| i.i.d., ρ = 0.6, Gaussian | 0.409 | 0.410 | 0.410 |
| i.i.d., ρ = 0.6, Laplace (κ = 3) | 0.625 | 0.410 | 0.631 |
| i.i.d., ρ = 0.6, centred exponential (κ = 6) | 0.854 | 0.410 | 0.852 |
| independent AR(1) 0.9, exponential innovations | 9.14 | 9.53 | 9.53 (no cumulant term) |
| independent AR(1) 0.9, t_5 innovations | 9.49 | 9.53 | 9.53 |
| VAR(1) relatives (0→1, λ = 0.7, w = 0.5), exponential | 1.769 | 1.719 | 1.818 |

The plug-in size at level 0.05 (n = 4,000, 3,000 reps) is 0.048 (exponential) and 0.054 (t_5), and the nominal size is 0.51 to 0.53 against a prediction of 0.525. The VAR-relatives row is inconclusive: the MC relative se of a variance with 3,000 reps is about 2.6%, i.e. ±0.046, and the result lies between W and W^{ng}. The i.i.d. rows confirm the formula sharply.

**E. B vs W (Sec. 4).**

| AR(1) coefficients | W | B |
|---|---|---|
| (0.8, −0.8) | 0.2195 | 4.5556 |
| (0.9, −0.5) | 0.3793 | 2.6364 |
| equal signs | equal | equal |

**F. Sensitivity of the code's fixed K on real data (post hoc, descriptive; `k_sensitivity_b3_results.txt`).** B3 primary data: ar_1_uniform_*, 30 segments of 3,000 rows, 2,730 unit-pairs.
- B̂ at K = 50 (used) has median 2.45, 90% quantile 4.15 and maximum 11.9.
- The ratio B̂(K)/B̂(50) has median 1.000 at K = 55, 1.007 at K = 122 and 1.026 at K = 300, with 90% quantile 1.116 at K = 300.
- No unfloored sum is below 1.
- The declared status of **0 of 2,730** unit-pairs changes when K is replaced by 55, 122 or 300.

So on these data the fixed truncation K = 50 is immaterial, and the plug-in is in the regime of Lemma 2. This was not checked for study 3, study 4 (E1) or the rivers.

**G. ER table, row c = 1 (`er_c1_conv_results.txt`).** Same generator, seeds and definitions as `ext/verify/x1_er.py`. The results agree with it exactly on seeds 100000-100002 at p = 20,000. The source-set comparison is chunked, so it is exact without hashing.

| p | seeds | L exact | S exact | L formula | S formula |
|---|---|---|---|---|---|
| 20,000 | 6 | 0.74857 ± 0.00076 | 0.65574 ± 0.00126 | 0.75000 | 0.65803 |
| 200,000 | 6 | 0.74972 ± 0.00031 | 0.65784 ± 0.00043 (se 0.00018) | 0.75000 | 0.65803 |
| 1,000,000 | 1 | 0.74989 | 0.65796 | 0.75000 | 0.65803 |

Deviation: 3 seeds were planned at p = 10⁶. The run took 952 s per seed, so it was stopped by hand after one seed to stay within the ~30-minute single-core budget. The gap S − S_formula shrinks from −0.0023 (p = 2·10⁴) to −0.0002 (2·10⁵, about 1 se) to −0.0001 (10⁶). The 1.8-s.d. gap at p = 20,000 is a finite-p effect at the critical point c = 1, as for L.

## 6. Citations: what was and was not opened

- Rudelson and Vershynin, "Hanson-Wright inequality and sub-gaussian concentration", arXiv 1306.2872 (v3). Opened (pdftotext). Thm 1.1 is quoted in Sec. 3. The journal version (Electron. Commun. Probab. 18 (2013), no. 82) is recalled from memory and was not verified.
- Andrews (1991), "Heteroskedasticity and autocorrelation consistent covariance matrix estimation", Econometrica 59(3), 817-858, doi 10.2307/2938229. Only the bibliographic record was verified (Econometric Society and RePEc pages). The text was not opened, and none of its conditions are quoted.
- Newey and West (1987), Econometrica 55(3), 703-708. Not opened. Context only.
- Brockwell and Davis, *Time Series: Theory and Methods* (2nd ed., 1991), Thm 11.2.2 (cross-correlation of independent linear processes) and Prop. 7.3.1 (Bartlett with fourth moment). Not opened. Thm 11.2.2 is attributed through arXiv 2310.08575. Used only to label steps as sketched.
- Arcones (1994) and Nourdin, Peccati and Podolskij (2011, Thm 1.1). As already cited in the paper. Not re-opened here.
- Isserlis, the moment-cumulant formula, Parseval, Slutsky, Cauchy-Schwarz and Chebyshev are standard and are used without citation.
