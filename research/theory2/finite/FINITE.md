# Theory 2, part 2 (Phase F): finite-sample relatives under autocorrelation, saturation by a hidden driver, conditioning on a proxy

Plan: `../PLAN2.md` (commit 6cbecc3). Part 1: `../THEORY.md` (Thm 1.9, Cor 1.10, Prop 2.8, Cor 2.9 are used here). Code: this folder (`common.py`, `f1_sim.py`, `f2_sim.py`, `f3_sim.py`, `f_informative.py`, each with a `*_results.txt`). This file was written and pushed before any CausalRivers data for seeds 8000-8029 were sampled or loaded. Section 7 fixes how P4-P6 are tested.

Labels: **proved** means a complete proof is given here. **Sketched** means the argument depends on a standard asymptotic result (a CLT for quadratic forms of stationary Gaussian processes) that is not proved here, or it contains a heuristic step that is named. **Conjectured** means not proved. "Checked" means compared with a fixed-seed simulation, which never replaces a proof.

## 0. Setting and notation

- X(t) = (X_1(t), ..., X_d(t)), t ∈ Z, is a zero-mean, jointly Gaussian, strictly stationary process. Its autocovariances γ_ab(h) = E X_a(t+h) X_b(t) satisfy Σ_h |γ_ab(h)| < ∞. Without loss of generality γ_aa(0) = 1, since every quantity below is a correlation and is scale invariant. ρ_ab = γ_ab(0).
- The main example is the stable Gaussian VAR on a summary graph with self-loops (THEORY.md, assumptions (A)). By Prop 2.8 of THEORY.md, ρ_ab = 0 whenever a and b are not summary relatives, and the converse holds for almost every parameter.
- We observe X(1), ..., X(n). The sample covariance is centred and normalised by 1/n: Ĉ_ab = (1/n) Σ_t (X_a(t) − X̄_a)(X_b(t) − X̄_b). The sample correlation is r̂_ab = Ĉ_ab / √(Ĉ_aa Ĉ_bb).
- **Estimator.** For thresholds τ_ab ∈ (0,1), declare a and b relatives iff |r̂_ab| > τ_ab. R̂ is the set of declared pairs, R = {a ≠ b : ρ_ab ≠ 0}, and ρ̂(v) = 1 + #{w : {v,w} ∈ R̂}. The estimated rel-sortability ŝ is the edge average of 1, 1/2, 0 for ρ̂(u) <, =, > ρ̂(v), as in THEORY.md Section 6. Fisher z with Bonferroni over m = d(d−1)/2 pairs and sample size n' is the threshold τ = tanh(z_c / √(n' − 3 − k)), with z_c = Φ^{-1}(1 − α/(2m)) and k conditioning variables. "Nominal" means n' = n. "Effective" means n' = n_eff,ab (defined in 1.2).
- Constants (process-level, for standardised X):
  - B_ab = Σ_{h∈Z} |γ_aa(h)γ_bb(h) + γ_ab(h)γ_ab(−h)| for a ≠ b. For an independent pair this is the Bartlett sum Σ_h ρ_a(h)ρ_b(h), where ρ_a is the autocorrelation function.
  - D = max_a 2 Σ_h γ_aa(h)².
  - κ_1 = max_a Σ_h |γ_aa(h)|.
  - κ_2 = max_{a≠b} (Σ_h |γ_aa(h)| + Σ_h |γ_ab(h)|).
  For i.i.d. sampling (γ(h) = 0 for h ≠ 0): B_ab = 1 + ρ_ab², D = 2, κ_1 = 1, κ_2 = 1 + max|ρ|. For an AR(1)-like series with coefficient λ: Σ_h ρ(h)² = (1+λ²)/(1−λ²) and Σ_h |ρ(h)| = (1+λ)/(1−λ).

## 1. F1: finite-sample recovery of the relatives sets

**Lemma 1.1 (Gaussian chaos tail; proved).** Let g ~ N(0, I_N), A symmetric N×N, S = gᵀAg − tr A. For every x > 0,
P(S ≥ 2‖A‖_F √x + 2‖A‖_op x) ≤ e^{−x}, and the same bound holds for −S.
*Proof.* Diagonalise A = Σ λ_i u_i u_iᵀ. By rotation invariance S has the law of Σ λ_i (g_i² − 1). For 2sλ < 1, E e^{sλ(g²−1)} = (1 − 2sλ)^{-1/2} e^{−sλ} (chi-square MGF). Write φ(u) = −½ log(1 − 2u) − u. For |u| < 1/2, φ(u) = Σ_{k≥2} (2u)^k/(2k) ≤ Σ_{k≥2} |2u|^k/4 = u²/(1 − 2|u|), because 1/(2k) ≤ 1/4 for k ≥ 2. With c = 2‖A‖_op, v = 2‖A‖_F² and 0 ≤ s < 1/c, log E e^{sS} = Σ_i φ(sλ_i) ≤ s² Σλ_i² /(1 − 2s max|λ_i|) = v s² / (2(1 − cs)). By Markov, P(S ≥ t) ≤ exp(−(st − vs²/(2(1−cs)))). Take t = √(2vx) + cx, q = √(2x/v) and s = q/(1 + cq), so that cs < 1 and 1 − cs = 1/(1+cq). Then vs²/(2(1−cs)) = x/(1+cq). Since √(2vx) = vq and cx = cvq²/2, st = 2x(1 + cq/2)/(1 + cq). Hence st − vs²/(2(1−cs)) = x(2 + cq − 1)/(1 + cq) = x. Note √(2vx) = 2‖A‖_F√x and cx = 2‖A‖_op x. For −S, replace A by −A. ∎
Source check: this is the Gaussian case of Hsu, Kakade, Zhang, "A tail inequality for quadratic forms of subgaussian random vectors", arXiv 1110.2842, Proposition 1 (opened). There it is stated for A ⪰ 0 and the upper tail, and derived from Laurent and Massart (2000, Ann. Statist. 28(5), Lemma 1), as quoted in that paper's Appendix A. Laurent-Massart itself was not opened (download blocked). The proof above is self-contained and does not rely on either source.

**Lemma 1.2 (sample covariances as Gaussian quadratic forms; proved).** Fix a ≠ b and set Y = (X_a(1..n), X_b(1..n)) ~ N(0, C), where C is the 2n×2n covariance. Let γ̃_ab = (1/n) Σ_t X_a(t) X_b(t), the uncentred estimate.
(i) γ̃_ab = Yᵀ M Y with M = (1/2n)[[0, I],[I, 0]]. With Y = C^{1/2} g this is gᵀ A g with A = C^{1/2} M C^{1/2}, tr A = γ_ab(0), and
n·Var(γ̃_ab) = 2n‖A‖_F² = Σ_{|h|<n} (1 − |h|/n) [γ_aa(h)γ_bb(h) + γ_ab(h)γ_ab(−h)] ≤ B_ab
(the exact finite-n Gaussian variance; its limit is Bartlett's covariance formula for lag 0). Also ‖A‖_op ≤ ‖M‖_op ‖C‖_op ≤ κ_2/(2n), since ‖C‖_op is at most the maximal absolute row sum of C.
(ii) For a = b: γ̃_aa = (1/n) X_aᵀX_a, A = C_aa^{1/2} C_aa^{1/2}/n, 2n‖A‖_F² = (2/n) Σ_{s,t} γ_aa(s−t)² ≤ D and ‖A‖_op ≤ κ_1/n.
(iii) The sample means satisfy X̄_a ~ N(0, σ̄_a²) with σ̄_a² = (1/n²) Σ_{s,t} γ_aa(s−t) ≤ κ_1/n.
*Proof.* (i) Yᵀ M Y = (1/2n)·2 Σ_t X_a(t)X_b(t). tr A = tr(MC) = (1/n) Σ_t γ_ab(0). ‖A‖_F² = tr(MCMC). Write C = [[C_aa, C_ab],[C_ba, C_bb]] with (C_ab)_{st} = γ_ab(s−t). Then 4n² tr(MCMC) = tr(C_ba C_ba) + 2 tr(C_aa C_bb) + tr(C_ab C_ab) = 2 Σ_{s,t} [γ_aa(s−t)γ_bb(s−t) + γ_ab(s−t)γ_ab(t−s)]. Counting pairs with s − t = h gives the weights n − |h|. The bound by B_ab is the triangle inequality. ‖A‖_op ≤ ‖C^{1/2}‖² ‖M‖ = ‖C‖‖M‖, and ‖M‖_op = 1/(2n). The row sums of C are at most Σ_h |γ_aa(h)| + Σ_h |γ_ab(h)| ≤ κ_2, or the same with a and b swapped. (ii) The same computation. (iii) X̄_a is a linear functional of a Gaussian vector. ∎

**Lemma 1.3 (deviation event; proved).** For x, y > 0 define
ε_ab(x,y) = √(2B_ab x/n) + κ_2 x/n + 2κ_1 y/n,  ε_d(y) = √(2D y/n) + 4κ_1 y/n.
Let E(x,y) be the event that |Ĉ_ab − ρ_ab| ≤ ε_ab(x,y) for all a ≠ b and |Ĉ_aa − 1| ≤ ε_d(y) for all a. Then
P(E(x,y)^c) ≤ d(d−1) e^{−x} + 4d e^{−y}.
*Proof.* Ĉ_ab = γ̃_ab − X̄_a X̄_b. By Lemma 1.1 and Lemma 1.2(i), each of the m = d(d−1)/2 off-diagonal γ̃_ab deviates by more than √(2B_ab x/n) + κ_2 x/n with probability at most 2e^{−x}. By 1.1 and 1.2(ii), each diagonal γ̃_aa deviates by more than √(2Dy/n) + 2κ_1 y/n with probability at most 2e^{−y}. By 1.2(iii) and the Chernoff bound P(|N(0,1)| ≥ t) ≤ 2e^{−t²/2}, |X̄_a| ≤ √(2κ_1 y/n) for all a except with probability at most 2d e^{−y}. On that event, |X̄_aX̄_b| ≤ 2κ_1 y/n and X̄_a² ≤ 2κ_1 y/n. Take a union bound. ∎

**Lemma 1.4 (correlation perturbation; proved, deterministic).** Let a', b' ∈ [1 − ε_d, 1 + ε_d] with ε_d < 1, |c' − ρ| ≤ ε and |ρ| ≤ 1. Then |c'/√(a'b') − ρ| ≤ (ε + |ρ| ε_d)/(1 − ε_d).
*Proof.* |c'/√(a'b') − ρ| ≤ |c' − ρ|/√(a'b') + |ρ| |1/√(a'b') − 1|. Here 1/√(a'b') ≤ 1/(1−ε_d), and |1/√(a'b') − 1| ≤ max(1/(1−ε_d) − 1, 1 − 1/(1+ε_d)) = ε_d/(1−ε_d). ∎

**Theorem 1.5 (F1, finite-sample recovery; proved).** Let x, y > 0 with ε_d(y) < 1, and set δ_ab = (ε_ab(x,y) + |ρ_ab| ε_d(y))/(1 − ε_d(y)). Suppose the thresholds satisfy
 δ_ab ≤ τ_ab for every pair with ρ_ab = 0, and |ρ_ab| − δ_ab > τ_ab for every pair with ρ_ab ≠ 0.
Then P(R̂ = R) ≥ 1 − d(d−1) e^{−x} − 4d e^{−y}. More generally, for any target set T of pairs, if |ρ_ab| − δ_ab > τ_ab on T and |ρ_ab| + δ_ab ≤ τ_ab off T, then P(R̂ = T) ≥ 1 − d(d−1)e^{−x} − 4de^{−y}.
*Proof.* On E(x,y), Lemma 1.4 with ε = ε_ab gives |r̂_ab − ρ_ab| ≤ δ_ab for all pairs. Hence |r̂_ab| ≤ |ρ_ab| + δ_ab ≤ τ_ab off T, so the pair is not declared, and |r̂_ab| ≥ |ρ_ab| − δ_ab > τ_ab on T, so it is declared. Apply Lemma 1.3. ∎
Explicit form. Let ρ_min = min_R |ρ_ab|, ρ_max = max |ρ_ab|, B = max B_ab, and use a common threshold τ ∈ (0, ρ_min). Take y = x + log(4/(d−1))_+, so that the failure probability is at most 2d(d−1)e^{−x}. All the lower-order terms (κ x/n, κ y/n) are O(x/n). The conditions then reduce, up to factors 1 + O(√(x/n)), to √(2Bx/n) + ρ_max√(2Dx/n) ≤ min(τ, ρ_min − τ). Hence, for n large with x/n small,
 P(R̂ ≠ R) ≤ 2d(d−1) exp(−n·min(τ, ρ_min − τ)²/(2(√B + ρ_max√D)²)·(1 + o(1))).
The exponent is linear in n/B. This is the sense in which n/B is the effective sample size of the bound. `common.f1_bound` evaluates the exact non-asymptotic bound, optimising over (x, y), with no o(1) term.

**Corollary 1.6 (rel-sortability; proved).** On {R̂ = R}, ρ̂(v) = ρ(v) for every v, so ŝ = s_pop exactly. Hence P(ŝ = s_pop) ≥ P(R̂ = R) ≥ 1 − d(d−1)e^{−x} − 4de^{−y} under the conditions of Theorem 1.5. For a VAR on a summary graph observed as a long stationary series, s_pop is the stationary value of THEORY.md Thm 1.9, computed over the non-self-loop edges (Cor 2.9 there). The same holds for snapshots, with B, D, κ from the i.i.d. formulas (i.i.d. contrast: B = 1 + ρ², D = 2, κ_1 = 1).

**Proposition 1.7 (effective sample size, asymptotic; sketched).** Let n → ∞. Then √n (r̂_ab − ρ_ab) ⇒ N(0, W_ab). By the delta method, with gradient (−ρ/2, −ρ/2, 1) in (Ĉ_aa, Ĉ_bb, Ĉ_ab),
W_ab = V(ab,ab) − ρ[V(ab,aa) + V(ab,bb)] + (ρ²/4)[V(aa,aa) + 2V(aa,bb) + V(bb,bb)], where V(ab,cd) = Σ_h [γ_ac(h)γ_bd(h) + γ_ad(h)γ_bc(h)].
The covariance limit is proved: by Isserlis, n Cov(γ̃_ab, γ̃_cd) → V(ab,cd), exactly as in Lemma 1.2. The centring is O(1/n). The Gaussian limit is the standard CLT for quadratic forms of a stationary Gaussian process with summable covariances; it is not proved or source-checked here, hence "sketched". For an independent pair, W_ab = Σ_h ρ_a(h)ρ_b(h) = B_ab (Bartlett). For i.i.d. data, W = (1 − ρ²)². Define n_eff,ab = n (1 − ρ_ab²)² / W_ab. This makes the Fisher z of r̂_ab approximately N(atanh ρ_ab, 1/n_eff,ab). For a null pair it reduces to n/B_ab. Checked: the empirical n·Var(r̂)/W_ab at n = 5000 lies in [0.93, 1.11] across 5 processes and all pairs (f1_results.txt).

**Corollary 1.8 (the nominal-n test is inconsistent under autocorrelation; sketched).** With the nominal threshold τ = tanh(z_c/√(n−3)) and a null pair (ρ_ab = 0), P(pair declared) → 2Φ̄(z_c/√B_ab) as n → ∞. This is not α/m, and it does not tend to 0. So P(R̂ = R) does not tend to 1 when some B_ab > 1. With the oracle effective threshold tanh(z_c/√(n_eff − 3)), the per-pair false-positive rate tends to α/m, and P(R̂ = R) → 1 − O(α) whenever ρ_min > 0. A fixed threshold τ < ρ_min gives P(R̂ = R) → 1. *Proof sketch.* Prop 1.7 with ρ = 0. Checked: the predicted asymptotic false-positive rates agree with the simulated ones to within ±0.02 for n ≥ 500 in all cases (e.g. 0.33 vs 0.32-0.33 at λ = 0.9; 0.60 vs 0.59-0.61 at λ = 0.97).

**Remark 1.9 (two failure modes, separated).** Under autocorrelation with a nominal test, false positives among true non-relatives occur at rate 2Φ̄(z_c/√B_ab) even with no hidden driver (Cor 1.8). A hidden driver makes ρ_ab ≠ 0 for every pair (Section 2), so false positives occur even with the effective test. Both make ŝ drift towards 1/2 (Section 2).

**Remark 1.10 (weak but nonzero relatives; checked).** In a VAR(1), relatives along a lagged edge have contemporaneous correlation of order λ·w. With little autocorrelation, ρ_min can be tiny: ρ_min = 0.0011 in case C5 (λ = 0.05), where no method recovers R at n ≤ 20,000. This matches the caveat to THEORY.md Prop 2.8.

### Simulation check of F1 (`f1_sim.py`, `f1_results.txt`; 1000 replications per cell, seeds 1000 + 100·case + n-index)

Five processes: d = 5 in-forest (0→2←1, 2→3, node 4 isolated) with λ = 0.5, 0.9 and 0.97, and edge weights 0.5, 0.3 and 0.1 (C1-C3; C3 is the break attempt with near-unit-root persistence and weak edges); d = 10 in-forest (C4); near-white-noise (C5). Each was run at n ∈ {200, ..., 20000} with nominal, oracle-effective and fixed (ρ_min/2) thresholds.
- The bound is never violated: in every cell the empirical P(R̂ = R) ≥ the F1 lower bound, and the empirical probability of the deviation event of the proof ≥ 1 − fail(x*,y*) (e.g. 0.998 vs 0.266 in C1 at n = 20000).
- Tightness. The bound is valid but loose. The smallest n at which the bound reaches 0.9 is several times the n at which the empirical probability does. With a fixed threshold ρ_min/2: C1 32,695 vs about 10,000 (about 3×), C2 62,794 vs about 5,000 (13×), C3 201,297 vs about 15,000 (13×), C4 166,809 vs more than 20,000. With the oracle Bonferroni threshold it is much looser (C1: 2.1·10^7). The reason is that at a Bonferroni threshold the bound's false-positive term d(d−1)e^{−z_c²/2} already uses most of a 0.1 budget: Chernoff loses the Mills-ratio factor z_c√(π/2) ≈ 4 against the Gaussian tail. The bound only falls below 0.1 once the lower-order terms (κ x/n) have become negligible. The exponent in n/B is the right one (Remark after Thm 1.5); the constants are not.
- The asymptotic approximation (Prop 1.7, pairs treated as independent) tracks the empirical P(R̂ = R) closely. Example: C3 oracle at n = 10,000 gives 0.684 vs 0.640, and at n = 20,000 0.989 vs 0.985.
- Cor 1.8 is confirmed. With the nominal test, P(R̂ = R) plateaus at 0.94 (C1), 0.22 (C2), 0.03 (C3) and 0.27 (C4) as n grows.

## 2. F2: saturation by a latent driver

**Model D (additive common driver).** X_v(t) = Y_v(t) + b_v C(t). Y is a stable Gaussian VAR on the river (summary) graph, C is a stationary Gaussian process with unit variance and summable autocovariances, and C is independent of Y. Write s_v² = Var Y_v + b_v² and ℓ_v = b_v²/s_v² (the driver's share of variance, "loading" on the R² scale). This is a one-factor model: the driver enters every node through the same process C. Lagged and filtered entries are treated in Section 3, Prop 3.6.

**Proposition 2.1 (population saturation; proved).** Under Model D, for river non-relatives u, v: ρ_uv = b_u b_v/(s_u s_v) = sgn(b_u b_v)√(ℓ_u ℓ_v) =: κ_uv. If every b_v ≠ 0, every pair of nodes in the weak component is a population relative, except on the Lebesgue-null set where γ^Y_uv(0) = −b_u b_v for some river-relative pair. Then ρ(v) = d for all v, and the population rel-sortability of every edge set is exactly 1/2. This is Cor 1.10(i) of THEORY.md for the augmented graph with source {C}.
*Proof.* Cov(X_u, X_v) = γ^Y_uv(0) + b_u b_v, and γ^Y_uv(0) = 0 for river non-relatives (THEORY.md Prop 2.8). For river relatives, the sum is nonzero except on the stated null set. All counts are equal, so every edge has value 1/2. ∎

**Theorem 2.2 (finite-sample saturation and its rate; proved).** Apply Theorem 1.5 with target T = all pairs. If |ρ_ab| − δ_ab > τ_ab for every pair, then P(ŝ = 1/2) ≥ P(R̂ = all pairs) ≥ 1 − d(d−1)e^{−x} − 4de^{−y}. For any threshold sequence with τ_n → 0, in particular nominal or effective Fisher z (τ_n = O(√(log m / n))), and with κ_min = min over pairs of |ρ_ab| > 0, choose x = y = c n with c small enough that δ ≤ κ_min/3 and τ_n ≤ κ_min/3 eventually. Then
P(ŝ_n ≠ 1/2) ≤ (d(d−1) + 4d) e^{−c n},  so |E ŝ_n − 1/2| ≤ ½ (d² + 3d) e^{−cn}.
For large n one can take c ≈ κ_min²/(18(√B + √D)²). The rate is exponential in n/B, the effective sample size. *Proof.* Theorem 1.5. ŝ = 1/2 on {R̂ = all pairs} because all counts equal d, and |ŝ − 1/2| ≤ 1/2 always. ∎

**Proposition 2.3 (onset sample size vs loading; sketched).** Let a non-relative pair have driver correlation κ = √(ℓ_u ℓ_v) (κ = ℓ for equal shares). By Prop 1.7, its probability of being declared is approximately Φ̄((atanh τ_n − atanh κ)/σ_n) + Φ((−atanh τ_n − atanh κ)/σ_n), with σ_n² = W/(n(1−κ²)²).
(a) Nominal test: the probability crosses 1/2 at n_1/2 = 3 + (z_c / atanh κ)². This does not depend on autocorrelation, which only sets the width of the transition (∝ √W). For equal shares, n_1/2 ≈ z_c²/ℓ² for small ℓ.
(b) Effective test: the crossing is at n_eff = 3 + (z_c/atanh κ)², that is n_1/2 ≈ (W/(1−κ²)²)·(3 + (z_c/atanh κ)²).
(c) Window of recovery (proved bracket via Thm 1.5, asymptotic location via (a)). The river relatives are recoverable by marginal tests only while τ_n stays between max_nonrel |κ| and min_rel |ρ|. With τ_n → 0, this window closes at n of order n_1/2(κ_max). Beyond it, ŝ → 1/2 by Thm 2.2.
Checked (`f2_results.txt`, seeds 2000 + 100·ℓ-index + n-index, 1000 replications). For ℓ = 0.02, 0.1, 0.3, 0.64, the empirical fraction of declared non-relative pairs matches the asymptotic prediction to within 0.02 at n ≥ 200 (to within 0.12 at n = 30, ℓ = 0.64). The predicted half-points are n_1/2 = 27,065, 1,079, 116 and 22. The empirical fraction at n = 1000 is 0.479 for ℓ = 0.1, and at n = 100 it is 0.387 for ℓ = 0.3. P(ŝ = 1/2) rises to 0.91 at n = 5000 for ℓ = 0.1 and to 0.997 at n = 200 for ℓ = 0.64. The nonasymptotic saturation bound (Thm 2.2) reaches 0.9 at n = 23,455 (ℓ = 0.1), 7,277 (ℓ = 0.3) and 4,854 (ℓ = 0.64). It is loose by factors of about 5 to 50, but valid in every cell, and its bound on P(ŝ ≠ 1/2) decays exponentially (e.g. ℓ = 0.64: 1e-1, 2e-4, 7e-10 at n = 4854, 9708, 19416).
Consequence for Phase 2 (post hoc, since those data were seen): the stage0 diagnostic and THEORY/README report non-relative correlations of about 0.64. With z_c ≈ 3.3 to 3.7, n_1/2 = 3 + (z_c/atanh 0.64)² ≈ 22 to 27 time steps. The actual n was 700 to 3,600. Phase 2 sat about two orders of magnitude inside the saturated regime, so Theorem 2.2 predicts ŝ = 1/2 with probability close to 1, as observed (accuracy 0.501 / 0.498).

## 3. F3: conditioning on a proxy of the driver

**Definition.** For a scalar series Z, u and v are *conditional relatives given Z* if the partial correlation π_uv = corr(X_u, X_v | Z) at lag 0 is nonzero. The estimator declares them if |π̂_uv| > τ'_uv, where π̂ is the sample partial correlation (equivalently, the correlation of the OLS residuals on Z with intercept) and τ' is the Fisher threshold with k = 1. For three variables, π = (ρ_uv − ρ_uZ ρ_vZ)/√((1 − ρ_uZ²)(1 − ρ_vZ²)).

**Proxy model.** Z(t) = √r C(t) + √(1−r) e(t), where e has unit variance and is independent of (Y, C) at lag 0 (e may be autocorrelated). Then corr(Z, C)² = r, the reliability.

**Proposition 3.1 (exact conditional correlation; proved).** Under Model D and the proxy model, for every u ≠ v,
π_uv = (γ^Y_uv(0) + (1−r) b_u b_v) / √((Var Y_u + (1−r)b_u²)(Var Y_v + (1−r)b_v²)).
In standardised form, with ρ^Y the driver-free correlation,
π_uv = (ρ^Y_uv √((1−ℓ_u)(1−ℓ_v)) + (1−r) sgn(b_ub_v)√(ℓ_uℓ_v)) / √((1 − rℓ_u)(1 − rℓ_v)).
*Proof.* Cov(X_u, Z) = b_u Cov(C, Z) = b_u √r and Var Z = 1. The partial covariance is Cov(X_u,X_v) − Cov(X_u,Z)Cov(X_v,Z) = γ^Y_uv + b_ub_v − r b_ub_v. The partial variance is Var Y_u + b_u² − r b_u². Divide, and use Var Y_u = s_u²(1−ℓ_u) and b_u² = s_u² ℓ_u. ∎

**Corollary 3.2 (exact recovery at r = 1; proved for the population, sketched for samples).** If r = 1 (Z = C), then π_uv = ρ^Y_uv for all pairs. The conditional relatives are exactly the river (summary) relatives, generically (THEORY.md Prop 2.8), and the population conditional rel-sortability equals the stationary value of THEORY.md Thm 1.9 (via Cor 2.9). Finite samples (sketched): the residuals X_u − β̂_u Z differ from Y_u by (β_u − β̂_u)Z with β̂_u − β_u = O_P(n^{-1/2}). Theorem 1.5 applied to (X, Z) together with Lemma 3.4 gives a nonasymptotic version (Thm 3.5). Checked: at r = 1 the empirical fraction of declared non-relative pairs stays at the nominal autocorrelation false-positive level (0.07 to 0.08, predicted 0.07) for n up to 5000. The conditional rel-sortability accuracy rises towards the river value (0.80 to 0.81 at n = 5000 vs s = 0.833; P(ŝ_Z = s) = 0.77 to 0.80, limited by the weak relative ρ_min).

**Proposition 3.3 (residual partial correlation between non-relatives; proved).** For river non-relatives under the assumptions of 3.1,
π_uv = (1 − r) sgn(b_ub_v) √(ℓ_uℓ_v) / √((1 − rℓ_u)(1 − rℓ_v)).
Hence (1 − r)√(ℓ_uℓ_v) ≤ |π_uv| ≤ (1 − r)√(ℓ_uℓ_v)/(1 − ℓ_max), and also |π_uv| ≤ |κ_uv|(1−r)/(1 − rℓ_max) < |κ_uv| for r > 0. |π_uv| decreases strictly in r on [0,1] when ℓ_u, ℓ_v < 1, and π_uv = 0 iff r = 1. For equal shares ℓ, π = (1−r)ℓ/(1 − rℓ).
*Proof.* Set ρ^Y = 0 in 3.1. Use 1 − rℓ ∈ [1 − ℓ, 1]. Monotonicity: |π_uv| = √(ℓ_uℓ_v)·√(f_u f_v) with f_w(r) = (1 − r)/(1 − rℓ_w), and f_w'(r) = (ℓ_w − 1)/(1 − rℓ_w)² < 0. ∎
Checked: the population π computed from the exact autocovariances equals the formula to 5 digits for ℓ ∈ {0.3, 0.64} and r ∈ {1, 0.99, 0.95, 0.9, 0.8} (`f3_results.txt`).

**Lemma 3.4 (Lipschitz bound for the partial correlation; proved, deterministic).** Let g(a,b,c) = (a − bc)/√((1−b²)(1−c²)). Take two points (a,b,c) and (a',b',c') with a, a' ∈ [−1,1] and |b|, |b'|, |c|, |c'| ≤ q < 1. Then
|g(a',b',c') − g(a,b,c)| ≤ L(q) max(|a'−a|, |b'−b|, |c'−c|), with L(q) = 1/(1−q²) + 4q/(1−q²)².
*Proof.* The segment between the points stays in the convex set [−1,1] × [−q,q]². ∂g/∂a = 1/√((1−b²)(1−c²)) ≤ 1/(1−q²). ∂g/∂b = (ab − c)/((1−b²)^{3/2}(1−c²)^{1/2}), and |ab − c| ≤ |b| + |c| ≤ 2q. The denominator is at least (1−q²)², because each of the four half-power factors is at most 1 and at least √(1−q²). The same holds for ∂g/∂c. Apply the mean value theorem along the segment and the triangle inequality. ∎ Checked on 19,144 random correlation matrices and perturbations: the worst ratio is 0.92 ≤ 1.

**Theorem 3.5 (critical n for spurious conditional relatives; bracket proved, location sketched).** Apply Lemma 1.3 to the (d+1)-dimensional process (X, Z) with its constants B, D, κ_1, κ_2. On E(x,y), every sample correlation is within δ_ab (Thm 1.5). By Lemma 3.4 with q' = max(|ρ_uZ| + δ_uZ, |ρ_vZ| + δ_vZ) < 1, |π̂_uv − π_uv| ≤ Δ_uv := L(q') max(δ_uv, δ_uZ, δ_vZ). Therefore, with probability at least 1 − (d+1)d e^{−x} − 4(d+1)e^{−y}:
(i) no spurious conditional relative appears, and the true ones are all found, if |π_uv| + Δ_uv ≤ τ'_n for non-relatives and |π_uv| − Δ_uv > τ'_n for relatives;
(ii) every non-relative pair is spuriously declared if |π_uv| − Δ_uv > τ'_n for all of them.
Since τ'_n ↓ 0 and Δ ∝ n^{-1/2}, (ii) eventually holds whenever r < 1, so spurious conditional relatives appear for every r < 1 once n is large. The critical n lies between the last n satisfying (i) and the first satisfying (ii). Asymptotic location (sketched, as in Prop 2.3): the nominal test crosses 1/2 at
 n_crit = 4 + (z_c / atanh|π_uv|)².
The effective test crosses at n_eff = 4 + (z_c/atanh|π_uv|)². Equivalently, for a given n the reliability must exceed
 r_crit(n, ℓ) = 1 − τ'(1 − ℓ)/(ℓ(1 − τ')), with τ' = tanh(z_c/√(n' − 4)) (equal shares; from π = τ' in 3.3),
for fewer than half of the non-relative pairs to test dependent. Example: ℓ = 0.64 and τ' = 0.082 give r_crit = 0.95.
Checked (`f3_results.txt`, seeds 3000 + 1000·ℓ-index + 100·r-index + n-index, 600 replications). The predicted n_crit = 472 (ℓ = 0.64, r = 0.9) gives an empirical declared fraction of 0.495 at n = 471. The other cases: 1,622 → 0.496 (ℓ = 0.64, r = 0.95); 154 → 0.526 (r = 0.8); 6,408 → 0.481 (ℓ = 0.3, r = 0.9); 1,734 → 0.511 (ℓ = 0.3, r = 0.8). Over all 38 cells the asymptotic prediction is within 0.03 of the empirical fraction. The triad estimator (Prop 3.8) recovers r to within 0.02 at n ≥ 1000. The nonasymptotic bracket probability in (i) and (ii) is 0 at every n ≤ 17,742 simulated: it is valid but uninformative at these sizes, because of the Lipschitz factor L(q') (q' ≈ 0.8 gives L ≈ 27) on top of the F1 looseness.

**Proposition 3.6 (one scalar proxy suffices only for a rank-one driver; proved).** Let X_v = Y_v + D_v with a driver part D independent of Y, and let Z be any variable independent of Y. For river non-relatives, π_uv ∝ Cov(D_u − P_Z D_u, D_v − P_Z D_v), where P_Z is the L² projection on Z. So π_uv = 0 for all non-relative pairs iff the residual driver components are pairwise uncorrelated. That holds when D_v(t) = b_v F(t) for a single process F with Z = F (rank one, r = 1). It fails in general when the driver reaches nodes through different filters, for example with different travel times: D_v(t) = b_v C(t − k_v). With C an AR(1) process with coefficient φ and Z = C(t) exactly,
π_uv = b_ub_v(φ^{|k_u−k_v|} − φ^{k_u+k_v}) / √((s_u² − b_u²φ^{2k_u})(s_v² − b_v²φ^{2k_v})),
which is nonzero whenever k_u, k_v > 0 and φ ∈ (0,1). Conditioning on the contemporaneous proxy then cannot remove the driver, even at reliability 1 with respect to C(t).
*Proof.* Partial covariance = Cov(X_u, X_v) − Cov(X_u,Z)Cov(X_v,Z)/Var Z, and the Y terms vanish for non-relatives because Y ⟂ (D, Z). The AR(1) case: Cov(C(t−k_u), C(t−k_v)) = φ^{|k_u−k_v|} and Cov(C(t−k), C(t)) = φ^k. ∎
Checked (break attempt 1): lags k = (0,1,2,4,8), φ = 0.9, ℓ = 0.64. The formula matches the exact population π to 5 digits (0.089, 0.164, 0.299 for the pairs with lags (1,8), (2,8), (4,8)). The declared fraction of non-relative pairs rises to 0.55 at n = 2000 and 0.63 at n = 10,000, although r = 1.

**Proposition 3.7 (proxy contaminated by river relatives; proved).** Let Z have unit variance, c_Z = Cov(C, Z) (so r = c_Z²), and q_u = Cov(Y_u, Z). Then for river non-relatives u, v the partial-covariance numerator is
b_ub_v(1 − r) − c_Z(b_u q_v + b_v q_u) − q_u q_v, with absolute value at most (1−r)|b_ub_v| + √r(|b_u q_v| + |b_v q_u|) + |q_u q_v|.
q_u = 0 unless some proxy station is a river relative of u (Prop 2.8 of THEORY.md). A proxy that contains a common river descendant of u and v (for example a station far downstream of a confluence) opens a collider path. *Proof.* Expand Cov(X_u, Z) = b_u c_Z + q_u. ∎ Checked (break attempt 2): adding 0.1 or 0.3 of the standardised confluence-downstream river component to Z (with C fully present) moves π for the two headwaters (river non-relatives) from 0 to −0.021 and +0.037.

**Proposition 3.8 (reliability is identified from three correlations; proved).** Under Model D and the proxy model (q = 0), for river non-relatives u, v with b_ub_v ≠ 0: ρ_uZ ρ_vZ / ρ_uv = r, and ℓ_u = ρ_uZ²/r.
*Proof.* ρ_uZ = b_u√r/s_u and ρ_uv = b_ub_v/(s_us_v). ∎
Estimator used in Phase E: r̂ = median over non-relative pairs with |r̂_uv| ≥ 0.1 of r̂_uZ r̂_vZ / r̂_uv, clipped to [0,1]. Under multi-filter drivers or contamination, r̂ is a one-factor summary, not the reliability with respect to any single variable.

## 4. Informativeness at CausalRivers sample sizes (pre-data, synthetic constants; `f_informative.py`)

CausalRivers subgraph windows (stage0 sampler) have n of roughly 700 to 3,650 at 24 h and 1,400 to 7,300 at 12 h. `f_informative_results.txt` evaluates the exact bounds with synthetic AR(1)-like constants of persistence λ ∈ {0.5, 0.9, 0.97, 0.99}. Here B_ab is set to twice the null Bartlett sum. That is a stand-in, not a bound.
- F1 (fixed threshold ρ_min/2, half the pairs related). The bound reaches 0.9 at n_eff = n/B_null of about 1,500 (ρ_min = 0.6), 3,100 (0.3) and 14,000 to 17,600 (0.1), essentially independent of λ. In raw n this is 2,400 to 1.75·10⁶. With λ = 0.5 and ρ_min ≥ 0.3 it would be informative at the largest CausalRivers windows. For λ ≥ 0.9 it is not.
- F2 (saturation, nominal test). The bound reaches 0.9 at n = 1,145 to 70,143 for κ = 0.64, against the asymptotic half-point of 22 to 27. With κ = 0.64 and λ = 0.5, the bound alone certifies saturation at CausalRivers sizes. For λ ≥ 0.9 it does not.
**Conclusion (pre-data).** The nonasymptotic F1-F3 bounds become informative only at effective sample sizes in the thousands (F1, F3). At CausalRivers sizes they are informative only if the river series were weakly persistent. The asymptotic statements (Prop 1.7, Cor 1.8, Prop 2.3, the location in Thm 3.5) are accurate in every simulation, and Phase E uses them for predictions. The observed effective sample sizes are reported in Phase E (README_part2.md).

## 5. Lean 4 candidates (deterministic or algebraic; for the Lean agent)

Feasible with Mathlib:
1. Lemma 1.4 (correlation perturbation inequality over ℝ).
2. The scalar inequality φ(u) = −½log(1−2u) − u ≤ u²/(1−2|u|) for |u| < 1/2 (via the log series or a derivative argument).
3. The Chernoff algebra in Lemma 1.1: with q = √(2x/v) and s = q/(1+cq), s(√(2vx) + cx) − vs²/(2(1−cs)) = x and cs < 1 (field_simp / nlinarith).
4. The deterministic core of Thm 1.5: if |r̂_ab − ρ_ab| ≤ δ_ab and the thresholds separate, the decision set equals the target.
5. Cor 1.6 and the saturation step of Thm 2.2: equal relatives sets give equal rel-sortability, and all counts equal give exactly 1/2 (finite sums over an edge Finset; this reuses the Lean definitions of s_ρ, if the Lean agent has them).
6. Prop 3.1 / 3.3 identities: π = (1−r)√(ℓ_uℓ_v)/√((1−rℓ_u)(1−rℓ_v)), π = 0 ⇔ r = 1 (for ℓ < 1), the two-sided bounds and monotonicity in r.
7. Prop 3.8 triad identity and the r_crit algebra of Thm 3.5.
8. Lemma 3.4 Lipschitz bound (needs a mean-value argument on a convex set; moderate).
9. Prop 3.6 AR(1) residual formula (algebra with powers φ^k).
10. The trace identity of Lemma 1.2 (block-matrix algebra; moderate).
Not formalised with reasonable effort (reason): Lemma 1.1 as a probability statement (needs Gaussian MGFs and rotation invariance of multivariate normals; Mathlib's support for multivariate Gaussians is limited), Lemma 1.2(iii) and the Isserlis covariance, the CLT behind Prop 1.7, Cor 1.8, Prop 2.3 and the location in Thm 3.5 (stationary-process CLT not in Mathlib).

## 6. Citations

- Hsu, Kakade, Zhang (2012), arXiv 1110.2842, Proposition 1. Opened (pdftotext). It matches Lemma 1.1 for A ⪰ 0.
- Laurent and Massart (2000), Ann. Statist. 28(5), 1302-1338, Lemma 1. Cited through HKZ, Appendix A. The original PDF was not opened (Project Euclid download blocked).
- Bartlett's formula and the CLT for sample autocorrelations of linear processes (Brockwell and Davis, *Time Series: Theory and Methods*, Ch. 11). Not opened, and the theorem number is not verified. This file therefore derives the finite-n variance itself (Lemma 1.2, proved) and labels every use of asymptotic normality as sketched.

## 7. Fixed before Phase E: how P4-P6 are tested (written before any data for seeds 8000-8029 were sampled or loaded)

- **Subgraphs.** `empirical2/sample2.py` is `empirical/sample.py` with SEEDS = 8000..8029 (same stage0 sampler, same regions East Germany and Bavaria, d = 5 and 10, same full-graph stationary prediction s̃ and confluence flag). Output: `subgraphs_8000.json`.
- **Proxy Z** (per subgraph and resolution). Excluded stations: those in S, plus every station at undirected river-graph distance ≤ 2 from a station of S (edges of the regional river graph, direction ignored). Candidates: all other station ids of the region cache with at least 90% valid values in the subgraph window at that resolution. Each candidate is standardised over its valid values in the window, and Z(t) is the mean of the available standardised candidates at t. Time steps with no candidate are dropped (count reported). Subgraphs with fewer than 3 candidates are excluded from P4-P6 and listed. Diagnostic: the fraction of candidates that are full-graph river relatives of some station in S (Prop 3.7 contamination).
- **Tests.** α = 0.01, two-sided, Bonferroni over d(d−1)/2 pairs. Conditional: the partial correlation of (X_u, X_v) given Z from the sample correlation matrix of the standardised window series. Fisher z with √(n' − 4).
  - **Primary** (the F1 estimator; by Cor 1.8 the nominal test is inconsistent under autocorrelation): n' = n_eff,uv = n / max(1, 1 + 2 Σ_{h=1}^{K} ρ̂_eu(h) ρ̂_ev(h)), where e_u and e_v are the OLS residuals on Z and K = 100 steps at 12 h and 50 at 24 h (same 50-day horizon).
  - **Secondary:** nominal n' = n (the Phase 2 estimator).
  - The unconditional baseline uses the same two versions with marginal correlations, raw-series ACFs and √(n' − 3).
- **Non-relative pairs:** anc_full(u) ∩ anc_full(v) = ∅ in the full regional river graph (as for s̃).
- **P4.** Per resolution: the pooled fraction of non-relative pairs declared dependent after conditioning, over all included subgraphs, pairs weighted equally. P4 holds if this is < 0.5 at both 12 h and 24 h (primary test), and fails at a resolution if ≥ 0.5. A cluster-bootstrap CI over subgraphs (2,000 resamples, seed 0) is reported. The unconditioned fraction for the same pairs is reported for comparison.
- **P5.** The conditional count ρ̂_Z(v) = 1 + #{significant partners}, and the edge accuracy (1 / ½ / 0) on the induced edges of confluence subgraphs, pooled over edges. Passes if accuracy > 0.5 with the cluster-bootstrap 95% CI lower bound > 0.5 at ≥ 1 resolution (primary test).
- **P6.** Spearman(s̃, per-subgraph conditional accuracy) over all included subgraphs with ≥ 1 edge. Passes if ≥ 0.3 at ≥ 1 resolution (primary test).
- **Kill rule** (plan): if P4 fails at both resolutions, the conditioning remedy is reported as unable to remove the hidden driver on these data, and P5 and P6 are reported but not interpreted as support.
- Secondary-test verdicts are reported alongside. They do not change the verdict.
- **F-theory checks (exploratory, fixed now).** Per subgraph: r̂ (Prop 3.8), ℓ̂_u = min(1, r̂_uZ²/r̂), predicted π_uv from Prop 3.3 with ℓ̂, predicted n_crit (Thm 3.5) for each non-relative pair, and the predicted fraction of non-relative pairs declared (asymptotic formula, nominal test: Φ-form with W replaced by the pair's Bartlett B̂; effective test: with n_eff). These are compared with the observed fractions. Also reported: observed n and n_eff distributions (raw and residual), the median B̂ and r_crit(n_eff, ℓ̂).

**Pre-data predictions from the F-theory** (conditional on quantities still unseen):
(a) Nominal test: by Cor 1.8, even a perfect proxy leaves a non-relative false-positive rate of about 2Φ̄(z_c/√B_res). This exceeds 0.5 once B_res > (z_c/0.674)² ≈ 24 (d = 5) to 30 (d = 10). With river persistence at 12 h and 24 h, the secondary (nominal) P4 is expected to fail unless B_res is small.
(b) Effective test: P4 should hold iff the typical non-relative pair has r̂ > r_crit(n_eff, ℓ̂), and Props 3.6 and 3.7 predict a residual floor. Different travel times (multi-filter driver) and contaminated proxies keep π away from 0 even if r̂ is high.
(c) P5 and P6 can hold only where (b) holds and the river relatives keep detectable partial correlations at n_eff (Cor 3.2 with Thm 1.5).

## 8. Erratum (Lean check, added after the Lean formalisation of Section 5)

The deterministic and algebraic statements listed in Section 5 were machine-checked in Lean 4 (`../lean/RelSort/Finite*.lean`; the map from statements to Lean names is in `../lean/README.md`). No statement was found to be false. Four statements need a nondegeneracy hypothesis that the text leaves implicit:

- **F1 (Prop 3.3).** The strict claims need b_u b_v ≠ 0, i.e. ℓ_u, ℓ_v > 0. These claims are |π_uv| < |κ_uv| for r > 0, "π_uv = 0 iff r = 1" and "|π_uv| decreases strictly in r". If b_u b_v = 0, then π_uv = κ_uv = 0 for every r. With ℓ_u, ℓ_v ∈ (0,1) all three hold (`absPi_lt_kappa`, `absPi_eq_zero_iff`, `absPi_strictAntiOn`). The equal-shares formula holds for |π|: π = sgn(b_u b_v)(1−r)ℓ/(1−rℓ) (`absPi_equal`).
- **F2 (Prop 3.8).** ℓ_u = ρ_uZ²/r needs r > 0. At r = 0, ρ_uZ = 0 and ℓ_u is not identified (`prop38_loading`). The ratio identity ρ_uZ ρ_vZ/ρ_uv = r holds for all r ∈ [0,1] when b_u b_v ≠ 0 (`prop38_ratio`).
- **F3 (Prop 3.6, AR(1) case).** "π_uv is nonzero whenever k_u, k_v > 0 and φ ∈ (0,1)" also needs b_u b_v ≠ 0 (`prop36_ar1_ne_zero`). The formula needs positive partial variances. Var Y_u, Var Y_v > 0 and 0 ≤ φ ≤ 1 suffice (`prop36_ar1`).
- **F4 (Prop 3.1).** The partial correlation is defined only when the partial variances Var Y_u + (1−r)b_u² are positive. This can fail at r = 1 with Var Y_u = 0. Var Y_u, Var Y_v > 0 suffices, and the standardised form needs it anyway because it uses ρ^Y (`prop31`, `prop31_std`).

Clarifications (no change to any statement):
- Lemma 1.4 does not need |ρ| ≤ 1.
- The explicit form after Theorem 1.5 (y = x + log(4/(d−1))₊) uses d ≥ 2.
- Lemma 1.2(i) uses that γ_aa and γ_bb are even, which autocovariances are.
- Theorem 3.5 applies Lemma 3.4 with a = r̂_uv. That needs |r̂_uv| ≤ 1, which holds for sample correlations by Cauchy-Schwarz.
