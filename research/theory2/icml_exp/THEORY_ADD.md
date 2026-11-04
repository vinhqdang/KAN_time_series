# Theory addendum for PLAN4 (E1, E2)

Builds on `../THEORY.md` (Prop 2.8, Thm 1.9, Cor 1.10, Cor 2.9) and `../finite/FINITE.md` (Lemma 1.2, Lemma 1.3, Thm 1.5, Prop 1.7, erratum §8). Written before any dependence statistic of a PLAN4 dataset was computed. Labels as in FINITE.md: **proved**, **sketched** (uses the stationary Gaussian quadratic-form CLT, not proved here), **assumed** (a physical modelling assumption about a sensor, not checkable from the design documents).

## A. Lagged relatives

Setting: a strictly stationary process X on the summary graph G_S. Assumption (Z): X_i(t) is a measurable function of the innovation sequences {ε_k(s) : k ∈ anc_S(i), s ≤ t}, and the innovation sequences of different nodes are independent. This is what the zero half of THEORY.md Prop 2.8 uses. It holds for the VAR model (A) of THEORY.md and for any per-node time-invariant filter of it (block means, differencing).

For L ∈ {0, 1, 2, …} ∪ {∞}, define the **lag-L relatives** of u as rel_L(u) = {u} ∪ {v ≠ u : γ_uv(h) ≠ 0 for some |h| ≤ L}, with γ_uv(h) = Cov(X_u(t+h), X_v(t)).

**Proposition A1 (same population sets; proved).**
(i) Under (Z), rel_0(u) ⊆ rel_L(u) ⊆ rel_∞(u) ⊆ rel_S(u) for every L.
(ii) Under the VAR model (A) with a given summary graph, rel_0(u) = rel_S(u) for Lebesgue-almost every parameter value. So rel_L(u) = rel_S(u) for every L, generically.
(iii) More weakly, rel_L(u) = rel_S(u) holds whenever every pair of summary relatives has a non-zero cross-covariance at some lag |h| ≤ L ("lag-L faithfulness"). Lag-0 faithfulness implies lag-L faithfulness, and the converse fails: in a VAR(1) with a white-noise source s and an edge s → j without a self-loop at s, γ_sj(0) = 0 while γ_js(1) ≠ 0. This is the THEORY.md caveat to Prop 2.8 and FINITE Rem 1.10.
(iv) Consequently, Thm 1.9, Cor 1.10 and Cor 2.9 of THEORY.md hold verbatim with "relatives" read as rel_L for any L ≥ 0, under lag-L faithfulness. The predicted s_ρ for any node set and any edge set is unchanged.

*Proof.* (i) The inclusions in L are immediate from the definition. If v ∉ rel_S(u), then anc_S(u) ∩ anc_S(v) = ∅. Under (Z), the processes X_u and X_v are then functions of independent innovation families, so they are independent and γ_uv(h) = 0 for every h. (ii) This is THEORY.md Prop 2.8, generic half, which covers Γ(0) (and Γ(h), h ≥ 0). (iii) Immediate. The example: X_s = ε_s and X_j(t) = a X_s(t−1) + b X_j(t−1) + ε_j(t) with |b| < 1. Then γ_sj(0) = Cov(ε_s(t), X_j(t)) = 0, because X_j(t) depends only on ε_s(t−1), ε_s(t−2), …. And Cov(X_j(t+1), X_s(t)) = a σ_s² ≠ 0. (iv) Thm 1.9 and Cor 1.10 are statements about the sets rel_S. Cor 2.9 identifies them with the dependence-defined sets; replace "non-zero marginal correlation" by "non-zero cross-covariance at some |h| ≤ L" and use (i) and (iii). ∎

**Estimator (used in PLAN4 E2).** Let r̂_uv(h) be the sample correlation of (X_u(t+h), X_v(t)) over the n − |h| overlapping times. Let B̂_uv = max(1, 1 + 2 Σ_{k=1}^{K} r̂_u(k) r̂_v(k)), and set n̂_eff,uv(h) = (n − |h|)/B̂_uv. Declare u and v relatives iff

max_{|h| ≤ L} |atanh r̂_uv(h)| · √(n̂_eff,uv(h) − 3) > z_c(L), with z_c(L) = Φ^{-1}(1 − α/(2m(2L+1))),

that is, Bonferroni over the m pairs and the 2L+1 lags. For L = 0 this is exactly the PLAN3 test (FINITE §7 form).

**Lemma A2 (null variance at every lag; proved for finite n, as in FINITE Lemma 1.2).** Let a and b be independent, stationary and Gaussian, with γ_aa(0) = γ_bb(0) = 1. Fix h, write N = n − |h|, and let γ̃_ab(h) = (1/N) Σ_t X_a(t+h) X_b(t). Then

N·Var γ̃_ab(h) = Σ_{|k|<N} (1 − |k|/N) γ_aa(k) γ_bb(k) ≤ B_ab := Σ_k |γ_aa(k) γ_bb(k)|.

The bound is the same for every h. *Proof.* Independence gives E[X_a(t+h)X_b(t)X_a(s+h)X_b(s)] = γ_aa(t−s)γ_bb(t−s). Count the pairs (t,s) with t − s = k. ∎

So the null Bartlett correction of FINITE Prop 1.7 applies at each lag with N in place of n. This is why n̂_eff,uv(h) uses the same B̂_uv for every h.

**Proposition A3 (F1 with lags; deterministic part proved, constants sketched).** Apply FINITE Lemma 1.3 to the d(d−1)(2L+1)/2 lagged cross-covariances (each a Gaussian quadratic form, Lemma 1.2 with the shift matrix M_h in place of M) together with the d variances. With the same x and y, the deviation event holds except with probability at most d(d−1)(2L+1)e^{−x} + 4de^{−y}. The proof of FINITE Thm 1.5 then goes through with these changes:
- the recovery condition for a relative pair becomes max_{|h|≤L} (|ρ_ab(h)| − δ_ab(h)) > τ;
- the condition for a non-relative pair becomes δ_ab(h) ≤ τ for all |h| ≤ L, since every ρ_ab(h) = 0 by A1(i).

In the explicit form after FINITE Thm 1.5:
- ρ_min is replaced by ρ*_min,L = min over relative pairs of max_{|h|≤L} |ρ_ab(h)|, which is ≥ ρ_min. This is the gain.
- The failure probability carries a factor 2L+1. Equivalently, the Bonferroni threshold rises from z_c(0) to z_c(L) ≈ √(z_c(0)² + 2 log(2L+1)). This is the cost.
- B is the maximum over pairs and lags of the lag-h Bartlett constants. For non-relative pairs these equal the lag-0 constants (Lemma A2). For relative pairs they contain the extra terms γ_ab(k+h)γ_ab(k−h), which are not computed here (sketched).

*Asymptotic consequence (sketched, FINITE Prop 1.7 at each lag).* The per-lag statistic is approximately N(atanh ρ_ab(h)·√n_eff(h), 1). A pair that is undetectable at lag 0 becomes detectable with lags roughly when

max_h |atanh ρ_ab(h)| √n_eff(h) / z_c(L) > |atanh ρ_ab(0)| √n_eff(0) / z_c(0).

For the PLAN4 settings (m = 105 pairs, α = 0.01): z_c(0) = 3.90, z_c(3) = 4.35, z_c(30) = 4.80. So a lagged correlation must exceed the lag-0 correlation by a factor of about 1.12 (L = 3) or 1.23 (L = 30) to help. It must exceed z_c(L)/√n_eff in absolute size to be detected at all.

**Prediction formula used in the out-of-sample F checks.** First-half plug-in per lag: p(h) = Φ̄(z_c(L) − μ_h) + Φ(−z_c(L) − μ_h), with μ_h = |atanh r̂¹_uv(h)| √((n₂ − |h|)/B̂¹_uv − 3). A pair is then predicted to be declared with probability between p_max = max_h p(h) (a lower bound) and p_ind = 1 − Π_h (1 − p(h)). The second value treats the lags as independent; the lag statistics are positively correlated, so it overstates the probability.

## B. Linear-zero lemma for the citris_1 experiment (E1 primary)

Design (generator `lt_crl_benchmark_v1/generators/citris_1.py`, read; the inputs in the data were regenerated exactly, `check_citris_design_results.txt`):
- The flag F(t) is i.i.d. on {0,…,5}. F(t) = k resets actuator k to a uniform draw.
- Otherwise red follows a reflected random walk on [0,255] with steps U(−50,50).
- green moves halfway towards red and adds a U(−50,50) step; blue moves a quarter of the way towards green and adds a U(−50,50) step.
- pol_1 follows a reflected random walk on [−90,90] with steps U(−10,10).
- pol_2 moves a quarter of the way towards pol_1 if blue > red, and the same distance away from pol_1 otherwise, plus a U(−5,5) step, reflected on [−90,90].
- The burn-in starts all actuators at 0, and the set values are rounded (colours with int(), angles to 0.1).

Write C for the colour paths (red, green, blue) and Q for the polariser paths (pol_1, pol_2).

**Lemma B1 (proved for the actuators).** Conditionally on (F, C), the law of Q is invariant under Q ↦ −Q. Hence E[g(Q) | F, C] = 0 for every odd integrable g, and E[pol_k(t)] = 0.
*Proof.* The pol updates are compositions of:
- a uniform draw on a symmetric interval (resets and steps);
- the maps x ↦ x + c(y − x) with c ∈ {1/4, −1/4} chosen by sign(blue − red), a function of C only;
- reflection at ±90;
- rounding to 0.1.
Each of these maps commutes with negation. The rounding uses Python's `:0.1f` formatting, round-half-even on the binary value; it is odd up to ties of measure zero under the continuous draws. The initial value 0 is symmetric, and F and C enter only through the coefficient sign and the reset times, which negation leaves unchanged. ∎

**Assumption B2 (sensors; assumed).**
- current, ir_1, vis_1, ir_2 and vis_2 are functions of the colour settings and of sensor noise that is independent of Q.
- ir_3 and vis_3 depend on Q only through an even function of θ_1 − θ_2 (Malus' law with the two polariser zero marks aligned), plus noise independent of Q.
- angle_k = a_k + b_k pol_k + e_k, with e_k independent of everything else.

**Corollary B3.** Under B1 and B2, every variable of the colour block 𝒞 = {red, green, blue, ir_1, vis_1, ir_2, vis_2, ir_3, vis_3, current} has zero cross-covariance, at every pair of times, with every variable of the polariser block 𝒬 = {pol_1, pol_2, angle_1, angle_2}.
*Proof.* Take Y ∈ 𝒞 at time t. Y is a function of (F, C, the polariser-difference cosines, independent noise), and that function is invariant under Q ↦ −Q. Take W ∈ 𝒬 at time s. Then W = a + b·pol_k(s) + e. E[W] = a by B1, so Cov(Y, W) = b·E[Y·pol_k(s)] = b·E[ E[Y·pol_k(s) | F, C, sensor noise] ]. For fixed (F, C, noise), Y·pol_k(s) is an odd function of the path Q, because Y is even in Q and pol_k(s) is odd. The sensor noise is independent of Q given (F, C). So the inner conditional expectation is zero by B1. ∎

**Consequence for the prediction.** The causal edges red → pol_2, blue → pol_2 and pol_k → ir_3, vis_3 are real. But they carry no linear (Pearson, any lag) dependence. The population relatives that any correlation-based estimator targets are therefore those of the **linear-dependence graph** G_lin, which is the design graph G_D with these six edges removed. G_lin has two weak components: the colour block, whose unique source is red, and the polariser block, whose unique source is pol_1. So:
- By Cor 1.10(i) of THEORY.md, every edge inside a block has predicted value ½ for every node set. This covers the chain red → green → blue, the colour → sensor edges and pol_1 → pol_2.
- The six cross-block edges are predicted by comparing block sizes within the node set. For the full 14-node set, ρ(red) = ρ(blue) = 10 > ρ(pol_2) = 4, so red → pol_2 and blue → pol_2 are predicted 0 (anti-sorted). pol_k → ir_3 and pol_k → vis_3 are predicted 1.
- Inside the blocks, the non-zero half (faithfulness) is assumed, as in Prop 2.8.
- The pol_1–pol_2 correlation is the weakest link: attraction and repulsion alternate with sign(blue − red). If it is near zero, pol_1, pol_2, angle_1 and angle_2 split into two pairs and every pol-block edge still has value ½.
- Risks: a non-zero zero-mark offset between the polarisers, or a non-affine angle sensor, would break B2. ir_3 and vis_3 could then become linearly related to the polariser block. That would change the prediction towards G_D, which is analysed as a secondary.

**Remark B4 (flag).** The flag is a measured common input of all five actuators. Conditioning on F in B1 makes its dependence irrelevant for linear cross-covariances between blocks. Within a block all pairs are relatives anyway. The flag is excluded from the node set.

## C. Why a tie is the discriminating prediction

Inside a weak component with a single source SCC, Thm 1.9 / Cor 1.10(i) predict equal relatives counts for every node: s_ρ = ½ on every edge, including chains (red → green → blue) and edges into sensors. The competitor predictors give a strict order on these edges whenever their score increases along the edge:
- (a) actuator/sensor indicator;
- (b) number of ancestors;
- (c) in-degree;
- (d) depth;
- the finite-G_T limit (all crossing edges 1, Thm 1.5).

So the theory and the competitors disagree exactly where the theory predicts a tie.

A tie in the *estimated* counts can also arise from lack of power (no partner detected at either endpoint). PLAN4 therefore adds three safeguards:
- a power prerequisite;
- a robustness analysis restricted to edges whose own endpoint pair is declared dependent;
- a report of the "all estimated counts equal" baseline, which shows how large the theory's advantage would be under zero power.
