# Lean 4 verification of THEORY.md (part L of PLAN2.md) and of the deterministic parts of finite/FINITE.md

Lean 4.35.0, Mathlib `v4.35.0` (commit 56a0058). Build: `lake exe cache get && LEAN_NUM_THREADS=2 lake build`.
Axiom check: `lake env lean Axioms.lean`. The full log of a clean rebuild plus the axiom check is in
`BUILD_LOG.txt`, and the `#print axioms` lines are in `AXIOMS.txt`. Every listed theorem depends only on
`propext`, `Classical.choice` and `Quot.sound`. The sources contain no `sorry`, no `admit` and no `axiom`
(the grep is at the end of the log). The only build warning is Lake's generic "consider adding `module`"
notice.

"Verified" below means the theorem was machine-checked with no `sorry` and with the axioms above.

## Modelling choices

- **Graphs.** A summary graph is a relation `E : V → V → Prop` on a `Fintype V`. The self-loop
  assumption is the hypothesis `hl : ∀ v, E v v`. The main theorems need `Nonempty V` (the paper has
  p ≥ 2), so `|E_S| ≥ 1`. Edge sets are `edges R = {(x,y) : R x y}`.
- **Reisach et al.'s definitions** (`Basic.lean`), stated for any finite digraph:
  - `anc` and `desc` are `Relation.ReflTransGen`. A directed path from x to y exists iff a directed
    walk does, and `P_xx = {x}` is the reflexive case.
  - `IsRel R y z := ∃ x, x ⇝ y ∧ x ⇝ z` is literally `z ∈ desc(anc(y))`.
  - `rho` is `|rel|`, `score` is 1, 1/2 or 0, and `sRho` is the average of the edge scores.
  - The only standard fact used without proof is "path exists ⇔ walk exists". It is how
    `ReflTransGen` reads `P_xy ≠ ∅`.
- **Unrolled DAG.** `G_T` with T = n + 1 slices is `unrolled E n` on `V × Fin (n+1)`. Lean slice `t` is
  the paper's time `t+1`, and the edges are `(i,t) → (j,t+1)` for every `E i j`. So T ≥ 2 is n ≥ 1, and
  the paper's edge `(i(t), j(t+1))` is Lean's `(i, s.castSucc) → (j, s.succ)` with `s = t − 1 : Fin n`.
  `tendsto_sRho_GT` states the limit with the slice count T itself.
- **Distance.** `d(u,v)` is `dist E u v : ℕ∞`, the shortest walk length (⊤ if v is unreachable).
  `dist_eq_dist_noLoop` proves that it equals the same quantity with self-loops deleted, which is
  THEORY.md's "ignoring self-loops". The fact that "shortest walk = shortest path" is not stated
  separately: a shortest walk never repeats a vertex.
- **Strongly connected components** are mutual reachability (`SameSCC`). Weak connectivity is
  `ReflTransGen` of the symmetrised relation (`WeakConn`).
- **Source SCCs** are represented by their member nodes. `IsSource E x` means that every ancestor of x is
  reachable from x, i.e. the SCC of x has no incoming edge from another SCC. `Src(v)` is represented by
  the predicate `x ↦ IsSource x ∧ x ⇝ v`. This is a restatement and is flagged as such. Two sets of SCCs
  are equal iff the unions of their nodes are equal, because SCCs are disjoint and nonempty, and a
  source SCC reaches v iff each of its nodes does. This equivalence is not proved in Lean; it is
  elementary.
- **Stationary windows.**
  - Burn-in: `G_{B+T}` is `unrolled E (B+m)`. The window holds the last T = m + 1 slices (`burnEmb`).
    Relatives are computed in the full DAG and restricted to the window (`relBurn`), and the edges are
    the edges of `G_{B+T}` inside the window (`edgeBurn`, proved equal to a copy of `E_T`).
  - Infinite past: `zGraph` lives on `V × ℤ`, with window `V × {0..m}`.

## Results

| THEORY.md | Lean name(s) | file | status |
|---|---|---|---|
| Lemma 1.1 (ancestors in G_T) | `unrolled_reach_iff`, `unrolled_reach_iff_dist` | Unrolled | verified |
| Lemma 1.2 (roots; desc of u(1)) | `isRoot_unrolled_iff`, `roots_unrolled_iff`, `desc_root_unrolled_iff`; the form used for relatives: `isRel_unrolled_iff` | Extras, Unrolled | verified |
| Reisach Thm 1 (edge ⇒ rel ⊆) | `relSet_subset_of_edge`, `relSet_subset_of_reach` (any digraph) | Basic | verified |
| **Thm 1.3** (per-edge value, every T) | `edge_score` (walk form `StrictCond`), `edge_score_dist` (the paper's `∃u, d(u,j) ≤ t ≤ d(u,i)`), `strictCond_iff_dist` | Unrolled | verified |
| Prop 1.4 (relative counts) | `rho_unrolled_eq` (ρ = Σ_w max(T − m_t, 0)), `mT_le_iff` (m_t is the stated minimum), `mT_large_t`, `rho_unrolled_large_t` (t ≥ p form: ρ + Σ_{rel_S} d(anc_S(v), w) = T·\|rel_S(v)\|) | Extras | verified |
| **Thm 1.5** exact finite-T value | `sRho_unrolled_exact` (and raw form `sRho_unrolled_eq`) | Limit | verified |
| **Thm 1.5** bounds | `sRho_unrolled_bounds`: 0 ≤ s − L ≤ min(p−1,T−1)\|E_∘\|/(2(T−1)\|E_S\|) ≤ (p−1)/(2(T−1)) | Limit | verified |
| **Thm 1.5** limit | `tendsto_sRho_unrolled` (in n = T−1), `tendsto_sRho_GT` (in T); `limitValue E = 1/2 + \|E_×\|/(2\|E_S\|)` | Limit | verified |
| **Cor 1.6** iff (both directions) | `tendsto_half_iff`; equivalent conditions `crossEdges_eq_empty_tfae` (E_× = ∅; i→j ⇒ j⇝i; i⇝j ⇒ j⇝i; weak components strongly connected) | Limit | verified |
| Cor 1.6 uniform bound | `sRho_unrolled_ge_of_cross` | Limit | verified |
| Reisach Thm 4 | `reisach_theorem4` (corollary) | Limit | verified |
| Cor 1.7 (DAG + loops: L = 1/2 + m/(2(m+p))) | `limitValue_dag` | Limit | verified |
| Cor 1.7 examples (3-node collider/fork/chain: 0.7) | `collider_limit`, `fork_limit`, `chain3_limit` | Examples | verified |
| Cor 1.7 forest count m = p − k, general chain formula | — | — | not formalised (graph counting only; it follows from `limitValue_dag` once m is known) |
| Remark 1.8 | uses Prop 1.4 (verified). The qualitative "O(1) out of O(T)" reading is not a formal statement | — | partially |
| **Thm 1.9 (a)** ρ_W = T·\|rel_S(v)\| | `rhoBurn_eq` (B ≥ p−1), `rhoZ_eq` (infinite past) | Stationary | verified |
| **Thm 1.9 (b)** s = 1/2 + \|E_↑\|/(2\|E_S\|), every T ≥ 2 | `sBurn_eq`, `sZ_eq` (generic core `sOf_window`) | Stationary | verified |
| Thm 1.9 (b) E_↑ via source SCCs | `relSet_eq_iff_src` (for i ⇝ j: rel_S(i) = rel_S(j) ⇔ Src(i) = Src(j)), `mem_upEdges_iff`, `exists_source` | Stationary | verified |
| **Cor 1.10 (i)** iff | `sBurn_eq_half_iff` (unique source SCC per weak component), `sBurn_eq_half_iff_common_anc` (common ancestor form), `sZ_eq_half_iff`; equivalences `upEdges_eq_empty_tfae` | Stationary | verified |
| Cor 1.10 (ii) | `stat_half_of_strong` (Reisach's condition), `stat_half_of_outForest` (out-trees, out-forests, chains, forks; acyclicity not needed) | Stationary | verified |
| Cor 1.10 (iii) in-forests | `inForest_isRel_iff` (rel = anc ∪ desc), `inForest_up_iff` (E_↑ edge ⇔ head is a confluence), `statValue_inForest` (formula), `statValue_inForest_gt_iff` (> 1/2 ⇔ a confluence exists) | Stationary | verified |
| Cor 1.10 (iv) | `collider_stat` (0.7), `fork_stat` (1/2), `chain3_stat` (1/2) | Examples | verified |
| Remark 1.11 | numerical (enumeration) | — | not formalised (it is a report of computations, not a theorem) |
| Remark 1.12 | `edgeLoops_limit` (with loops: limit 2/3), `edgeNoLoops_sRho` (without loops: s_ρ(G_T) = 1/2 for every T ≥ 2) | Extras | verified |
| Prop 2.1 (Fejér) | `fejer_identity`, `fejer_identity'`: Σ_{u,w<D} Γ(u−w) = DΓ(0) + Σ_{h=1}^{D−1}(D−h)(Γ(h)+Γ(−h)), for any Γ : ℤ → M | T2Algebra | partially: the counting identity is verified; the step Cov(Σ_u X(sD+u)) = Σ_{u,w} Γ(u−w) (bilinearity of covariance plus stationarity) is not formalised |
| Thm 2.2 (aggregation limit and rate) | — | — | not formalised (needs the MA(∞) representation, geometric decay of Γ(h) and summability; analytic and probabilistic) |
| Prop 2.3 (long-run SEM) | `longrun_factor` (I − A(1) = (I − Λ)(I − B̃)), `longrun_inv`, `selfloop_sum_lt_one` (λ_j < 1 from no root of φ_j on [0,1], via the intermediate value theorem) | T2Algebra | partially: the algebraic factorisation and λ_j < 1 are verified; that the AR polynomial factorises under (A-DAG), and S_LR = Ψ(1)ΣΨ(1)ᵀ, are not |
| Cor 2.4, Prop 2.5, Cor 2.6, Prop 2.7, Prop 2.8, Cor 2.9 | — | — | not formalised. They involve continuity of sortability in covariance matrices, an R² formula via matrix inverses, the SVAR covariance structure, and real-analytic genericity. 2.5 is matrix algebra and feasible but was not done; 2.7–2.9 need the stochastic model |
| PLAN2 F1–F3 (finite-sample) | see the next section (FINITE.md) | Finite* | deterministic and algebraic parts verified; probabilistic parts not formalised |

## FINITE.md (Phase F): deterministic and algebraic lemmas

Files `RelSort/FiniteCorr.lean`, `RelSort/FiniteTrace.lean`, `RelSort/FinitePartial.lean`,
`RelSort/FiniteLipschitz.lean`. Everything is stated over ℝ with explicit hypotheses.

Modelling choices (Phase F):

- **Covariances as inner products.** In `FinitePartial` the zero-mean, finite-variance variables
  (Y_u, C, e, Z, ...) are vectors of an arbitrary real inner product space `E` (for example L²(Ω)),
  and the lag-0 covariance is the inner product. Independence or uncorrelatedness at lag 0 enters
  only as vanishing inner products, which are hypotheses (`ProxyHyp`, `AR1Hyp`). FINITE.md's proofs
  of Props 2.1, 3.1, 3.6, 3.7 and 3.8 use nothing else (bilinearity of the covariance).
- **Partial correlation** `pcorr x y z` is FINITE.md's three-variable formula
  `g(ρ_xy, ρ_xz, ρ_yz) = (a − bc)/√((1−b²)(1−c²))` (`pc`). `pcorr_eq_pcov` proves it equals the
  partial covariance over the partial standard deviations, and `pcorr_eq_corr_resid` that it equals
  the correlation of the OLS residuals on `z` (FINITE.md's "equivalently"), under positive variances
  and positive partial variances.
- **Shares and signs.** `ℓ_u = b_u²/(Var Y_u + b_u²)` is written out; `sgn` is Mathlib's
  `SignType.sign`. `|π_uv|` for non-relatives is the function `absPi r ℓ_u ℓ_v`; `prop33_abs` proves
  `|π_uv| = absPi ...` in the model.
- **Counts and sortability (Cor 1.6, Thm 2.2).** `cntDecl D v = 1 + #{w ≠ v : D v w}` is ρ̂(v) for a
  declared relation `D`, and sortability reuses `sOf` from `Basic.lean`.
- **Lemma 1.2 matrices.** `covBlock` is C = [[C_aa, C_ab],[C_abᵀ, C_bb]] with Toeplitz blocks
  `(s,t) ↦ γ(s − t)` on `Fin n`, and `Mswap` is M = (1/2n)[[0,I],[I,0]]. ‖A‖_F² is represented by
  tr(MCMC); the step ‖C^{1/2}MC^{1/2}‖_F² = tr(MCMC) (trace cyclicity with the matrix square root) is
  not formalised.
- **Lemma 3.4.** The proof goes coordinate by coordinate, (a,b,c) → (a,b,c') → (a,b',c') →
  (a',b',c'), with the one-dimensional mean value theorem on [−q,q]. The path stays in the same box
  and gives the same constant L(q) = 1/(1−q²) + 4q/(1−q²)² as FINITE.md's segment argument.
- **n_crit.** The threshold is τ(n) = tanh(z/√(n − k₀)) with k₀ = 3 + k. Lean proves where τ(n)
  equals (or falls below) the population |π| or κ. That "the probability crosses 1/2 there" is the
  asymptotic (sketched) part and is not formalised.

| FINITE.md | Lean name(s) | file | status |
|---|---|---|---|
| Lemma 1.1, series bound φ(u) ≤ u²/(1−2\|u\|) (§5 item 2) | `phiC`, `phi_le` | FiniteCorr | verified |
| Lemma 1.1, log-MGF bound Σφ(sλ_i) ≤ vs²/(2(1−cs)) | `chernoff_logmgf_le` | FiniteCorr | verified |
| Lemma 1.1, Chernoff algebra (§5 item 3): cs < 1 and exponent = x; √(2vx) = 2‖A‖_F√x | `chernoff_choice`, `sqrt_two_v_x` | FiniteCorr | verified |
| Lemma 1.1 as a probability statement | — | — | not formalised (Gaussian MGF, rotation invariance) |
| Lemma 1.2(i): γ̃_ab = YᵀMY, tr A = γ_ab(0) | `quad_M`, `trace_MC` | FiniteTrace | verified |
| Lemma 1.2(i) trace identity (§5 item 10): 2n tr(MCMC) = (1/n)Σ_{s,t}[...] = Σ_{\|h\|<n}(1−\|h\|/n)[...] ≤ B_ab | `lemma12_trace`, `lemma12_fejer`, `avg_toeplitz_le_tsum`, `lemma12_le_B` | FiniteTrace | verified (needs γ_aa, γ_bb even, which autocovariances are) |
| Lemma 1.2(ii): 2n‖A‖_F² = (2/n)Σγ_aa(s−t)² ≤ D | `lemma12_diag`, `lemma12_diag_le_D` | FiniteTrace | verified |
| Lemma 1.2: Var = 2‖A‖_F², operator-norm bounds, (iii) | — | — | not formalised (Gaussian moments; operator norms) |
| Lemma 1.3, deterministic steps: Ĉ_ab = γ̃_ab − X̄_aX̄_b; deviations ε_ab, ε_d on the event; thresholds from Lemma 1.2; union-bound count | `centred_cov_eq`, `centred_dev_offdiag`, `centred_dev_diag`, `lemma13_eps`, `lemma13_count` | FiniteCorr | verified |
| Lemma 1.3, probability of the event | — | — | not formalised (Gaussian tails) |
| **Lemma 1.4** (§5 item 1) | `corr_perturb` | FiniteCorr | verified; the hypothesis \|ρ\| ≤ 1 is not needed |
| **Thm 1.5** deterministic core (§5 item 4) | `decide_eq_target` (any target T), `decide_eq_support` (T = R), `thm15_det` (from Ĉ with Lemma 1.4's δ_ab) | FiniteCorr | verified |
| Thm 1.5 explicit form: y = x + log(4/(d−1))₊ gives failure ≤ 2d(d−1)e^{−x} | `explicit_y` (d ≥ 2) | FiniteCorr | verified; the (1 + o(1)) exponent is asymptotic, not formalised |
| **Cor 1.6** (§5 item 5) | `sOf_cntDecl_congr` | FiniteCorr | verified |
| Prop 1.7, Cor 1.8 | — | — | not formalised (CLT, Isserlis, delta method) |
| **Prop 2.1**: ρ_uv = b_ub_v/(s_us_v) = sgn(b_ub_v)√(ℓ_uℓ_v); all counts equal ⇒ s = 1/2 | `prop21_corr`, `prop21_corr_std`, `sOf_saturated`, `sOf_const` | FinitePartial, FiniteCorr | verified; the Lebesgue-null exception set is not formalised |
| **Thm 2.2** saturation step and constants | `sOf_saturated`, `abs_sOf_sub_half_le` (\|ŝ − 1/2\| ≤ 1/2), `thm22_const` | FiniteCorr | verified; the probability bound is Thm 1.5 + Lemma 1.3 (probabilistic part not formalised) |
| Prop 2.3 (a) location n_1/2 = 3 + (z_c/atanh κ)² | `ncrit_iff`, `ncrit_eq_iff` with k₀ = 3 | FinitePartial | deterministic location verified; "probability 1/2 there" is asymptotic, not formalised |
| **Prop 3.1** raw and standardised forms (§5 item 6) | `prop31`, `prop31_std` | FinitePartial | verified (needs Var Y_u, Var Y_v > 0) |
| Definition §3: π equals the correlation of OLS residuals | `pcorr_eq_pcov`, `pcorr_eq_corr_resid`, `inner_resid` | FinitePartial | verified |
| **Cor 3.2** (r = 1 ⇒ π = ρ^Y) | `cor32` | FinitePartial | population part verified; the finite-sample part is sketched in FINITE.md and not formalised |
| **Prop 3.3** formula, bounds, π = 0 ⇔ r = 1, strict monotonicity, equal shares | `prop33`, `prop33_abs`, `absPi_lower`, `absPi_upper`, `absPi_le_kappa`, `absPi_lt_kappa`, `absPi_eq_zero_iff`, `absPi_strictAntiOn`, `absPi_equal` | FinitePartial | verified, with the extra hypothesis b_ub_v ≠ 0 for the strict statements (erratum F1 in FINITE.md) |
| **Lemma 3.4** Lipschitz bound (§5 item 8) | `lipL`, `pc_lipschitz` (helpers `pc_lip_a`, `pc_lip_b`) | FiniteLipschitz | verified |
| Thm 3.5 bracket \|π̂ − π\| ≤ L(q′)max(δ) and decisions (i)/(ii) | `thm35_bracket`; (i)/(ii) are `decide_eq_target` applied to π̂ | FiniteLipschitz, FiniteCorr | verified (deterministic part) |
| Thm 3.5 n_crit = 4 + (z_c/atanh\|π\|)² and r_crit (§5 item 7) | `ncrit_iff`, `ncrit_eq_iff` (k₀ = 4), `rcrit`, `rcrit_iff`, `rcrit_eq_iff`, `rcrit_example` (ℓ = 0.64, τ′ = 0.082 ⇒ r_crit ∈ (0.949, 0.951)) | FinitePartial | verified (deterministic location; asymptotic reading not formalised) |
| **Prop 3.6** general form, rank one, AR(1) formula (§5 item 9), nonvanishing | `prop36_pcov`, `prop36_zero_iff`, `prop36_rank_one`, `prop36_ar1_pcov`, `prop36_ar1_pvar`, `prop36_ar1`, `prop36_ar1_ne_zero` | FinitePartial | verified (nonvanishing needs b_ub_v ≠ 0, erratum F3) |
| **Prop 3.7** numerator and bound | `prop37_pcov`, `prop37_bound` | FinitePartial | verified (r ≤ 1 from Cauchy-Schwarz) |
| **Prop 3.8** ρ_uZρ_vZ/ρ_uv = r and ℓ_u = ρ_uZ²/r | `prop38_ratio`, `prop38_loading` | FinitePartial | verified; ℓ_u = ρ_uZ²/r needs r > 0 (erratum F2) |
| Section 4, simulations, Section 7 | — | — | not formal statements |

The errata found while formalising are recorded in FINITE.md, section "Erratum (Lean check)". None
of them changes a result used later in FINITE.md; they are missing nondegeneracy hypotheses.

## Findings about THEORY.md

No statement of Section 1 was found to be false. Every Section 1 statement that was formalised (all of
them except the forest edge count in Cor 1.7 and the numerical Remark 1.11) was proved as stated, with
the clarifications below.

1. **Thm 1.3, proof route.** The Lean proof does not use Reisach et al.'s Lemma 1 (eq. 10). It
   characterises rel in G_T directly (`isRel_unrolled_iff`: w(r) ∈ rel(v(t)) iff some u reaches v and w
   by walks of lengths t−1 and r−1) and compares the sets. The statement is the same.
2. **Prop 1.4, t ≥ p.** THEORY.md's proof uses "whenever T ≥ p". That hypothesis is redundant, because
   t ≤ T. `rho_unrolled_large_t` needs only t ≥ p (Lean: t ≥ p − 1).
3. **Cor 1.10 (iii), in-forests.** The formalisation assumes acyclicity apart from self-loops, plus
   out-degree at most 1. This is the meaning of "forest". Out-degree ≤ 1 alone (the parenthetical in
   THEORY.md) is not enough. Counterexample: a ⇄ b with c → a. Every node has at most one downstream
   neighbour, E_↑ = ∅ (the only source SCC is {c}), so s = 1/2, yet a has two parents. This is a
   clarification of the hypothesis, not an error in the statement for forests. THEORY.md is therefore
   left unchanged.
4. **Cor 1.10 (ii), out-forests.** These need no acyclicity: "at most one parent besides itself" is
   enough (`stat_half_of_outForest`).
5. **Window edges and self-loops.** In Thm 1.9 the edge set is a copy of E_T, self-loops included, as
   THEORY.md states. The Phase-2 operationalisation (THEORY.md §6, non-self-loop edges only) is not part
   of the Lean statements.

## Files

- `RelSort/Basic.lean`: definitions (anc, desc, rel, ρ, s_ρ), Reisach Thm 1.
- `RelSort/Walk.lean`: walks of a given length, `dist`, and the bound "reachable ⇒ walk of length p − 1".
- `RelSort/Unrolled.lean`: G_T, Lemma 1.1, Thm 1.3, edge-set reindexing.
- `RelSort/Limit.lean`: Thm 1.5, Cor 1.6, Cor 1.7.
- `RelSort/Stationary.lean`: Thm 1.9 (burn-in and infinite past), source SCCs, Cor 1.10.
- `RelSort/Examples.lean`: 3-node collider, fork and chain.
- `RelSort/Extras.lean`: Lemma 1.2, Prop 1.4, Remark 1.12.
- `RelSort/T2Algebra.lean`: algebraic parts of Prop 2.1 and Prop 2.3.
- `RelSort/FiniteCorr.lean`: FINITE.md Lemmas 1.1 (algebra), 1.3 (deterministic steps), 1.4, Thm 1.5 core, Cor 1.6, Thm 2.2 saturation.
- `RelSort/FiniteTrace.lean`: FINITE.md Lemma 1.2 block-matrix and Toeplitz algebra.
- `RelSort/FinitePartial.lean`: FINITE.md Props 2.1, 3.1-3.3, 3.6-3.8, r_crit and n_crit, in an inner-product model.
- `RelSort/FiniteLipschitz.lean`: FINITE.md Lemma 3.4 and the bracket of Thm 3.5.
- `Axioms.lean`: `#print axioms` for each main theorem. Output: `AXIOMS.txt`, `BUILD_LOG.txt`.
