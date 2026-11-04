# Lean 4 verification of THEORY.md (part L of PLAN2.md)

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
| PLAN2 F1–F3 (finite-sample) | — | — | not formalised. They were not yet written in THEORY.md when this was done. They are probabilistic (Fisher-z tests, Bartlett variance, partial correlations) |

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
- `Axioms.lean`: `#print axioms` for each main theorem. Output: `AXIOMS.txt`, `BUILD_LOG.txt`.
