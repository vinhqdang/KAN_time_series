/-
THEORY.md Theorem 1.5 (exact finite-`T` value of `s_ρ(G_T)`, explicit bound, limit),
Corollary 1.6 (iff condition for the limit `1/2`, uniform bound) and Corollary 1.7
(DAGs with self-loops).

`n = T - 1` is the number of edge layers of `G_T`; `T ≥ 2` is `n ≥ 1`.
-/
import RelSort.Unrolled

open Finset Filter Topology

set_option linter.unusedSectionVars false

namespace RelSort

variable {V : Type*} [Fintype V] {E : V → V → Prop}

/-- Reachability in `G_S`. -/
abbrev Reach (E : V → V → Prop) : V → V → Prop := Relation.ReflTransGen E

/-- Same strongly connected component. -/
def SameSCC (E : V → V → Prop) (i j : V) : Prop := Reach E i j ∧ Reach E j i

open Classical in
/-- `E_×`: summary edges joining two different SCCs. -/
noncomputable def crossEdges (E : V → V → Prop) : Finset (V × V) :=
  (edges E).filter (fun e => ¬ SameSCC E e.1 e.2)

open Classical in
/-- `E_∘ = E_S \ E_×`: self-loops and edges inside an SCC. -/
noncomputable def innerEdges (E : V → V → Prop) : Finset (V × V) :=
  (edges E).filter (fun e => SameSCC E e.1 e.2)

open Classical in
/-- `N_T(i,j)`: number of the `T - 1` copies of `(i,j)` with score `1`. -/
noncomputable def numStrict (E : V → V → Prop) (n : ℕ) (i j : V) : ℕ :=
  (univ.filter (fun s : Fin n => StrictCond E s i j)).card

/-- The limit `L(G_S) = 1/2 + |E_×| / (2 |E_S|)`. -/
noncomputable def limitValue (E : V → V → Prop) : ℝ :=
  1 / 2 + ((crossEdges E).card : ℝ) / (2 * (edges E).card)

lemma edges_card_pos [Nonempty V] (hl : ∀ v, E v v) : 0 < (edges E).card := by
  obtain ⟨v⟩ := ‹Nonempty V›
  exact Finset.card_pos.mpr ⟨(v, v), mem_edges.mpr (hl v)⟩

/-- On an edge, "different SCC" is "the head does not reach the tail". -/
lemma sameSCC_edge_iff {i j : V} (hij : E i j) : SameSCC E i j ↔ Reach E j i :=
  ⟨fun h => h.2, fun h => ⟨Relation.ReflTransGen.single hij, h⟩⟩

/-- Crossing edges: every copy has score `1`. -/
lemma strictCond_of_cross (hl : ∀ v, E v v) {i j : V} (hij : E i j) (hc : ¬ SameSCC E i j)
    (s : ℕ) : StrictCond E s i j := by
  rw [sameSCC_edge_iff hij] at hc
  exact ⟨j, walkN_mono hl (Nat.zero_le _) (rfl : WalkN E 0 j j),
    fun h => hc (walkN_reach h)⟩

/-- Edges inside an SCC: only the layers `s < p - 1` (paper: `t ≤ p - 1`) can have score `1`. -/
lemma lt_of_strictCond_inner (hl : ∀ v, E v v) {i j : V} (hc : SameSCC E i j) {s : ℕ}
    (h : StrictCond E s i j) : s < Fintype.card V - 1 := by
  obtain ⟨u, h1, h2⟩ := h
  have hui : Reach E u i := (walkN_reach h1).trans hc.2
  by_contra hs
  exact h2 (walkN_of_reach hl hui (by omega))

lemma numStrict_cross (hl : ∀ v, E v v) {i j : V} (hij : E i j) (hc : ¬ SameSCC E i j)
    (n : ℕ) : numStrict E n i j = n := by
  classical
  unfold numStrict
  have : ∀ s : Fin n, StrictCond E s i j := fun s => strictCond_of_cross hl hij hc s
  simp [this]

lemma numStrict_le (n : ℕ) (i j : V) : numStrict E n i j ≤ n := by
  classical
  unfold numStrict
  exact (card_filter_le _ _).trans (by simp)

lemma numStrict_inner_le (hl : ∀ v, E v v) {i j : V} (hc : SameSCC E i j) (n : ℕ) :
    numStrict E n i j ≤ min (Fintype.card V - 1) n := by
  classical
  refine le_min ?_ (numStrict_le n i j)
  unfold numStrict
  calc (univ.filter (fun s : Fin n => StrictCond E s i j)).card
      ≤ (range (Fintype.card V - 1)).card := by
        apply card_le_card_of_injOn (fun s : Fin n => (s : ℕ))
        · intro s hs
          simp only [coe_filter, mem_univ, true_and, Set.mem_ofPred_eq] at hs
          simpa using lt_of_strictCond_inner hl hc hs
        · intro a _ b _ h; exact Fin.ext h
    _ = Fintype.card V - 1 := card_range _

/-- Edges inside an SCC, sharp form: only the layers `s < Δ` (paper: `t ≤ Δ`) can have
score `1`, where `Δ = maxFinDist E` is the largest finite distance. -/
lemma lt_of_strictCond_inner_maxFinDist (hl : ∀ v, E v v) {i j : V} (hc : SameSCC E i j)
    {s : ℕ} (h : StrictCond E s i j) : s < maxFinDist E := by
  obtain ⟨u, h1, h2⟩ := h
  have hui : Reach E u i := (walkN_reach h1).trans hc.2
  by_contra hs
  exact h2 (walkN_of_reach_maxFinDist hl hui (by omega))

lemma numStrict_inner_le_maxFinDist (hl : ∀ v, E v v) {i j : V} (hc : SameSCC E i j) (n : ℕ) :
    numStrict E n i j ≤ min (maxFinDist E) n := by
  classical
  refine le_min ?_ (numStrict_le n i j)
  unfold numStrict
  calc (univ.filter (fun s : Fin n => StrictCond E s i j)).card
      ≤ (range (maxFinDist E)).card := by
        apply card_le_card_of_injOn (fun s : Fin n => (s : ℕ))
        · intro s hs
          simp only [coe_filter, mem_univ, true_and, Set.mem_ofPred_eq] at hs
          simpa using lt_of_strictCond_inner_maxFinDist hl hc hs
        · intro a _ b _ h; exact Fin.ext h
    _ = maxFinDist E := card_range _

/-- Sum of the scores of the copies of one summary edge. -/
lemma sum_layer_score (hl : ∀ v, E v v) {n : ℕ} {i j : V} (hij : E i j) :
    ∑ s : Fin n, score (rho (unrolled E n) (i, s.castSucc)) (rho (unrolled E n) (j, s.succ)) =
      (n : ℝ) / 2 + (numStrict E n i j : ℝ) / 2 := by
  classical
  have : ∀ s : Fin n, score (rho (unrolled E n) (i, s.castSucc))
      (rho (unrolled E n) (j, s.succ)) = 1 / 2 + 1 / 2 * (if StrictCond E s i j then 1 else 0) := by
    intro s
    rw [edge_score hl hij s]
    split_ifs <;> norm_num
  simp_rw [this]
  rw [sum_add_distrib, ← mul_sum, sum_boole]
  simp [numStrict]
  ring

/-- Exact finite-`T` value, raw form:
`s_ρ(G_T) = 1/2 + Σ_{(i,j) ∈ E_S} N_T(i,j) / (2 (T-1) |E_S|)`. -/
theorem sRho_unrolled_eq (hl : ∀ v, E v v) [Nonempty V] {n : ℕ} (hn : 1 ≤ n) :
    sRho (unrolled E n) =
      1 / 2 + (∑ e ∈ edges E, (numStrict E n e.1 e.2 : ℝ)) / (2 * n * (edges E).card) := by
  have hE : (0 : ℝ) < (edges E).card := by exact_mod_cast edges_card_pos hl
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  unfold sRho sOf
  rw [sum_edges_unrolled, card_edges_unrolled]
  rw [sum_congr rfl (fun e he => sum_layer_score hl (mem_edges.mp he))]
  rw [sum_add_distrib, sum_const, ← sum_div]
  simp only [nsmul_eq_mul]
  push_cast
  field_simp

/-- **Theorem 1.5, exact finite-`T` value**:
`s_ρ(G_T) = 1/2 + [ (T-1)|E_×| + Σ_{(i,j) ∈ E_∘} N_T(i,j) ] / (2 (T-1) |E_S|)`. -/
theorem sRho_unrolled_exact (hl : ∀ v, E v v) [Nonempty V] {n : ℕ} (hn : 1 ≤ n) :
    sRho (unrolled E n) =
      1 / 2 + ((n : ℝ) * (crossEdges E).card + ∑ e ∈ innerEdges E, (numStrict E n e.1 e.2 : ℝ))
        / (2 * n * (edges E).card) := by
  classical
  have hs : ∑ e ∈ edges E, (numStrict E n e.1 e.2 : ℝ) =
      (n : ℝ) * (crossEdges E).card + ∑ e ∈ innerEdges E, (numStrict E n e.1 e.2 : ℝ) := by
    rw [← sum_filter_not_add_sum_filter (edges E) (fun e => SameSCC E e.1 e.2)]
    have hc : ∑ e ∈ (edges E).filter (fun e => ¬ SameSCC E e.1 e.2), (numStrict E n e.1 e.2 : ℝ)
        = ∑ _e ∈ crossEdges E, (n : ℝ) := by
      apply sum_congr rfl
      intro e he
      rw [mem_filter, mem_edges] at he
      rw [numStrict_cross hl he.1 he.2]
    rw [hc, sum_const, nsmul_eq_mul, mul_comm]
    rfl
  rw [sRho_unrolled_eq hl hn, hs]

/-- `s_ρ(G_T) - L = Σ_{E_∘} N_T / (2 (T-1) |E_S|)`. -/
lemma sRho_unrolled_sub (hl : ∀ v, E v v) [Nonempty V] {n : ℕ} (hn : 1 ≤ n) :
    sRho (unrolled E n) - limitValue E =
      (∑ e ∈ innerEdges E, (numStrict E n e.1 e.2 : ℝ)) / (2 * n * (edges E).card) := by
  have hE : (0 : ℝ) < (edges E).card := by exact_mod_cast edges_card_pos hl
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  rw [sRho_unrolled_exact hl hn, limitValue]
  field_simp
  ring

/-- **Theorem 1.5, explicit bound, sharp form**: with `Δ = maxFinDist E` the largest finite
shortest-path distance in `G_S` (`Δ ≤ p - 1`, `maxFinDist_le_card_sub_one`), for every `T ≥ 2`,
`0 ≤ s_ρ(G_T) - L ≤ min(Δ, T-1) |E_∘| / (2 (T-1) |E_S|) ≤ Δ / (2 (T-1))`. -/
theorem sRho_unrolled_bounds_maxFinDist (hl : ∀ v, E v v) [Nonempty V] {n : ℕ} (hn : 1 ≤ n) :
    0 ≤ sRho (unrolled E n) - limitValue E ∧
    sRho (unrolled E n) - limitValue E ≤
      (min (maxFinDist E) n : ℕ) * (innerEdges E).card / (2 * n * (edges E).card) ∧
    (min (maxFinDist E) n : ℕ) * ((innerEdges E).card : ℝ) / (2 * n * (edges E).card) ≤
      (maxFinDist E : ℝ) / (2 * n) := by
  classical
  have hE : (0 : ℝ) < (edges E).card := by exact_mod_cast edges_card_pos hl
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  have hden : (0 : ℝ) < 2 * n * (edges E).card := by positivity
  rw [sRho_unrolled_sub hl hn]
  refine ⟨by positivity, ?_, ?_⟩
  · apply div_le_div_of_nonneg_right _ hden.le
    have : ∑ e ∈ innerEdges E, (numStrict E n e.1 e.2 : ℝ) ≤
        ∑ _e ∈ innerEdges E, ((min (maxFinDist E) n : ℕ) : ℝ) := by
      apply sum_le_sum
      intro e he
      unfold innerEdges at he
      rw [mem_filter] at he
      exact_mod_cast numStrict_inner_le_maxFinDist hl he.2 n
    simpa [mul_comm] using this
  · have hin : ((innerEdges E).card : ℝ) ≤ (edges E).card := by
      exact_mod_cast card_filter_le _ _
    have hmin : ((min (maxFinDist E) n : ℕ) : ℝ) ≤ (maxFinDist E : ℝ) := by
      exact_mod_cast min_le_left _ _
    rw [div_le_div_iff₀ hden (by positivity)]
    calc (min (maxFinDist E) n : ℕ) * ((innerEdges E).card : ℝ) * (2 * n)
        ≤ (maxFinDist E : ℝ) * (edges E).card * (2 * n) := by
          apply mul_le_mul_of_nonneg_right _ (by positivity)
          exact mul_le_mul hmin hin (by positivity) (by positivity)
      _ = (maxFinDist E : ℝ) * (2 * n * (edges E).card) := by ring

/-- **Theorem 1.5, explicit bound** (`p - 1` form; corollary of
`sRho_unrolled_bounds_maxFinDist`, since `Δ ≤ p - 1`): for every `T ≥ 2`,
`0 ≤ s_ρ(G_T) - L ≤ min(p-1, T-1) |E_∘| / (2 (T-1) |E_S|) ≤ (p-1) / (2 (T-1))`. -/
theorem sRho_unrolled_bounds (hl : ∀ v, E v v) [Nonempty V] {n : ℕ} (hn : 1 ≤ n) :
    0 ≤ sRho (unrolled E n) - limitValue E ∧
    sRho (unrolled E n) - limitValue E ≤
      (min (Fintype.card V - 1) n : ℕ) * (innerEdges E).card / (2 * n * (edges E).card) ∧
    (min (Fintype.card V - 1) n : ℕ) * ((innerEdges E).card : ℝ) / (2 * n * (edges E).card) ≤
      ((Fintype.card V - 1 : ℕ) : ℝ) / (2 * n) := by
  classical
  obtain ⟨h1, h2, -⟩ := sRho_unrolled_bounds_maxFinDist hl hn
  have hE : (0 : ℝ) < (edges E).card := by exact_mod_cast edges_card_pos hl
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  have hden : (0 : ℝ) < 2 * n * (edges E).card := by positivity
  have hΔ := maxFinDist_le_card_sub_one hl (E := E)
  refine ⟨h1, h2.trans ?_, ?_⟩
  · apply div_le_div_of_nonneg_right _ hden.le
    apply mul_le_mul_of_nonneg_right _ (by positivity)
    exact_mod_cast min_le_min_right n hΔ
  · have hin : ((innerEdges E).card : ℝ) ≤ (edges E).card := by
      exact_mod_cast card_filter_le _ _
    have hmin : ((min (Fintype.card V - 1) n : ℕ) : ℝ) ≤ ((Fintype.card V - 1 : ℕ) : ℝ) := by
      exact_mod_cast min_le_left _ _
    rw [div_le_div_iff₀ hden (by positivity)]
    calc (min (Fintype.card V - 1) n : ℕ) * ((innerEdges E).card : ℝ) * (2 * n)
        ≤ ((Fintype.card V - 1 : ℕ) : ℝ) * (edges E).card * (2 * n) := by
          apply mul_le_mul_of_nonneg_right _ (by positivity)
          exact mul_le_mul hmin hin (by positivity) (by positivity)
      _ = ((Fintype.card V - 1 : ℕ) : ℝ) * (2 * n * (edges E).card) := by ring

/-- **Theorem 1.5, limit** (`n = T - 1 → ∞`). -/
theorem tendsto_sRho_unrolled (hl : ∀ v, E v v) [Nonempty V] :
    Tendsto (fun n : ℕ => sRho (unrolled E n)) atTop (𝓝 (limitValue E)) := by
  have hup : Tendsto (fun n : ℕ => limitValue E + (((Fintype.card V - 1 : ℕ) : ℝ) / 2) / n)
      atTop (𝓝 (limitValue E)) := by
    simpa using tendsto_const_nhds.add (tendsto_const_div_atTop_nhds_zero_nat
      (((Fintype.card V - 1 : ℕ) : ℝ) / 2))
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds hup ?_ ?_
  · filter_upwards [eventually_ge_atTop 1] with n hn
    linarith [(sRho_unrolled_bounds hl hn).1]
  · filter_upwards [eventually_ge_atTop 1] with n hn
    obtain ⟨_, h2, h3⟩ := sRho_unrolled_bounds hl hn
    have : ((Fintype.card V - 1 : ℕ) : ℝ) / (2 * n) = (((Fintype.card V - 1 : ℕ) : ℝ) / 2) / n := by
      rw [div_div]
    linarith

/-- **Theorem 1.5, limit**, indexed by the number of slices `T` (`G_T = unrolled E (T-1)`). -/
theorem tendsto_sRho_GT (hl : ∀ v, E v v) [Nonempty V] :
    Tendsto (fun T : ℕ => sRho (unrolled E (T - 1))) atTop (𝓝 (limitValue E)) :=
  (tendsto_sRho_unrolled hl).comp (tendsto_sub_atTop_nat 1)

/-! ### Corollary 1.6 -/

/-- Weak connectivity (connected in the underlying undirected graph). -/
def WeakConn (E : V → V → Prop) : V → V → Prop :=
  Relation.ReflTransGen (fun a b => E a b ∨ E b a)

/-- The equivalent forms of "every weakly connected component of `G_S` is strongly
connected". -/
theorem crossEdges_eq_empty_tfae :
    List.TFAE [crossEdges E = ∅,
      ∀ i j, E i j → Reach E j i,
      ∀ i j, Reach E i j → Reach E j i,
      ∀ i j, WeakConn E i j → SameSCC E i j] := by
  classical
  tfae_have 1 → 2 := by
    intro h i j hij
    by_contra hc
    have : (i, j) ∈ crossEdges E := by
      unfold crossEdges
      rw [mem_filter, mem_edges, sameSCC_edge_iff hij]
      exact ⟨hij, hc⟩
    rw [h] at this
    exact absurd this (Finset.notMem_empty _)
  tfae_have 2 → 3 := by
    intro h i j hij
    induction hij with
    | refl => exact Relation.ReflTransGen.refl
    | tail _ hbc ih => exact (h _ _ hbc).trans ih
  tfae_have 3 → 4 := by
    intro h i j hij
    have hr : Reach E i j := by
      induction hij with
      | refl => exact Relation.ReflTransGen.refl
      | tail _ hbc ih =>
        rcases hbc with hbc | hbc
        · exact ih.tail hbc
        · exact ih.trans (h _ _ (Relation.ReflTransGen.single hbc))
    exact ⟨hr, h _ _ hr⟩
  tfae_have 4 → 1 := by
    intro h
    unfold crossEdges
    rw [Finset.filter_eq_empty_iff]
    intro e he
    rw [mem_edges] at he
    exact not_not.mpr (h _ _ (Relation.ReflTransGen.single (Or.inl he)))
  tfae_finish

lemma limitValue_eq_half_iff (hl : ∀ v, E v v) [Nonempty V] :
    limitValue E = 1 / 2 ↔ crossEdges E = ∅ := by
  have hE : (0 : ℝ) < (edges E).card := by exact_mod_cast edges_card_pos hl
  unfold limitValue
  constructor
  · intro h
    have : ((crossEdges E).card : ℝ) / (2 * (edges E).card) = 0 := by linarith
    rw [div_eq_zero_iff] at this
    rcases this with h1 | h1
    · exact Finset.card_eq_zero.mp (by exact_mod_cast h1)
    · exfalso; linarith
  · intro h; simp [h]

/-- **Corollary 1.6**: `lim_{T→∞} s_ρ(G_T) = 1/2` iff every weakly connected component of
`G_S` is strongly connected (iff `E_× = ∅`). The "if" direction is Reisach et al.'s
Theorem 4; the "only if" direction is new. -/
theorem tendsto_half_iff (hl : ∀ v, E v v) [Nonempty V] :
    Tendsto (fun n : ℕ => sRho (unrolled E n)) atTop (𝓝 (1 / 2)) ↔
      ∀ i j, WeakConn E i j → SameSCC E i j := by
  have key := (crossEdges_eq_empty_tfae (E := E)).out 1 4
  rw [← key, ← limitValue_eq_half_iff hl]
  constructor
  · intro h; exact tendsto_nhds_unique (tendsto_sRho_unrolled hl) h
  · intro h; rw [← h]; exact tendsto_sRho_unrolled hl

/-- Reisach et al.'s Theorem 4, in their form (`i ⇝ j ⇒ j ⇝ i`). -/
theorem reisach_theorem4 (hl : ∀ v, E v v) [Nonempty V]
    (h : ∀ i j, Reach E i j → Reach E j i) :
    Tendsto (fun n : ℕ => sRho (unrolled E n)) atTop (𝓝 (1 / 2)) :=
  (tendsto_half_iff hl).mpr ((crossEdges_eq_empty_tfae (E := E)).out 3 4 |>.mp h)

/-- **Corollary 1.6, uniform bound**: if some weak component is not strongly connected, then
`s_ρ(G_T) ≥ 1/2 + 1/(2|E_S|)` for every `T ≥ 2`. -/
theorem sRho_unrolled_ge_of_cross (hl : ∀ v, E v v) [Nonempty V] (h : crossEdges E ≠ ∅)
    {n : ℕ} (hn : 1 ≤ n) :
    1 / 2 + 1 / (2 * ((edges E).card : ℝ)) ≤ sRho (unrolled E n) := by
  have hE : (0 : ℝ) < (edges E).card := by exact_mod_cast edges_card_pos hl
  have h1 : (1 : ℝ) ≤ (crossEdges E).card := by
    exact_mod_cast Finset.card_pos.mpr (Finset.nonempty_iff_ne_empty.mpr h)
  have h2 := (sRho_unrolled_bounds hl hn).1
  have h3 : 1 / (2 * ((edges E).card : ℝ)) ≤ ((crossEdges E).card : ℝ) / (2 * (edges E).card) :=
    div_le_div_of_nonneg_right h1 (by positivity)
  unfold limitValue at h2
  linarith

/-! ### Corollary 1.7 (summary graph acyclic apart from self-loops) -/

open Classical in
/-- Number of non-self-loop summary edges. -/
noncomputable def numProperEdges (E : V → V → Prop) : ℕ :=
  ((edges E).filter (fun e => e.1 ≠ e.2)).card

/-- **Corollary 1.7**: if `G_S` is acyclic apart from self-loops, with `m` non-self-loop edges,
then `L = 1/2 + m / (2 (m + p))`. -/
theorem limitValue_dag (hl : ∀ v, E v v)
    (hac : ∀ i j, Reach E i j → Reach E j i → i = j) :
    limitValue E = 1 / 2 + (numProperEdges E : ℝ) / (2 * (numProperEdges E + Fintype.card V)) := by
  classical
  have hcross : crossEdges E = (edges E).filter (fun e => e.1 ≠ e.2) := by
    unfold crossEdges
    apply Finset.filter_congr
    intro e he
    rw [mem_edges] at he
    constructor
    · intro h heq; apply h; rw [← heq]; exact ⟨Relation.ReflTransGen.refl, Relation.ReflTransGen.refl⟩
    · intro h hs; exact h (hac _ _ hs.1 hs.2)
  have hloops : ((edges E).filter (fun e => ¬ e.1 ≠ e.2)).card = Fintype.card V := by
    have : (edges E).filter (fun e => ¬ e.1 ≠ e.2) = univ.image (fun v : V => (v, v)) := by
      ext ⟨a, b⟩
      simp only [mem_filter, mem_edges, not_not, mem_image, mem_univ, true_and, Prod.mk.injEq]
      constructor
      · rintro ⟨-, rfl⟩; exact ⟨a, rfl, rfl⟩
      · rintro ⟨v, rfl, rfl⟩; exact ⟨hl v, rfl⟩
    rw [this, card_image_of_injective _ (fun a b h => (Prod.mk.inj h).1), card_univ]
  have htot : (edges E).card = numProperEdges E + Fintype.card V := by
    rw [← hloops, numProperEdges, card_filter_add_card_filter_not]
  unfold limitValue
  rw [hcross, htot]
  push_cast
  rfl

end RelSort
