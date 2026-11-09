/-
THEORY.md Theorem 1.9 (stationary / burn-in version) and Corollary 1.10.

* Burn-in: `G_{B+T}` is `unrolled E (B + m)` (`B + m + 1` slices); the window `W` is the last
  `T = m + 1` slices, embedded by `t ↦ B + t`. Relatives are those of the full DAG, restricted
  to `W`; edges are the edges of `G_{B+T}` inside `W`.
* Infinite past: `G_Z` on `V × ℤ`, window `V × {0,…,m}`.

Source SCCs are represented by their member nodes: `x` is a *source node* iff every ancestor
of `x` is reachable from `x` (its SCC has no incoming edge from another SCC), and
`Src(v)` is represented by the predicate `x ↦ (x source ∧ x ⇝ v)`. Since SCCs are disjoint and
a source SCC reaches `v` iff each of its nodes does, `Src(i) = Src(j)` as sets of SCCs iff these
predicates agree.
-/
import RelSort.Limit

open Finset

set_option linter.unusedSectionVars false

namespace RelSort

variable {V : Type*} [Fintype V] {E : V → V → Prop}

open Classical in
/-- `E_↑ = {(i,j) ∈ E_S : rel_S(i) ≠ rel_S(j)}`. -/
noncomputable def upEdges (E : V → V → Prop) : Finset (V × V) :=
  (edges E).filter (fun e => relSet E e.1 ≠ relSet E e.2)

/-- The stationary value `1/2 + |E_↑| / (2 |E_S|)`. -/
noncomputable def statValue (E : V → V → Prop) : ℝ :=
  1 / 2 + ((upEdges E).card : ℝ) / (2 * (edges E).card)

/-! ### A window whose relatives are the summary relatives -/

section Window

variable {m : ℕ}

lemma setOf'_window {P : V × Fin (m + 1) → V × Fin (m + 1) → Prop}
    (hP : ∀ v t w r, P (v, t) (w, r) ↔ IsRel E v w) (v : V) (t : Fin (m + 1)) :
    setOf' P (v, t) = relSet E v ×ˢ (univ : Finset (Fin (m + 1))) := by
  ext ⟨w, r⟩
  simp [hP, mem_relSet]

lemma cntOf_window {P : V × Fin (m + 1) → V × Fin (m + 1) → Prop}
    (hP : ∀ v t w r, P (v, t) (w, r) ↔ IsRel E v w) (v : V) (t : Fin (m + 1)) :
    cntOf P (v, t) = (m + 1) * rho E v := by
  unfold cntOf
  rw [setOf'_window hP, card_product, card_univ, Fintype.card_fin, mul_comm]
  rfl

/-- Generic form of Theorem 1.9 (b): any window model whose relatives are the summary
relatives has `s_ρ = 1/2 + |E_↑| / (2|E_S|)` for every `T = m + 1 ≥ 2`. -/
theorem sOf_window (hl : ∀ v, E v v) [Nonempty V] (hm : 1 ≤ m)
    {P : V × Fin (m + 1) → V × Fin (m + 1) → Prop}
    (hP : ∀ v t w r, P (v, t) (w, r) ↔ IsRel E v w) :
    sOf (unrolled E m) (cntOf P) = statValue E := by
  classical
  have hE : (0 : ℝ) < (edges E).card := by exact_mod_cast edges_card_pos hl
  have hm' : (0 : ℝ) < m := by exact_mod_cast hm
  have hedge : ∀ e ∈ edges E, ∀ s : Fin m,
      score (cntOf P (e.1, s.castSucc)) (cntOf P (e.2, s.succ)) =
        1 / 2 + 1 / 2 * (if relSet E e.1 ≠ relSet E e.2 then 1 else 0) := by
    intro e he s
    rw [mem_edges] at he
    rw [cntOf_window hP, cntOf_window hP, rho_eq_card, rho_eq_card,
      score_mul_of_subset (relSet_subset_of_edge he) _ (Nat.succ_pos m)]
    by_cases h : relSet E e.1 = relSet E e.2
    · simp [h]
    · simp [h]; norm_num
  unfold sOf statValue
  rw [sum_edges_unrolled, card_edges_unrolled]
  rw [sum_congr rfl (fun e he => sum_congr rfl (fun s _ => hedge e he s))]
  simp only [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
  rw [← mul_sum, sum_add_distrib, ← mul_sum, sum_boole, sum_const, nsmul_eq_mul]
  unfold upEdges
  push_cast
  field_simp

end Window

/-! ### Burn-in version -/

section BurnIn

variable (E) (B m : ℕ)

/-- Embedding of the observed window (`T = m + 1` slices) into `G_{B+T}`. -/
def burnEmb (a : V × Fin (m + 1)) : V × Fin (B + m + 1) := (a.1, ⟨B + a.2, by omega⟩)

/-- Relatives in the full DAG `G_{B+T}`, restricted to the window. -/
def relBurn (a b : V × Fin (m + 1)) : Prop :=
  IsRel (unrolled E (B + m)) (burnEmb B m a) (burnEmb B m b)

/-- Edges of `G_{B+T}` inside the window. -/
def edgeBurn (a b : V × Fin (m + 1)) : Prop :=
  unrolled E (B + m) (burnEmb B m a) (burnEmb B m b)

/-- `ρ_W` for the burn-in window. -/
noncomputable def rhoBurn : V × Fin (m + 1) → ℕ := cntOf (relBurn E B m)

/-- `s_ρ^{W,B}`. -/
noncomputable def sBurn : ℝ := sOf (edgeBurn E B m) (rhoBurn E B m)

variable {E B m}

/-- The window edges are exactly a copy of `E_T`. -/
lemma edgeBurn_eq : edgeBurn E B m = unrolled E m := by
  funext a b
  apply propext
  simp only [edgeBurn, unrolled, burnEmb]
  constructor
  · rintro ⟨h1, h2⟩; exact ⟨h1, by omega⟩
  · rintro ⟨h1, h2⟩; exact ⟨h1, by omega⟩

/-- With burn-in `B ≥ Δ` (`Δ = maxFinDist E`, the largest finite distance), window relatives
are the summary relatives. -/
lemma relBurn_iff_maxFinDist (hl : ∀ v, E v v) (hB : maxFinDist E ≤ B) (v : V)
    (t : Fin (m + 1)) (w : V) (r : Fin (m + 1)) : relBurn E B m (v, t) (w, r) ↔ IsRel E v w := by
  unfold relBurn burnEmb
  rw [isRel_unrolled_iff hl]
  simp only
  constructor
  · rintro ⟨u, h1, h2⟩; exact ⟨u, walkN_reach h1, walkN_reach h2⟩
  · rintro ⟨u, h1, h2⟩
    exact ⟨u, walkN_of_reach_maxFinDist hl h1 (by omega),
      walkN_of_reach_maxFinDist hl h2 (by omega)⟩

/-- With burn-in `B ≥ p - 1`, window relatives are the summary relatives
(corollary of `relBurn_iff_maxFinDist`). -/
lemma relBurn_iff (hl : ∀ v, E v v) (hB : Fintype.card V - 1 ≤ B) (v : V)
    (t : Fin (m + 1)) (w : V) (r : Fin (m + 1)) : relBurn E B m (v, t) (w, r) ↔ IsRel E v w :=
  relBurn_iff_maxFinDist hl ((maxFinDist_le_card_sub_one hl).trans hB) v t w r

/-- **Theorem 1.9 (a)**, burn-in, sharp form: if `B ≥ Δ`, `ρ_W(v(t)) = T · |rel_S(v)|`. -/
theorem rhoBurn_eq_maxFinDist (hl : ∀ v, E v v) (hB : maxFinDist E ≤ B) (v : V)
    (t : Fin (m + 1)) : rhoBurn E B m (v, t) = (m + 1) * rho E v :=
  cntOf_window (relBurn_iff_maxFinDist hl hB) v t

/-- **Theorem 1.9 (a)**, burn-in: `ρ_W(v(t)) = T · |rel_S(v)|` (`B ≥ p - 1` form). -/
theorem rhoBurn_eq (hl : ∀ v, E v v) (hB : Fintype.card V - 1 ≤ B) (v : V)
    (t : Fin (m + 1)) : rhoBurn E B m (v, t) = (m + 1) * rho E v :=
  rhoBurn_eq_maxFinDist hl ((maxFinDist_le_card_sub_one hl).trans hB) v t

/-- **Theorem 1.9 (b)**, burn-in, sharp form: if `B ≥ Δ` (the largest finite distance), then for
every `T ≥ 2`, `s_ρ^{W,B} = 1/2 + |E_↑| / (2|E_S|)`. -/
theorem sBurn_eq_maxFinDist (hl : ∀ v, E v v) [Nonempty V] (hB : maxFinDist E ≤ B)
    (hm : 1 ≤ m) : sBurn E B m = statValue E := by
  unfold sBurn rhoBurn
  rw [edgeBurn_eq]
  exact sOf_window hl hm (relBurn_iff_maxFinDist hl hB)

/-- **Theorem 1.9 (b)**, burn-in: if `B ≥ p - 1`, then for every `T ≥ 2`,
`s_ρ^{W,B} = 1/2 + |E_↑| / (2|E_S|)` (corollary of `sBurn_eq_maxFinDist`). -/
theorem sBurn_eq (hl : ∀ v, E v v) [Nonempty V] (hB : Fintype.card V - 1 ≤ B) (hm : 1 ≤ m) :
    sBurn E B m = statValue E :=
  sBurn_eq_maxFinDist hl ((maxFinDist_le_card_sub_one hl).trans hB) hm

end BurnIn

/-! ### Infinite-past version `G_Z` -/

section InfinitePast

variable (E) (m : ℕ)

/-- The infinite-past DAG `G_Z` on `V × ℤ`. -/
def zGraph (a b : V × ℤ) : Prop := E a.1 b.1 ∧ b.2 = a.2 + 1

/-- Window embedding `V × {0,…,m} → V × ℤ`. -/
def zEmb (a : V × Fin (m + 1)) : V × ℤ := (a.1, (a.2 : ℤ))

def relZ (a b : V × Fin (m + 1)) : Prop := IsRel (zGraph E) (zEmb m a) (zEmb m b)

def edgeZ (a b : V × Fin (m + 1)) : Prop := zGraph E (zEmb m a) (zEmb m b)

noncomputable def sZ : ℝ := sOf (edgeZ E m) (cntOf (relZ E m))

variable {E m}

lemma edgeZ_eq : edgeZ E m = unrolled E m := by
  funext a b
  apply propext
  simp only [edgeZ, zGraph, unrolled, zEmb]
  constructor
  · rintro ⟨h1, h2⟩; exact ⟨h1, by omega⟩
  · rintro ⟨h1, h2⟩; exact ⟨h1, by omega⟩

lemma zGraph_reach_proj {a b : V × ℤ} (h : Relation.ReflTransGen (zGraph E) a b) :
    Reach E a.1 b.1 := by
  induction h with
  | refl => exact Relation.ReflTransGen.refl
  | tail _ hbc ih => exact ih.tail hbc.1

lemma zGraph_reach_of_walkN {k : ℕ} {u v : V} (h : WalkN E k u v) (a : ℤ) :
    Relation.ReflTransGen (zGraph E) (u, a) (v, a + k) := by
  induction k generalizing v with
  | zero => rw [walkN_zero] at h; subst h; simpa using Relation.ReflTransGen.refl
  | succ k ih =>
    obtain ⟨w, hw, hwv⟩ := h
    exact (ih hw).tail ⟨hwv, by push_cast; ring⟩

/-- In `G_Z`, relatives of window nodes are the summary relatives, for every pair of times. -/
lemma isRel_zGraph_iff (hl : ∀ v, E v v) (v w : V) (t r : ℤ) :
    IsRel (zGraph E) (v, t) (w, r) ↔ IsRel E v w := by
  constructor
  · rintro ⟨x, h1, h2⟩; exact ⟨x.1, zGraph_reach_proj h1, zGraph_reach_proj h2⟩
  · rintro ⟨u, h1, h2⟩
    obtain ⟨k1, hk1⟩ := reach_walkN h1
    obtain ⟨k2, hk2⟩ := reach_walkN h2
    let a : ℤ := min t r - k1 - k2
    have e1 : a + ((t - a).toNat : ℕ) = t := by omega
    have e2 : a + ((r - a).toNat : ℕ) = r := by omega
    refine ⟨(u, a), ?_, ?_⟩
    · have := zGraph_reach_of_walkN (walkN_mono hl (show k1 ≤ (t - a).toNat by omega) hk1) a
      rwa [e1] at this
    · have := zGraph_reach_of_walkN (walkN_mono hl (show k2 ≤ (r - a).toNat by omega) hk2) a
      rwa [e2] at this

/-- **Theorem 1.9 (a)**, infinite past. -/
theorem rhoZ_eq (hl : ∀ v, E v v) (v : V) (t : Fin (m + 1)) :
    cntOf (relZ E m) (v, t) = (m + 1) * rho E v :=
  cntOf_window (fun v t w r => isRel_zGraph_iff hl v w t r) v t

/-- **Theorem 1.9 (b)**, infinite past: `s_ρ = 1/2 + |E_↑| / (2|E_S|)` for every `T ≥ 2`. -/
theorem sZ_eq (hl : ∀ v, E v v) [Nonempty V] (hm : 1 ≤ m) : sZ E m = statValue E := by
  unfold sZ
  rw [edgeZ_eq]
  exact sOf_window hl hm (fun v t w r => isRel_zGraph_iff hl v w t r)

end InfinitePast

/-! ### Source SCCs -/

/-- `x` lies in a source SCC: every ancestor of `x` is reachable from `x`. -/
def IsSource (E : V → V → Prop) (x : V) : Prop := ∀ y, Reach E y x → Reach E x y

/-- `x` is a node of a source SCC in `Src(v)`. -/
def SrcOf (E : V → V → Prop) (v x : V) : Prop := IsSource E x ∧ Reach E x v

/-- Every node is reached from a source SCC (finite graph). -/
lemma exists_source (v : V) : ∃ x, IsSource E x ∧ Reach E x v := by
  classical
  let anc : V → Finset V := fun x => univ.filter (fun y => Reach E y x)
  have hS : (univ.filter (fun x => Reach E x v)).Nonempty :=
    ⟨v, by simp [Relation.ReflTransGen.refl]⟩
  obtain ⟨x, hx, hmin⟩ := Finset.exists_min_image _ (fun x => (anc x).card) hS
  rw [mem_filter] at hx
  refine ⟨x, ?_, hx.2⟩
  intro y hyx
  have hsub : anc y ⊆ anc x := by
    intro z hz; simp only [anc, mem_filter, mem_univ, true_and] at hz ⊢; exact hz.trans hyx
  have hy : y ∈ univ.filter (fun x => Reach E x v) := by simp [hyx.trans hx.2]
  have heq : anc y = anc x := Finset.eq_of_subset_of_card_le hsub (hmin y hy)
  have : x ∈ anc y := by rw [heq]; simp [anc, Relation.ReflTransGen.refl]
  simpa [anc] using this

/-- **Theorem 1.9, source-SCC form**: for `i ⇝ j`, `rel_S(i) = rel_S(j)` iff
`Src(i) = Src(j)`. -/
theorem relSet_eq_iff_src {i j : V} (hij : Reach E i j) :
    relSet E i = relSet E j ↔ ∀ x, SrcOf E i x ↔ SrcOf E j x := by
  constructor
  · intro h x
    refine ⟨fun hx => ⟨hx.1, hx.2.trans hij⟩, fun hx => ⟨hx.1, ?_⟩⟩
    have hmem : x ∈ relSet E j := mem_relSet.mpr ⟨x, hx.2, Relation.ReflTransGen.refl⟩
    rw [← h, mem_relSet] at hmem
    obtain ⟨u, hui, hux⟩ := hmem
    exact (hx.1 u hux).trans hui
  · intro h
    apply le_antisymm (relSet_subset_of_reach hij)
    intro w hw
    rw [mem_relSet] at hw ⊢
    obtain ⟨u, huj, huw⟩ := hw
    obtain ⟨x, hxs, hxu⟩ := exists_source (E := E) u
    have : SrcOf E i x := (h x).mpr ⟨hxs, hxu.trans huj⟩
    exact ⟨x, this.2, hxu.trans huw⟩

/-- `E_↑ = {(i,j) ∈ E_S : Src(j) ≠ Src(i)}`. -/
theorem mem_upEdges_iff {e : V × V} :
    e ∈ upEdges E ↔ E e.1 e.2 ∧ ¬ (∀ x, SrcOf E e.1 x ↔ SrcOf E e.2 x) := by
  classical
  unfold upEdges
  rw [mem_filter, mem_edges]
  constructor
  · rintro ⟨h1, h2⟩
    exact ⟨h1, fun h => h2 ((relSet_eq_iff_src (Relation.ReflTransGen.single h1)).mpr h)⟩
  · rintro ⟨h1, h2⟩
    exact ⟨h1, fun h => h2 ((relSet_eq_iff_src (Relation.ReflTransGen.single h1)).mp h)⟩

/-! ### Corollary 1.10 -/

lemma weakConn_of_reach {a b : V} (h : Reach E a b) : WeakConn E a b := by
  induction h with
  | refl => exact Relation.ReflTransGen.refl
  | tail _ hbc ih => exact ih.tail (Or.inl hbc)

lemma weakConn_symm {a b : V} (h : WeakConn E a b) : WeakConn E b a := by
  induction h with
  | refl => exact Relation.ReflTransGen.refl
  | tail _ hbc ih => exact Relation.ReflTransGen.head (hbc.symm) ih

/-- **Corollary 1.10 (i)**, equivalent conditions: `E_↑ = ∅`; `Src` constant on weak components;
every weak component has exactly one source SCC (existence always holds, `exists_source`);
every two nodes in the same weak component have a common ancestor. -/
theorem upEdges_eq_empty_tfae :
    List.TFAE [upEdges E = ∅,
      ∀ i j, WeakConn E i j → ∀ x, SrcOf E i x ↔ SrcOf E j x,
      ∀ x y, IsSource E x → IsSource E y → WeakConn E x y → SameSCC E x y,
      ∀ i j, WeakConn E i j → ∃ u, Reach E u i ∧ Reach E u j] := by
  classical
  tfae_have 1 → 2 := by
    intro h i j hij
    have hedge : ∀ a b, E a b → ∀ x, SrcOf E a x ↔ SrcOf E b x := by
      intro a b hab
      by_contra hc
      have : (a, b) ∈ upEdges E := mem_upEdges_iff.mpr ⟨hab, hc⟩
      rw [h] at this
      exact Finset.notMem_empty _ this
    induction hij with
    | refl => exact fun _ => Iff.rfl
    | tail _ hbc ih =>
      intro x
      rcases hbc with hbc | hbc
      · exact (ih x).trans (hedge _ _ hbc x)
      · exact (ih x).trans (hedge _ _ hbc x).symm
  tfae_have 2 → 4 := by
    intro h i j hij
    obtain ⟨x, hxs, hxi⟩ := exists_source (E := E) i
    exact ⟨x, hxi, ((h i j hij x).mp ⟨hxs, hxi⟩).2⟩
  tfae_have 4 → 3 := by
    intro h x y hx hy hxy
    obtain ⟨u, hux, huy⟩ := h x y hxy
    have hxy' : Reach E x y := (hx u hux).trans huy
    exact ⟨hxy', hy x hxy'⟩
  tfae_have 3 → 1 := by
    intro h
    unfold upEdges
    rw [Finset.filter_eq_empty_iff]
    intro e he
    rw [mem_edges] at he
    rw [not_not, relSet_eq_iff_src (Relation.ReflTransGen.single he)]
    intro x
    refine ⟨fun hx => ⟨hx.1, hx.2.tail he⟩, fun hx => ⟨hx.1, ?_⟩⟩
    obtain ⟨x', hx's, hx'i⟩ := exists_source (E := E) e.1
    have hw : WeakConn E x x' :=
      (weakConn_of_reach hx.2).trans (weakConn_symm (weakConn_of_reach (hx'i.tail he)))
    exact (h x x' hx.1 hx's hw).1.trans hx'i
  tfae_finish

lemma statValue_eq_half_iff (hl : ∀ v, E v v) [Nonempty V] :
    statValue E = 1 / 2 ↔ upEdges E = ∅ := by
  have hE : (0 : ℝ) < (edges E).card := by exact_mod_cast edges_card_pos hl
  unfold statValue
  constructor
  · intro h
    have : ((upEdges E).card : ℝ) / (2 * (edges E).card) = 0 := by linarith
    rw [div_eq_zero_iff] at this
    rcases this with h1 | h1
    · exact Finset.card_eq_zero.mp (by exact_mod_cast h1)
    · exfalso; linarith
  · intro h; simp [h]

/-- **Corollary 1.10 (i)**: with burn-in `B ≥ p - 1` and any `T ≥ 2`, `s_ρ^{W,B} = 1/2` iff every
weakly connected component has exactly one source SCC. -/
theorem sBurn_eq_half_iff (hl : ∀ v, E v v) [Nonempty V] {B m : ℕ}
    (hB : Fintype.card V - 1 ≤ B) (hm : 1 ≤ m) :
    sBurn E B m = 1 / 2 ↔
      ∀ x y, IsSource E x → IsSource E y → WeakConn E x y → SameSCC E x y := by
  have key := (upEdges_eq_empty_tfae (E := E)).out 1 3
  rw [sBurn_eq hl hB hm, statValue_eq_half_iff hl]
  exact key

/-- **Corollary 1.10 (i)**, common-ancestor form. -/
theorem sBurn_eq_half_iff_common_anc (hl : ∀ v, E v v) [Nonempty V] {B m : ℕ}
    (hB : Fintype.card V - 1 ≤ B) (hm : 1 ≤ m) :
    sBurn E B m = 1 / 2 ↔ ∀ i j, WeakConn E i j → ∃ u, Reach E u i ∧ Reach E u j := by
  have key := (upEdges_eq_empty_tfae (E := E)).out 1 4
  rw [sBurn_eq hl hB hm, statValue_eq_half_iff hl]
  exact key

/-- **Corollary 1.10 (i)**, sharp burn-in form: with `B ≥ Δ` (largest finite distance) and any
`T ≥ 2`, `s_ρ^{W,B} = 1/2` iff every weakly connected component has exactly one source SCC. -/
theorem sBurn_eq_half_iff_maxFinDist (hl : ∀ v, E v v) [Nonempty V] {B m : ℕ}
    (hB : maxFinDist E ≤ B) (hm : 1 ≤ m) :
    sBurn E B m = 1 / 2 ↔
      ∀ x y, IsSource E x → IsSource E y → WeakConn E x y → SameSCC E x y := by
  have key := (upEdges_eq_empty_tfae (E := E)).out 1 3
  rw [sBurn_eq_maxFinDist hl hB hm, statValue_eq_half_iff hl]
  exact key

/-- **Corollary 1.10 (i)**, infinite-past version. -/
theorem sZ_eq_half_iff (hl : ∀ v, E v v) [Nonempty V] {m : ℕ} (hm : 1 ≤ m) :
    sZ E m = 1 / 2 ↔
      ∀ x y, IsSource E x → IsSource E y → WeakConn E x y → SameSCC E x y := by
  have key := (upEdges_eq_empty_tfae (E := E)).out 1 3
  rw [sZ_eq hl hm, statValue_eq_half_iff hl]
  exact key

/-! #### Corollary 1.10 (ii): graph classes with value exactly `1/2` -/

/-- Last-step decomposition of a path. -/
lemma reach_last {u b : V} (h : Reach E u b) : u = b ∨ ∃ k, k ≠ b ∧ Reach E u k ∧ E k b := by
  induction h with
  | refl => exact Or.inl rfl
  | @tail c d hc hcd ih =>
    by_cases hcd' : c = d
    · subst hcd'; exact ih
    · exact Or.inr ⟨c, hcd', hc, hcd⟩

/-- First-step decomposition of a path. -/
lemma reach_first {x b : V} (h : Reach E x b) : x = b ∨ ∃ y, y ≠ x ∧ E x y ∧ Reach E y b := by
  induction h using Relation.ReflTransGen.head_induction_on with
  | refl => exact Or.inl rfl
  | @head a c hac hcb ih =>
    by_cases hac' : c = a
    · subst hac'; exact ih
    · exact Or.inr ⟨c, hac', hac, hcb⟩

/-- Strongly connected weak components (Reisach et al.'s condition) satisfy (i). -/
theorem stat_half_of_strong (hl : ∀ v, E v v) [Nonempty V] (h : ∀ i j, Reach E i j → Reach E j i) :
    statValue E = 1 / 2 := by
  have key := (upEdges_eq_empty_tfae (E := E)).out 1 3
  rw [statValue_eq_half_iff hl, key]
  have h4 := ((crossEdges_eq_empty_tfae (E := E)).out 3 4).mp h
  exact fun x y _ _ hxy => h4 x y hxy

/-- Out-forest: every node has at most one parent other than itself. -/
def OutForest (E : V → V → Prop) : Prop := ∀ v a b, E a v → E b v → a ≠ v → b ≠ v → a = b

/-- Out-trees, out-forests, chains and forks satisfy (i) and give exactly `1/2`
(no acyclicity needed). -/
theorem stat_half_of_outForest (hl : ∀ v, E v v) [Nonempty V] (hf : OutForest E) :
    statValue E = 1 / 2 := by
  have key := (upEdges_eq_empty_tfae (E := E)).out 1 4
  rw [statValue_eq_half_iff hl, key]
  intro i j hij
  induction hij with
  | refl => exact ⟨i, Relation.ReflTransGen.refl, Relation.ReflTransGen.refl⟩
  | @tail b c hib hbc ih =>
    obtain ⟨u, hui, hub⟩ := ih
    rcases hbc with hbc | hcb
    · exact ⟨u, hui, hub.tail hbc⟩
    · by_cases hcb' : c = b
      · subst hcb'; exact ⟨u, hui, hub⟩
      · rcases reach_last hub with rfl | ⟨k, hkb, huk, hk⟩
        · exact ⟨c, (Relation.ReflTransGen.single hcb).trans hui, Relation.ReflTransGen.refl⟩
        · have : k = c := hf _ _ _ hk hcb hkb hcb'
          subst this
          exact ⟨u, hui, huk⟩

/-! #### Corollary 1.10 (iii): in-forests -/

/-- In-forest: acyclic apart from self-loops, and every node has at most one child other than
itself (river networks). -/
def InForest (E : V → V → Prop) : Prop :=
  (∀ v a b, E v a → E v b → a ≠ v → b ≠ v → a = b) ∧ (∀ i j, Reach E i j → Reach E j i → i = j)

lemma inForest_linear (hf : InForest E) {x a b : V} (ha : Reach E x a) (hb : Reach E x b) :
    Reach E a b ∨ Reach E b a := by
  induction ha using Relation.ReflTransGen.head_induction_on generalizing b with
  | refl => exact Or.inl hb
  | @head x y hxy hya ih =>
    rcases reach_first hb with rfl | ⟨z, hzx, hxz, hzb⟩
    · exact Or.inr ((Relation.ReflTransGen.single hxy).trans hya)
    · by_cases hyx : y = x
      · subst hyx
        exact ih ((Relation.ReflTransGen.single hxz).trans hzb)
      · have : z = y := hf.1 _ _ _ hxz hxy hzx hyx
        subst this
        exact ih hzb

/-- In an in-forest, `rel_S(v) = anc_S(v) ∪ desc_S(v)`. -/
theorem inForest_isRel_iff (hf : InForest E) {v w : V} :
    IsRel E v w ↔ Reach E w v ∨ Reach E v w := by
  constructor
  · rintro ⟨u, huv, huw⟩
    exact (inForest_linear hf huv huw).symm
  · rintro (h | h)
    · exact ⟨w, h, Relation.ReflTransGen.refl⟩
    · exact ⟨v, Relation.ReflTransGen.refl, h⟩

/-- In an in-forest, a non-self-loop edge `(i,j)` is in `E_↑` iff `j` is a confluence
(has a parent other than `i` and itself). -/
theorem inForest_up_iff (hf : InForest E) {i j : V} (hij : E i j) (hne : i ≠ j) :
    relSet E i ≠ relSet E j ↔ ∃ k, E k j ∧ k ≠ i ∧ k ≠ j := by
  rw [Ne, relSet_eq_iff_src (Relation.ReflTransGen.single hij)]
  constructor
  · intro h
    have : ∃ x, SrcOf E j x ∧ ¬ SrcOf E i x := by
      by_contra hc
      push Not at hc
      exact h (fun x => ⟨fun hx => ⟨hx.1, hx.2.tail hij⟩, hc x⟩)
    obtain ⟨x, ⟨hxs, hxj⟩, hxi⟩ := this
    rcases reach_last hxj with rfl | ⟨k, hkj, hxk, hk⟩
    · exact absurd ⟨hxs, hxs i (Relation.ReflTransGen.single hij)⟩ hxi
    · refine ⟨k, hk, ?_, hkj⟩
      rintro rfl
      exact hxi ⟨hxs, hxk⟩
  · rintro ⟨k, hkj, hki, hkj'⟩ h
    obtain ⟨x, hxs, hxk⟩ := exists_source (E := E) k
    have hxj : SrcOf E j x := ⟨hxs, hxk.tail hkj⟩
    have hxi : Reach E x i := ((h x).mpr hxj).2
    rcases inForest_linear hf hxi hxk with hik | hki'
    · rcases reach_first hik with rfl | ⟨y, hyi, hiy, hyk⟩
      · exact hki rfl
      · have : y = j := hf.1 _ _ _ hiy hij hyi (Ne.symm hne)
        subst this
        exact hkj' (hf.2 _ _ (Relation.ReflTransGen.single hkj) hyk)
    · rcases reach_first hki' with rfl | ⟨y, hyk, hky, hyi⟩
      · exact hki rfl
      · have : y = j := hf.1 _ _ _ hky hkj hyk (Ne.symm hkj')
        subst this
        exact hne (hf.2 _ _ (Relation.ReflTransGen.single hij) hyi)

open Classical in
/-- Edges into confluences: non-self-loop edges `(i,j)` such that `j` has another parent. -/
noncomputable def confluenceEdges (E : V → V → Prop) : Finset (V × V) :=
  (edges E).filter (fun e => e.1 ≠ e.2 ∧ ∃ k, E k e.2 ∧ k ≠ e.1 ∧ k ≠ e.2)

/-- **Corollary 1.10 (iii)**: for an in-forest,
`s_ρ^{W,B} = 1/2 + #{edges into confluences} / (2|E_S|)`. -/
theorem statValue_inForest (hf : InForest E) :
    statValue E = 1 / 2 + ((confluenceEdges E).card : ℝ) / (2 * (edges E).card) := by
  classical
  have : upEdges E = confluenceEdges E := by
    unfold upEdges confluenceEdges
    apply Finset.filter_congr
    intro e he
    rw [mem_edges] at he
    by_cases hne : e.1 = e.2
    · have : relSet E e.1 = relSet E e.2 := by rw [hne]
      simp [hne]
    · rw [inForest_up_iff hf he hne]
      simp [hne]
  rw [statValue, this]

/-- **Corollary 1.10 (iii)**: an in-forest has `s_ρ > 1/2` iff it contains a confluence. -/
theorem statValue_inForest_gt_iff (hl : ∀ v, E v v) [Nonempty V] (hf : InForest E) :
    1 / 2 < statValue E ↔ ∃ i j k, E i j ∧ E k j ∧ i ≠ j ∧ k ≠ i ∧ k ≠ j := by
  classical
  have hE : (0 : ℝ) < (edges E).card := by exact_mod_cast edges_card_pos hl
  rw [statValue_inForest hf]
  constructor
  · intro h
    have hpos : (0 : ℝ) < (confluenceEdges E).card := by
      by_contra hc
      push Not at hc
      have : ((confluenceEdges E).card : ℝ) / (2 * (edges E).card) ≤ 0 :=
        div_nonpos_of_nonpos_of_nonneg hc (by positivity)
      linarith
    obtain ⟨e, he⟩ := Finset.card_pos.mp (by exact_mod_cast hpos)
    unfold confluenceEdges at he
    rw [mem_filter, mem_edges] at he
    obtain ⟨h1, h2, k, hk1, hk2, hk3⟩ := he
    exact ⟨e.1, e.2, k, h1, hk1, h2, hk2, hk3⟩
  · rintro ⟨i, j, k, hij, hkj, hne, hki, hkj'⟩
    have hmem : (i, j) ∈ confluenceEdges E := by
      unfold confluenceEdges
      rw [mem_filter, mem_edges]
      exact ⟨hij, hne, k, hkj, hki, hkj'⟩
    have : (0 : ℝ) < (confluenceEdges E).card := by exact_mod_cast Finset.card_pos.mpr ⟨_, hmem⟩
    have : 0 < ((confluenceEdges E).card : ℝ) / (2 * (edges E).card) := by positivity
    linarith

end RelSort
