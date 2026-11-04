/-
X2 (research/theory2/ext/EXT.md, Theorem X2.1): the exact value of every edge of `G_T`, and of
`s_ρ(G_T)`, for an ARBITRARY summary graph `E` (no self-loop hypothesis at all).

`Fresh E u` : `u` has no incoming edge in `E` (in-degree 0, self-loop included). Such a node is a
root of `G_T` at every time slice, not only at slice 1.

`StrictNL E s i j` (paper time `t = s + 1`):
  (∃ u, WalkN E (s+1) u j ∧ ¬ WalkN E s u i)                              -- the root `u(1)`
  ∨ (∃ u, Fresh E u ∧ ∃ m ≤ s, WalkN E (m+1) u j ∧ ¬ WalkN E m u i)       -- a fresh root `u(s-m)`
-/
import RelSort.Limit

open Finset

set_option linter.unusedSectionVars false

namespace RelSort

variable {V : Type*} [Fintype V] {E : V → V → Prop}

/-- No incoming edge at all (self-loop included). -/
def Fresh (E : V → V → Prop) (u : V) : Prop := ∀ w, ¬ E w u

lemma fresh_walk {u x : V} {k : ℕ} (hf : Fresh E u) (h : WalkN E k x u) : k = 0 ∧ x = u := by
  cases k with
  | zero => exact ⟨rfl, h⟩
  | succ k => obtain ⟨w, _, hw⟩ := h; exact absurd hw (hf w)

/-- Strictness condition without any self-loop assumption (`s = t - 1`). -/
def StrictNL (E : V → V → Prop) (s : ℕ) (i j : V) : Prop :=
  (∃ u, WalkN E (s + 1) u j ∧ ¬ WalkN E s u i) ∨
  (∃ u, Fresh E u ∧ ∃ m, m ≤ s ∧ WalkN E (m + 1) u j ∧ ¬ WalkN E m u i)

/-- Relatives in `G_T` without loops: a common ancestor `u(a)` (Lean slice `a`). -/
theorem isRel_unrolled_iff_NL {n : ℕ} {v w : V} {t r : Fin (n + 1)} :
    IsRel (unrolled E n) (v, t) (w, r) ↔
      ∃ (u : V) (a : ℕ), a ≤ t ∧ a ≤ r ∧ WalkN E (t - a) u v ∧ WalkN E (r - a) u w := by
  constructor
  · rintro ⟨⟨u, a⟩, h1, h2⟩
    rw [unrolled_reach_iff] at h1 h2
    exact ⟨u, a, h1.1, h2.1, h1.2, h2.2⟩
  · rintro ⟨u, a, ha1, ha2, h1, h2⟩
    have ha : a < n + 1 := by have := t.isLt; omega
    refine ⟨(u, ⟨a, ha⟩), ?_, ?_⟩
    · rw [unrolled_reach_iff]; exact ⟨ha1, h1⟩
    · rw [unrolled_reach_iff]; exact ⟨ha2, h2⟩

/-- The key reduction: a common ancestor of `j(s+1)` and `w(r)` can be pushed back to slice 0
(a non-fresh node has a parent) or kept at a fresh node. -/
lemma reduce_ancestor {i j w : V} (hij : E i j) {s r : ℕ} (hn : ¬ StrictNL E s i j) :
    ∀ a : ℕ, ∀ x : V, a ≤ s + 1 → a ≤ r → WalkN E (s + 1 - a) x j → WalkN E (r - a) x w →
      ∃ (x' : V) (a' : ℕ), a' ≤ s ∧ a' ≤ r ∧ WalkN E (s - a') x' i ∧ WalkN E (r - a') x' w := by
  intro a
  induction a with
  | zero =>
    intro x _ _ h1 h2
    refine ⟨x, 0, Nat.zero_le _, Nat.zero_le _, ?_, by simpa using h2⟩
    by_contra hc
    exact hn (Or.inl ⟨x, by simpa using h1, by simpa using hc⟩)
  | succ a0 ih =>
    intro x ha1 ha2 h1 h2
    by_cases hf : Fresh E x
    · have has : a0 + 1 ≤ s := by
        by_contra hc
        have e : s + 1 - (a0 + 1) = 0 := by omega
        rw [e, walkN_zero] at h1
        subst h1
        exact hf i hij
      refine ⟨x, a0 + 1, has, ha2, ?_, h2⟩
      by_contra hc
      refine hn (Or.inr ⟨x, hf, s - (a0 + 1), by omega, ?_, hc⟩)
      have e : s - (a0 + 1) + 1 = s + 1 - (a0 + 1) := by omega
      rw [e]; exact h1
    · simp only [Fresh, not_forall, not_not] at hf
      obtain ⟨y, hy⟩ := hf
      apply ih y (by omega) (by omega)
      · have e : s + 1 - a0 = (s + 1 - (a0 + 1)) + 1 := by omega
        rw [e, walkN_succ']; exact ⟨x, hy, h1⟩
      · have e : r - a0 = (r - (a0 + 1)) + 1 := by omega
        rw [e, walkN_succ']; exact ⟨x, hy, h2⟩

lemma relSet_unrolled_eq_iff_NL {n : ℕ} {i j : V} (hij : E i j) (s : Fin n) :
    relSet (unrolled E n) (i, s.castSucc) = relSet (unrolled E n) (j, s.succ) ↔
      ¬ StrictNL E s i j := by
  constructor
  · intro heq hS
    rcases hS with ⟨u, hu1, hu2⟩ | ⟨u, hf, m, hm, hu1, hu2⟩
    · have hmem : (u, (0 : Fin (n + 1))) ∈ relSet (unrolled E n) (j, s.succ) := by
        rw [mem_relSet, isRel_unrolled_iff_NL]
        exact ⟨u, 0, by simp, by simp, by simpa using hu1, by simp⟩
      rw [← heq, mem_relSet, isRel_unrolled_iff_NL] at hmem
      obtain ⟨u', a, ha1, ha2, h1, h2⟩ := hmem
      have ha : a = 0 := by simpa using ha2
      subst ha
      simp only [Fin.val_zero, Nat.sub_zero, walkN_zero] at h2
      subst h2
      exact hu2 (by simpa using h1)
    · have hsn : s < n := s.isLt
      have hlt : s - m < n + 1 := by omega
      have hmem : (u, (⟨s - m, hlt⟩ : Fin (n + 1))) ∈ relSet (unrolled E n) (j, s.succ) := by
        rw [mem_relSet, isRel_unrolled_iff_NL]
        refine ⟨u, s - m, by simp; omega, le_refl _, ?_, by simp⟩
        have e : ((s.succ : Fin (n + 1)) : ℕ) - (s - m) = m + 1 := by simp; omega
        rw [e]; exact hu1
      rw [← heq, mem_relSet, isRel_unrolled_iff_NL] at hmem
      obtain ⟨u', a, ha1, ha2, h1, h2⟩ := hmem
      simp only at ha2 h2
      have hk := fresh_walk hf (x := u') (k := s - m - a) h2
      have hu' : u' = u := hk.2
      subst hu'
      have ha : a = s - m := by omega
      subst ha
      apply hu2
      have e : ((s.castSucc : Fin (n + 1)) : ℕ) - (s - m) = m := by simp; omega
      rw [e] at h1; exact h1
  · intro hnot
    apply le_antisymm (relSet_subset_of_edge (unrolled_edge hij s))
    rintro ⟨w, r⟩ hw
    rw [mem_relSet, isRel_unrolled_iff_NL] at hw ⊢
    obtain ⟨x, a, ha1, ha2, h1, h2⟩ := hw
    have hs : ((s.succ : Fin (n + 1)) : ℕ) = s + 1 := by simp
    have hc : ((s.castSucc : Fin (n + 1)) : ℕ) = s := by simp
    rw [hs] at ha1 h1
    obtain ⟨x', a', b1, b2, g1, g2⟩ :=
      reduce_ancestor hij hnot a x ha1 ha2 h1 h2
    exact ⟨x', a', by rw [hc]; exact b1, b2, by rw [hc]; exact g1, g2⟩

open Classical in
/-- **Theorem X2.1 (edge value without any loop assumption).** For the edge `(i(t), j(t+1))`,
`t = s + 1`, of `G_T` the score is `1` if `StrictNL E s i j` and `1/2` otherwise. -/
theorem edge_score_NL {n : ℕ} {i j : V} (hij : E i j) (s : Fin n) :
    score (rho (unrolled E n) (i, s.castSucc)) (rho (unrolled E n) (j, s.succ)) =
      if StrictNL E s i j then 1 else 1 / 2 := by
  classical
  rw [rho_eq_card, rho_eq_card, score_of_subset (relSet_subset_of_edge (unrolled_edge hij s))]
  by_cases h : StrictNL E s i j
  · have : ¬ relSet (unrolled E n) (i, s.castSucc) = relSet (unrolled E n) (j, s.succ) := by
      rw [relSet_unrolled_eq_iff_NL hij]; exact not_not.mpr h
    simp only [this, h, ↓reduceIte]
  · have : relSet (unrolled E n) (i, s.castSucc) = relSet (unrolled E n) (j, s.succ) := by
      rw [relSet_unrolled_eq_iff_NL hij]; exact h
    simp only [this, h, ↓reduceIte]

open Classical in
/-- Number of copies of `(i,j)` with score `1` (no loop assumption). -/
noncomputable def numStrictNL (E : V → V → Prop) (n : ℕ) (i j : V) : ℕ :=
  (univ.filter (fun s : Fin n => StrictNL E s i j)).card

lemma sum_layer_score_NL {n : ℕ} {i j : V} (hij : E i j) :
    ∑ s : Fin n, score (rho (unrolled E n) (i, s.castSucc)) (rho (unrolled E n) (j, s.succ)) =
      (n : ℝ) / 2 + (numStrictNL E n i j : ℝ) / 2 := by
  classical
  have : ∀ s : Fin n, score (rho (unrolled E n) (i, s.castSucc))
      (rho (unrolled E n) (j, s.succ)) = 1 / 2 + 1 / 2 * (if StrictNL E s i j then 1 else 0) := by
    intro s
    rw [edge_score_NL hij s]
    split_ifs <;> norm_num
  simp_rw [this]
  rw [sum_add_distrib, ← mul_sum, sum_boole]
  simp [numStrictNL]
  ring

/-- **Theorem X2.1 (exact finite-`T` value, arbitrary summary graph).**
`s_ρ(G_T) = 1/2 + Σ_{(i,j) ∈ E_S} N^{NL}_T(i,j) / (2 (T-1) |E_S|)`, `T = n + 1 ≥ 2`,
for every summary graph with at least one edge. -/
theorem sRho_unrolled_eq_NL (hE : 0 < (edges E).card) {n : ℕ} (hn : 1 ≤ n) :
    sRho (unrolled E n) =
      1 / 2 + (∑ e ∈ edges E, (numStrictNL E n e.1 e.2 : ℝ)) / (2 * n * (edges E).card) := by
  have hE' : (0 : ℝ) < (edges E).card := by exact_mod_cast hE
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  unfold sRho sOf
  rw [sum_edges_unrolled, card_edges_unrolled]
  rw [sum_congr rfl (fun e he => sum_layer_score_NL (mem_edges.mp he))]
  rw [sum_add_distrib, sum_const, ← sum_div]
  simp only [nsmul_eq_mul]
  push_cast
  field_simp

/-- Hence `s_ρ(G_T) ≥ 1/2`: relatives never favour the wrong direction. -/
theorem sRho_unrolled_ge_half_NL (hE : 0 < (edges E).card) {n : ℕ} (hn : 1 ≤ n) :
    1 / 2 ≤ sRho (unrolled E n) := by
  rw [sRho_unrolled_eq_NL hE hn]
  have : 0 ≤ (∑ e ∈ edges E, (numStrictNL E n e.1 e.2 : ℝ)) / (2 * n * (edges E).card) := by
    positivity
  linarith

/-- With a self-loop at the head, a crossing edge is strict at every layer (this is where the
loop at `j` is used; for a head without closed walks it can fail, see `edgeNoLoops_sRho`). -/
lemma strictNL_of_closed_walk {i j : V} {s : ℕ} (hc : WalkN E (s + 1) j j)
    (hnr : ¬ Reach E j i) : StrictNL E s i j :=
  Or.inl ⟨j, hc, fun h => hnr (walkN_reach h)⟩

/-- An edge `i → j` where `j` has a fresh parent `u ≠ i` is strict at every layer. -/
lemma strictNL_of_fresh_parent {i j u : V} (hu : Fresh E u) (huj : E u j) (hne : u ≠ i)
    (s : ℕ) : StrictNL E s i j := by
  refine Or.inr ⟨u, hu, 0, Nat.zero_le _, ?_, ?_⟩
  · exact ⟨u, rfl, huj⟩
  · intro h
    simp only [walkN_zero] at h
    exact hne h


/-- If every layer of every edge is strict, `s_ρ(G_T) = 1`. -/
theorem sRho_unrolled_eq_one_NL (hE : 0 < (edges E).card) {n : ℕ} (hn : 1 ≤ n)
    (h : ∀ i j, E i j → ∀ s : ℕ, StrictNL E s i j) : sRho (unrolled E n) = 1 := by
  classical
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  have hE' : (0 : ℝ) < (edges E).card := by exact_mod_cast hE
  have hnum : ∀ e ∈ edges E, (numStrictNL E n e.1 e.2 : ℝ) = n := by
    intro e he
    have : ∀ s : Fin n, StrictNL E s e.1 e.2 := fun s => h _ _ (mem_edges.mp he) s
    simp [numStrictNL, this]
  rw [sRho_unrolled_eq_NL hE hn, sum_congr rfl hnum]
  simp only [sum_const, nsmul_eq_mul]
  field_simp
  ring

/-- Collider `0 → 2 ← 1` WITHOUT self-loops: `s_ρ(G_T) = 1` for every `T ≥ 2`
(with self-loops the limit is `7/10`). -/
def colliderNL : Fin 3 → Fin 3 → Prop := fun a b => (a = 0 ∨ a = 1) ∧ b = 2

instance : DecidableRel colliderNL := fun a b => by unfold colliderNL; infer_instance

theorem colliderNL_sRho {n : ℕ} (hn : 1 ≤ n) : sRho (unrolled colliderNL n) = 1 := by
  classical
  have hE : 0 < (edges colliderNL).card := by
    rw [show edges colliderNL = univ.filter (fun p : Fin 3 × Fin 3 => colliderNL p.1 p.2) by
      ext p; simp]
    decide
  refine sRho_unrolled_eq_one_NL hE hn ?_
  intro i j hij s
  have hfresh : ∀ u : Fin 3, u = 0 ∨ u = 1 → Fresh colliderNL u := by
    intro u hu w hw
    unfold colliderNL at hw
    omega
  rcases hij with ⟨hi, hj⟩
  rcases hi with rfl | rfl
  · exact strictNL_of_fresh_parent (u := 1) (hfresh 1 (Or.inr rfl)) ⟨Or.inr rfl, hj⟩ (by decide) s
  · exact strictNL_of_fresh_parent (u := 0) (hfresh 0 (Or.inl rfl)) ⟨Or.inl rfl, hj⟩ (by decide) s

/-- The edge `0 → 1` with a self-loop only at the head `1`: `s_ρ(G_T) = 1` for every `T ≥ 2`
(loops at both nodes: limit `2/3`; no loops: `1/2`; loop only at the tail: `1/2`). -/
def edgeLoopHead : Fin 2 → Fin 2 → Prop := fun a b => b = 1 ∧ (a = 0 ∨ a = 1)

instance : DecidableRel edgeLoopHead := fun a b => by unfold edgeLoopHead; infer_instance

theorem edgeLoopHead_sRho {n : ℕ} (hn : 1 ≤ n) : sRho (unrolled edgeLoopHead n) = 1 := by
  classical
  have hE : 0 < (edges edgeLoopHead).card := by
    rw [show edges edgeLoopHead = univ.filter (fun p : Fin 2 × Fin 2 => edgeLoopHead p.1 p.2) by
      ext p; simp]
    decide
  refine sRho_unrolled_eq_one_NL hE hn ?_
  intro i j hij s
  have hf0 : Fresh edgeLoopHead 0 := by
    intro w hw; unfold edgeLoopHead at hw; omega
  rcases hij with ⟨rfl, hi⟩
  rcases hi with rfl | rfl
  · -- crossing edge 0 → 1: closed walk at 1 (the loop) and 1 does not reach 0
    refine strictNL_of_closed_walk (s := s) ?_ ?_
    · have : ∀ k : ℕ, WalkN edgeLoopHead k (1 : Fin 2) 1 := by
        intro k
        induction k with
        | zero => rfl
        | succ k ih => exact ⟨1, ih, ⟨rfl, Or.inr rfl⟩⟩
      exact this _
    · intro hr
      have : ∀ b : Fin 2, Relation.ReflTransGen edgeLoopHead (1 : Fin 2) b → b = 1 := by
        intro b hb
        induction hb with
        | refl => rfl
        | tail _ hbc _ => exact hbc.1
      have := this _ hr
      exact absurd this (by decide)
  · exact strictNL_of_fresh_parent (u := 0) hf0 ⟨rfl, Or.inl rfl⟩ (by decide) s

end RelSort
