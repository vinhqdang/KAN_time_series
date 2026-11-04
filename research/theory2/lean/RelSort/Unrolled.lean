/-
The time-unrolled DAG `G_T` (Reisach et al., Sec. 5) and THEORY.md Lemma 1.1, Lemma 1.2
(in the form used), Theorem 1.3.

Indexing convention: `G_T` with `T = n + 1` slices has node type `V × Fin (n+1)`;
the Lean slice `t : Fin (n+1)` is the paper's time `t + 1`. Edges are `(i, t) → (j, t+1)` for
every summary edge `E i j` (self-loops included), so `|E_T| = n · |E_S| = (T-1)|E_S|`.
-/
import RelSort.Walk

open Finset

set_option linter.unusedSectionVars false

namespace RelSort

variable {V : Type*} (E : V → V → Prop)

/-- The unrolled DAG `G_T`, `T = n + 1`. -/
def unrolled (n : ℕ) (a b : V × Fin (n + 1)) : Prop := E a.1 b.1 ∧ (b.2 : ℕ) = a.2 + 1

variable {E}

variable [Fintype V]

/-- **Lemma 1.1** (ancestors in `G_T`): `u(s) ∈ anc(v(t))` iff `s ≤ t` and there is a walk of
length `t - s` from `u` to `v` in `G_S` (equivalently, with self-loops, `d(u,v) ≤ t - s`,
see `unrolled_reach_iff_dist`). -/
theorem unrolled_reach_iff {n : ℕ} {u v : V} {s t : Fin (n + 1)} :
    Relation.ReflTransGen (unrolled E n) (u, s) (v, t) ↔ (s : ℕ) ≤ t ∧ WalkN E (t - s) u v := by
  constructor
  · intro h
    suffices H : ∀ b : V × Fin (n + 1), Relation.ReflTransGen (unrolled E n) (u, s) b →
        (s : ℕ) ≤ b.2 ∧ WalkN E (b.2 - s) u b.1 from H _ h
    intro b hb
    induction hb with
    | refl => simp
    | @tail b c _ hbc ih =>
      obtain ⟨hle, hw⟩ := ih
      obtain ⟨he, ht⟩ := hbc
      refine ⟨by omega, ?_⟩
      have e : (c.2 : ℕ) - s = ((b.2 : ℕ) - s) + 1 := by omega
      rw [e]
      exact ⟨_, hw, he⟩
  · rintro ⟨hle, hw⟩
    suffices H : ∀ k (v : V) (t : Fin (n + 1)), (s : ℕ) + k = t → WalkN E k u v →
        Relation.ReflTransGen (unrolled E n) (u, s) (v, t) from
      H (t - s) v t (by omega) hw
    intro k
    induction k with
    | zero =>
      intro v t hst hw
      rw [walkN_zero] at hw
      subst hw
      have : s = t := Fin.ext (by omega)
      subst this
      exact Relation.ReflTransGen.refl
    | succ k ih =>
      intro v t hst hw
      obtain ⟨w, hw, hwv⟩ := hw
      have hr : (s : ℕ) + k < n + 1 := by omega
      exact (ih w ⟨s + k, hr⟩ rfl hw).tail ⟨hwv, by simp; omega⟩

theorem unrolled_reach_iff_dist (hl : ∀ v, E v v) {n : ℕ} {u v : V} {s t : Fin (n + 1)} :
    Relation.ReflTransGen (unrolled E n) (u, s) (v, t) ↔
      (s : ℕ) ≤ t ∧ dist E u v ≤ ((t - s : ℕ) : ℕ∞) := by
  rw [unrolled_reach_iff, dist_le_iff hl]

/-- **Lemma 1.2 consequence** (relatives in `G_T`): `w(r) ∈ rel(v(t))` iff some `u` has walks of
lengths `t` and `r` (Lean slices, i.e. paper's `t-1`, `r-1`) to `v` and `w`; i.e. iff some
root `u(1)` is a common ancestor. -/
theorem isRel_unrolled_iff (hl : ∀ v, E v v) {n : ℕ} {v w : V} {t r : Fin (n + 1)} :
    IsRel (unrolled E n) (v, t) (w, r) ↔ ∃ u, WalkN E t u v ∧ WalkN E r u w := by
  constructor
  · rintro ⟨⟨u, a⟩, h1, h2⟩
    rw [unrolled_reach_iff] at h1 h2
    exact ⟨u, walkN_mono hl (Nat.sub_le _ _) h1.2, walkN_mono hl (Nat.sub_le _ _) h2.2⟩
  · rintro ⟨u, h1, h2⟩
    refine ⟨(u, 0), ?_, ?_⟩
    · rw [unrolled_reach_iff]; simpa using h1
    · rw [unrolled_reach_iff]; simpa using h2

/-- The condition of Theorem 1.3 in walk form, for the edge `(i(t), j(t+1))` with paper time
`t = s + 1`: some `u` has `d(u,j) ≤ t` and `d(u,i) > t - 1`. -/
def StrictCond (E : V → V → Prop) (s : ℕ) (i j : V) : Prop :=
  ∃ u, WalkN E (s + 1) u j ∧ ¬ WalkN E s u i

/-- The same condition in the paper's distance form `∃ u, d(u,j) ≤ t ≤ d(u,i)`, `t = s + 1`. -/
theorem strictCond_iff_dist (hl : ∀ v, E v v) {s : ℕ} {i j : V} :
    StrictCond E s i j ↔
      ∃ u, dist E u j ≤ ((s + 1 : ℕ) : ℕ∞) ∧ ((s + 1 : ℕ) : ℕ∞) ≤ dist E u i := by
  unfold StrictCond
  refine exists_congr fun u => and_congr (dist_le_iff hl).symm ?_
  rw [← dist_le_iff hl, not_le, Nat.cast_add_one, ENat.add_one_le_iff (ENat.natCast_ne_top s)]

/-- The edge `(i(t), j(t+1))` of `G_T`, written with Lean indices. -/
lemma unrolled_edge {n : ℕ} {i j : V} (hij : E i j) (s : Fin n) :
    unrolled E n (i, s.castSucc) (j, s.succ) := ⟨hij, by simp⟩

lemma relSet_unrolled_eq_iff (hl : ∀ v, E v v) {n : ℕ} {i j : V} (hij : E i j) (s : Fin n) :
    relSet (unrolled E n) (i, s.castSucc) = relSet (unrolled E n) (j, s.succ) ↔
      ¬ StrictCond E s i j := by
  constructor
  · rintro heq ⟨u, hu1, hu2⟩
    have hmem : (u, (0 : Fin (n + 1))) ∈ relSet (unrolled E n) (j, s.succ) := by
      rw [mem_relSet, isRel_unrolled_iff hl]
      exact ⟨u, by simpa using hu1, rfl⟩
    rw [← heq, mem_relSet, isRel_unrolled_iff hl] at hmem
    obtain ⟨u', h1, h2⟩ := hmem
    have : u' = u := h2
    subst this
    exact hu2 (by simpa using h1)
  · intro hnot
    apply le_antisymm (relSet_subset_of_edge (unrolled_edge hij s))
    rintro ⟨w, r⟩ hw
    rw [mem_relSet, isRel_unrolled_iff hl] at hw ⊢
    obtain ⟨u, h1, h2⟩ := hw
    refine ⟨u, ?_, h2⟩
    by_contra hc
    exact hnot ⟨u, by simpa using h1, by simpa using hc⟩

open Classical in
/-- **Theorem 1.3** (exact value of every edge of `G_T`, every `T`): for the edge
`(i(t), j(t+1))`, `t = s + 1`, the score is `1` if `∃ u, d(u,j) ≤ t ≤ d(u,i)` and `1/2`
otherwise; it never is `0`, and it does not depend on `T`. -/
theorem edge_score (hl : ∀ v, E v v) {n : ℕ} {i j : V} (hij : E i j) (s : Fin n) :
    score (rho (unrolled E n) (i, s.castSucc)) (rho (unrolled E n) (j, s.succ)) =
      if StrictCond E s i j then 1 else 1 / 2 := by
  classical
  rw [rho_eq_card, rho_eq_card, score_of_subset (relSet_subset_of_edge (unrolled_edge hij s))]
  by_cases h : StrictCond E s i j
  · have : ¬ relSet (unrolled E n) (i, s.castSucc) = relSet (unrolled E n) (j, s.succ) := by
      rw [relSet_unrolled_eq_iff hl hij]; exact not_not.mpr h
    simp only [this, h, ↓reduceIte]
  · have : relSet (unrolled E n) (i, s.castSucc) = relSet (unrolled E n) (j, s.succ) := by
      rw [relSet_unrolled_eq_iff hl hij]; exact h
    simp only [this, h, ↓reduceIte]

open Classical in
/-- **Theorem 1.3**, in the paper's distance form. -/
theorem edge_score_dist (hl : ∀ v, E v v) {n : ℕ} {i j : V} (hij : E i j) (s : Fin n) :
    score (rho (unrolled E n) (i, s.castSucc)) (rho (unrolled E n) (j, s.succ)) =
      if (∃ u, dist E u j ≤ ((s + 1 : ℕ) : ℕ∞) ∧ ((s + 1 : ℕ) : ℕ∞) ≤ dist E u i)
      then 1 else 1 / 2 := by
  classical
  rw [edge_score hl hij s]
  by_cases h : StrictCond E s i j
  · have h' := (strictCond_iff_dist hl).mp h
    simp only [h, h', ↓reduceIte]
  · have h' : ¬ ∃ u, dist E u j ≤ ((s + 1 : ℕ) : ℕ∞) ∧ ((s + 1 : ℕ) : ℕ∞) ≤ dist E u i :=
      fun h' => h ((strictCond_iff_dist hl).mpr h')
    simp only [h, h', ↓reduceIte]

/-! ### Edge set of `G_T` -/

/-- Lean-index form of an edge copy: summary edge `e` at layer `s`. -/
def edgeCopy {n : ℕ} (q : (V × V) × Fin n) : (V × Fin (n + 1)) × (V × Fin (n + 1)) :=
  ((q.1.1, q.2.castSucc), (q.1.2, q.2.succ))

lemma edgeCopy_injective {n : ℕ} : Function.Injective (edgeCopy (V := V) (n := n)) := by
  rintro ⟨⟨a, b⟩, s⟩ ⟨⟨a', b'⟩, s'⟩ h
  simp only [edgeCopy, Prod.mk.injEq, Fin.castSucc_inj] at h
  obtain ⟨⟨rfl, rfl⟩, rfl, -⟩ := h
  rfl

lemma edges_unrolled (n : ℕ) [DecidableEq V] :
    edges (unrolled E n) = (edges E ×ˢ (univ : Finset (Fin n))).image edgeCopy := by
  ext ⟨⟨a, x⟩, ⟨b, y⟩⟩
  simp only [mem_edges, unrolled, mem_image, mem_product, mem_univ, and_true, Prod.exists,
    edgeCopy, Prod.mk.injEq]
  constructor
  · rintro ⟨hab, hy⟩
    have hx : (x : ℕ) < n := by have := y.isLt; omega
    refine ⟨a, b, ⟨x, hx⟩, hab, ⟨rfl, Fin.ext rfl⟩, rfl, Fin.ext (by simp [hy])⟩
  · rintro ⟨a', b', s, hab, ⟨rfl, rfl⟩, rfl, rfl⟩
    exact ⟨hab, by simp⟩

lemma sum_edges_unrolled (n : ℕ) (g : (V × Fin (n + 1)) × (V × Fin (n + 1)) → ℝ) :
    ∑ p ∈ edges (unrolled E n), g p =
      ∑ e ∈ edges E, ∑ s : Fin n, g ((e.1, s.castSucc), (e.2, s.succ)) := by
  classical
  rw [edges_unrolled, sum_image (fun x _ y _ h => edgeCopy_injective h), sum_product]
  rfl

lemma card_edges_unrolled (n : ℕ) :
    (edges (unrolled E n)).card = n * (edges E).card := by
  classical
  rw [edges_unrolled, card_image_of_injective _ edgeCopy_injective, card_product, card_univ,
    Fintype.card_fin, mul_comm]

end RelSort
