/-
X3 (research/theory2/ext/EXT.md): predictive meaning of `s_ρ`.

A *pairwise count rule* is `f : ℕ → ℕ → ℝ` with `0 ≤ f a b ≤ 1`: shown an unordered edge in a
uniformly random order as `(first, second)` with relative-counts `(a, b)`, it declares
"`first` is the tail" with probability `f a b`.  For a true edge `x → y`, the probability of a
correct orientation is `1/2 · f(ρx, ρy) + 1/2 · (1 − f(ρy, ρx))` (order `(x,y)` or `(y,x)`).
`accRule` is the edge average. The oracle sort-and-orient rule with fair-coin ties is `sortRule`.

* `accRule_sortRule`: the accuracy of the oracle rule is exactly `sOf` (so `s_ρ` for ρ = relatives).
* `accRule_le_sRho`: no pairwise count rule beats it on any directed graph (Reisach et al.'s
  Theorem 1 gives `ρ x ≤ ρ y` on edges, and ties are forced to `1/2` by the random order).
-/
import RelSort.Basic

open Finset

set_option linter.unusedSectionVars false

namespace RelSort

variable {W : Type*} [Fintype W]

/-- Probability of a correct orientation of the true edge `x → y` (random presentation order). -/
noncomputable def edgeAcc (f : ℕ → ℕ → ℝ) (a b : ℕ) : ℝ := 1 / 2 * f a b + 1 / 2 * (1 - f b a)

/-- The oracle sort-and-orient rule: smaller count first, ties by a fair coin. -/
noncomputable def sortRule (a b : ℕ) : ℝ := if a < b then 1 else if a = b then 1 / 2 else 0

/-- Edge-averaged accuracy of the rule `f` given counts `ρ`. -/
noncomputable def accRule (R : W → W → Prop) (ρ : W → ℕ) (f : ℕ → ℕ → ℝ) : ℝ :=
  (∑ p ∈ edges R, edgeAcc f (ρ p.1) (ρ p.2)) / (edges R).card

lemma edgeAcc_sortRule (a b : ℕ) : edgeAcc sortRule a b = score a b := by
  unfold edgeAcc sortRule score
  rcases lt_trichotomy a b with h | h | h
  · simp [h, not_lt.mpr h.le, h.ne']; norm_num
  · subst h; norm_num
  · have h1 : ¬ a < b := not_lt.mpr h.le
    have h2 : a ≠ b := h.ne'
    have h3 : ¬ b < a → False := fun hh => hh h
    simp [h1, h2, h]

/-- **X3, identity**: the accuracy of the oracle sort rule is exactly the rel-sortability. -/
theorem accRule_sortRule (R : W → W → Prop) (ρ : W → ℕ) :
    accRule R ρ sortRule = sOf R ρ := by
  unfold accRule sOf
  simp_rw [edgeAcc_sortRule]

lemma edgeAcc_le_score {f : ℕ → ℕ → ℝ} (h0 : ∀ a b, 0 ≤ f a b) (h1 : ∀ a b, f a b ≤ 1) {a b : ℕ}
    (hab : a ≤ b) : edgeAcc f a b ≤ score a b := by
  unfold edgeAcc score
  rcases hab.lt_or_eq with h | h
  · simp only [h, ↓reduceIte]
    linarith [h1 a b, h0 b a]
  · subst h; simp; linarith

/-- **X3, optimality among pairwise count rules**: if `ρ x ≤ ρ y` on every edge, every pairwise
count rule has accuracy at most `sOf R ρ`. -/
theorem accRule_le_sOf (R : W → W → Prop) (ρ : W → ℕ) (hρ : ∀ x y, R x y → ρ x ≤ ρ y)
    {f : ℕ → ℕ → ℝ} (h0 : ∀ a b, 0 ≤ f a b) (h1 : ∀ a b, f a b ≤ 1) :
    accRule R ρ f ≤ sOf R ρ := by
  unfold accRule sOf
  apply div_le_div_of_nonneg_right _ (Nat.cast_nonneg _)
  apply sum_le_sum
  intro p hp
  exact edgeAcc_le_score h0 h1 (hρ _ _ (mem_edges.mp hp))

/-- Specialisation to relatives: no pairwise count rule beats `s_ρ(G)` on any digraph. -/
theorem accRule_le_sRho (R : W → W → Prop) {f : ℕ → ℕ → ℝ} (h0 : ∀ a b, 0 ≤ f a b)
    (h1 : ∀ a b, f a b ≤ 1) : accRule R (rho R) f ≤ sRho R := by
  refine accRule_le_sOf R (rho R) ?_ h0 h1
  intro x y hxy
  rw [rho_eq_card, rho_eq_card]
  exact Finset.card_le_card (relSet_subset_of_edge hxy)

/-- Oracle rule on relatives has accuracy `s_ρ`. -/
theorem accRule_sortRule_rho (R : W → W → Prop) : accRule R (rho R) sortRule = sRho R :=
  accRule_sortRule R (rho R)

/-- A strictly sorted edge is never mis-oriented by the oracle rule (precision 1 on decided edges),
and the reverse reading is impossible: `ρ y < ρ x` cannot happen on an edge `x → y`. -/
theorem edgeAcc_sortRule_strict {a b : ℕ} (h : a < b) : edgeAcc sortRule a b = 1 := by
  rw [edgeAcc_sortRule]; simp [score, h]

theorem rho_not_gt_of_edge (R : W → W → Prop) {x y : W} (h : R x y) : ¬ rho R y < rho R x := by
  rw [rho_eq_card, rho_eq_card]
  exact not_lt.mpr (Finset.card_le_card (relSet_subset_of_edge h))

end RelSort
