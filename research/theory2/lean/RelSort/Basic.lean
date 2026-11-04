/-
Basic definitions (Reisach, Chambaz, Blanchard, Weichwald, arXiv 2605.06288, Sec. 2.1-2.2),
for an arbitrary finite directed graph given by a relation `R` on a `Fintype`.

* `anc(y) = {x : P_xy ≠ ∅}` is `Relation.ReflTransGen R x y` (a directed path from `x` to `y`
  exists iff a directed walk exists; `P_yy = {y}` is the reflexive case).
* `rel(y) = desc(anc(y))`: `IsRel R y z` iff `∃ x, x ∈ anc(y) ∧ z ∈ desc(x)`. This is literally the
  paper's definition (a union over `x ∈ anc(y)` of `desc(x)`).
* `ρ(y) = |rel(y)|`, and `s_ρ(G) = (1/|E|) Σ_{(x,y) ∈ E} s_ρ(x,y)` with `s_ρ(x,y) = 1, 1/2, 0`
  according as `ρ(x) <, =, > ρ(y)`.
-/
import Mathlib

open Finset

set_option linter.unusedSectionVars false

namespace RelSort

section General

variable {W : Type*} [Fintype W]

/-- `IsRel R y z`: `z ∈ rel(y) = desc(anc(y))`. -/
def IsRel (R : W → W → Prop) (y z : W) : Prop :=
  ∃ x, Relation.ReflTransGen R x y ∧ Relation.ReflTransGen R x z

open Classical in
/-- The finite set `{z : P y z}`. -/
noncomputable def setOf' (P : W → W → Prop) (y : W) : Finset W := univ.filter (fun z => P y z)

/-- The count `|{z : P y z}|`; with `P = IsRel R` this is `ρ(y)`. -/
noncomputable def cntOf (P : W → W → Prop) (y : W) : ℕ := (setOf' P y).card

/-- `rel(y)` as a finset. -/
noncomputable def relSet (R : W → W → Prop) (y : W) : Finset W := setOf' (IsRel R) y

/-- `ρ(y) = |rel(y)|`. -/
noncomputable def rho (R : W → W → Prop) (y : W) : ℕ := cntOf (IsRel R) y

/-- Edge score `s_ρ(x,y)`: 1, 1/2, 0 when `a <, =, > b`. -/
noncomputable def score (a b : ℕ) : ℝ := if a < b then 1 else if a = b then 1 / 2 else 0

open Classical in
/-- The edge set of `R`. -/
noncomputable def edges (R : W → W → Prop) : Finset (W × W) := univ.filter (fun p => R p.1 p.2)

/-- Edge average of the scores for an arbitrary count function `ρ`. -/
noncomputable def sOf (R : W → W → Prop) (ρ : W → ℕ) : ℝ :=
  (∑ p ∈ edges R, score (ρ p.1) (ρ p.2)) / (edges R).card

/-- Reisach et al.'s rel-sortability `s_ρ(G)`. -/
noncomputable def sRho (R : W → W → Prop) : ℝ := sOf R (rho R)

@[simp] lemma mem_setOf' {P : W → W → Prop} {y z : W} : z ∈ setOf' P y ↔ P y z := by
  classical
  unfold setOf'
  simp

@[simp] lemma mem_edges {R : W → W → Prop} {p : W × W} : p ∈ edges R ↔ R p.1 p.2 := by
  classical
  unfold edges
  simp

lemma mem_relSet {R : W → W → Prop} {y z : W} : z ∈ relSet R y ↔ IsRel R y z := mem_setOf'

lemma rho_eq_card (R : W → W → Prop) (y : W) : rho R y = (relSet R y).card := rfl

/-- Reisach et al. Theorem 1 (here for any directed graph): an edge `x → y` gives
`rel(x) ⊆ rel(y)`. -/
lemma relSet_subset_of_edge {R : W → W → Prop} {x y : W} (h : R x y) :
    relSet R x ⊆ relSet R y := by
  intro z hz
  rw [mem_relSet] at hz ⊢
  obtain ⟨u, hux, huz⟩ := hz
  exact ⟨u, hux.tail h, huz⟩

/-- Same, along any directed path. -/
lemma relSet_subset_of_reach {R : W → W → Prop} {x y : W} (h : Relation.ReflTransGen R x y) :
    relSet R x ⊆ relSet R y := by
  intro z hz
  rw [mem_relSet] at hz ⊢
  obtain ⟨u, hux, huz⟩ := hz
  exact ⟨u, hux.trans h, huz⟩

open Classical in
/-- Score of a pair with nested sets: `1/2` if equal, `1` otherwise. -/
lemma score_of_subset {A B : Finset W} (h : A ⊆ B) :
    score A.card B.card = if A = B then 1 / 2 else 1 := by
  unfold score
  by_cases hAB : A = B
  · subst hAB; simp
  · have hss : A ⊂ B := Finset.ssubset_iff_subset_ne.mpr ⟨h, hAB⟩
    have := Finset.card_lt_card hss
    simp [hAB, this]

open Classical in
lemma score_mul_of_subset {A B : Finset W} (h : A ⊆ B) (c : ℕ) (hc : 0 < c) :
    score (c * A.card) (c * B.card) = if A = B then 1 / 2 else 1 := by
  rw [← score_of_subset h]
  unfold score
  have h1 : c * A.card < c * B.card ↔ A.card < B.card := Nat.mul_lt_mul_left hc
  have h2 : c * A.card = c * B.card ↔ A.card = B.card := by
    constructor
    · intro e; exact Nat.eq_of_mul_eq_mul_left hc e
    · intro e; rw [e]
  simp only [h1, h2]

end General

end RelSort
