/-
Named examples of Corollaries 1.7 and 1.10 (iv) on three nodes `0, 1, 2` with self-loops:
collider `0 → 2 ← 1`, fork `2 ← 0 → 1`, chain `0 → 1 → 2`.
Finite-`G_T` limit (Cor. 1.7): `0.7` for all three. Stationary value (Cor. 1.10): `0.7` for the
collider, `1/2` for the fork and the chain.
-/
import RelSort.Stationary

open Finset

set_option linter.unusedSectionVars false

namespace RelSort

variable {V : Type*} [Fintype V] {E : V → V → Prop}

/-- A strictly increasing rank along non-loop edges makes the graph acyclic apart from loops. -/
lemma acyclic_of_rank (f : V → ℕ) (hf : ∀ a b, E a b → a = b ∨ f a < f b) :
    ∀ i j, Reach E i j → Reach E j i → i = j := by
  have key : ∀ i j, Reach E i j → i = j ∨ f i < f j := by
    intro i j h
    induction h with
    | refl => exact Or.inl rfl
    | @tail b c _ hbc ih =>
      rcases ih with rfl | h1 <;> rcases hf _ _ hbc with rfl | h2
      · exact Or.inl rfl
      · exact Or.inr h2
      · exact Or.inr h1
      · exact Or.inr (h1.trans h2)
  intro i j h1 h2
  rcases key i j h1 with h | h
  · exact h
  · rcases key j i h2 with h' | h'
    · exact h'.symm
    · omega

/-- Collider `0 → 2 ← 1`. -/
def collider : Fin 3 → Fin 3 → Prop := fun a b => a = b ∨ (a = 0 ∧ b = 2) ∨ (a = 1 ∧ b = 2)
/-- Fork `2 ← 0 → 1`. -/
def fork : Fin 3 → Fin 3 → Prop := fun a b => a = b ∨ (a = 0 ∧ b = 1) ∨ (a = 0 ∧ b = 2)
/-- Chain `0 → 1 → 2`. -/
def chain3 : Fin 3 → Fin 3 → Prop := fun a b => a = b ∨ (a = 0 ∧ b = 1) ∨ (a = 1 ∧ b = 2)

instance : DecidableRel collider := fun a b => by unfold collider; infer_instance
instance : DecidableRel fork := fun a b => by unfold fork; infer_instance
instance : DecidableRel chain3 := fun a b => by unfold chain3; infer_instance

lemma edges_eq_filter (R : Fin 3 → Fin 3 → Prop) [DecidableRel R] :
    edges R = univ.filter (fun p : Fin 3 × Fin 3 => R p.1 p.2) := by
  ext p; simp

lemma numProperEdges_eq_filter (R : Fin 3 → Fin 3 → Prop) [DecidableRel R] :
    numProperEdges R = (univ.filter (fun p : Fin 3 × Fin 3 => R p.1 p.2 ∧ p.1 ≠ p.2)).card := by
  classical
  unfold numProperEdges
  congr 1
  ext p; simp

lemma confluenceEdges_eq_filter (R : Fin 3 → Fin 3 → Prop) [DecidableRel R] :
    confluenceEdges R = univ.filter (fun p : Fin 3 × Fin 3 =>
      R p.1 p.2 ∧ p.1 ≠ p.2 ∧ ∃ k, R k p.2 ∧ k ≠ p.1 ∧ k ≠ p.2) := by
  classical
  unfold confluenceEdges
  ext p; simp

lemma collider_dag : ∀ i j, Reach collider i j → Reach collider j i → i = j :=
  acyclic_of_rank (fun a => if a = 2 then 1 else 0) (by decide)
lemma fork_dag : ∀ i j, Reach fork i j → Reach fork j i → i = j :=
  acyclic_of_rank (fun a => if a = 0 then 0 else 1) (by decide)
lemma chain3_dag : ∀ i j, Reach chain3 i j → Reach chain3 j i → i = j :=
  acyclic_of_rank (fun a => (a : ℕ)) (by decide)

/-- Cor. 1.7: the 3-node collider has finite-`G_T` limit `0.7`. -/
theorem collider_limit : limitValue collider = 7 / 10 := by
  rw [limitValue_dag (E := collider) (fun v => Or.inl rfl) collider_dag, numProperEdges_eq_filter]
  rw [show (univ.filter (fun p : Fin 3 × Fin 3 => collider p.1 p.2 ∧ p.1 ≠ p.2)).card = 2 by
    decide]
  norm_num

/-- Cor. 1.7: the 3-node fork has finite-`G_T` limit `0.7`. -/
theorem fork_limit : limitValue fork = 7 / 10 := by
  rw [limitValue_dag (E := fork) (fun v => Or.inl rfl) fork_dag, numProperEdges_eq_filter]
  rw [show (univ.filter (fun p : Fin 3 × Fin 3 => fork p.1 p.2 ∧ p.1 ≠ p.2)).card = 2 by
    decide]
  norm_num

/-- Cor. 1.7: the 3-node chain has finite-`G_T` limit `0.7`. -/
theorem chain3_limit : limitValue chain3 = 7 / 10 := by
  rw [limitValue_dag (E := chain3) (fun v => Or.inl rfl) chain3_dag, numProperEdges_eq_filter]
  rw [show (univ.filter (fun p : Fin 3 × Fin 3 => chain3 p.1 p.2 ∧ p.1 ≠ p.2)).card = 2 by
    decide]
  norm_num

/-- Cor. 1.10 (iv): the collider has stationary value `1/2 + 2/(2·5) = 0.7`. -/
theorem collider_stat : statValue collider = 7 / 10 := by
  have hf : InForest collider := ⟨by decide, collider_dag⟩
  rw [statValue_inForest hf, confluenceEdges_eq_filter, edges_eq_filter]
  rw [show (univ.filter (fun p : Fin 3 × Fin 3 => collider p.1 p.2 ∧ p.1 ≠ p.2 ∧
      ∃ k, collider k p.2 ∧ k ≠ p.1 ∧ k ≠ p.2)).card = 2 by decide]
  rw [show (univ.filter (fun p : Fin 3 × Fin 3 => collider p.1 p.2)).card = 5 by decide]
  norm_num

/-- Cor. 1.10 (iv): the fork has stationary value `1/2`. -/
theorem fork_stat : statValue fork = 1 / 2 :=
  stat_half_of_outForest (E := fork) (fun v => Or.inl rfl) (by unfold OutForest; decide)

/-- Cor. 1.10 (ii): the chain has stationary value `1/2`. -/
theorem chain3_stat : statValue chain3 = 1 / 2 :=
  stat_half_of_outForest (E := chain3) (fun v => Or.inl rfl) (by unfold OutForest; decide)

end RelSort
