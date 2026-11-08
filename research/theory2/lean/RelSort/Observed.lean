/-
The summary-level estimand `s̃` of the real-data studies (ICML draft, Appendix "Real-data
estimand"), for an arbitrary finite summary graph `E` and an observed node set `O`.

* Population relatives of an observed node `v` among the observed nodes: `rel_S(v) ∩ O`
  (relatives computed in the full graph, so unobserved nodes act as latent common ancestors).
* Scored edges: the induced non-self-loop edges `E_O = {(i,j) ∈ E : i, j ∈ O, i ≠ j}`.
* `s̃ = (1/|E_O|) Σ_{(i,j) ∈ E_O} score(|rel_S(i) ∩ O|, |rel_S(j) ∩ O|)`.

Results: every scored edge has score `1/2` or `1` (`score_obsEdge`); `s̃ = 1/2 + |E_O^↑|/(2|E_O|)`
with `E_O^↑` the scored edges whose observed relative sets differ (`sTilde_eq`); `E_O^↑ ⊆ E_↑`
(`upObs_subset_upEdges`), so `s̃ > 1/2` forces an up-edge (stationary sense) inside the induced
structure (`sTilde_gt_half_iff`, `exists_upEdge_of_sTilde_gt_half`); an observed relative of the
head that is not a relative of the tail is a sufficient witness (`mem_upObs_of_witness`); with
every node observed, `E_O^↑` is `E_↑` without self-loops (`upObs_univ`).
No self-loop hypothesis is needed.
-/
import RelSort.Stationary

open Finset

set_option linter.unusedSectionVars false

namespace RelSort

variable {V : Type*} [Fintype V] {E : V → V → Prop}

open Classical in
/-- `rel_S(v) ∩ O`. -/
noncomputable def relObs (E : V → V → Prop) (O : Finset V) (v : V) : Finset V :=
  (relSet E v).filter (fun w => w ∈ O)

/-- `|rel_S(v) ∩ O|`, the population lag-0 relatives count among observed nodes. -/
noncomputable def rhoObs (E : V → V → Prop) (O : Finset V) (v : V) : ℕ := (relObs E O v).card

open Classical in
/-- Induced non-self-loop edges `E_O`. -/
noncomputable def obsEdges (E : V → V → Prop) (O : Finset V) : Finset (V × V) :=
  (edges E).filter (fun e => e.1 ∈ O ∧ e.2 ∈ O ∧ e.1 ≠ e.2)

open Classical in
/-- `E_O^↑`: scored edges whose observed relative sets differ. -/
noncomputable def upObs (E : V → V → Prop) (O : Finset V) : Finset (V × V) :=
  (obsEdges E O).filter (fun e => relObs E O e.1 ≠ relObs E O e.2)

/-- The estimand `s̃`. -/
noncomputable def sTilde (E : V → V → Prop) (O : Finset V) : ℝ :=
  (∑ e ∈ obsEdges E O, score (rhoObs E O e.1) (rhoObs E O e.2)) / (obsEdges E O).card

lemma mem_obsEdges {O : Finset V} {e : V × V} :
    e ∈ obsEdges E O ↔ E e.1 e.2 ∧ e.1 ∈ O ∧ e.2 ∈ O ∧ e.1 ≠ e.2 := by
  classical
  unfold obsEdges
  rw [mem_filter, mem_edges]

/-- Monotonicity (Reisach et al. Thm 1) survives restriction to `O`. -/
theorem relObs_subset_of_edge {O : Finset V} {i j : V} (h : E i j) :
    relObs E O i ⊆ relObs E O j := by
  classical
  unfold relObs
  exact Finset.filter_subset_filter _ (relSet_subset_of_edge h)

open Classical in
/-- Every scored edge has score `1/2` (equal observed relative sets) or `1`. -/
theorem score_obsEdge {O : Finset V} {e : V × V} (he : e ∈ obsEdges E O) :
    score (rhoObs E O e.1) (rhoObs E O e.2) =
      if relObs E O e.1 = relObs E O e.2 then 1 / 2 else 1 := by
  classical
  have h := (mem_obsEdges.mp he).1
  unfold rhoObs
  exact score_of_subset (relObs_subset_of_edge h)

/-- `s̃ = 1/2 + |E_O^↑| / (2 |E_O|)` whenever `E_O` is nonempty. -/
theorem sTilde_eq {O : Finset V} (hne : (obsEdges E O).Nonempty) :
    sTilde E O = 1 / 2 + ((upObs E O).card : ℝ) / (2 * (obsEdges E O).card) := by
  classical
  have hpos : (0 : ℝ) < (obsEdges E O).card := by exact_mod_cast hne.card_pos
  have hsum : (∑ e ∈ obsEdges E O, score (rhoObs E O e.1) (rhoObs E O e.2)) =
      ∑ e ∈ obsEdges E O, ((1 : ℝ) / 2 +
        (if relObs E O e.1 ≠ relObs E O e.2 then (1 : ℝ) / 2 else 0)) := by
    refine Finset.sum_congr rfl (fun e he => ?_)
    rw [score_obsEdge he]
    split_ifs <;> first | contradiction | norm_num
  unfold sTilde
  rw [hsum, Finset.sum_add_distrib, Finset.sum_const, ← Finset.sum_filter]
  unfold upObs
  rw [Finset.sum_const, nsmul_eq_mul, nsmul_eq_mul]
  field_simp

/-- `E_O^↑ ⊆ E_↑`: an edge that the observed counts can orient is an up-edge of the full
summary graph (stationary sense of Theorem 1.9). -/
theorem upObs_subset_upEdges {O : Finset V} : upObs E O ⊆ upEdges E := by
  classical
  intro e he
  unfold upObs at he
  rw [mem_filter] at he
  obtain ⟨he1, hne⟩ := he
  unfold upEdges
  rw [mem_filter]
  refine ⟨mem_edges.mpr (mem_obsEdges.mp he1).1, fun h => hne ?_⟩
  unfold relObs
  rw [h]

/-- `s̃ > 1/2` iff some scored edge has different observed relative sets. -/
theorem sTilde_gt_half_iff {O : Finset V} (hne : (obsEdges E O).Nonempty) :
    1 / 2 < sTilde E O ↔ (upObs E O).Nonempty := by
  classical
  rw [sTilde_eq hne]
  have hpos : (0 : ℝ) < 2 * (obsEdges E O).card := by
    have : (0 : ℝ) < (obsEdges E O).card := by exact_mod_cast hne.card_pos
    linarith
  constructor
  · intro h
    by_contra hc
    rw [Finset.not_nonempty_iff_eq_empty] at hc
    rw [hc] at h
    simp at h
  · intro h
    have : (0 : ℝ) < (upObs E O).card := by exact_mod_cast h.card_pos
    have := div_pos this hpos
    linarith

/-- `s̃ ≥ 1/2` always (when `E_O` is nonempty). -/
theorem sTilde_ge_half {O : Finset V} (hne : (obsEdges E O).Nonempty) :
    1 / 2 ≤ sTilde E O := by
  classical
  rw [sTilde_eq hne]
  have : (0 : ℝ) ≤ ((upObs E O).card : ℝ) / (2 * (obsEdges E O).card) := by positivity
  linarith

/-- Sign relation: `s̃ > 1/2` forces an induced non-loop edge that is an up-edge of the full
summary graph. (The converse fails: the extra relatives of the head may all be unobserved.) -/
theorem exists_upEdge_of_sTilde_gt_half {O : Finset V} (hne : (obsEdges E O).Nonempty)
    (h : 1 / 2 < sTilde E O) : ∃ e ∈ obsEdges E O, e ∈ upEdges E := by
  classical
  obtain ⟨e, he⟩ := (sTilde_gt_half_iff hne).mp h
  refine ⟨e, ?_, upObs_subset_upEdges he⟩
  unfold upObs at he
  exact (mem_filter.mp he).1

/-- If no induced non-loop edge is an up-edge, `s̃ = 1/2`. -/
theorem sTilde_eq_half_of_no_up {O : Finset V} (hne : (obsEdges E O).Nonempty)
    (h : ∀ e ∈ obsEdges E O, e ∉ upEdges E) : sTilde E O = 1 / 2 := by
  have hle := sTilde_ge_half hne
  by_contra hc
  obtain ⟨e, he, hup⟩ := exists_upEdge_of_sTilde_gt_half hne (lt_of_le_of_ne hle (Ne.symm hc))
  exact h e he hup

/-- Sufficient witness: an observed relative of the head that is not a relative of the tail. -/
theorem mem_upObs_of_witness {O : Finset V} {e : V × V} {w : V} (he : e ∈ obsEdges E O)
    (hw : w ∈ O) (hj : IsRel E e.2 w) (hi : ¬ IsRel E e.1 w) : e ∈ upObs E O := by
  classical
  unfold upObs
  rw [mem_filter]
  refine ⟨he, fun h => hi ?_⟩
  have hwj : w ∈ relObs E O e.2 := by
    unfold relObs; rw [mem_filter]; exact ⟨mem_relSet.mpr hj, hw⟩
  rw [← h] at hwj
  unfold relObs at hwj
  exact mem_relSet.mp (mem_filter.mp hwj).1

open Classical in
/-- Full observation: `E_O^↑` is `E_↑` without self-loops. -/
theorem upObs_univ : upObs E univ = (upEdges E).filter (fun e => e.1 ≠ e.2) := by
  classical
  ext e
  unfold upObs upEdges relObs
  simp only [mem_filter, mem_obsEdges, mem_edges, mem_univ, true_and, filter_true_of_mem,
    implies_true]
  tauto

end RelSort
