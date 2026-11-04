/-
Deterministic lemmas used in x1proof/X1PROOF.md (random summary graphs, target T2 and T3).

`E_↑ = {(i,j) ∈ E_S : Src(j) ≠ Src(i)}` (Theorem 1.9) is characterised by *witnesses*:

* (a) an edge `(i,j)` of `E_↑` has a second parent `k ∉ {i,j}` of `j` and a source node `x`
  reaching `k` but not `i` (`upEdge_witness`); hence `E_↑` is contained in the set of edges into
  nodes that have a parent other than the tail and other than themselves (`upEdges_subset_confluenceNL`);
* (b) conversely, a source reaching some parent `k` of `j` but not `i` puts `(i,j)` into `E_↑`
  (`mem_upEdges_of_witness`), and so does a parent `k` of `j` that has no common ancestor with `i`
  (`mem_upEdges_of_disjoint_anc`).

No self-loop hypothesis is needed.
-/
import RelSort.Stationary

open Finset

set_option linter.unusedSectionVars false

namespace RelSort

variable {V : Type*} [Fintype V] {E : V → V → Prop}

/-- (a) Witness form of `E_↑`: second parent `k` of `j` and a source `x ⇝ k` with `x ⇝̸ i`. -/
theorem upEdge_witness {i j : V} (hij : E i j) (hup : (i, j) ∈ upEdges E) :
    ∃ k x, k ≠ i ∧ k ≠ j ∧ E k j ∧ SrcOf E k x ∧ ¬ Reach E x i := by
  rw [mem_upEdges_iff] at hup
  obtain ⟨-, hne⟩ := hup
  -- a source in Src(j) \ Src(i)
  obtain ⟨x, hx⟩ : ∃ x, SrcOf E j x ∧ ¬ SrcOf E i x := by
    by_contra hcon
    push Not at hcon
    apply hne
    intro x
    exact ⟨fun h => ⟨h.1, h.2.trans (Relation.ReflTransGen.single hij)⟩, hcon x⟩
  have hxs : IsSource E x := hx.1.1
  have hxi : ¬ Reach E x i := fun h => hx.2 ⟨hxs, h⟩
  have hxj : Reach E x j := hx.1.2
  rcases reach_last hxj with hxeq | ⟨k, hkj, hxk, hEk⟩
  · exfalso
    subst hxeq
    exact hxi (hxs i (Relation.ReflTransGen.single hij))
  · refine ⟨k, x, ?_, hkj, hEk, ⟨hxs, hxk⟩, hxi⟩
    rintro rfl
    exact hxi hxk

/-- (b) A source reaching a parent `k` of `j` but not `i` makes `(i,j)` an `E_↑` edge. -/
theorem mem_upEdges_of_witness {i j k x : V} (hij : E i j) (hkj : E k j)
    (hx : SrcOf E k x) (hxi : ¬ Reach E x i) : (i, j) ∈ upEdges E := by
  rw [mem_upEdges_iff]
  refine ⟨hij, fun h => hxi ?_⟩
  exact ((h x).mpr ⟨hx.1, hx.2.trans (Relation.ReflTransGen.single hkj)⟩).2

/-- (b') A parent `k` of `j` with no common ancestor with `i` makes `(i,j)` an `E_↑` edge. -/
theorem mem_upEdges_of_disjoint_anc {i j k : V} (hij : E i j) (hkj : E k j)
    (hd : ∀ u, Reach E u k → ¬ Reach E u i) : (i, j) ∈ upEdges E := by
  obtain ⟨x, hxs, hxk⟩ := exists_source (E := E) k
  exact mem_upEdges_of_witness hij hkj ⟨hxs, hxk⟩ (hd x hxk)

open Classical in
/-- Edges `(i,j)` whose head has a parent `k` different from `i` and from `j` itself. -/
noncomputable def confluenceNL (E : V → V → Prop) : Finset (V × V) :=
  (edges E).filter (fun e => ∃ k, k ≠ e.1 ∧ k ≠ e.2 ∧ E k e.2)

/-- `E_↑` is contained in the edges into confluences (non-loop parents only). -/
theorem upEdges_subset_confluenceNL : upEdges E ⊆ confluenceNL E := by
  classical
  intro e he
  have he' := he
  rw [mem_upEdges_iff] at he'
  obtain ⟨k, x, hk1, hk2, hk3, -, -⟩ := upEdge_witness (i := e.1) (j := e.2) he'.1 he
  unfold confluenceNL
  rw [mem_filter, mem_edges]
  exact ⟨he'.1, k, hk1, hk2, hk3⟩

theorem card_upEdges_le_confluenceNL : (upEdges E).card ≤ (confluenceNL E).card :=
  card_le_card upEdges_subset_confluenceNL

end RelSort
