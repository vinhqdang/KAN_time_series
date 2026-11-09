/-
Further deterministic statements of THEORY.md Section 1:
* Lemma 1.2 (roots of `G_T` and descendants of time-1 nodes);
* Proposition 1.4 (exact relative counts in `G_T`), first formula, and its `t ≥ p` form;
* Remark 1.12 (scope: the single edge `0 → 1` with and without self-loops).
-/
import RelSort.Examples

open Finset

set_option linter.unusedSectionVars false

namespace RelSort

variable {V : Type*} [Fintype V] {E : V → V → Prop}

/-! ### Lemma 1.2 -/

/-- `x` is a root: `anc(x) = {x}`. -/
def IsRoot {W : Type*} (R : W → W → Prop) (x : W) : Prop :=
  ∀ y, Relation.ReflTransGen R y x → y = x

/-- The roots of `G_T` are exactly the time-1 nodes (Lean slice 0). -/
theorem isRoot_unrolled_iff (hl : ∀ v, E v v) {n : ℕ} {u : V} {s : Fin (n + 1)} :
    IsRoot (unrolled E n) (u, s) ↔ (s : ℕ) = 0 := by
  constructor
  · intro h
    by_contra hs
    have hlt : (s : ℕ) - 1 < n + 1 := by omega
    have := h (u, ⟨s - 1, hlt⟩) (Relation.ReflTransGen.single ⟨hl u, by simp; omega⟩)
    simp only [Prod.mk.injEq, true_and] at this
    have := congrArg Fin.val this
    simp at this
    omega
  · intro hs ⟨w, r⟩ hy
    rw [unrolled_reach_iff] at hy
    have hr : (r : ℕ) = 0 := by omega
    have hrs : r = s := Fin.ext (by omega)
    subst hrs
    rw [hr] at hy
    have : w = u := hy.2
    rw [this]

/-- **Lemma 1.2**: `roots(v(t)) = {u(1) : d(u,v) ≤ t - 1}` (Lean: slice 0, `d(u,v) ≤ t`). -/
theorem roots_unrolled_iff (hl : ∀ v, E v v) {n : ℕ} {u v : V} {s t : Fin (n + 1)} :
    (IsRoot (unrolled E n) (u, s) ∧ Relation.ReflTransGen (unrolled E n) (u, s) (v, t)) ↔
      (s : ℕ) = 0 ∧ dist E u v ≤ (t : ℕ) := by
  rw [isRoot_unrolled_iff hl, unrolled_reach_iff, dist_le_iff hl]
  constructor
  · rintro ⟨h0, -, hw⟩; rw [h0, Nat.sub_zero] at hw; exact ⟨h0, hw⟩
  · rintro ⟨h0, hw⟩; exact ⟨h0, by omega, by rw [h0, Nat.sub_zero]; exact hw⟩

/-- **Lemma 1.2**: `desc(u(1)) = {w(s) : d(u,w) ≤ s - 1}`. -/
theorem desc_root_unrolled_iff (hl : ∀ v, E v v) {n : ℕ} {u w : V} {r : Fin (n + 1)} :
    Relation.ReflTransGen (unrolled E n) (u, 0) (w, r) ↔ dist E u w ≤ (r : ℕ) := by
  rw [unrolled_reach_iff, dist_le_iff hl]
  simp

/-! ### Proposition 1.4 -/

open Classical in
/-- `m_t(v,w) = min{d(u,w) : d(u,v) ≤ t - 1}` (Lean slice `t`, i.e. `d(u,v) ≤ t`), `⊤` if empty. -/
noncomputable def mT (E : V → V → Prop) (t : ℕ) (v w : V) : ℕ∞ :=
  if h : ∃ r u, WalkN E t u v ∧ WalkN E r u w then (Nat.find h : ℕ∞) else ⊤

/-- `m_t(v,w)` is the minimum of `d(u,w)` over `u` with `d(u,v) ≤ t`. -/
theorem mT_le_iff (hl : ∀ v, E v v) {t r : ℕ} {v w : V} :
    mT E t v w ≤ r ↔ ∃ u, dist E u v ≤ (t : ℕ) ∧ dist E u w ≤ r := by
  classical
  simp only [dist_le_iff hl]
  unfold mT
  split_ifs with h
  · rw [Nat.cast_le, Nat.find_le_iff]
    constructor
    · rintro ⟨k, hk, u, h1, h2⟩; exact ⟨u, h1, walkN_mono hl hk h2⟩
    · rintro ⟨u, h1, h2⟩; exact ⟨r, le_rfl, u, h1, h2⟩
  · simp only [top_le_iff, ENat.natCast_ne_top, false_iff]
    rintro ⟨u, h1, h2⟩; exact h ⟨r, u, h1, h2⟩

/-- The count `max(T - m, 0)` with `m ∈ ℕ∞`. -/
def truncCount (T : ℕ) (m : ℕ∞) : ℕ := if m = ⊤ then 0 else T - m.toNat

lemma card_filter_ge (T k : ℕ) :
    (univ.filter (fun r : Fin T => k ≤ (r : ℕ))).card = T - k := by
  classical
  have : (univ.filter (fun r : Fin T => k ≤ (r : ℕ))).map Fin.valEmbedding = Ico k T := by
    ext x
    simp only [mem_map, mem_filter, mem_univ, true_and, Fin.valEmbedding_apply, mem_Ico]
    constructor
    · rintro ⟨r, hr, rfl⟩; exact ⟨hr, r.isLt⟩
    · rintro ⟨h1, h2⟩; exact ⟨⟨x, h2⟩, h1, rfl⟩
  rw [← card_map, this, Nat.card_Ico]

/-- **Proposition 1.4**: `ρ(v(t)) = Σ_{w ∈ V_S} max(T - m_t(v,w), 0)`. -/
theorem rho_unrolled_eq (hl : ∀ v, E v v) {n : ℕ} (v : V) (t : Fin (n + 1)) :
    rho (unrolled E n) (v, t) = ∑ w : V, truncCount (n + 1) (mT E t v w) := by
  classical
  rw [rho_eq_card]
  have hcard : (relSet (unrolled E n) (v, t)).card =
      ∑ w : V, (univ.filter (fun r : Fin (n + 1) => mT E t v w ≤ (r : ℕ))).card := by
    rw [← card_sigma]
    apply card_bij (fun p _ => ⟨p.1, p.2⟩)
    · intro p hp
      rw [mem_relSet, isRel_unrolled_iff hl] at hp
      simp only [mem_sigma, mem_univ, mem_filter, true_and]
      rw [mT_le_iff hl]; simp only [dist_le_iff hl]; exact hp
    · intro a _ b _ h
      simp only [Sigma.mk.injEq, heq_eq_eq] at h
      exact Prod.ext h.1 h.2
    · rintro ⟨w, r⟩ hp
      refine ⟨(w, r), ?_, rfl⟩
      simp only [mem_sigma, mem_univ, mem_filter, true_and] at hp
      rw [mT_le_iff hl] at hp; simp only [dist_le_iff hl] at hp
      rw [mem_relSet, isRel_unrolled_iff hl]; exact hp
  rw [hcard]
  apply sum_congr rfl
  intro w _
  unfold truncCount
  split_ifs with htop
  · rw [htop]; simp
  · obtain ⟨k, hk⟩ := ENat.ne_top_iff_exists.mp htop
    rw [← hk, ENat.toNat_natCast, ← card_filter_ge]
    congr 1

/-- For `t ≥ p - 1` (paper: `t ≥ p`), `m_t(v,w) = d(anc_S(v), w)` is finite iff
`w ∈ rel_S(v)`, and then at most `p - 1`. -/
theorem mT_large_t (hl : ∀ v, E v v) {t : ℕ} (ht : Fintype.card V - 1 ≤ t) (v w : V) :
    (mT E t v w ≠ ⊤ ↔ IsRel E v w) ∧
    (∀ r : ℕ, mT E t v w ≤ r ↔ ∃ u, Reach E u v ∧ dist E u w ≤ r) ∧
    (IsRel E v w → mT E t v w ≤ (Fintype.card V - 1 : ℕ)) := by
  have hiff : ∀ r : ℕ, mT E t v w ≤ r ↔ ∃ u, Reach E u v ∧ dist E u w ≤ r := by
    intro r
    rw [mT_le_iff hl]
    refine exists_congr fun u => and_congr ?_ Iff.rfl
    rw [dist_le_iff hl]
    exact ⟨walkN_reach, fun h => walkN_of_reach hl h ht⟩
  have hrel : IsRel E v w → mT E t v w ≤ (Fintype.card V - 1 : ℕ) := by
    rintro ⟨u, huv, huw⟩
    exact (hiff _).mpr ⟨u, huv, dist_le_card_sub_one hl huw⟩
  refine ⟨⟨fun h => ?_, fun h => ne_top_of_le_ne_top (ENat.natCast_ne_top _) (hrel h)⟩,
    hiff, hrel⟩
  obtain ⟨k, hk⟩ := ENat.ne_top_iff_exists.mp h
  obtain ⟨u, huv, huw⟩ := (hiff k).mp (le_of_eq hk.symm)
  exact ⟨u, huv, dist_ne_top_iff.mp (ne_top_of_le_ne_top (ENat.natCast_ne_top _) huw)⟩

/-- **Proposition 1.4**, `t ≥ p` form (Lean `t ≥ p - 1`; the paper's extra hypothesis `T ≥ p`
is implied, since `t ≤ T`):
`ρ(v(t)) + Σ_{w ∈ rel_S(v)} d(anc_S(v), w) = T · |rel_S(v)|`. -/
theorem rho_unrolled_large_t (hl : ∀ v, E v v) {n : ℕ}
    (v : V) (t : Fin (n + 1)) (ht : Fintype.card V - 1 ≤ t) :
    rho (unrolled E n) (v, t) + ∑ w ∈ relSet E v, (mT E t v w).toNat =
      (n + 1) * (relSet E v).card := by
  classical
  rw [rho_unrolled_eq hl, ← sum_filter_add_sum_filter_not univ (fun w => IsRel E v w)]
  have hnot : ∑ w ∈ univ.filter (fun w => ¬ IsRel E v w), truncCount (n + 1) (mT E t v w) = 0 := by
    apply sum_eq_zero
    intro w hw
    rw [mem_filter] at hw
    have : mT E t v w = ⊤ := by
      by_contra h; exact hw.2 ((mT_large_t hl ht v w).1.mp h)
    simp [truncCount, this]
  have hrs : relSet E v = univ.filter (fun w => IsRel E v w) := by
    ext w; simp [mem_relSet]
  rw [hnot, add_zero, hrs, ← sum_add_distrib, card_eq_sum_ones, mul_sum, mul_one]
  apply sum_congr rfl
  intro w hw
  rw [mem_filter] at hw
  have hle := (mT_large_t hl ht v w).2.2 hw.2
  have hne : mT E t v w ≠ ⊤ := (mT_large_t hl ht v w).1.mpr hw.2
  have : (mT E t v w).toNat ≤ Fintype.card V - 1 := by
    obtain ⟨k, hk⟩ := ENat.ne_top_iff_exists.mp hne
    rw [← hk] at hle ⊢
    rw [ENat.toNat_natCast]
    exact Nat.cast_le.mp hle
  simp only [truncCount, hne, ↓reduceIte]
  omega

/-! ### Remark 1.12 -/

/-- The single edge `0 → 1` with self-loops. -/
def edgeLoops : Fin 2 → Fin 2 → Prop := fun a b => a = b ∨ (a = 0 ∧ b = 1)
/-- The single edge `0 → 1` without self-loops. -/
def edgeNoLoops : Fin 2 → Fin 2 → Prop := fun a b => a = 0 ∧ b = 1

instance : DecidableRel edgeLoops := fun a b => by unfold edgeLoops; infer_instance
instance : DecidableRel edgeNoLoops := fun a b => by unfold edgeNoLoops; infer_instance

/-- **Remark 1.12**, with self-loops: the finite-`G_T` limit is `2/3`. -/
theorem edgeLoops_limit : limitValue edgeLoops = 2 / 3 := by
  classical
  rw [limitValue_dag (E := edgeLoops) (fun v => Or.inl rfl)
    (acyclic_of_rank (fun a => (a : ℕ)) (by decide))]
  have h1 : numProperEdges edgeLoops =
      (univ.filter (fun p : Fin 2 × Fin 2 => edgeLoops p.1 p.2 ∧ p.1 ≠ p.2)).card := by
    unfold numProperEdges; congr 1; ext p; simp
  have : numProperEdges edgeLoops = 1 := by rw [h1]; decide
  rw [this]
  norm_num

lemma reach_edgeNoLoops {n : ℕ} {a b : Fin 2 × Fin (n + 1)}
    (h : Relation.ReflTransGen (unrolled edgeNoLoops n) a b) :
    a = b ∨ unrolled edgeNoLoops n a b := by
  induction h with
  | refl => exact Or.inl rfl
  | @tail b c _ hbc ih =>
    rcases ih with rfl | hab
    · exact Or.inr hbc
    · exact absurd hab.1.2 (by rw [hbc.1.1]; decide)

/-- **Remark 1.12**, without self-loops: `s_ρ(G_T) = 1/2` for every `T ≥ 2` (the lemmas of
Section 1 need the self-loops). -/
theorem edgeNoLoops_sRho {n : ℕ} (hn : 1 ≤ n) : sRho (unrolled edgeNoLoops n) = 1 / 2 := by
  classical
  have hscore : ∀ e ∈ edges edgeNoLoops, ∀ s : Fin n,
      score (rho (unrolled edgeNoLoops n) (e.1, s.castSucc))
        (rho (unrolled edgeNoLoops n) (e.2, s.succ)) = 1 / 2 := by
    intro e he s
    rw [mem_edges] at he
    have hedge := unrolled_edge (E := edgeNoLoops) he s
    rw [rho_eq_card, rho_eq_card, score_of_subset (relSet_subset_of_edge hedge)]
    have : relSet (unrolled edgeNoLoops n) (e.1, s.castSucc) =
        relSet (unrolled edgeNoLoops n) (e.2, s.succ) := by
      apply le_antisymm (relSet_subset_of_edge hedge)
      intro z hz
      rw [mem_relSet] at hz ⊢
      obtain ⟨x, hx1, hx2⟩ := hz
      rcases reach_edgeNoLoops hx1 with rfl | hx
      · -- `x = e.2(s+1)`, which has no out-edges: `z = x`
        rcases reach_edgeNoLoops hx2 with rfl | hz
        · exact ⟨_, Relation.ReflTransGen.refl, Relation.ReflTransGen.single hedge⟩
        · have h0 := hz.1.1
          simp only at h0
          rw [he.2] at h0
          exact absurd h0 (by decide)
      · -- `x` is a parent of `e.2(s+1)`, i.e. `x = e.1(s)`
        have hx1' : x = (e.1, s.castSucc) := by
          obtain ⟨⟨h1, -⟩, h2⟩ := hx
          ext
          · rw [h1, he.1]
          · simp at h2; simp; omega
        subst hx1'
        exact ⟨_, Relation.ReflTransGen.refl, hx2⟩
    simp [this]
  have hE : (edges edgeNoLoops).card = 1 := by
    rw [show edges edgeNoLoops = univ.filter (fun p : Fin 2 × Fin 2 => edgeNoLoops p.1 p.2) by
      ext p; simp]
    decide
  unfold sRho sOf
  rw [sum_edges_unrolled, card_edges_unrolled,
    sum_congr rfl (fun e he => sum_congr rfl (fun s _ => hscore e he s)), hE]
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  simp [hE]
  field_simp

/-! ### `Δ` cannot be lowered in Theorem 1.9 (burn-in) -/

/-- With burn-in `B = 0`, window relatives are the relatives of `G_T` itself. -/
lemma relBurn_zero_iff (hl : ∀ v, E v v) {m : ℕ} (a b : V × Fin (m + 1)) :
    relBurn E 0 m a b ↔ IsRel (unrolled E m) a b := by
  obtain ⟨v, t⟩ := a
  obtain ⟨w, r⟩ := b
  unfold relBurn burnEmb
  rw [isRel_unrolled_iff hl, isRel_unrolled_iff hl]
  simp

/-- With burn-in `B = 0`, the windowed score is the finite-`G_T` score. -/
lemma sBurn_zero (hl : ∀ v, E v v) (m : ℕ) : sBurn E 0 m = sRho (unrolled E m) := by
  have h : rhoBurn E 0 m = rho (unrolled E m) := by
    funext a
    unfold rhoBurn rho cntOf setOf'
    congr 1
    ext b
    simp only [mem_filter, mem_univ, true_and]
    exact relBurn_zero_iff hl a b
  unfold sBurn sRho
  rw [edgeBurn_eq, h]

lemma edgeLoops_loops : ∀ v, edgeLoops v v := fun _ => Or.inl rfl

/-- The single edge `0 → 1` with self-loops has `Δ = 1`. -/
lemma maxFinDist_edgeLoops : maxFinDist edgeLoops = 1 := by
  have h1 := maxFinDist_le_card_sub_one edgeLoops_loops
  simp only [Fintype.card_fin] at h1
  have h0 : maxFinDist edgeLoops ≠ 0 := by
    intro h0
    have hd := dist_le_maxFinDist (E := edgeLoops) (u := 0) (v := 1)
      (Relation.ReflTransGen.single (Or.inr ⟨rfl, rfl⟩))
    rw [h0, dist_le_iff edgeLoops_loops, walkN_zero] at hd
    exact absurd hd (by decide)
  omega

/-- **Sharpness of `B ≥ Δ` in Theorem 1.9**: for the single edge `0 → 1` with self-loops,
`Δ = 1`; with burn-in `B = Δ = 1` the windowed score equals the stationary value `1/2`, but with
`B = Δ - 1 = 0` it is strictly larger (at least `1/2 + 1/(2|E_S|) = 2/3`) for every `T ≥ 2`. So
the hypothesis `Δ ≤ B` of `sBurn_eq_maxFinDist` cannot be weakened to `Δ - 1 ≤ B` in general. -/
theorem edgeLoops_burn_sharp {m : ℕ} (hm : 1 ≤ m) :
    maxFinDist edgeLoops = 1 ∧ statValue edgeLoops = 1 / 2 ∧
    sBurn edgeLoops 1 m = statValue edgeLoops ∧
    statValue edgeLoops < sBurn edgeLoops (maxFinDist edgeLoops - 1) m := by
  classical
  have hst : statValue edgeLoops = 1 / 2 :=
    stat_half_of_outForest (E := edgeLoops) edgeLoops_loops (by unfold OutForest; decide)
  refine ⟨maxFinDist_edgeLoops, hst,
    sBurn_eq_maxFinDist edgeLoops_loops maxFinDist_edgeLoops.le hm, ?_⟩
  rw [maxFinDist_edgeLoops, Nat.sub_self, sBurn_zero edgeLoops_loops, hst]
  have hac := acyclic_of_rank (E := edgeLoops) (fun a => (a : ℕ)) (by decide)
  have hcross : crossEdges edgeLoops ≠ ∅ := by
    intro h
    have hmem : ((0 : Fin 2), (1 : Fin 2)) ∈ crossEdges edgeLoops := by
      unfold crossEdges
      rw [mem_filter, mem_edges]
      refine ⟨Or.inr ⟨rfl, rfl⟩, fun hs => ?_⟩
      exact absurd (hac 0 1 hs.1 hs.2) (by decide)
    rw [h] at hmem
    exact Finset.notMem_empty _ hmem
  have hge := sRho_unrolled_ge_of_cross edgeLoops_loops hcross hm
  have hE : (0 : ℝ) < (edges edgeLoops).card := by exact_mod_cast edges_card_pos edgeLoops_loops
  have : (0 : ℝ) < 1 / (2 * ((edges edgeLoops).card : ℝ)) := by positivity
  linarith

end RelSort
