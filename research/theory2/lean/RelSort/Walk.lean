/-
Walks of a given length in the summary graph `E` (self-loops allowed as steps), the distance
`d(u,v)`, and the bound "reachable ⇒ reachable by a walk of length `|V| - 1`".
-/
import RelSort.Basic

open Finset

set_option linter.unusedSectionVars false

namespace RelSort

variable {V : Type*} (E : V → V → Prop)

/-- `WalkN E n u v`: there is a directed walk of length exactly `n` from `u` to `v` in `E`. -/
def WalkN : ℕ → V → V → Prop
  | 0, u, v => u = v
  | n + 1, u, v => ∃ w, WalkN n u w ∧ E w v

variable {E}

@[simp] lemma walkN_zero {u v : V} : WalkN E 0 u v ↔ u = v := Iff.rfl

lemma walkN_succ {n : ℕ} {u v : V} : WalkN E (n + 1) u v ↔ ∃ w, WalkN E n u w ∧ E w v := Iff.rfl

lemma walkN_add {m n : ℕ} {u v : V} :
    WalkN E (m + n) u v ↔ ∃ w, WalkN E m u w ∧ WalkN E n w v := by
  induction n generalizing v with
  | zero => simp
  | succ n ih =>
    rw [← add_assoc, walkN_succ]
    constructor
    · rintro ⟨w, hw, hwv⟩
      obtain ⟨x, hx, hxw⟩ := ih.mp hw
      exact ⟨x, hx, w, hxw, hwv⟩
    · rintro ⟨x, hx, w, hxw, hwv⟩
      exact ⟨w, ih.mpr ⟨x, hx, hxw⟩, hwv⟩

lemma walkN_succ' {n : ℕ} {u v : V} : WalkN E (n + 1) u v ↔ ∃ w, E u w ∧ WalkN E n w v := by
  rw [add_comm, walkN_add]
  simp [WalkN]

lemma walkN_reach {n : ℕ} {u v : V} (h : WalkN E n u v) : Relation.ReflTransGen E u v := by
  induction n generalizing v with
  | zero => rw [walkN_zero] at h; subst h; exact Relation.ReflTransGen.refl
  | succ n ih =>
    obtain ⟨w, hw, hwv⟩ := h
    exact (ih hw).tail hwv

lemma reach_walkN {u v : V} (h : Relation.ReflTransGen E u v) : ∃ n, WalkN E n u v := by
  induction h with
  | refl => exact ⟨0, rfl⟩
  | tail _ hbc ih =>
    obtain ⟨n, hn⟩ := ih
    exact ⟨n + 1, _, hn, hbc⟩

lemma reach_iff_walkN {u v : V} : Relation.ReflTransGen E u v ↔ ∃ n, WalkN E n u v :=
  ⟨reach_walkN, fun ⟨_, h⟩ => walkN_reach h⟩

/-- With a self-loop at every node, walk lengths can be padded. -/
lemma walkN_mono (hl : ∀ v, E v v) {m n : ℕ} (hmn : m ≤ n) {u v : V} (h : WalkN E m u v) :
    WalkN E n u v := by
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hmn
  induction k with
  | zero => simpa using h
  | succ k ih =>
    rw [← add_assoc, walkN_succ]
    exact ⟨v, ih (Nat.le_add_right _ _), hl v⟩

/-! ### Distance -/

open Classical in
/-- `dist E u v = d(u,v)`: the length of a shortest walk from `u` to `v` (`⊤` if none).
Self-loops never shorten a walk, so this is the shortest path length ignoring self-loops
(`dist_eq_dist_noLoop`). -/
noncomputable def dist (E : V → V → Prop) (u v : V) : ℕ∞ :=
  if h : ∃ n, WalkN E n u v then (Nat.find h : ℕ∞) else ⊤

lemma dist_le_iff_exists {u v : V} {n : ℕ} :
    dist E u v ≤ n ↔ ∃ m ≤ n, WalkN E m u v := by
  classical
  unfold dist
  split_ifs with h
  · rw [Nat.cast_le, Nat.find_le_iff]
  · simp only [top_le_iff, ENat.natCast_ne_top, false_iff]
    rintro ⟨m, -, hm⟩
    exact h ⟨m, hm⟩

/-- With self-loops, `d(u,v) ≤ n ↔` a walk of length exactly `n` exists. -/
lemma dist_le_iff (hl : ∀ v, E v v) {u v : V} {n : ℕ} : dist E u v ≤ n ↔ WalkN E n u v := by
  rw [dist_le_iff_exists]
  exact ⟨fun ⟨m, hm, h⟩ => walkN_mono hl hm h, fun h => ⟨n, le_rfl, h⟩⟩

lemma dist_ne_top_iff {u v : V} : dist E u v ≠ ⊤ ↔ Relation.ReflTransGen E u v := by
  classical
  rw [reach_iff_walkN]
  unfold dist
  split_ifs with h
  · simp [h]
  · simp [h]

/-- The graph with self-loops removed. -/
def noLoop (E : V → V → Prop) : V → V → Prop := fun u v => E u v ∧ u ≠ v

lemma walkN_noLoop_of_walkN {n : ℕ} {u v : V} (h : WalkN E n u v) :
    ∃ m ≤ n, WalkN (noLoop E) m u v := by
  induction n generalizing v with
  | zero => exact ⟨0, le_rfl, h⟩
  | succ n ih =>
    obtain ⟨w, hw, hwv⟩ := h
    obtain ⟨m, hm, hm'⟩ := ih hw
    by_cases hwv' : w = v
    · subst hwv'; exact ⟨m, by omega, hm'⟩
    · exact ⟨m + 1, by omega, w, hm', hwv, hwv'⟩

lemma walkN_of_walkN_noLoop {n : ℕ} {u v : V} (h : WalkN (noLoop E) n u v) : WalkN E n u v := by
  induction n generalizing v with
  | zero => exact h
  | succ n ih =>
    obtain ⟨w, hw, hwv, -⟩ := h
    exact ⟨w, ih hw, hwv⟩

/-- `d` computed with or without self-loops is the same (THEORY.md: "ignoring self-loops"). -/
lemma dist_eq_dist_noLoop (u v : V) : dist E u v = dist (noLoop E) u v := by
  have key : ∀ n : ℕ, dist E u v ≤ n ↔ dist (noLoop E) u v ≤ n := by
    intro n
    rw [dist_le_iff_exists, dist_le_iff_exists]
    constructor
    · rintro ⟨m, hm, h⟩
      obtain ⟨k, hk, h'⟩ := walkN_noLoop_of_walkN h
      exact ⟨k, hk.trans hm, h'⟩
    · rintro ⟨m, hm, h⟩
      exact ⟨m, hm, walkN_of_walkN_noLoop h⟩
  apply le_antisymm
  · induction h : dist (noLoop E) u v using ENat.recTopCoe with
    | top => exact le_top
    | coe n => exact (key n).mpr (le_of_eq h)
  · induction h : dist E u v using ENat.recTopCoe with
    | top => exact le_top
    | coe n => exact (key n).mp (le_of_eq h)

/-! ### Bound: reachable implies reachable in `|V| - 1` steps (with self-loops) -/

section Bound

variable [Fintype V]

open Classical in
/-- Nodes reachable from `u` by a walk of length `n`. -/
noncomputable def ball (E : V → V → Prop) (u : V) (n : ℕ) : Finset V :=
  univ.filter (fun v => WalkN E n u v)

lemma mem_ball {u v : V} {n : ℕ} : v ∈ ball E u n ↔ WalkN E n u v := by
  classical
  unfold ball
  simp

lemma ball_mono (hl : ∀ v, E v v) (u : V) (n : ℕ) : ball E u n ⊆ ball E u (n + 1) := by
  intro v hv
  rw [mem_ball] at hv ⊢
  exact walkN_mono hl (Nat.le_succ n) hv

lemma ball_succ_eq {u : V} {n : ℕ} (h : ball E u n = ball E u (n + 1)) :
    ball E u (n + 1) = ball E u (n + 2) := by
  ext v
  rw [mem_ball, mem_ball, walkN_succ (n := n + 1), walkN_succ]
  constructor
  · rintro ⟨w, hw, hwv⟩
    refine ⟨w, ?_, hwv⟩
    rw [← mem_ball, ← h, mem_ball]; exact hw
  · rintro ⟨w, hw, hwv⟩
    refine ⟨w, ?_, hwv⟩
    rw [← mem_ball, h, mem_ball]; exact hw

lemma ball_stable {u : V} {n : ℕ} (h : ball E u n = ball E u (n + 1)) :
    ∀ k, ball E u (n + k) = ball E u n := by
  intro k
  induction k with
  | zero => rfl
  | succ k ih =>
    have : ∀ j, ball E u (n + j) = ball E u (n + j + 1) := by
      intro j
      induction j with
      | zero => exact h
      | succ j ihj => exact ball_succ_eq ihj
    rw [← add_assoc, ← this k, ih]

lemma card_ball_ge (hl : ∀ v, E v v) (u : V) :
    ∀ n, (∀ k < n, ball E u k ≠ ball E u (k + 1)) → n + 1 ≤ (ball E u n).card := by
  intro n
  induction n with
  | zero =>
    intro _
    have : u ∈ ball E u 0 := by rw [mem_ball]; rfl
    exact Finset.card_pos.mpr ⟨u, this⟩
  | succ n ih =>
    intro hne
    have h1 := ih (fun k hk => hne k (by omega))
    have hss : ball E u n ⊂ ball E u (n + 1) :=
      Finset.ssubset_iff_subset_ne.mpr ⟨ball_mono hl u n, hne n (by omega)⟩
    have := Finset.card_lt_card hss
    omega

/-- With a self-loop at every node, reachability is witnessed by a walk of length
`|V| - 1`. -/
lemma walkN_card_sub_one (hl : ∀ v, E v v) {u v : V} (h : Relation.ReflTransGen E u v) :
    WalkN E (Fintype.card V - 1) u v := by
  obtain ⟨m, hm⟩ := reach_walkN h
  have hpos : 0 < Fintype.card V := Fintype.card_pos_iff.mpr ⟨u⟩
  -- some `k < |V|` has `ball k = ball (k+1)`
  have hex : ∃ k < Fintype.card V, ball E u k = ball E u (k + 1) := by
    by_contra hcon
    push Not at hcon
    have := card_ball_ge hl u (Fintype.card V) hcon
    have := Finset.card_le_univ (ball E u (Fintype.card V))
    omega
  obtain ⟨k, hk, hkeq⟩ := hex
  have hvm : v ∈ ball E u m := by rw [mem_ball]; exact hm
  rw [← mem_ball]
  by_cases hmle : m ≤ Fintype.card V - 1
  · rw [mem_ball] at hvm ⊢; exact walkN_mono hl hmle hvm
  · have e1 := ball_stable hkeq (m - k)
    have e2 := ball_stable hkeq (Fintype.card V - 1 - k)
    rw [show k + (m - k) = m by omega] at e1
    rw [show k + (Fintype.card V - 1 - k) = Fintype.card V - 1 by omega] at e2
    rw [e2, ← e1]; exact hvm

/-- Reachable ⇒ a walk of every length `n ≥ |V| - 1`. -/
lemma walkN_of_reach (hl : ∀ v, E v v) {u v : V} (h : Relation.ReflTransGen E u v) {n : ℕ}
    (hn : Fintype.card V - 1 ≤ n) : WalkN E n u v :=
  walkN_mono hl hn (walkN_card_sub_one hl h)

/-- `d(u,v) ≤ p - 1` whenever `v` is reachable from `u`. -/
lemma dist_le_card_sub_one (hl : ∀ v, E v v) {u v : V} (h : Relation.ReflTransGen E u v) :
    dist E u v ≤ (Fintype.card V - 1 : ℕ) :=
  (dist_le_iff hl).mpr (walkN_card_sub_one hl h)

/-! ### `Δ`: the largest finite distance -/

/-- `Δ = maxFinDist E`: the largest *finite* shortest-path distance `d(u,v)` over ordered pairs
with `v` reachable from `u` (unreachable pairs, `d = ⊤`, contribute `0` through `ENat.toNat`). -/
noncomputable def maxFinDist (E : V → V → Prop) : ℕ :=
  univ.sup (fun q : V × V => (dist E q.1 q.2).toNat)

/-- Every finite distance is at most `Δ`. -/
lemma dist_le_maxFinDist {u v : V} (h : Relation.ReflTransGen E u v) :
    dist E u v ≤ (maxFinDist E : ℕ∞) := by
  have hne : dist E u v ≠ ⊤ := dist_ne_top_iff.mpr h
  have hle : (dist E u v).toNat ≤ maxFinDist E :=
    Finset.le_sup (f := fun q : V × V => (dist E q.1 q.2).toNat) (mem_univ (u, v))
  rw [← ENat.natCast_toNat hne]
  exact_mod_cast hle

/-- `Δ` is attained: some reachable pair has `d(u,v) = Δ`. -/
lemma exists_dist_eq_maxFinDist [Nonempty V] :
    ∃ u v, Relation.ReflTransGen E u v ∧ dist E u v = (maxFinDist E : ℕ∞) := by
  obtain ⟨q, -, hq⟩ := Finset.exists_mem_eq_sup (univ : Finset (V × V)) univ_nonempty
    (fun q : V × V => (dist E q.1 q.2).toNat)
  by_cases htop : dist E q.1 q.2 = ⊤
  · obtain ⟨x⟩ := ‹Nonempty V›
    refine ⟨x, x, Relation.ReflTransGen.refl, ?_⟩
    have h0 : dist E x x = 0 := by
      have : dist E x x ≤ (0 : ℕ) := dist_le_iff_exists.mpr ⟨0, le_rfl, rfl⟩
      simpa using this
    rw [h0, maxFinDist, hq, htop]
    simp
  · refine ⟨q.1, q.2, dist_ne_top_iff.mp htop, ?_⟩
    rw [maxFinDist, hq, ENat.natCast_toNat htop]

/-- `Δ ≤ p - 1` (with self-loops at every node). -/
lemma maxFinDist_le_card_sub_one (hl : ∀ v, E v v) : maxFinDist E ≤ Fintype.card V - 1 := by
  unfold maxFinDist
  apply Finset.sup_le
  intro q _
  by_cases htop : dist E q.1 q.2 = ⊤
  · simp [htop]
  · have h := dist_le_card_sub_one hl (dist_ne_top_iff.mp htop)
    rw [← ENat.natCast_toNat htop] at h
    exact_mod_cast h

/-- Reachable ⇒ a walk of every length `n ≥ Δ` (sharpening of `walkN_of_reach`). -/
lemma walkN_of_reach_maxFinDist (hl : ∀ v, E v v) {u v : V} (h : Relation.ReflTransGen E u v)
    {n : ℕ} (hn : maxFinDist E ≤ n) : WalkN E n u v :=
  (dist_le_iff hl).mp ((dist_le_maxFinDist h).trans (by exact_mod_cast hn))

end Bound

end RelSort
