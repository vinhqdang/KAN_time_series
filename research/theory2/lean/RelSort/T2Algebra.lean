/-
Algebraic cores of THEORY.md Section 2 (T2). The probabilistic parts (stationarity, the
covariance of a block sum, MA(∞) representation, genericity) are not formalised; see README.

* Prop. 2.1 (Fejér identity): the counting identity behind
  `Cov(Σ_u X(sD+u)) = Σ_{u,w} Γ(u-w) = Σ_{|h|<D} (D-|h|) Γ(h)`, for any `Γ : ℤ → M`.
* Prop. 2.3: the factorisation `I - A(1) = (I - Λ)(I - B̃)` with `B̃ = (I - Λ)⁻¹ N`, and
  `λ_j < 1` from stability of the univariate AR polynomial on `[0,1]`.
-/
import Mathlib

open Finset

namespace RelSort

section Fejer

variable {M : Type*} [AddCommGroup M]

lemma fejer_row (f : ℤ → M) (D : ℕ) :
    ∑ w ∈ range D, f ((D : ℤ) - w) = ∑ h ∈ range (D + 1), f h - f 0 := by
  induction D with
  | zero => simp
  | succ D ih =>
    rw [sum_range_succ', sum_range_succ]
    push_cast
    have : ∀ i : ℕ, ((D : ℤ) + 1 - (i + 1)) = (D : ℤ) - i := fun i => by ring
    simp only [this, sub_zero]
    rw [ih]
    abel

lemma fejer_col (f : ℤ → M) (D : ℕ) :
    ∑ u ∈ range D, f ((u : ℤ) - D) = ∑ h ∈ range (D + 1), f (-(h : ℤ)) - f 0 := by
  induction D with
  | zero => simp
  | succ D ih =>
    rw [sum_range_succ', sum_range_succ]
    push_cast
    have : ∀ i : ℕ, ((i : ℤ) + 1 - (D + 1)) = (i : ℤ) - D := fun i => by ring
    simp only [this, zero_sub]
    rw [ih]
    abel

/-- **Prop. 2.1 (Fejér identity), counting form**:
`Σ_{u,w=0}^{D-1} Γ(u-w) = Σ_{h=0}^{D-1} (D-h)(Γ(h) + Γ(-h)) - D Γ(0)`
`= D Γ(0) + Σ_{h=1}^{D-1} (D-h)(Γ(h) + Γ(-h)) = Σ_{|h|<D} (D-|h|) Γ(h)`. -/
theorem fejer_identity (f : ℤ → M) (D : ℕ) :
    ∑ u ∈ range D, ∑ w ∈ range D, f ((u : ℤ) - w) =
      ∑ h ∈ range D, (D - h) • (f h + f (-(h : ℤ))) - D • f 0 := by
  induction D with
  | zero => simp
  | succ D ih =>
    have hL : ∑ u ∈ range (D + 1), ∑ w ∈ range (D + 1), f ((u : ℤ) - w) =
        ∑ u ∈ range D, ∑ w ∈ range D, f ((u : ℤ) - w) + ∑ w ∈ range D, f ((D : ℤ) - w) +
          ∑ u ∈ range D, f ((u : ℤ) - D) + f 0 := by
      simp only [sum_range_succ, sum_add_distrib, sub_self]
      abel
    have hR : ∑ h ∈ range (D + 1), (D + 1 - h) • (f h + f (-(h : ℤ))) =
        ∑ h ∈ range D, (D - h) • (f h + f (-(h : ℤ))) +
          ∑ h ∈ range (D + 1), (f h + f (-(h : ℤ))) := by
      have e : ∀ h ∈ range (D + 1), (D + 1 - h) • (f h + f (-(h : ℤ))) =
          (D - h) • (f h + f (-(h : ℤ))) + (f h + f (-(h : ℤ))) := by
        intro h hh
        rw [mem_range] at hh
        rw [show D + 1 - h = (D - h) + 1 by omega, add_smul, one_smul]
      rw [sum_congr rfl e, sum_add_distrib,
        sum_range_succ (fun h => (D - h) • (f h + f (-(h : ℤ))))]
      simp
    rw [hL, ih, fejer_row, fejer_col, hR, sum_add_distrib, succ_nsmul]
    abel

/-- The same identity in the symmetric form `D Γ(0) + Σ_{h=1}^{D-1} (D-h)(Γ(h) + Γ(-h))`. -/
theorem fejer_identity' (f : ℤ → M) (D : ℕ) :
    ∑ u ∈ range D, ∑ w ∈ range D, f ((u : ℤ) - w) =
      D • f 0 + ∑ h ∈ Ico 1 D, (D - h) • (f h + f (-(h : ℤ))) := by
  rw [fejer_identity]
  rcases Nat.eq_zero_or_pos D with rfl | hD
  · simp
  · rw [range_eq_Ico, ← sum_Ico_consecutive _ (Nat.zero_le 1) hD, Nat.Ico_zero_eq_range,
      range_one, sum_singleton]
    simp only [Nat.cast_zero, neg_zero, Nat.sub_zero, nsmul_add]
    abel

end Fejer

section LongRun

variable {d : ℕ}

/-- **Prop. 2.3, factorisation**: with `A(1) = Λ + N` and `I - Λ` invertible,
`I - A(1) = (I - Λ)(I - B̃)` where `B̃ = (I - Λ)⁻¹ N`. -/
theorem longrun_factor (A Λ N : Matrix (Fin d) (Fin d) ℝ) (hA : A = Λ + N)
    (hinv : IsUnit (1 - Λ).det) :
    1 - A = (1 - Λ) * (1 - (1 - Λ)⁻¹ * N) := by
  rw [Matrix.mul_sub, Matrix.mul_one, ← Matrix.mul_assoc, Matrix.mul_nonsing_inv _ hinv,
    Matrix.one_mul, hA]
  abel

/-- **Prop. 2.3, inverse**: `(I - A(1))⁻¹ = (I - B̃)⁻¹ (I - Λ)⁻¹`, so
`S_LR = (I - A(1))⁻¹ Σ (I - A(1))⁻ᵀ = (I - B̃)⁻¹ (I - Λ)⁻¹ Σ (I - Λ)⁻ᵀ (I - B̃)⁻ᵀ`. -/
theorem longrun_inv (A Λ N : Matrix (Fin d) (Fin d) ℝ) (hA : A = Λ + N)
    (hinv : IsUnit (1 - Λ).det) :
    (1 - A)⁻¹ = (1 - (1 - Λ)⁻¹ * N)⁻¹ * (1 - Λ)⁻¹ := by
  rw [longrun_factor A Λ N hA hinv, Matrix.mul_inv_rev]

/-- **Prop. 2.3, `λ_j < 1`**: if the univariate AR polynomial
`φ(z) = 1 - Σ_{τ=1}^{L} a_τ z^τ` has no root in `[0,1]` (implied by stability: no root in the
closed unit disk), then `λ = Σ_τ a_τ < 1`. -/
theorem selfloop_sum_lt_one (L : ℕ) (a : ℕ → ℝ)
    (hstab : ∀ z : ℝ, 0 ≤ z → z ≤ 1 → 1 - ∑ τ ∈ Icc 1 L, a τ * z ^ τ ≠ 0) :
    ∑ τ ∈ Icc 1 L, a τ < 1 := by
  set φ : ℝ → ℝ := fun z => 1 - ∑ τ ∈ Icc 1 L, a τ * z ^ τ with hφ
  have hcont : ContinuousOn φ (Set.Icc 0 1) := by
    apply Continuous.continuousOn
    simp only [hφ]
    fun_prop
  have h0 : φ 0 = 1 := by
    simp only [hφ]
    rw [sum_eq_zero (fun τ hτ => by
      rw [mem_Icc] at hτ
      rw [zero_pow (by omega), mul_zero])]
    simp
  have h1 : φ 1 = 1 - ∑ τ ∈ Icc 1 L, a τ := by simp [hφ]
  by_contra hcon
  push Not at hcon
  have hmem : (0 : ℝ) ∈ Set.Icc (φ 1) (φ 0) := ⟨by rw [h1]; linarith, by rw [h0]; norm_num⟩
  obtain ⟨c, ⟨hc0, hc1⟩, hc⟩ := intermediate_value_Icc' (zero_le_one' ℝ) hcont hmem
  exact hstab c hc0 hc1 hc

end LongRun

end RelSort
