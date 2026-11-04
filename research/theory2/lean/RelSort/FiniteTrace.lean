/-
The block-matrix algebra of Lemma 1.2 in `research/theory2/finite/FINITE.md`.

`Y = (X_a(1..n), X_b(1..n))` has covariance `C = [[C_aa, C_ab],[C_ba, C_bb]]` with Toeplitz blocks
`(C_ab)_{st} = γ_ab(s - t)` and `C_ba = C_abᵀ`, and `M = (1/2n)[[0, I],[I, 0]]`.

* `quad_M`: `Yᵀ M Y = (1/n) Σ_t X_a(t) X_b(t)` (`γ̃_ab` is a quadratic form).
* `trace_MC`: `tr(MC) = γ_ab(0)`.
* `lemma12_trace`: `2n tr(MCMC) = (1/n) Σ_{s,t} [γ_aa(s-t)γ_bb(s-t) + γ_ab(s-t)γ_ab(t-s)]`.
* `lemma12_fejer`: the same as `Σ_{|h|<n} (1 - |h|/n) [...]` (via `fejer_identity'`).
* `lemma12_le_B`: it is at most `B_ab = Σ_h |γ_aa(h)γ_bb(h) + γ_ab(h)γ_ab(-h)|`.
* `lemma12_diag`, `lemma12_diag_le_D`: Lemma 1.2(ii), `2n‖C_aa/n‖_F² = (2/n) Σ_{s,t} γ_aa(s-t)² ≤ D`.

Not formalised: `‖A‖_F² = tr(MCMC)` for `A = C^{1/2} M C^{1/2}` (cyclicity of the trace with the
matrix square root), `Var(gᵀAg) = 2‖A‖_F²` (a Gaussian moment identity), and the operator-norm
bounds.
-/
import RelSort.T2Algebra

open Finset Matrix

namespace RelSort

variable {n : ℕ}

/-- The Toeplitz block `(s,t) ↦ Γ(s - t)`. -/
def toep (n : ℕ) (Γ : ℤ → ℝ) : Matrix (Fin n) (Fin n) ℝ := Matrix.of fun s t => Γ ((s : ℤ) - t)

/-- The covariance matrix `C` of `(X_a(1..n), X_b(1..n))`. -/
def covBlock (n : ℕ) (Γaa Γbb Γab : ℤ → ℝ) : Matrix (Fin n ⊕ Fin n) (Fin n ⊕ Fin n) ℝ :=
  Matrix.fromBlocks (toep n Γaa) (toep n Γab) (toep n Γab)ᵀ (toep n Γbb)

/-- `M = (1/2n) [[0, I],[I, 0]]`. -/
noncomputable def Mswap (n : ℕ) : Matrix (Fin n ⊕ Fin n) (Fin n ⊕ Fin n) ℝ :=
  (1 / (2 * n : ℝ)) • Matrix.fromBlocks 0 1 1 0

/-- `Yᵀ M Y = (1/n) Σ_t x_a(t) x_b(t)`. -/
theorem quad_M (hn : 0 < n) (xa xb : Fin n → ℝ) :
    Sum.elim xa xb ⬝ᵥ (Mswap n *ᵥ Sum.elim xa xb) = (1 / (n : ℝ)) * ∑ t, xa t * xb t := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  unfold Mswap
  rw [Matrix.smul_mulVec, Matrix.fromBlocks_mulVec, dotProduct_smul]
  simp only [Matrix.zero_mulVec, Matrix.one_mulVec, zero_add, add_zero, smul_eq_mul,
    dotProduct, Fintype.sum_sum_type, Function.comp_apply, Sum.elim_inl, Sum.elim_inr]
  rw [show ∑ x, xb x * xa x = ∑ x, xa x * xb x from sum_congr rfl (fun _ _ => mul_comm _ _)]
  field_simp
  ring

/-- `tr(MC) = γ_ab(0)`. -/
theorem trace_MC (hn : 0 < n) (Γaa Γbb Γab : ℤ → ℝ) :
    Matrix.trace (Mswap n * covBlock n Γaa Γbb Γab) = Γab 0 := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  unfold Mswap covBlock
  rw [Matrix.smul_mul, Matrix.fromBlocks_multiply, Matrix.trace_smul]
  simp [Matrix.trace, Fintype.sum_sum_type, toep]
  field_simp
  ring

/-- **Lemma 1.2(i), trace identity.** For even `γ_aa`, `γ_bb` (autocovariances):
`2n tr(MCMC) = (1/n) Σ_{s,t} [γ_aa(s-t)γ_bb(s-t) + γ_ab(s-t)γ_ab(t-s)]`, i.e.
`4n² tr(MCMC) = 2 Σ_{s,t} [...]`. -/
theorem lemma12_trace (hn : 0 < n) (Γaa Γbb Γab : ℤ → ℝ) (haa : ∀ h, Γaa (-h) = Γaa h)
    (hbb : ∀ h, Γbb (-h) = Γbb h) :
    2 * n * Matrix.trace (Mswap n * covBlock n Γaa Γbb Γab * Mswap n * covBlock n Γaa Γbb Γab) =
      (1 / (n : ℝ)) * ∑ s : Fin n, ∑ t : Fin n,
        (Γaa ((s : ℤ) - t) * Γbb ((s : ℤ) - t) + Γab ((s : ℤ) - t) * Γab ((t : ℤ) - s)) := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  have hneg : ∀ s t : Fin n, ((t : ℤ) - s) = -((s : ℤ) - t) := fun s t => by ring
  have hMC : Mswap n * covBlock n Γaa Γbb Γab = (1 / (2 * n : ℝ)) •
      Matrix.fromBlocks (toep n Γab)ᵀ (toep n Γbb) (toep n Γaa) (toep n Γab) := by
    unfold Mswap covBlock
    rw [Matrix.smul_mul, Matrix.fromBlocks_multiply]
    simp
  rw [Matrix.mul_assoc (Mswap n * covBlock n Γaa Γbb Γab), hMC,
    Matrix.smul_mul, Matrix.mul_smul, Matrix.fromBlocks_multiply, Matrix.trace_smul,
    Matrix.trace_smul]
  simp only [smul_eq_mul, Matrix.trace, Fintype.sum_sum_type, Matrix.diag,
    Matrix.fromBlocks_apply₁₁, Matrix.fromBlocks_apply₂₂, Matrix.add_apply, Matrix.mul_apply, toep,
    Matrix.of_apply, Matrix.transpose_apply, Finset.sum_add_distrib]
  have e1 : ∑ x : Fin n, ∑ j : Fin n, Γbb ((x : ℤ) - j) * Γaa ((j : ℤ) - x) =
      ∑ s : Fin n, ∑ t : Fin n, Γaa ((s : ℤ) - t) * Γbb ((s : ℤ) - t) := by
    refine sum_congr rfl (fun s _ => sum_congr rfl (fun t _ => ?_))
    rw [hneg, haa]; ring
  have e2 : ∑ x : Fin n, ∑ j : Fin n, Γaa ((x : ℤ) - j) * Γbb ((j : ℤ) - x) =
      ∑ s : Fin n, ∑ t : Fin n, Γaa ((s : ℤ) - t) * Γbb ((s : ℤ) - t) := by
    refine sum_congr rfl (fun s _ => sum_congr rfl (fun t _ => ?_))
    rw [hneg s t, hbb]
  have e3 : ∑ x : Fin n, ∑ j : Fin n, Γab ((j : ℤ) - x) * Γab ((x : ℤ) - j) =
      ∑ s : Fin n, ∑ t : Fin n, Γab ((s : ℤ) - t) * Γab ((t : ℤ) - s) := by
    refine sum_congr rfl (fun s _ => sum_congr rfl (fun t _ => ?_))
    ring
  rw [e1, e2, e3, ← sum_add_distrib]
  simp only [← sum_add_distrib]
  rw [Finset.mul_sum, Finset.mul_sum]
  field_simp
  rw [← Finset.sum_div, mul_div_assoc', div_eq_iff (by positivity), Finset.mul_sum, Finset.sum_mul]
  refine sum_congr rfl (fun s _ => ?_)
  rw [Finset.mul_sum, Finset.sum_mul]
  refine sum_congr rfl (fun t _ => ?_)
  ring

/-- **Lemma 1.2(i), Fejér form.** `(1/n) Σ_{s,t} g(s-t) = g(0) + Σ_{h=1}^{n-1} (1 - h/n)(g(h) + g(-h))`,
which is `Σ_{|h|<n} (1 - |h|/n) g(h)`. -/
theorem lemma12_fejer (hn : 0 < n) (g : ℤ → ℝ) :
    (1 / (n : ℝ)) * ∑ s : Fin n, ∑ t : Fin n, g ((s : ℤ) - t) =
      g 0 + ∑ h ∈ Ico 1 n, (1 - (h : ℝ) / n) * (g h + g (-(h : ℤ))) := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  have hsum : ∑ s : Fin n, ∑ t : Fin n, g ((s : ℤ) - t) =
      ∑ u ∈ range n, ∑ w ∈ range n, g ((u : ℤ) - w) := by
    rw [← Fin.sum_univ_eq_sum_range (fun u => ∑ w ∈ range n, g ((u : ℤ) - w))]
    refine sum_congr rfl (fun s _ => ?_)
    rw [← Fin.sum_univ_eq_sum_range (fun w => g ((s : ℕ) - (w : ℤ)))]
  rw [hsum, fejer_identity']
  simp only [nsmul_eq_mul, mul_add, Finset.mul_sum]
  have hmem : ∀ h ∈ Ico 1 n, ((n - h : ℕ) : ℝ) = n - h := fun h hh => by
    rw [Nat.cast_sub (mem_Ico.mp hh).2.le]
  rw [sum_congr rfl (fun h hh => by rw [hmem h hh])]
  field_simp

/-- `(1/n) Σ_{s,t} g(s-t) ≤ Σ_{h∈ℤ} |g(h)|` for summable `|g|`. -/
theorem avg_toeplitz_le_tsum (hn : 0 < n) (g : ℤ → ℝ) (hg : Summable fun h => |g h|) :
    (1 / (n : ℝ)) * ∑ s : Fin n, ∑ t : Fin n, g ((s : ℤ) - t) ≤ ∑' h, |g h| := by
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  have hrow : ∀ s : Fin n, ∑ t : Fin n, g ((s : ℤ) - t) ≤ ∑' h, |g h| := by
    intro s
    have hinj : Function.Injective (fun t : Fin n => (s : ℤ) - t) := by
      intro t t' h
      simp only [sub_right_inj, Nat.cast_inj] at h
      exact Fin.ext h
    calc ∑ t : Fin n, g ((s : ℤ) - t) ≤ ∑ t : Fin n, |g ((s : ℤ) - t)| :=
          sum_le_sum (fun t _ => le_abs_self _)
      _ = ∑ h ∈ univ.image (fun t : Fin n => (s : ℤ) - t), |g h| := by
          rw [sum_image (fun x _ y _ h => hinj h)]
      _ ≤ ∑' h, |g h| := hg.sum_le_tsum _ (fun _ _ => abs_nonneg _)
  calc (1 / (n : ℝ)) * ∑ s : Fin n, ∑ t : Fin n, g ((s : ℤ) - t)
      ≤ (1 / (n : ℝ)) * ∑ _s : Fin n, ∑' h, |g h| := by
        gcongr with s; exact hrow s
    _ = ∑' h, |g h| := by
        rw [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]; field_simp

/-- **Lemma 1.2(i), bound by `B_ab`.** `2n tr(MCMC) ≤ B_ab = Σ_h |γ_aa(h)γ_bb(h) + γ_ab(h)γ_ab(-h)|`
when that series converges. -/
theorem lemma12_le_B (hn : 0 < n) (Γaa Γbb Γab : ℤ → ℝ) (haa : ∀ h, Γaa (-h) = Γaa h)
    (hbb : ∀ h, Γbb (-h) = Γbb h)
    (hB : Summable fun h => |Γaa h * Γbb h + Γab h * Γab (-h)|) :
    2 * n * Matrix.trace (Mswap n * covBlock n Γaa Γbb Γab * Mswap n * covBlock n Γaa Γbb Γab) ≤
      ∑' h, |Γaa h * Γbb h + Γab h * Γab (-h)| := by
  rw [lemma12_trace hn Γaa Γbb Γab haa hbb]
  have := avg_toeplitz_le_tsum hn (fun h => Γaa h * Γbb h + Γab h * Γab (-h)) hB
  simp only [neg_sub] at this
  exact this

/-- **Lemma 1.2(ii), trace identity.** For `A = C_aa/n` (the matrix of `γ̃_aa`) and even `γ_aa`:
`2n ‖A‖_F² = 2n tr(A A) = (2/n) Σ_{s,t} γ_aa(s-t)²`. -/
theorem lemma12_diag (hn : 0 < n) (Γ : ℤ → ℝ) (hΓ : ∀ h, Γ (-h) = Γ h) :
    2 * n * Matrix.trace (((1 / (n : ℝ)) • toep n Γ) * ((1 / (n : ℝ)) • toep n Γ)) =
      (2 / (n : ℝ)) * ∑ s : Fin n, ∑ t : Fin n, Γ ((s : ℤ) - t) ^ 2 := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  rw [Matrix.smul_mul, Matrix.mul_smul, Matrix.trace_smul, Matrix.trace_smul]
  simp only [smul_eq_mul, Matrix.trace, Matrix.diag, Matrix.mul_apply, toep, Matrix.of_apply]
  have e : ∀ s t : Fin n, Γ ((s : ℤ) - t) * Γ ((t : ℤ) - s) = Γ ((s : ℤ) - t) ^ 2 := by
    intro s t
    rw [show ((t : ℤ) - s) = -((s : ℤ) - t) by ring, hΓ, sq]
  simp only [e]
  field_simp

/-- **Lemma 1.2(ii), bound by `D`.** `(2/n) Σ_{s,t} γ_aa(s-t)² ≤ 2 Σ_h γ_aa(h)²`. -/
theorem lemma12_diag_le_D (hn : 0 < n) (Γ : ℤ → ℝ) (hs : Summable fun h => Γ h ^ 2) :
    (2 / (n : ℝ)) * ∑ s : Fin n, ∑ t : Fin n, Γ ((s : ℤ) - t) ^ 2 ≤ 2 * ∑' h, Γ h ^ 2 := by
  have hs' : Summable fun h => |Γ h ^ 2| := by simp only [abs_pow, sq_abs]; exact hs
  have := avg_toeplitz_le_tsum hn (fun h => Γ h ^ 2) hs'
  simp only [abs_pow, sq_abs] at this
  rw [show (2 / (n : ℝ)) = 2 * (1 / n) by ring, mul_assoc]
  linarith

end RelSort
