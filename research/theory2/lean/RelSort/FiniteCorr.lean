/-
Deterministic parts of F1 and F2 in `research/theory2/finite/FINITE.md` (Sections 1-2).

* Lemma 1.4 (correlation perturbation): `corr_perturb`.
* Lemma 1.3, the deterministic steps on the deviation event: `centred_cov_eq` (the centred sample
  covariance is the uncentred one minus the product of the means), `centred_dev_offdiag`,
  `centred_dev_diag`, the thresholds `lemma13_eps`, and the union-bound count `lemma13_count`.
* Lemma 1.1, the analytic and algebraic steps: `phi_le` (the bound on
  `φ(u) = -½ log(1-2u) - u`), `chernoff_logmgf_le` (the sum bound on the log-MGF),
  `chernoff_choice` (the choice of `s` gives exponent exactly `x`).
* Theorem 1.5, deterministic core: `decide_eq_target`, `decide_eq_support`, `thm15_det`; the
  choice of `y` in the explicit form: `explicit_y`. Theorem 2.2 constant: `thm22_const`.
* Corollary 1.6 and the saturation step of Theorem 2.2: `cntDecl`, `sOf_cntDecl_congr`,
  `sOf_const`, `sOf_saturated`, `abs_sOf_sub_half_le`.
-/
import RelSort.Basic

open Finset Real

set_option linter.unusedSectionVars false

namespace RelSort

/-! ### Lemma 1.4 -/

/-- **Lemma 1.4 (correlation perturbation).** If `a', b' ∈ [1-ε_d, 1+ε_d]` with `ε_d < 1` and
`|c' - ρ| ≤ ε`, then `|c'/√(a'b') - ρ| ≤ (ε + |ρ| ε_d)/(1 - ε_d)`. The hypothesis `|ρ| ≤ 1` of
FINITE.md is not needed. -/
theorem corr_perturb {a b c ρ ε εd : ℝ} (hεd : εd < 1)
    (ha : a ∈ Set.Icc (1 - εd) (1 + εd)) (hb : b ∈ Set.Icc (1 - εd) (1 + εd))
    (hc : |c - ρ| ≤ ε) :
    |c / √(a * b) - ρ| ≤ (ε + |ρ| * εd) / (1 - εd) := by
  obtain ⟨ha1, ha2⟩ := ha
  obtain ⟨hb1, hb2⟩ := hb
  have hpos : 0 < 1 - εd := by linarith
  have hε : 0 ≤ ε := le_trans (abs_nonneg _) hc
  have hεd0 : 0 ≤ εd := by linarith
  have hsl : 1 - εd ≤ √(a * b) := by
    rw [show 1 - εd = √((1 - εd) ^ 2) from (Real.sqrt_sq hpos.le).symm]
    exact Real.sqrt_le_sqrt (by nlinarith)
  have hsu : √(a * b) ≤ 1 + εd := by
    rw [show 1 + εd = √((1 + εd) ^ 2) from (Real.sqrt_sq (by linarith)).symm]
    exact Real.sqrt_le_sqrt (by nlinarith)
  set s := √(a * b) with hs
  have spos : 0 < s := lt_of_lt_of_le hpos hsl
  have h1 : |1 - s| ≤ εd := abs_le.mpr ⟨by linarith, by linarith⟩
  have key : c / s - ρ = ((c - ρ) + ρ * (1 - s)) / s := by field_simp; ring
  have hnum : |(c - ρ) + ρ * (1 - s)| ≤ ε + |ρ| * εd := by
    calc |(c - ρ) + ρ * (1 - s)| ≤ |c - ρ| + |ρ * (1 - s)| := abs_add_le _ _
      _ ≤ ε + |ρ| * εd := by rw [abs_mul]; gcongr
  rw [key, abs_div, abs_of_pos spos, div_le_div_iff₀ spos hpos]
  have h0 : 0 ≤ ε + |ρ| * εd := by positivity
  calc |(c - ρ) + ρ * (1 - s)| * (1 - εd) ≤ (ε + |ρ| * εd) * (1 - εd) := by gcongr
    _ ≤ (ε + |ρ| * εd) * s := by gcongr

/-! ### Lemma 1.3, deterministic steps -/

/-- The centred sample covariance `(1/n) Σ (x_t - x̄)(y_t - ȳ)` equals `(1/n) Σ x_t y_t - x̄ ȳ`
(`Ĉ_ab = γ̃_ab - X̄_a X̄_b`), for `n ≥ 1` observations. -/
theorem centred_cov_eq {n : ℕ} (hn : 0 < n) (x y : Fin n → ℝ) :
    (1 / (n : ℝ)) * ∑ t, (x t - (1 / (n : ℝ)) * ∑ s, x s) * (y t - (1 / (n : ℝ)) * ∑ s, y s) =
      (1 / (n : ℝ)) * ∑ t, x t * y t -
        ((1 / (n : ℝ)) * ∑ s, x s) * ((1 / (n : ℝ)) * ∑ s, y s) := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  set X := ∑ s, x s
  set Y := ∑ s, y s
  have e : ∀ t, (x t - 1 / (n : ℝ) * X) * (y t - 1 / (n : ℝ) * Y) =
      x t * y t - (1 / (n : ℝ) * Y) * x t - (1 / (n : ℝ) * X) * y t +
        (1 / (n : ℝ) * X) * (1 / (n : ℝ) * Y) := fun t => by ring
  simp only [e, sum_add_distrib, sum_sub_distrib, ← mul_sum, sum_const, card_univ,
    Fintype.card_fin, nsmul_eq_mul]
  field_simp
  ring

/-- If `|m| ≤ √t` and `|m'| ≤ √t` with `t ≥ 0`, then `|m m'| ≤ t`. -/
lemma abs_mul_le_of_sqrt {m m' t : ℝ} (ht : 0 ≤ t) (hm : |m| ≤ √t) (hm' : |m'| ≤ √t) :
    |m * m'| ≤ t := by
  rw [abs_mul]
  calc |m| * |m'| ≤ √t * √t := mul_le_mul hm hm' (abs_nonneg _) (Real.sqrt_nonneg _)
    _ = t := Real.mul_self_sqrt ht

/-- **Lemma 1.3, off-diagonal step.** On the event where the uncentred `γ̃_ab` is within
`√(2B x/n) + κ₂ x/n` of `ρ_ab` and both sample means are at most `√(2κ₁ y/n)` in absolute value,
`|Ĉ_ab - ρ_ab| ≤ ε_ab(x,y) = √(2B x/n) + κ₂ x/n + 2κ₁ y/n`. -/
theorem centred_dev_offdiag {g ma mb ρ B κ1 κ2 x y n : ℝ} (hκ1 : 0 ≤ κ1) (hy : 0 ≤ y) (hn : 0 < n)
    (hg : |g - ρ| ≤ √(2 * B * x / n) + κ2 * x / n)
    (ha : |ma| ≤ √(2 * κ1 * y / n)) (hb : |mb| ≤ √(2 * κ1 * y / n)) :
    |(g - ma * mb) - ρ| ≤ √(2 * B * x / n) + κ2 * x / n + 2 * κ1 * y / n := by
  have ht : 0 ≤ 2 * κ1 * y / n := by positivity
  have hm := abs_mul_le_of_sqrt ht ha hb
  calc |(g - ma * mb) - ρ| = |(g - ρ) + (-(ma * mb))| := by ring_nf
    _ ≤ |g - ρ| + |-(ma * mb)| := abs_add_le _ _
    _ ≤ _ := by rw [abs_neg]; linarith

/-- **Lemma 1.3, diagonal step.** `|Ĉ_aa - 1| ≤ ε_d(y) = √(2D y/n) + 4κ₁ y/n` on the event. -/
theorem centred_dev_diag {g m D κ1 y n : ℝ} (hκ1 : 0 ≤ κ1) (hy : 0 ≤ y) (hn : 0 < n)
    (hg : |g - 1| ≤ √(2 * D * y / n) + 2 * κ1 * y / n)
    (hm : |m| ≤ √(2 * κ1 * y / n)) :
    |(g - m * m) - 1| ≤ √(2 * D * y / n) + 4 * κ1 * y / n := by
  have ht : 0 ≤ 2 * κ1 * y / n := by positivity
  have hmm := abs_mul_le_of_sqrt ht hm hm
  have h4 : 4 * κ1 * y / n = 2 * κ1 * y / n + 2 * κ1 * y / n := by ring
  calc |(g - m * m) - 1| = |(g - 1) + (-(m * m))| := by ring_nf
    _ ≤ |g - 1| + |-(m * m)| := abs_add_le _ _
    _ ≤ _ := by rw [abs_neg]; linarith

/-- **Lemma 1.3, union-bound count.** `m = d(d-1)/2` off-diagonal pairs at `2e^{-x}` each, `d`
diagonal terms at `2e^{-y}` and `d` sample means at `2e^{-y}` add up to
`d(d-1) e^{-x} + 4 d e^{-y}`. -/
theorem lemma13_count (d : ℕ) (x y : ℝ) :
    (d.choose 2 : ℝ) * (2 * exp (-x)) + d * (2 * exp (-y)) + d * (2 * exp (-y)) =
      d * (d - 1) * exp (-x) + 4 * d * exp (-y) := by
  rw [Nat.cast_choose_two ℝ]
  ring

/-- **Lemma 1.3, thresholds from Lemma 1.2.** If `2n‖A‖_F² ≤ B` and `2n‖A‖_op ≤ κ` (`n > 0`),
the Lemma 1.1 deviation `2‖A‖_F √x + 2‖A‖_op x` is at most `√(2Bx/n) + κx/n`. (Off-diagonal:
`B = B_ab`, `κ = κ₂`; diagonal: `B = D`, `κ = 2κ₁`, since `‖A‖_op ≤ κ₁/n`.) -/
theorem lemma13_eps {F O B κ x n : ℝ} (hF : 0 ≤ F) (hx : 0 ≤ x) (hn : 0 < n)
    (hB : 2 * n * F ^ 2 ≤ B) (hO : 2 * n * O ≤ κ) :
    2 * F * √x + 2 * O * x ≤ √(2 * B * x / n) + κ * x / n := by
  have h1 : 2 * F * √x ≤ √(2 * B * x / n) := by
    rw [show 2 * F * √x = √((2 * F) ^ 2 * x) by
      rw [Real.sqrt_mul (sq_nonneg _), Real.sqrt_sq (by positivity)]]
    apply Real.sqrt_le_sqrt
    rw [le_div_iff₀ hn]
    nlinarith
  have h2 : 2 * O * x ≤ κ * x / n := by
    rw [le_div_iff₀ hn]
    nlinarith
  linarith

/-- **Theorem 1.5, explicit form, choice of `y`.** For `d ≥ 2` and
`y = x + max(log(4/(d-1)), 0)`: `d(d-1)e^{-x} + 4d e^{-y} ≤ 2d(d-1)e^{-x}`. -/
theorem explicit_y (d : ℕ) (hd : 2 ≤ d) (x : ℝ) :
    (d : ℝ) * (d - 1) * exp (-x) + 4 * d * exp (-(x + max (Real.log (4 / (d - 1))) 0)) ≤
      2 * d * (d - 1) * exp (-x) := by
  have hd1 : (1 : ℝ) ≤ (d : ℝ) - 1 := by
    have : (2 : ℝ) ≤ d := by exact_mod_cast hd
    linarith
  have hdpos : (0 : ℝ) < d := by linarith
  have hex : 0 < exp (-x) := exp_pos _
  rw [neg_add, Real.exp_add]
  have key : 4 * exp (-max (Real.log (4 / (d - 1))) 0) ≤ (d : ℝ) - 1 := by
    rcases le_total (Real.log (4 / (d - 1))) 0 with h | h
    · rw [max_eq_right h, neg_zero, Real.exp_zero, mul_one]
      have : 4 / ((d : ℝ) - 1) ≤ 1 := by
        have hpos : 0 < 4 / ((d : ℝ) - 1) := by positivity
        exact (Real.log_nonpos_iff hpos.le).mp h
      rw [div_le_one (by linarith)] at this
      exact this
    · rw [max_eq_left h, Real.exp_neg, Real.exp_log (by positivity)]
      field_simp
      linarith
  have : 4 * (d : ℝ) * (exp (-x) * exp (-max (Real.log (4 / (d - 1))) 0)) ≤
      d * (d - 1) * exp (-x) := by
    have := mul_le_mul_of_nonneg_left key (le_of_lt (mul_pos hdpos hex))
    nlinarith
  linarith

/-- **Theorem 2.2, constant.** `½ (d(d-1) + 4d) = ½ (d² + 3d)`. -/
theorem thm22_const (d : ℝ) : (1 / 2) * (d * (d - 1) + 4 * d) = (1 / 2) * (d ^ 2 + 3 * d) := by
  ring

/-! ### Lemma 1.1, analytic and algebraic steps -/

/-- `φ(u) = -½ log(1 - 2u) - u`, the cumulant of `u (g² - 1)` for `g ~ N(0,1)`. -/
noncomputable def phiC (u : ℝ) : ℝ := -(1 / 2) * Real.log (1 - 2 * u) - u

/-- **Lemma 1.1, series bound.** For `|u| < 1/2`, `φ(u) ≤ u²/(1 - 2|u|)`. The proof compares the
log series `-log(1 - x) - x = Σ_{k≥2} x^k/k` termwise with `Σ_{k≥2} |x|^k/2`. -/
theorem phi_le {u : ℝ} (hu : |u| < 1 / 2) : phiC u ≤ u ^ 2 / (1 - 2 * |u|) := by
  set x := 2 * u with hxdef
  have hxabs : |x| = 2 * |u| := by rw [hxdef, abs_mul]; norm_num
  have hx : |x| < 1 := by rw [hxabs]; linarith
  have h1 := Real.hasSum_pow_div_log_of_abs_lt_one hx
  have h2 := (hasSum_nat_add_iff' 1).mpr h1
  simp only [Finset.range_one, Finset.sum_singleton, zero_add, pow_one, Nat.cast_zero,
    div_one] at h2
  have hg : HasSum (fun n : ℕ => x ^ 2 / 2 * |x| ^ n) (x ^ 2 / 2 * (1 - |x|)⁻¹) :=
    (hasSum_geometric_of_lt_one (abs_nonneg x) hx).mul_left (x ^ 2 / 2)
  have hle : ∀ n : ℕ, x ^ (n + 1 + 1) / ((n + 1 : ℕ) + 1 : ℝ) ≤ x ^ 2 / 2 * |x| ^ n := by
    intro n
    have hk : x ^ (n + 1 + 1) ≤ |x| ^ (n + 1 + 1) := by
      rw [← abs_pow]; exact le_abs_self _
    have hd : (2 : ℝ) ≤ ((n + 1 : ℕ) + 1 : ℝ) := by push_cast; linarith [(n.cast_nonneg : (0:ℝ) ≤ n)]
    have hpos : (0 : ℝ) < ((n + 1 : ℕ) + 1 : ℝ) := by linarith
    calc x ^ (n + 1 + 1) / ((n + 1 : ℕ) + 1 : ℝ) ≤ |x| ^ (n + 1 + 1) / ((n + 1 : ℕ) + 1 : ℝ) :=
          div_le_div_of_nonneg_right hk hpos.le
      _ ≤ |x| ^ (n + 1 + 1) / 2 := div_le_div_of_nonneg_left (by positivity) (by norm_num) hd
      _ = x ^ 2 / 2 * |x| ^ n := by
          rw [show n + 1 + 1 = 2 + n by ring, pow_add, sq_abs]; ring
  have key := hasSum_le hle h2 hg
  have hden : 0 < 1 - |x| := by linarith
  have hden' : 0 < 1 - 2 * |u| := by linarith
  unfold phiC
  rw [← hxdef]
  have : -(1 / 2) * Real.log (1 - x) - u = (1 / 2) * (-Real.log (1 - x) - x) := by
    rw [hxdef]; ring
  rw [this]
  calc (1 / 2) * (-Real.log (1 - x) - x) ≤ (1 / 2) * (x ^ 2 / 2 * (1 - |x|)⁻¹) := by
        gcongr
    _ = u ^ 2 / (1 - 2 * |u|) := by
        rw [hxabs, hxdef]; field_simp

/-- **Lemma 1.1, log-MGF bound.** For eigenvalues `λ_i` with `2|λ_i| ≤ c` (`c = 2‖A‖_op`),
`v = 2 Σ λ_i²` (`= 2‖A‖_F²`) and `0 ≤ s`, `cs < 1`:
`Σ_i φ(s λ_i) ≤ v s² / (2(1 - cs))`. -/
theorem chernoff_logmgf_le {ι : Type*} (I : Finset ι) (lam : ι → ℝ) {c s : ℝ} (hs : 0 ≤ s)
    (hcs : c * s < 1) (hlam : ∀ i ∈ I, 2 * |lam i| ≤ c) :
    ∑ i ∈ I, phiC (s * lam i) ≤ (2 * ∑ i ∈ I, lam i ^ 2) * s ^ 2 / (2 * (1 - c * s)) := by
  have hden : 0 < 1 - c * s := by linarith
  have hterm : ∀ i ∈ I, phiC (s * lam i) ≤ s ^ 2 * lam i ^ 2 / (1 - c * s) := by
    intro i hi
    have h2 : 2 * |s * lam i| ≤ c * s := by
      rw [abs_mul, abs_of_nonneg hs]; nlinarith [hlam i hi]
    have hu : |s * lam i| < 1 / 2 := by linarith
    have hd : 0 < 1 - 2 * |s * lam i| := by linarith
    calc phiC (s * lam i) ≤ (s * lam i) ^ 2 / (1 - 2 * |s * lam i|) := phi_le hu
      _ ≤ (s * lam i) ^ 2 / (1 - c * s) :=
          div_le_div_of_nonneg_left (sq_nonneg _) hden (by linarith)
      _ = s ^ 2 * lam i ^ 2 / (1 - c * s) := by ring
  calc ∑ i ∈ I, phiC (s * lam i) ≤ ∑ i ∈ I, s ^ 2 * lam i ^ 2 / (1 - c * s) := sum_le_sum hterm
    _ = (2 * ∑ i ∈ I, lam i ^ 2) * s ^ 2 / (2 * (1 - c * s)) := by
        rw [← sum_div, ← mul_sum]; field_simp

/-- **Lemma 1.1, choice of `s`.** With `c ≥ 0`, `v > 0`, `x > 0`, `q = √(2x/v)` and
`s = q/(1 + cq)`: `0 ≤ s`, `cs < 1`, and `s (√(2vx) + cx) - v s²/(2(1 - cs)) = x`. -/
theorem chernoff_choice {c v x : ℝ} (hc : 0 ≤ c) (hv : 0 < v) (hx : 0 < x) :
    let q := √(2 * x / v)
    let s := q / (1 + c * q)
    0 ≤ s ∧ c * s < 1 ∧
      s * (√(2 * v * x) + c * x) - v * s ^ 2 / (2 * (1 - c * s)) = x := by
  intro q s
  have hq0 : 0 ≤ q := Real.sqrt_nonneg _
  have hq2 : q ^ 2 = 2 * x / v := Real.sq_sqrt (div_nonneg (by linarith) hv.le)
  have hqv : √(2 * v * x) = v * q := by
    rw [show 2 * v * x = v ^ 2 * (2 * x / v) by field_simp, Real.sqrt_mul (sq_nonneg v),
      Real.sqrt_sq hv.le]
  have hcq : 0 < 1 + c * q := by positivity
  have hs0 : 0 ≤ s := div_nonneg hq0 hcq.le
  have hcs : c * s < 1 := by
    show c * (q / (1 + c * q)) < 1
    rw [mul_div_assoc', div_lt_one hcq]; linarith
  refine ⟨hs0, hcs, ?_⟩
  have hxq : x = v * q ^ 2 / 2 := by rw [hq2]; field_simp
  have h1cs : 1 - c * s = 1 / (1 + c * q) := by
    show 1 - c * (q / (1 + c * q)) = 1 / (1 + c * q)
    field_simp; ring
  rw [hqv, h1cs]
  show q / (1 + c * q) * (v * q + c * x) - v * (q / (1 + c * q)) ^ 2 / (2 * (1 / (1 + c * q))) = x
  rw [hxq]
  field_simp
  ring

/-- `√(2vx) = 2‖A‖_F √x` when `v = 2‖A‖_F²`. -/
theorem sqrt_two_v_x {F x : ℝ} (hF : 0 ≤ F) :
    √(2 * (2 * F ^ 2) * x) = 2 * F * √x := by
  rw [show 2 * (2 * F ^ 2) * x = (2 * F) ^ 2 * x by ring, Real.sqrt_mul (sq_nonneg _),
    Real.sqrt_sq (by positivity)]

/-! ### Theorem 1.5, deterministic core -/

/-- **Theorem 1.5, deterministic core.** If every estimate is within `δ_p` of its target, every
pair of the target set `T` has `|ρ_p| - δ_p > τ_p` and every other pair has `|ρ_p| + δ_p ≤ τ_p`,
then the declared set `{p : |r̂_p| > τ_p}` is exactly `T`. -/
theorem decide_eq_target {P : Type*} (rhat ρ δ τ : P → ℝ) (T : Set P)
    (hdev : ∀ p, |rhat p - ρ p| ≤ δ p)
    (hin : ∀ p ∈ T, τ p < |ρ p| - δ p) (hout : ∀ p ∉ T, |ρ p| + δ p ≤ τ p) :
    {p | τ p < |rhat p|} = T := by
  ext p
  simp only [Set.mem_ofPred_eq]
  have h1 : |rhat p| - |ρ p| ≤ |rhat p - ρ p| := abs_sub_abs_le_abs_sub _ _
  have h2 : |ρ p| - |rhat p| ≤ |ρ p - rhat p| := abs_sub_abs_le_abs_sub _ _
  rw [abs_sub_comm] at h2
  have hd := hdev p
  by_cases hp : p ∈ T
  · simp only [hp, iff_true]; linarith [hin p hp]
  · simp only [hp, iff_false, not_lt]; linarith [hout p hp]

/-- **Theorem 1.5 with `T = R = {ρ ≠ 0}`.** -/
theorem decide_eq_support {P : Type*} (rhat ρ δ τ : P → ℝ)
    (hdev : ∀ p, |rhat p - ρ p| ≤ δ p)
    (hnull : ∀ p, ρ p = 0 → δ p ≤ τ p) (hrel : ∀ p, ρ p ≠ 0 → τ p < |ρ p| - δ p) :
    {p | τ p < |rhat p|} = {p | ρ p ≠ 0} := by
  apply decide_eq_target rhat ρ δ τ _ hdev
  · intro p hp; exact hrel p hp
  · intro p hp
    simp only [Set.mem_ofPred_eq, not_not] at hp
    rw [hp, abs_zero, zero_add]; exact hnull p hp

/-- **Theorem 1.5, deterministic form on the event `E(x,y)`.** Variables `V`, sample covariance
matrix `C`, population correlations `ρ`. If `|C_aa - 1| ≤ ε_d < 1` for every `a`,
`|C_ab - ρ_ab| ≤ ε_ab` for `a ≠ b`, and the thresholds separate the target `T` with margin
`δ_ab = (ε_ab + |ρ_ab| ε_d)/(1 - ε_d)`, then for every `a ≠ b`, the pair is declared
(`|C_ab/√(C_aa C_bb)| > τ_ab`) iff `T a b`. -/
theorem thm15_det {V : Type*} (C ρ ε τ : V → V → ℝ) (T : V → V → Prop) {εd : ℝ} (hεd : εd < 1)
    (hdiag : ∀ a, |C a a - 1| ≤ εd) (hoff : ∀ a b, a ≠ b → |C a b - ρ a b| ≤ ε a b)
    (hin : ∀ a b, a ≠ b → T a b → τ a b < |ρ a b| - (ε a b + |ρ a b| * εd) / (1 - εd))
    (hout : ∀ a b, a ≠ b → ¬ T a b → |ρ a b| + (ε a b + |ρ a b| * εd) / (1 - εd) ≤ τ a b)
    (a b : V) (hab : a ≠ b) :
    τ a b < |C a b / √(C a a * C b b)| ↔ T a b := by
  have hI : ∀ c, C c c ∈ Set.Icc (1 - εd) (1 + εd) := fun c => by
    have := abs_le.mp (hdiag c); constructor <;> linarith
  have hdev := corr_perturb hεd (hI a) (hI b) (hoff a b hab)
  have key := decide_eq_target (P := Unit) (fun _ => C a b / √(C a a * C b b)) (fun _ => ρ a b)
    (fun _ => (ε a b + |ρ a b| * εd) / (1 - εd)) (fun _ => τ a b) {_u | T a b}
    (fun _ => hdev) (fun _ h => hin a b hab h) (fun _ h => hout a b hab h)
  have := congrArg (fun S : Set Unit => () ∈ S) key
  simpa using this

/-! ### Corollary 1.6 and the saturation step of Theorem 2.2 -/

section Counts

variable {V : Type*} [Fintype V] [DecidableEq V]

open Classical in
/-- The count `ρ̂(v) = 1 + #{w ≠ v : v, w declared}` of a (declared or population) relative
relation `D`. -/
noncomputable def cntDecl (D : V → V → Prop) (v : V) : ℕ :=
  1 + (univ.filter (fun w => w ≠ v ∧ D v w)).card

/-- **Corollary 1.6.** If the declared relation agrees with the population relation off the
diagonal (`R̂ = R`), the counts agree and so does the rel-sortability of any edge set. -/
theorem sOf_cntDecl_congr (E : V → V → Prop) (D R : V → V → Prop)
    (h : ∀ v w, v ≠ w → (D v w ↔ R v w)) :
    cntDecl D = cntDecl R ∧ sOf E (cntDecl D) = sOf E (cntDecl R) := by
  have hc : cntDecl D = cntDecl R := by
    funext v
    unfold cntDecl
    congr 2
    ext w
    simp only [mem_filter, mem_univ, true_and]
    constructor
    · rintro ⟨hw, hd⟩; exact ⟨hw, (h v w (Ne.symm hw)).mp hd⟩
    · rintro ⟨hw, hd⟩; exact ⟨hw, (h v w (Ne.symm hw)).mpr hd⟩
  exact ⟨hc, by rw [hc]⟩

/-- Constant counts give rel-sortability exactly `1/2` on any nonempty edge set. -/
theorem sOf_const (E : V → V → Prop) (ρ : V → ℕ) (k : ℕ) (hρ : ∀ v, ρ v = k)
    (hE : (edges E).Nonempty) : sOf E ρ = 1 / 2 := by
  unfold sOf
  have : ∀ p ∈ edges E, score (ρ p.1) (ρ p.2) = 1 / 2 := by
    intro p _; rw [hρ, hρ]; simp [score]
  rw [sum_congr rfl this, sum_const, nsmul_eq_mul]
  have hc : ((edges E).card : ℝ) ≠ 0 := by exact_mod_cast hE.card_pos.ne'
  field_simp

open Classical in
/-- **Theorem 2.2, saturation step.** If every pair is declared (`R̂` = all pairs), every count
equals `d = |V|`, and the rel-sortability of every nonempty edge set is exactly `1/2`. -/
theorem sOf_saturated (E : V → V → Prop) (D : V → V → Prop) (h : ∀ v w, v ≠ w → D v w)
    (hE : (edges E).Nonempty) :
    (∀ v, cntDecl D v = Fintype.card V) ∧ sOf E (cntDecl D) = 1 / 2 := by
  have hc : ∀ v, cntDecl D v = Fintype.card V := by
    intro v
    unfold cntDecl
    have : univ.filter (fun w => w ≠ v ∧ D v w) = univ.erase v := by
      ext w
      simp only [mem_filter, mem_univ, true_and, mem_erase, and_true]
      exact ⟨fun h' => h'.1, fun hw => ⟨hw, h v w (Ne.symm hw)⟩⟩
    rw [this, card_erase_of_mem (mem_univ v), card_univ]
    have : 0 < Fintype.card V := Fintype.card_pos_iff.mpr ⟨v⟩
    omega
  exact ⟨hc, sOf_const E _ _ hc hE⟩

/-- `|ŝ - 1/2| ≤ 1/2` always (used for the bound on `|E ŝ - 1/2|` in Theorem 2.2). -/
theorem abs_sOf_sub_half_le (E : V → V → Prop) (ρ : V → ℕ) : |sOf E ρ - 1 / 2| ≤ 1 / 2 := by
  have hs : ∀ a b : ℕ, 0 ≤ score a b ∧ score a b ≤ 1 := by
    intro a b; unfold score; split_ifs <;> norm_num
  unfold sOf
  rcases Nat.eq_zero_or_pos (edges E).card with h0 | hpos
  · rw [h0]; simp
  · have hc : (0 : ℝ) < (edges E).card := by exact_mod_cast hpos
    have hlo : 0 ≤ ∑ p ∈ edges E, score (ρ p.1) (ρ p.2) := sum_nonneg (fun p _ => (hs _ _).1)
    have hhi : ∑ p ∈ edges E, score (ρ p.1) (ρ p.2) ≤ (edges E).card := by
      calc ∑ p ∈ edges E, score (ρ p.1) (ρ p.2) ≤ ∑ _p ∈ edges E, (1 : ℝ) :=
            sum_le_sum (fun p _ => (hs _ _).2)
        _ = (edges E).card := by simp
    rw [abs_le]
    constructor
    · have : 0 ≤ (∑ p ∈ edges E, score (ρ p.1) (ρ p.2)) / (edges E).card := div_nonneg hlo hc.le
      linarith
    · have : (∑ p ∈ edges E, score (ρ p.1) (ρ p.2)) / (edges E).card ≤ 1 := by
        rw [div_le_one hc]; exact hhi
      linarith

end Counts

end RelSort
