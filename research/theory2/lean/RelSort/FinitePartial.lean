/-
Covariance algebra of F2 and F3 in `research/theory2/finite/FINITE.md` (Sections 2-3).

Modelling choice: zero-mean, finite-variance random variables are vectors of a real inner product
space `E` (for instance `L²(Ω)`), and the covariance at lag 0 is the inner product. Independence or
uncorrelatedness at lag 0 is used only through vanishing inner products, which are hypotheses.
FINITE.md's proofs use nothing else (bilinearity of the covariance).

* Partial correlation: `pc` (the three-variable formula `g(a,b,c)`), `pcov`, `resid`,
  `pcorr_eq_pcov` (the formula equals the partial covariance over the partial standard deviations),
  `pcorr_eq_corr_resid` (and equals the correlation of the OLS residuals).
* Prop 2.1: `prop21_corr`, `prop21_corr_std`.
* Prop 3.1: `prop31`, `prop31_std`. Cor 3.2: `cor32`. Prop 3.3: `prop33`, `prop33_abs` and the
  properties of `absPi` (`absPi_lower`, `absPi_upper`, `absPi_le_kappa`, `absPi_lt_kappa`,
  `absPi_eq_zero_iff`, `absPi_strictAntiOn`, `absPi_equal`).
* Prop 3.6: `prop36_pcov`, `prop36_rank_one`, `prop36_ar1`, `prop36_ar1_ne_zero`.
* Prop 3.7: `prop37_pcov`, `prop37_bound`. Prop 3.8: `prop38_ratio`, `prop38_loading`.
* Thm 3.5 / Prop 2.3 algebra: `rcrit`, `rcrit_iff`, `rcrit_eq_iff`, `rcrit_example`,
  `ncrit_iff`, `ncrit_eq_iff`.
-/
import RelSort.FiniteCorr

open Real

set_option linter.unusedSectionVars false

namespace RelSort

/-! ### Pure-real helpers -/

/-- The three-variable partial correlation `g(a,b,c) = (a - bc)/√((1-b²)(1-c²))`. -/
noncomputable def pc (a b c : ℝ) : ℝ := (a - b * c) / √((1 - b ^ 2) * (1 - c ^ 2))

lemma pc_eq_aux {sx sy sz a b c : ℝ} (hx : 0 < sx) (hy : 0 < sy) (hz : 0 < sz)
    (hpx : 0 < sx ^ 2 - b ^ 2 / sz ^ 2) (hpy : 0 < sy ^ 2 - c ^ 2 / sz ^ 2) :
    pc (a / (sx * sy)) (b / (sx * sz)) (c / (sy * sz)) =
      (a - b * c / sz ^ 2) / √((sx ^ 2 - b ^ 2 / sz ^ 2) * (sy ^ 2 - c ^ 2 / sz ^ 2)) := by
  unfold pc
  have hin : (1 - (b / (sx * sz)) ^ 2) * (1 - (c / (sy * sz)) ^ 2) =
      ((sx ^ 2 - b ^ 2 / sz ^ 2) * (sy ^ 2 - c ^ 2 / sz ^ 2)) / (sx * sy) ^ 2 := by
    field_simp
  rw [hin, Real.sqrt_div' _ (sq_nonneg _), Real.sqrt_sq (by positivity)]
  have hP : 0 < √((sx ^ 2 - b ^ 2 / sz ^ 2) * (sy ^ 2 - c ^ 2 / sz ^ 2)) :=
    Real.sqrt_pos.mpr (mul_pos hpx hpy)
  field_simp

/-- `√(P / Q) = √P / √Q` packaged for `Q > 0`. -/
lemma sqrt_div_pos {P Q : ℝ} (hQ : 0 < Q) : √(P / Q) = √P / √Q := Real.sqrt_div' P hQ.le

/-! ### Inner-product model -/

section Inner

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

local notation "⟪" x ", " y "⟫" => @inner ℝ _ _ x y

/-- Correlation `⟪x,y⟫ / √(⟪x,x⟫⟪y,y⟫)`. -/
noncomputable def corr (x y : E) : ℝ := ⟪x, y⟫ / √(⟪x, x⟫ * ⟪y, y⟫)

/-- Partial correlation of `x, y` given `z`, by the three-variable formula of FINITE.md §3. -/
noncomputable def pcorr (x y z : E) : ℝ := pc (corr x y) (corr x z) (corr y z)

/-- Partial covariance `⟪x,y⟫ - ⟪x,z⟫⟪y,z⟫/⟪z,z⟫`. -/
noncomputable def pcov (x y z : E) : ℝ := ⟪x, y⟫ - ⟪x, z⟫ * ⟪y, z⟫ / ⟪z, z⟫

/-- The OLS residual of `x` on `z`. -/
noncomputable def resid (x z : E) : E := x - (⟪x, z⟫ / ⟪z, z⟫) • z

lemma inner_resid (x y z : E) (hz : ⟪z, z⟫ ≠ 0) : ⟪resid x z, resid y z⟫ = pcov x y z := by
  unfold resid pcov
  simp only [inner_sub_left, inner_sub_right, real_inner_smul_left, real_inner_smul_right]
  rw [real_inner_comm z x, real_inner_comm z y]
  field_simp
  ring

/-- **The three-variable formula is the partial covariance over the partial standard deviations.** -/
theorem pcorr_eq_pcov (x y z : E) (hx : 0 < ⟪x, x⟫) (hy : 0 < ⟪y, y⟫) (hz : 0 < ⟪z, z⟫)
    (hpx : 0 < pcov x x z) (hpy : 0 < pcov y y z) :
    pcorr x y z = pcov x y z / √(pcov x x z * pcov y y z) := by
  unfold pcorr corr
  rw [Real.sqrt_mul hx.le, Real.sqrt_mul hx.le, Real.sqrt_mul hy.le]
  have hX : ⟪x, x⟫ = √⟪x, x⟫ ^ 2 := (Real.sq_sqrt hx.le).symm
  have hY : ⟪y, y⟫ = √⟪y, y⟫ ^ 2 := (Real.sq_sqrt hy.le).symm
  have hZ : ⟪z, z⟫ = √⟪z, z⟫ ^ 2 := (Real.sq_sqrt hz.le).symm
  unfold pcov at hpx hpy ⊢
  set sx := √⟪x, x⟫
  set sy := √⟪y, y⟫
  set sz := √⟪z, z⟫
  rw [hX, hZ] at hpx
  rw [hY, hZ] at hpy
  rw [hX, hY, hZ, pc_eq_aux (Real.sqrt_pos.mpr hx) (Real.sqrt_pos.mpr hy) (Real.sqrt_pos.mpr hz)
    (by rw [pow_two ⟪x, z⟫]; exact hpx) (by rw [pow_two ⟪y, z⟫]; exact hpy)]
  rw [pow_two ⟪x, z⟫, pow_two ⟪y, z⟫]

/-- **The partial correlation is the correlation of the OLS residuals on `z`.** -/
theorem pcorr_eq_corr_resid (x y z : E) (hx : 0 < ⟪x, x⟫) (hy : 0 < ⟪y, y⟫) (hz : 0 < ⟪z, z⟫)
    (hpx : 0 < pcov x x z) (hpy : 0 < pcov y y z) :
    pcorr x y z = corr (resid x z) (resid y z) := by
  rw [pcorr_eq_pcov x y z hx hy hz hpx hpy]
  unfold corr
  rw [inner_resid x y z hz.ne', inner_resid x x z hz.ne', inner_resid y y z hz.ne']

/-! #### Model D and the proxy model -/

section ModelD

variable {Yu Yv C e : E} {bu bv r : ℝ}

/-- Covariance of two nodes of Model D, `X = Y + b C`, with `Y ⟂ C`. -/
lemma inner_modelD (huC : ⟪Yu, C⟫ = 0) (hvC : ⟪Yv, C⟫ = 0) (hC : ⟪C, C⟫ = 1) :
    ⟪Yu + bu • C, Yv + bv • C⟫ = ⟪Yu, Yv⟫ + bu * bv := by
  simp only [inner_add_left, inner_add_right, real_inner_smul_left, real_inner_smul_right]
  rw [real_inner_comm Yv C, huC, hvC, hC]
  ring

/-- `Var Z = 1` for the proxy `Z = √r C + √(1-r) e`. -/
lemma inner_proxy_self (hr0 : 0 ≤ r) (hr1 : r ≤ 1) (hC : ⟪C, C⟫ = 1) (he : ⟪e, e⟫ = 1)
    (hCe : ⟪C, e⟫ = 0) :
    ⟪√r • C + √(1 - r) • e, √r • C + √(1 - r) • e⟫ = 1 := by
  simp only [inner_add_left, inner_add_right, real_inner_smul_left, real_inner_smul_right]
  rw [real_inner_comm C e, hC, he, hCe]
  have h1 : √r * √r = r := Real.mul_self_sqrt hr0
  have h2 : √(1 - r) * √(1 - r) = 1 - r := Real.mul_self_sqrt (by linarith)
  linear_combination h1 + h2

/-- `Cov(X_u, Z) = b_u √r`. -/
lemma inner_node_proxy (huC : ⟪Yu, C⟫ = 0) (hue : ⟪Yu, e⟫ = 0) (hC : ⟪C, C⟫ = 1)
    (hCe : ⟪C, e⟫ = 0) :
    ⟪Yu + bu • C, √r • C + √(1 - r) • e⟫ = bu * √r := by
  simp only [inner_add_left, inner_add_right, real_inner_smul_left, real_inner_smul_right]
  rw [huC, hue, hC, hCe]
  ring

/-- **Prop 2.1 (population saturation), correlation.** For river non-relatives
(`Cov(Y_u, Y_v) = 0`), `ρ_uv = b_u b_v/(s_u s_v)` with `s_u² = Var Y_u + b_u²`. -/
theorem prop21_corr (huC : ⟪Yu, C⟫ = 0) (hvC : ⟪Yv, C⟫ = 0) (hC : ⟪C, C⟫ = 1)
    (huv : ⟪Yu, Yv⟫ = 0) :
    corr (Yu + bu • C) (Yv + bv • C) =
      bu * bv / (√(⟪Yu, Yu⟫ + bu ^ 2) * √(⟪Yv, Yv⟫ + bv ^ 2)) := by
  have h0 : 0 ≤ ⟪Yu, Yu⟫ := real_inner_self_nonneg
  unfold corr
  rw [inner_modelD huC hvC hC, inner_modelD huC huC hC, inner_modelD hvC hvC hC, huv,
    Real.sqrt_mul (by nlinarith)]
  ring_nf

/-- `sgn(b_u b_v) √(ℓ_u ℓ_v) = b_u b_v/(s_u s_v)` for `ℓ = b²/s²`. -/
lemma sign_sqrt_loading {Su Sv bu bv : ℝ} (hu : 0 < Su) (hv : 0 < Sv) :
    (SignType.sign (bu * bv) : ℝ) * √(bu ^ 2 / Su * (bv ^ 2 / Sv)) = bu * bv / (√Su * √Sv) := by
  rw [show bu ^ 2 / Su * (bv ^ 2 / Sv) = (bu * bv) ^ 2 / (Su * Sv) by field_simp,
    sqrt_div_pos (mul_pos hu hv), Real.sqrt_sq_eq_abs, Real.sqrt_mul hu.le, mul_div_assoc',
    sign_mul_abs]

/-- **Prop 2.1, standardised form** `ρ_uv = sgn(b_u b_v) √(ℓ_u ℓ_v) = κ_uv`. -/
theorem prop21_corr_std (huC : ⟪Yu, C⟫ = 0) (hvC : ⟪Yv, C⟫ = 0) (hC : ⟪C, C⟫ = 1)
    (huv : ⟪Yu, Yv⟫ = 0) (hu : 0 < ⟪Yu, Yu⟫ + bu ^ 2) (hv : 0 < ⟪Yv, Yv⟫ + bv ^ 2) :
    corr (Yu + bu • C) (Yv + bv • C) = (SignType.sign (bu * bv) : ℝ) *
      √(bu ^ 2 / (⟪Yu, Yu⟫ + bu ^ 2) * (bv ^ 2 / (⟪Yv, Yv⟫ + bv ^ 2))) := by
  rw [prop21_corr huC hvC hC huv, sign_sqrt_loading hu hv]

/-- Hypotheses of the proxy model at lag 0: `Y_u, Y_v ⟂ C, e`; `C ⟂ e`; unit variances. -/
structure ProxyHyp (Yu Yv C e : E) : Prop where
  huC : ⟪Yu, C⟫ = 0
  hvC : ⟪Yv, C⟫ = 0
  hue : ⟪Yu, e⟫ = 0
  hve : ⟪Yv, e⟫ = 0
  hC : ⟪C, C⟫ = 1
  he : ⟪e, e⟫ = 1
  hCe : ⟪C, e⟫ = 0

/-- **Prop 3.1 (exact conditional correlation).** Under Model D and the proxy model
`Z = √r C + √(1-r) e`, with `Var Y_u, Var Y_v > 0` and `r ∈ [0,1]`:
`π_uv = (γ^Y_uv + (1-r) b_u b_v)/√((Var Y_u + (1-r) b_u²)(Var Y_v + (1-r) b_v²))`. -/
theorem prop31 (h : ProxyHyp Yu Yv C e) (hr0 : 0 ≤ r) (hr1 : r ≤ 1)
    (hu : 0 < ⟪Yu, Yu⟫) (hv : 0 < ⟪Yv, Yv⟫) :
    pcorr (Yu + bu • C) (Yv + bv • C) (√r • C + √(1 - r) • e) =
      (⟪Yu, Yv⟫ + (1 - r) * bu * bv) /
        √((⟪Yu, Yu⟫ + (1 - r) * bu ^ 2) * (⟪Yv, Yv⟫ + (1 - r) * bv ^ 2)) := by
  have hZ := inner_proxy_self (C := C) (e := e) hr0 hr1 h.hC h.he h.hCe
  have huZ := inner_node_proxy (bu := bu) (r := r) h.huC h.hue h.hC h.hCe
  have hvZ := inner_node_proxy (bu := bv) (r := r) h.hvC h.hve h.hC h.hCe
  have hrr : √r * √r = r := Real.mul_self_sqrt hr0
  have pc_uv : pcov (Yu + bu • C) (Yv + bv • C) (√r • C + √(1 - r) • e) =
      ⟪Yu, Yv⟫ + (1 - r) * bu * bv := by
    unfold pcov; rw [inner_modelD h.huC h.hvC h.hC, huZ, hvZ, hZ]
    linear_combination (-(bu * bv)) * hrr
  have pc_uu : pcov (Yu + bu • C) (Yu + bu • C) (√r • C + √(1 - r) • e) =
      ⟪Yu, Yu⟫ + (1 - r) * bu ^ 2 := by
    unfold pcov; rw [inner_modelD h.huC h.huC h.hC, huZ, hZ]
    linear_combination (-(bu * bu)) * hrr
  have pc_vv : pcov (Yv + bv • C) (Yv + bv • C) (√r • C + √(1 - r) • e) =
      ⟪Yv, Yv⟫ + (1 - r) * bv ^ 2 := by
    unfold pcov; rw [inner_modelD h.hvC h.hvC h.hC, hvZ, hZ]
    linear_combination (-(bv * bv)) * hrr
  have h1r : 0 ≤ 1 - r := by linarith
  rw [pcorr_eq_pcov _ _ _ (by rw [inner_modelD h.huC h.huC h.hC]; nlinarith)
    (by rw [inner_modelD h.hvC h.hvC h.hC]; nlinarith) (by rw [hZ]; norm_num)
    (by rw [pc_uu]; positivity) (by rw [pc_vv]; positivity), pc_uv, pc_uu, pc_vv]

/-- **Prop 3.1, standardised form** (pure algebra). With `S_u = Var Y_u + b_u²`,
`ℓ_u = b_u²/S_u` and `ρ^Y = γ/√(Var Y_u Var Y_v)`, the raw form equals
`(ρ^Y √((1-ℓ_u)(1-ℓ_v)) + (1-r) sgn(b_u b_v) √(ℓ_u ℓ_v)) / √((1 - rℓ_u)(1 - rℓ_v))`. -/
theorem prop31_std {Vu Vv γ bu bv r : ℝ} (hu : 0 < Vu) (hv : 0 < Vv) (hr1 : r ≤ 1) :
    (γ + (1 - r) * bu * bv) / √((Vu + (1 - r) * bu ^ 2) * (Vv + (1 - r) * bv ^ 2)) =
      (γ / √(Vu * Vv) * √((1 - bu ^ 2 / (Vu + bu ^ 2)) * (1 - bv ^ 2 / (Vv + bv ^ 2))) +
        (1 - r) * (SignType.sign (bu * bv) : ℝ) *
          √(bu ^ 2 / (Vu + bu ^ 2) * (bv ^ 2 / (Vv + bv ^ 2)))) /
        √((1 - r * (bu ^ 2 / (Vu + bu ^ 2))) * (1 - r * (bv ^ 2 / (Vv + bv ^ 2)))) := by
  have hSu : 0 < Vu + bu ^ 2 := by positivity
  have hSv : 0 < Vv + bv ^ 2 := by positivity
  have hPu : 0 < Vu + (1 - r) * bu ^ 2 := by
    have : 0 ≤ (1 - r) * bu ^ 2 := mul_nonneg (by linarith) (sq_nonneg _)
    linarith
  have hPv : 0 < Vv + (1 - r) * bv ^ 2 := by
    have : 0 ≤ (1 - r) * bv ^ 2 := mul_nonneg (by linarith) (sq_nonneg _)
    linarith
  have e1 : (1 - bu ^ 2 / (Vu + bu ^ 2)) * (1 - bv ^ 2 / (Vv + bv ^ 2)) =
      (Vu * Vv) / ((Vu + bu ^ 2) * (Vv + bv ^ 2)) := by field_simp; ring
  have e2 : (1 - r * (bu ^ 2 / (Vu + bu ^ 2))) * (1 - r * (bv ^ 2 / (Vv + bv ^ 2))) =
      ((Vu + (1 - r) * bu ^ 2) * (Vv + (1 - r) * bv ^ 2)) / ((Vu + bu ^ 2) * (Vv + bv ^ 2)) := by
    field_simp; ring
  have hss := mul_pos hSu hSv
  rw [mul_assoc (1 - r) (SignType.sign (bu * bv) : ℝ), sign_sqrt_loading hSu hSv, e1, e2, sqrt_div_pos hss, sqrt_div_pos hss,
    Real.sqrt_mul hSu.le]
  have a1 : 0 < √(Vu * Vv) := Real.sqrt_pos.mpr (mul_pos hu hv)
  have a2 : 0 < √(Vu + bu ^ 2) := Real.sqrt_pos.mpr hSu
  have a3 : 0 < √(Vv + bv ^ 2) := Real.sqrt_pos.mpr hSv
  have a4 : 0 < √((Vu + (1 - r) * bu ^ 2) * (Vv + (1 - r) * bv ^ 2)) :=
    Real.sqrt_pos.mpr (mul_pos hPu hPv)
  field_simp

/-- **Cor 3.2 (r = 1).** With a perfect proxy (`Z = C`), `π_uv = ρ^Y_uv`, the driver-free
correlation. -/
theorem cor32 (h : ProxyHyp Yu Yv C e) (hu : 0 < ⟪Yu, Yu⟫) (hv : 0 < ⟪Yv, Yv⟫) :
    pcorr (Yu + bu • C) (Yv + bv • C) (√1 • C + √(1 - 1) • e) = corr Yu Yv := by
  rw [prop31 h (by norm_num) le_rfl hu hv]
  unfold corr
  simp

/-- **Prop 3.3 (residual partial correlation between non-relatives), signed form.** -/
theorem prop33 (h : ProxyHyp Yu Yv C e) (hr0 : 0 ≤ r) (hr1 : r ≤ 1)
    (hu : 0 < ⟪Yu, Yu⟫) (hv : 0 < ⟪Yv, Yv⟫) (huv : ⟪Yu, Yv⟫ = 0) :
    pcorr (Yu + bu • C) (Yv + bv • C) (√r • C + √(1 - r) • e) =
      (1 - r) * (SignType.sign (bu * bv) : ℝ) *
          √(bu ^ 2 / (⟪Yu, Yu⟫ + bu ^ 2) * (bv ^ 2 / (⟪Yv, Yv⟫ + bv ^ 2))) /
        √((1 - r * (bu ^ 2 / (⟪Yu, Yu⟫ + bu ^ 2))) *
          (1 - r * (bv ^ 2 / (⟪Yv, Yv⟫ + bv ^ 2)))) := by
  rw [prop31 h hr0 hr1 hu hv, prop31_std hu hv hr1, huv]
  simp

end ModelD

end Inner

/-! ### Prop 3.3: properties of `|π| = (1-r)√(ℓ_u ℓ_v)/√((1-rℓ_u)(1-rℓ_v))` -/

/-- `|π_uv|` for river non-relatives as a function of reliability `r` and shares `ℓ_u, ℓ_v`. -/
noncomputable def absPi (r lu lv : ℝ) : ℝ := (1 - r) * √(lu * lv) / √((1 - r * lu) * (1 - r * lv))

section AbsPi

variable {r lu lv : ℝ}

lemma one_sub_mul_pos (hr1 : r ≤ 1) (hl0 : 0 ≤ lu) (hl1 : lu < 1) : 0 < 1 - r * lu := by
  rcases le_or_gt r 0 with h | h
  · nlinarith
  · nlinarith

/-- `|π_uv| = absPi r ℓ_u ℓ_v` (the absolute value of the signed form of Prop 3.3). -/
theorem abs_signed_eq_absPi (s : SignType) (hs : s ≠ 0) (hr1 : r ≤ 1) :
    |(1 - r) * (s : ℝ) * √(lu * lv) / √((1 - r * lu) * (1 - r * lv))| = absPi r lu lv := by
  have hs1 : |(s : ℝ)| = 1 := by
    rcases s with _ | _ | _
    · exact absurd rfl hs
    · simp
    · simp
  unfold absPi
  rw [abs_div, abs_mul, abs_mul, hs1, abs_of_nonneg (by linarith : (0 : ℝ) ≤ 1 - r),
    abs_of_nonneg (Real.sqrt_nonneg _), abs_of_nonneg (Real.sqrt_nonneg _), mul_one]

/-- Lower bound `(1-r)√(ℓ_uℓ_v) ≤ |π_uv|`. -/
theorem absPi_lower (hr0 : 0 ≤ r) (hr1 : r ≤ 1) (hl0 : 0 ≤ lu) (hl1 : lu < 1) (hm0 : 0 ≤ lv)
    (hm1 : lv < 1) : (1 - r) * √(lu * lv) ≤ absPi r lu lv := by
  unfold absPi
  have hu := one_sub_mul_pos hr1 hl0 hl1
  have hv := one_sub_mul_pos hr1 hm0 hm1
  have hd : 0 < √((1 - r * lu) * (1 - r * lv)) := Real.sqrt_pos.mpr (mul_pos hu hv)
  have hd1 : √((1 - r * lu) * (1 - r * lv)) ≤ 1 := by
    rw [Real.sqrt_le_one]
    have : r * lu ≥ 0 := mul_nonneg hr0 hl0
    have : r * lv ≥ 0 := mul_nonneg hr0 hm0
    nlinarith
  rw [le_div_iff₀ hd]
  have : 0 ≤ (1 - r) * √(lu * lv) := mul_nonneg (by linarith) (Real.sqrt_nonneg _)
  nlinarith

/-- Denominator bound `√((1 - rℓ_u)(1 - rℓ_v)) ≥ 1 - r·max(ℓ_u,ℓ_v)`. -/
lemma sqrt_den_ge (hr0 : 0 ≤ r) (hr1 : r ≤ 1) (hl0 : 0 ≤ lu) (hl1 : lu < 1)
    (hm1 : lv < 1) : 1 - r * max lu lv ≤ √((1 - r * lu) * (1 - r * lv)) := by
  have hmax : max lu lv < 1 := max_lt hl1 hm1
  have h0 : 0 ≤ 1 - r * max lu lv := (one_sub_mul_pos hr1 (le_max_of_le_left hl0) hmax).le
  rw [show 1 - r * max lu lv = √((1 - r * max lu lv) ^ 2) from (Real.sqrt_sq h0).symm]
  apply Real.sqrt_le_sqrt
  have a1 : 1 - r * max lu lv ≤ 1 - r * lu := by nlinarith [le_max_left lu lv]
  have a2 : 1 - r * max lu lv ≤ 1 - r * lv := by nlinarith [le_max_right lu lv]
  rw [sq]; exact mul_le_mul a1 a2 h0 (le_trans h0 a1)

/-- `|π_uv| ≤ |κ_uv|(1-r)/(1 - r ℓ_max)` with `|κ_uv| = √(ℓ_uℓ_v)`. -/
theorem absPi_le_kappa (hr0 : 0 ≤ r) (hr1 : r ≤ 1) (hl0 : 0 ≤ lu) (hl1 : lu < 1) (_hm0 : 0 ≤ lv)
    (hm1 : lv < 1) : absPi r lu lv ≤ √(lu * lv) * (1 - r) / (1 - r * max lu lv) := by
  unfold absPi
  have hmax : max lu lv < 1 := max_lt hl1 hm1
  have h0 : 0 < 1 - r * max lu lv := one_sub_mul_pos hr1 (le_max_of_le_left hl0) hmax
  rw [mul_comm (√(lu * lv))]
  exact div_le_div_of_nonneg_left (mul_nonneg (by linarith) (Real.sqrt_nonneg _)) h0
    (sqrt_den_ge hr0 hr1 hl0 hl1 hm1)

/-- Upper bound `|π_uv| ≤ (1-r)√(ℓ_uℓ_v)/(1 - ℓ_max)`. -/
theorem absPi_upper (hr0 : 0 ≤ r) (hr1 : r ≤ 1) (hl0 : 0 ≤ lu) (hl1 : lu < 1) (hm0 : 0 ≤ lv)
    (hm1 : lv < 1) : absPi r lu lv ≤ (1 - r) * √(lu * lv) / (1 - max lu lv) := by
  have hmax : max lu lv < 1 := max_lt hl1 hm1
  have hm0' : 0 ≤ max lu lv := le_max_of_le_left hl0
  calc absPi r lu lv ≤ √(lu * lv) * (1 - r) / (1 - r * max lu lv) :=
        absPi_le_kappa hr0 hr1 hl0 hl1 hm0 hm1
    _ ≤ (1 - r) * √(lu * lv) / (1 - max lu lv) := by
        rw [mul_comm]
        apply div_le_div_of_nonneg_left (mul_nonneg (by linarith) (Real.sqrt_nonneg _))
          (by linarith)
        nlinarith

/-- **`|π_uv| < |κ_uv|` for `r > 0`.** This needs `ℓ_uℓ_v > 0` (i.e. `b_u b_v ≠ 0`); if
`b_u b_v = 0` both sides are 0. FINITE.md states it without this hypothesis (erratum F1). -/
theorem absPi_lt_kappa (hr0 : 0 < r) (hr1 : r ≤ 1) (hl0 : 0 < lu) (hl1 : lu < 1) (hm0 : 0 < lv)
    (hm1 : lv < 1) : absPi r lu lv < √(lu * lv) := by
  have hmax : max lu lv < 1 := max_lt hl1 hm1
  have h0 : 0 < 1 - r * max lu lv := one_sub_mul_pos hr1 (le_max_of_le_left hl0.le) hmax
  have hk : 0 < √(lu * lv) := Real.sqrt_pos.mpr (mul_pos hl0 hm0)
  calc absPi r lu lv ≤ √(lu * lv) * (1 - r) / (1 - r * max lu lv) :=
        absPi_le_kappa hr0.le hr1 hl0.le hl1 hm0.le hm1
    _ < √(lu * lv) := by
        rw [div_lt_iff₀ h0]
        have : r * max lu lv < r := by nlinarith
        nlinarith

/-- **`π_uv = 0 ⇔ r = 1`**, for `ℓ_u, ℓ_v ∈ (0,1)` and `r ∈ [0,1]` (needs `b_u b_v ≠ 0`). -/
theorem absPi_eq_zero_iff (hr1 : r ≤ 1) (hl0 : 0 < lu) (hl1 : lu < 1) (hm0 : 0 < lv)
    (hm1 : lv < 1) : absPi r lu lv = 0 ↔ r = 1 := by
  unfold absPi
  have hu := one_sub_mul_pos hr1 hl0.le hl1
  have hv := one_sub_mul_pos hr1 hm0.le hm1
  have hd : 0 < √((1 - r * lu) * (1 - r * lv)) := Real.sqrt_pos.mpr (mul_pos hu hv)
  have hk : 0 < √(lu * lv) := Real.sqrt_pos.mpr (mul_pos hl0 hm0)
  rw [div_eq_zero_iff, or_iff_left hd.ne', mul_eq_zero, or_iff_left hk.ne', sub_eq_zero, eq_comm]

/-- `f_ℓ(r) = (1 - r)/(1 - rℓ)` is strictly decreasing on `[0,1]` for `ℓ < 1`. -/
lemma f_strictAnti {l : ℝ} (hl0 : 0 ≤ l) (hl1 : l < 1) :
    StrictAntiOn (fun r : ℝ => (1 - r) / (1 - r * l)) (Set.Icc 0 1) := by
  intro a ha b hb hab
  simp only
  have h1 := one_sub_mul_pos ha.2 hl0 hl1
  have h2 := one_sub_mul_pos hb.2 hl0 hl1
  rw [div_lt_div_iff₀ h2 h1]
  nlinarith

/-- `absPi r ℓ_u ℓ_v = √(ℓ_u ℓ_v · f_u(r) f_v(r))`. -/
lemma absPi_eq_sqrt (hr1 : r ≤ 1) (hl0 : 0 ≤ lu) (hl1 : lu < 1) (hm0 : 0 ≤ lv) (hm1 : lv < 1) :
    absPi r lu lv = √(lu * lv * ((1 - r) / (1 - r * lu) * ((1 - r) / (1 - r * lv)))) := by
  unfold absPi
  have hu := one_sub_mul_pos hr1 hl0 hl1
  have hv := one_sub_mul_pos hr1 hm0 hm1
  rw [show lu * lv * ((1 - r) / (1 - r * lu) * ((1 - r) / (1 - r * lv))) =
      (1 - r) ^ 2 * (lu * lv) / ((1 - r * lu) * (1 - r * lv)) by field_simp,
    sqrt_div_pos (mul_pos hu hv), Real.sqrt_mul (sq_nonneg _), Real.sqrt_sq (by linarith)]

/-- **Monotonicity in `r`.** `|π_uv|` decreases strictly in `r` on `[0,1]` when
`ℓ_u, ℓ_v ∈ (0,1)` (FINITE.md says `ℓ_u, ℓ_v < 1`; positivity is also needed, erratum F1). -/
theorem absPi_strictAntiOn (hl0 : 0 < lu) (hl1 : lu < 1) (hm0 : 0 < lv) (hm1 : lv < 1) :
    StrictAntiOn (fun r => absPi r lu lv) (Set.Icc 0 1) := by
  intro a ha b hb hab
  simp only
  rw [absPi_eq_sqrt ha.2 hl0.le hl1 hm0.le hm1, absPi_eq_sqrt hb.2 hl0.le hl1 hm0.le hm1]
  apply Real.sqrt_lt_sqrt
  · have := one_sub_mul_pos hb.2 hl0.le hl1
    have := one_sub_mul_pos hb.2 hm0.le hm1
    have : 0 ≤ 1 - b := by linarith [hb.2]
    positivity
  · have fu := f_strictAnti hl0.le hl1 ha hb hab
    have fv := f_strictAnti hm0.le hm1 ha hb hab
    simp only at fu fv
    have hu := one_sub_mul_pos hb.2 hl0.le hl1
    have hv := one_sub_mul_pos hb.2 hm0.le hm1
    have gu : 0 ≤ (1 - b) / (1 - b * lu) := div_nonneg (by linarith [hb.2]) hu.le
    have gv : 0 ≤ (1 - b) / (1 - b * lv) := div_nonneg (by linarith [hb.2]) hv.le
    apply mul_lt_mul_of_pos_left _ (mul_pos hl0 hm0)
    exact mul_lt_mul'' fu fv gu gv

/-- **Equal shares:** `|π| = (1-r)ℓ/(1-rℓ)`. (FINITE.md writes `π`; the sign is `sgn(b_u b_v)`.) -/
theorem absPi_equal {l : ℝ} (hr1 : r ≤ 1) (hl0 : 0 ≤ l) (hl1 : l < 1) :
    absPi r l l = (1 - r) * l / (1 - r * l) := by
  unfold absPi
  have h := one_sub_mul_pos hr1 hl0 hl1
  rw [← sq, ← sq, Real.sqrt_sq hl0, Real.sqrt_sq h.le]

end AbsPi

/-! ### `r_crit` (Thm 3.5) -/

/-- `r_crit(ℓ, τ') = 1 - τ'(1-ℓ)/(ℓ(1-τ'))`. -/
noncomputable def rcrit (l τ : ℝ) : ℝ := 1 - τ * (1 - l) / (l * (1 - τ))

/-- **`r_crit`.** For equal shares `ℓ ∈ (0,1)`, `τ' < 1` and `r ≤ 1`: `|π| < τ'` iff `r > r_crit`. -/
theorem rcrit_iff {l τ r : ℝ} (hl0 : 0 < l) (hl1 : l < 1) (hτ1 : τ < 1) (hr1 : r ≤ 1) :
    (1 - r) * l / (1 - r * l) < τ ↔ rcrit l τ < r := by
  have hd := one_sub_mul_pos hr1 hl0.le hl1
  have hτ : 0 < 1 - τ := by linarith
  have hlτ : 0 < l * (1 - τ) := mul_pos hl0 hτ
  have key : rcrit l τ = (l - τ) / (l * (1 - τ)) := by
    unfold rcrit; field_simp; ring
  rw [div_lt_iff₀ hd, key, div_lt_iff₀ hlτ]
  constructor <;> intro h <;> nlinarith

/-- `|π| = τ'` iff `r = r_crit` (FINITE.md: "from π = τ' in 3.3"). -/
theorem rcrit_eq_iff {l τ r : ℝ} (hl0 : 0 < l) (hl1 : l < 1) (hτ1 : τ < 1) (hr1 : r ≤ 1) :
    (1 - r) * l / (1 - r * l) = τ ↔ r = rcrit l τ := by
  have hd := one_sub_mul_pos hr1 hl0.le hl1
  have hτ : 0 < 1 - τ := by linarith
  have hlτ : 0 < l * (1 - τ) := mul_pos hl0 hτ
  unfold rcrit
  rw [div_eq_iff hd.ne']
  constructor
  · intro h; field_simp; linarith
  · intro h; rw [h]; field_simp; ring

/-- The numerical example of Thm 3.5: `ℓ = 0.64`, `τ' = 0.082` give `r_crit ≈ 0.95`. -/
theorem rcrit_example : 0.949 < rcrit 0.64 0.082 ∧ rcrit 0.64 0.082 < 0.951 := by
  unfold rcrit; constructor <;> norm_num

/-! ### `n_crit` (Prop 2.3, Thm 3.5) -/

/-- **`n_crit` / `n_1/2`.** For the Fisher threshold `τ(n) = tanh(z/√(n - k₀))` (`k₀ = 3 + k`,
`k` conditioning variables) and a population correlation `p ∈ (0,1)` (`p = |π|` or `p = κ`):
the threshold is below `p` iff `n > k₀ + (z/artanh p)²`. With `k₀ = 4` this is
`n_crit = 4 + (z_c/atanh|π|)²`, and with `k₀ = 3` it is `n_1/2 = 3 + (z_c/atanh κ)²`. -/
theorem ncrit_iff {z p n k0 : ℝ} (hz : 0 < z) (hp0 : 0 < p) (hp1 : p < 1) (hn : k0 < n) :
    tanh (z / √(n - k0)) < p ↔ k0 + (z / artanh p) ^ 2 < n := by
  have hpI : p ∈ Set.Ioo (-1 : ℝ) 1 := ⟨by linarith, hp1⟩
  have hA : 0 < artanh p := artanh_pos ⟨hp0, hp1⟩
  have hs : 0 < √(n - k0) := Real.sqrt_pos.mpr (by linarith)
  have ht : ∀ x, tanh x < p ↔ x < artanh p := fun x => by
    rw [← artanh_lt_artanh_iff ⟨neg_one_lt_tanh x, tanh_lt_one x⟩ hpI, artanh_tanh]
  rw [ht, div_lt_iff₀ hs, ← div_lt_iff₀' hA, Real.lt_sqrt (by positivity)]
  constructor <;> intro h <;> linarith

/-- Equality form: the threshold equals `p` iff `n = k₀ + (z/artanh p)²`. -/
theorem ncrit_eq_iff {z p n k0 : ℝ} (hz : 0 < z) (hp0 : 0 < p) (hp1 : p < 1) (hn : k0 < n) :
    tanh (z / √(n - k0)) = p ↔ n = k0 + (z / artanh p) ^ 2 := by
  have hpI : p ∈ Set.Ioo (-1 : ℝ) 1 := ⟨by linarith, hp1⟩
  have hA : 0 < artanh p := artanh_pos ⟨hp0, hp1⟩
  have hs : 0 < √(n - k0) := Real.sqrt_pos.mpr (by linarith)
  have ht : ∀ x, tanh x = p ↔ x = artanh p := fun x => by
    constructor
    · intro h; rw [← h, artanh_tanh]
    · intro h; rw [h, tanh_artanh hpI]
  rw [ht, div_eq_iff hs.ne']
  rw [show z = artanh p * √(n - k0) ↔ √(n - k0) = z / artanh p by
    rw [eq_div_iff hA.ne', mul_comm]; exact eq_comm]
  constructor
  · intro h
    have := Real.sq_sqrt (show (0 : ℝ) ≤ n - k0 by linarith)
    rw [h] at this; linarith
  · intro h
    rw [show n - k0 = (z / artanh p) ^ 2 by linarith, Real.sqrt_sq (by positivity)]

/-! ### Props 3.3 (absolute form), 3.6, 3.7, 3.8 in the inner-product model -/

section Inner2

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

local notation "⟪" x ", " y "⟫" => @inner ℝ _ _ x y

/-- The share `ℓ = b²/(Var Y + b²)` lies in `[0,1)` when `Var Y > 0`. -/
lemma loading_mem {V b : ℝ} (hV : 0 < V) : 0 ≤ b ^ 2 / (V + b ^ 2) ∧ b ^ 2 / (V + b ^ 2) < 1 := by
  have h : 0 < V + b ^ 2 := by positivity
  exact ⟨by positivity, by rw [div_lt_one h]; linarith⟩

/-- **Prop 3.3, absolute form.** For river non-relatives with `b_u b_v ≠ 0`,
`|π_uv| = (1-r)√(ℓ_uℓ_v)/√((1-rℓ_u)(1-rℓ_v))`, to which `absPi_lower`, `absPi_upper`,
`absPi_le_kappa`, `absPi_lt_kappa`, `absPi_eq_zero_iff`, `absPi_strictAntiOn` apply. -/
theorem prop33_abs {Yu Yv C e : E} {bu bv r : ℝ} (h : ProxyHyp Yu Yv C e) (hr0 : 0 ≤ r)
    (hr1 : r ≤ 1) (hu : 0 < ⟪Yu, Yu⟫) (hv : 0 < ⟪Yv, Yv⟫) (huv : ⟪Yu, Yv⟫ = 0)
    (hb : bu * bv ≠ 0) :
    |pcorr (Yu + bu • C) (Yv + bv • C) (√r • C + √(1 - r) • e)| =
      absPi r (bu ^ 2 / (⟪Yu, Yu⟫ + bu ^ 2)) (bv ^ 2 / (⟪Yv, Yv⟫ + bv ^ 2)) := by
  rw [prop33 h hr0 hr1 hu hv huv]
  exact abs_signed_eq_absPi _ (sign_ne_zero.mpr hb) hr1

/-- `corr(X_u, Z) = b_u √r / s_u` in the proxy model. -/
lemma corr_node_proxy {Yu C e : E} {bu r : ℝ} (huC : ⟪Yu, C⟫ = 0) (hue : ⟪Yu, e⟫ = 0)
    (hC : ⟪C, C⟫ = 1) (he : ⟪e, e⟫ = 1) (hCe : ⟪C, e⟫ = 0) (hr0 : 0 ≤ r) (hr1 : r ≤ 1) :
    corr (Yu + bu • C) (√r • C + √(1 - r) • e) = bu * √r / √(⟪Yu, Yu⟫ + bu ^ 2) := by
  unfold corr
  rw [inner_node_proxy huC hue hC hCe, inner_proxy_self hr0 hr1 hC he hCe,
    inner_modelD huC huC hC, mul_one]
  ring_nf

/-- **Prop 3.8 (reliability from three correlations).** For river non-relatives with
`b_u b_v ≠ 0`: `ρ_uZ ρ_vZ / ρ_uv = r`. -/
theorem prop38_ratio {Yu Yv C e : E} {bu bv r : ℝ} (h : ProxyHyp Yu Yv C e) (hr0 : 0 ≤ r)
    (hr1 : r ≤ 1) (huv : ⟪Yu, Yv⟫ = 0) (hb : bu * bv ≠ 0) :
    corr (Yu + bu • C) (√r • C + √(1 - r) • e) * corr (Yv + bv • C) (√r • C + √(1 - r) • e) /
      corr (Yu + bu • C) (Yv + bv • C) = r := by
  have hu0 : 0 ≤ ⟪Yu, Yu⟫ := real_inner_self_nonneg
  have hv0 : 0 ≤ ⟪Yv, Yv⟫ := real_inner_self_nonneg
  have hbu : bu ≠ 0 := left_ne_zero_of_mul hb
  have hbv : bv ≠ 0 := right_ne_zero_of_mul hb
  have hSu : 0 < √(⟪Yu, Yu⟫ + bu ^ 2) := Real.sqrt_pos.mpr (by positivity)
  have hSv : 0 < √(⟪Yv, Yv⟫ + bv ^ 2) := Real.sqrt_pos.mpr (by positivity)
  rw [corr_node_proxy h.huC h.hue h.hC h.he h.hCe hr0 hr1,
    corr_node_proxy h.hvC h.hve h.hC h.he h.hCe hr0 hr1, prop21_corr h.huC h.hvC h.hC huv]
  have hrr : √r ^ 2 = r := Real.sq_sqrt hr0
  field_simp
  linear_combination hrr

/-- **Prop 3.8, loading.** For `r > 0`, `ℓ_u = ρ_uZ²/r` (FINITE.md omits `r > 0`; erratum F2). -/
theorem prop38_loading {Yu C e : E} {bu r : ℝ} (huC : ⟪Yu, C⟫ = 0) (hue : ⟪Yu, e⟫ = 0)
    (hC : ⟪C, C⟫ = 1) (he : ⟪e, e⟫ = 1) (hCe : ⟪C, e⟫ = 0) (hr0 : 0 < r) (hr1 : r ≤ 1) :
    bu ^ 2 / (⟪Yu, Yu⟫ + bu ^ 2) = corr (Yu + bu • C) (√r • C + √(1 - r) • e) ^ 2 / r := by
  rw [corr_node_proxy huC hue hC he hCe hr0.le hr1, div_pow, mul_pow,
    Real.sq_sqrt hr0.le, Real.sq_sqrt (by have : 0 ≤ ⟪Yu, Yu⟫ := real_inner_self_nonneg; positivity)]
  field_simp

/-- **Prop 3.6, general form.** If `Y ⟂ (D, Z)` across the two nodes and `Cov(Y_u, Y_v) = 0`
(river non-relatives), the partial covariance of `X = Y + D` given `Z` is the covariance of the
residual driver components `D - P_Z D`. -/
theorem prop36_pcov {Yu Yv Du Dv Z : E} (huv : ⟪Yu, Yv⟫ = 0) (huDv : ⟪Yu, Dv⟫ = 0)
    (hvDu : ⟪Yv, Du⟫ = 0) (huZ : ⟪Yu, Z⟫ = 0) (hvZ : ⟪Yv, Z⟫ = 0) (hZ : ⟪Z, Z⟫ ≠ 0) :
    pcov (Yu + Du) (Yv + Dv) Z = ⟪resid Du Z, resid Dv Z⟫ := by
  rw [inner_resid Du Dv Z hZ]
  unfold pcov
  simp only [inner_add_left, inner_add_right]
  rw [huv, huDv, real_inner_comm Yv Du, hvDu, huZ, hvZ]
  ring

/-- **Prop 3.6, iff.** Under positive variances and partial variances, `π_uv = 0` iff the
residual driver components are uncorrelated. -/
theorem prop36_zero_iff {Yu Yv Du Dv Z : E} (huv : ⟪Yu, Yv⟫ = 0) (huDv : ⟪Yu, Dv⟫ = 0)
    (hvDu : ⟪Yv, Du⟫ = 0) (huZ : ⟪Yu, Z⟫ = 0) (hvZ : ⟪Yv, Z⟫ = 0)
    (hx : 0 < ⟪Yu + Du, Yu + Du⟫) (hy : 0 < ⟪Yv + Dv, Yv + Dv⟫) (hz : 0 < ⟪Z, Z⟫)
    (hpx : 0 < pcov (Yu + Du) (Yu + Du) Z) (hpy : 0 < pcov (Yv + Dv) (Yv + Dv) Z) :
    pcorr (Yu + Du) (Yv + Dv) Z = 0 ↔ ⟪resid Du Z, resid Dv Z⟫ = 0 := by
  rw [pcorr_eq_pcov _ _ _ hx hy hz hpx hpy, prop36_pcov huv huDv hvDu huZ hvZ hz.ne',
    div_eq_zero_iff, or_iff_left (Real.sqrt_pos.mpr (mul_pos hpx hpy)).ne']

/-- **Prop 3.6, rank-one driver with `Z = F`.** The residual driver components vanish, so
`π_uv = 0` for every river non-relative pair (by `prop36_zero_iff`). -/
theorem prop36_rank_one (F : E) (bu bv : ℝ) (hF : ⟪F, F⟫ ≠ 0) :
    ⟪resid (bu • F) F, resid (bv • F) F⟫ = 0 := by
  have h : ∀ b : ℝ, resid (b • F) F = 0 := fun b => by
    unfold resid
    rw [real_inner_smul_left, mul_div_assoc, div_self hF, mul_one, sub_self]
  rw [h bu, inner_zero_left]

section AR1

variable {Yu Yv : E} {Cs : ℕ → E} {φ bu bv : ℝ} {ku kv : ℕ}

/-- Hypotheses of the AR(1) multi-filter example: `Cov(C(t-i), C(t-j)) = φ^|i-j|`, `Y ⟂ C`,
`Cov(Y_u, Y_v) = 0`. `Cs k` is `C(t-k)` and the proxy is `Z = C(t) = Cs 0`. -/
structure AR1Hyp (Yu Yv : E) (Cs : ℕ → E) (φ : ℝ) : Prop where
  hC : ∀ i j, ⟪Cs i, Cs j⟫ = φ ^ Nat.dist i j
  huC : ∀ k, ⟪Yu, Cs k⟫ = 0
  hvC : ∀ k, ⟪Yv, Cs k⟫ = 0
  huv : ⟪Yu, Yv⟫ = 0

lemma ar1_inner (hC : ∀ i j, ⟪Cs i, Cs j⟫ = φ ^ Nat.dist i j) {Y Y' : E}
    (hY : ∀ k, ⟪Y, Cs k⟫ = 0) (hY' : ∀ k, ⟪Y', Cs k⟫ = 0) (b b' : ℝ) (k k' : ℕ) :
    ⟪Y + b • Cs k, Y' + b' • Cs k'⟫ = ⟪Y, Y'⟫ + b * b' * φ ^ Nat.dist k k' := by
  simp only [inner_add_left, inner_add_right, real_inner_smul_left, real_inner_smul_right]
  rw [real_inner_comm Y' (Cs k), hY, hY', hC]
  ring

lemma ar1_proxy (hC : ∀ i j, ⟪Cs i, Cs j⟫ = φ ^ Nat.dist i j) {Y : E}
    (hY : ∀ k, ⟪Y, Cs k⟫ = 0) (b : ℝ) (k : ℕ) : ⟪Y + b • Cs k, Cs 0⟫ = b * φ ^ k := by
  simp only [inner_add_left, real_inner_smul_left]
  rw [hY, hC, Nat.dist_zero_right]
  ring

/-- **Prop 3.6, AR(1) example: partial covariance.**
`pcov = b_u b_v (φ^{|k_u - k_v|} - φ^{k_u + k_v})`. -/
theorem prop36_ar1_pcov (h : AR1Hyp Yu Yv Cs φ) :
    pcov (Yu + bu • Cs ku) (Yv + bv • Cs kv) (Cs 0) =
      bu * bv * (φ ^ Nat.dist ku kv - φ ^ (ku + kv)) := by
  unfold pcov
  rw [ar1_inner h.hC h.huC h.hvC, ar1_proxy h.hC h.huC, ar1_proxy h.hC h.hvC, h.hC,
    Nat.dist_self, h.huv, pow_zero, pow_add]
  ring

/-- Partial variance `s_u² - b_u² φ^{2k_u}` with `s_u² = Var Y_u + b_u²`. -/
theorem prop36_ar1_pvar (h : AR1Hyp Yu Yv Cs φ) :
    pcov (Yu + bu • Cs ku) (Yu + bu • Cs ku) (Cs 0) =
      (⟪Yu, Yu⟫ + bu ^ 2) - bu ^ 2 * φ ^ (2 * ku) := by
  unfold pcov
  rw [ar1_inner h.hC h.huC h.huC, ar1_proxy h.hC h.huC, h.hC, Nat.dist_self, Nat.dist_self,
    pow_zero]
  ring

/-- **Prop 3.6, AR(1) formula.** For `0 ≤ φ ≤ 1` and `Var Y > 0`,
`π_uv = b_u b_v (φ^{|k_u-k_v|} - φ^{k_u+k_v}) / √((s_u² - b_u²φ^{2k_u})(s_v² - b_v²φ^{2k_v}))`. -/
theorem prop36_ar1 (h : AR1Hyp Yu Yv Cs φ) (hφ0 : 0 ≤ φ) (hφ1 : φ ≤ 1)
    (hu : 0 < ⟪Yu, Yu⟫) (hv : 0 < ⟪Yv, Yv⟫) :
    pcorr (Yu + bu • Cs ku) (Yv + bv • Cs kv) (Cs 0) =
      bu * bv * (φ ^ Nat.dist ku kv - φ ^ (ku + kv)) /
        √(((⟪Yu, Yu⟫ + bu ^ 2) - bu ^ 2 * φ ^ (2 * ku)) *
          ((⟪Yv, Yv⟫ + bv ^ 2) - bv ^ 2 * φ ^ (2 * kv))) := by
  have h' : AR1Hyp Yv Yu Cs φ := ⟨h.hC, h.hvC, h.huC, by rw [real_inner_comm]; exact h.huv⟩
  have pu := prop36_ar1_pvar (bu := bu) (ku := ku) h
  have pv := prop36_ar1_pvar (bu := bv) (ku := kv) h'
  have hpos : ∀ (V b : ℝ) (k : ℕ), 0 < V → 0 < V + b ^ 2 - b ^ 2 * φ ^ (2 * k) := by
    intro V b k hV
    have : φ ^ (2 * k) ≤ 1 := pow_le_one₀ hφ0 hφ1
    nlinarith [sq_nonneg b]
  have hz : 0 < ⟪Cs 0, Cs 0⟫ := by rw [h.hC, Nat.dist_self, pow_zero]; norm_num
  rw [pcorr_eq_pcov _ _ _
    (by rw [ar1_inner h.hC h.huC h.huC, Nat.dist_self, pow_zero]; nlinarith [sq_nonneg bu])
    (by rw [ar1_inner h.hC h.hvC h.hvC, Nat.dist_self, pow_zero]; nlinarith [sq_nonneg bv])
    hz (by rw [pu]; exact hpos _ _ _ hu) (by rw [pv]; exact hpos _ _ _ hv),
    prop36_ar1_pcov h, pu, pv]

/-- **Prop 3.6, nonvanishing.** For `b_u b_v ≠ 0`, `0 < φ < 1` and lags `k_u, k_v > 0`, the
numerator `b_u b_v (φ^{|k_u-k_v|} - φ^{k_u+k_v})` is nonzero, so `π_uv ≠ 0` even though the proxy
is the driver itself (`r = 1` with respect to `C(t)`). FINITE.md leaves `b_u b_v ≠ 0` implicit. -/
theorem prop36_ar1_ne_zero (h : AR1Hyp Yu Yv Cs φ) (hb : bu * bv ≠ 0) (hφ0 : 0 < φ)
    (hφ1 : φ < 1) (hku : 0 < ku) (hkv : 0 < kv) (hu : 0 < ⟪Yu, Yu⟫) (hv : 0 < ⟪Yv, Yv⟫) :
    pcorr (Yu + bu • Cs ku) (Yv + bv • Cs kv) (Cs 0) ≠ 0 := by
  rw [prop36_ar1 h hφ0.le hφ1.le hu hv]
  have hd : Nat.dist ku kv < ku + kv := by unfold Nat.dist; omega
  have hlt : φ ^ (ku + kv) < φ ^ Nat.dist ku kv := pow_lt_pow_right_of_lt_one₀ hφ0 hφ1 hd
  have hpos : ∀ (V b : ℝ) (k : ℕ), 0 < V → 0 < V + b ^ 2 - b ^ 2 * φ ^ (2 * k) := by
    intro V b k hV
    have : φ ^ (2 * k) ≤ 1 := pow_le_one₀ hφ0.le hφ1.le
    nlinarith [sq_nonneg b]
  apply div_ne_zero
  · exact mul_ne_zero hb (by linarith)
  · exact (Real.sqrt_pos.mpr (mul_pos (hpos _ _ _ hu) (hpos _ _ _ hv))).ne'

end AR1

/-- **Prop 3.7 (proxy contaminated by river relatives), numerator.** With `Var Z = 1`,
`c_Z = Cov(C, Z)`, `q_u = Cov(Y_u, Z)`, for river non-relatives the partial covariance is
`b_u b_v (1 - c_Z²) - c_Z (b_u q_v + b_v q_u) - q_u q_v`. -/
theorem prop37_pcov {Yu Yv C Z : E} {bu bv : ℝ} (huC : ⟪Yu, C⟫ = 0) (hvC : ⟪Yv, C⟫ = 0)
    (hC : ⟪C, C⟫ = 1) (huv : ⟪Yu, Yv⟫ = 0) (hZ : ⟪Z, Z⟫ = 1) :
    pcov (Yu + bu • C) (Yv + bv • C) Z =
      bu * bv * (1 - ⟪C, Z⟫ ^ 2) - ⟪C, Z⟫ * (bu * ⟪Yv, Z⟫ + bv * ⟪Yu, Z⟫) -
        ⟪Yu, Z⟫ * ⟪Yv, Z⟫ := by
  unfold pcov
  rw [inner_modelD huC hvC hC, huv, hZ]
  simp only [inner_add_left, real_inner_smul_left]
  ring

/-- **Prop 3.7, bound.** With `r = c_Z²` (which is `≤ 1` by Cauchy-Schwarz),
`|numerator| ≤ (1-r)|b_u b_v| + √r(|b_u q_v| + |b_v q_u|) + |q_u q_v|`. -/
theorem prop37_bound {Yu Yv C Z : E} {bu bv : ℝ} (huC : ⟪Yu, C⟫ = 0) (hvC : ⟪Yv, C⟫ = 0)
    (hC : ⟪C, C⟫ = 1) (huv : ⟪Yu, Yv⟫ = 0) (hZ : ⟪Z, Z⟫ = 1) :
    |pcov (Yu + bu • C) (Yv + bv • C) Z| ≤
      (1 - ⟪C, Z⟫ ^ 2) * |bu * bv| + √(⟪C, Z⟫ ^ 2) * (|bu * ⟪Yv, Z⟫| + |bv * ⟪Yu, Z⟫|) +
        |⟪Yu, Z⟫ * ⟪Yv, Z⟫| := by
  rw [prop37_pcov huC hvC hC huv hZ, Real.sqrt_sq_eq_abs]
  have hcs : ⟪C, Z⟫ * ⟪C, Z⟫ ≤ ⟪C, C⟫ * ⟪Z, Z⟫ := real_inner_mul_inner_self_le C Z
  rw [hC, hZ] at hcs
  have h1 : 0 ≤ 1 - ⟪C, Z⟫ ^ 2 := by nlinarith
  set c := ⟪C, Z⟫
  have t1 := abs_sub (bu * bv * (1 - c ^ 2) - c * (bu * ⟪Yv, Z⟫ + bv * ⟪Yu, Z⟫))
    (⟪Yu, Z⟫ * ⟪Yv, Z⟫)
  have t2 := abs_sub (bu * bv * (1 - c ^ 2)) (c * (bu * ⟪Yv, Z⟫ + bv * ⟪Yu, Z⟫))
  have t3 : |bu * bv * (1 - c ^ 2)| = (1 - c ^ 2) * |bu * bv| := by
    rw [abs_mul (bu * bv), abs_of_nonneg h1, mul_comm]
  have t4 : |c * (bu * ⟪Yv, Z⟫ + bv * ⟪Yu, Z⟫)| ≤ |c| * (|bu * ⟪Yv, Z⟫| + |bv * ⟪Yu, Z⟫|) := by
    rw [abs_mul c]; exact mul_le_mul_of_nonneg_left (abs_add_le _ _) (abs_nonneg c)
  linarith

end Inner2

end RelSort
