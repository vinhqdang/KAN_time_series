/-
Lemma 3.4 of `research/theory2/finite/FINITE.md` (Lipschitz bound for the three-variable partial
correlation) and the deterministic bracket of Theorem 3.5.

The proof goes one coordinate at a time instead of along the segment. The path
`(a,b,c) → (a,b,c') → (a,b',c') → (a',b',c')` stays in `[-1,1] × [-q,q]²`, so the same three
partial-derivative bounds apply, and the constant is the same
`L(q) = 1/(1-q²) + 4q/(1-q²)²`.
-/
import RelSort.FinitePartial

open Real

namespace RelSort

/-- `L(q) = 1/(1 - q²) + 4q/(1 - q²)²`. -/
noncomputable def lipL (q : ℝ) : ℝ := 1 / (1 - q ^ 2) + 4 * q / (1 - q ^ 2) ^ 2

/-- `pc` is symmetric in its last two arguments. -/
lemma pc_symm (a b c : ℝ) : pc a b c = pc a c b := by
  unfold pc; rw [mul_comm b c, mul_comm (1 - b ^ 2)]

/-- `pc a b c = (a - bc)/(√(1-b²) √(1-c²))`. -/
lemma pc_eq_split {a b c : ℝ} (hb : b ^ 2 ≤ 1) :
    pc a b c = (a - b * c) / (√(1 - b ^ 2) * √(1 - c ^ 2)) := by
  unfold pc; rw [Real.sqrt_mul (by linarith)]

/-- Derivative of `b ↦ (a - bc)/(√(1-b²) K)`. -/
lemma hasDerivAt_pc_b (a c K b : ℝ) (hb : b ^ 2 < 1) (hK : K ≠ 0) :
    HasDerivAt (fun t => (a - t * c) / (√(1 - t ^ 2) * K))
      ((a * b - c) / ((1 - b ^ 2) * √(1 - b ^ 2) * K)) b := by
  have hpos : 0 < 1 - b ^ 2 := by linarith
  have hs : 0 < √(1 - b ^ 2) := Real.sqrt_pos.mpr hpos
  have h1 : HasDerivAt (fun t : ℝ => 1 - t ^ 2) (-(2 * b)) b := by
    have := (hasDerivAt_pow 2 b).const_sub 1
    convert this using 1; push_cast; ring
  have h2 := (h1.sqrt hpos.ne').mul_const K
  have h4 : HasDerivAt (fun t : ℝ => a - t * c) (-c) b := by
    simpa using ((hasDerivAt_id' b).mul_const c).const_sub a
  have h5 := h4.div h2 (mul_ne_zero hs.ne' hK)
  convert h5 using 1
  have hss : √(1 - b ^ 2) ^ 2 = 1 - b ^ 2 := Real.sq_sqrt hpos.le
  field_simp
  rw [hss]
  ring

/-- `√(1 - t²) ≥ √(1 - q²)` for `|t| ≤ q`. -/
lemma sqrt_one_sub_sq_ge {t q : ℝ} (ht : |t| ≤ q) : √(1 - q ^ 2) ≤ √(1 - t ^ 2) := by
  apply Real.sqrt_le_sqrt
  have : t ^ 2 ≤ q ^ 2 := by rw [← sq_abs]; exact pow_le_pow_left₀ (abs_nonneg t) ht 2
  linarith

/-- One-coordinate Lipschitz bound in the second argument: for `|a| ≤ 1`, `|c| ≤ q < 1`,
`|b|, |b'| ≤ q`: `|g(a,b',c) - g(a,b,c)| ≤ 2q/(1-q²)² · |b' - b|`. -/
lemma pc_lip_b {a b b' c q : ℝ} (hq : q < 1) (ha : |a| ≤ 1) (hb : |b| ≤ q) (hb' : |b'| ≤ q)
    (hc : |c| ≤ q) : |pc a b' c - pc a b c| ≤ 2 * q / (1 - q ^ 2) ^ 2 * |b' - b| := by
  have hq0 : 0 ≤ q := le_trans (abs_nonneg _) hb
  have hq2 : q ^ 2 < 1 := by nlinarith
  have hQ : 0 < 1 - q ^ 2 := by linarith
  have hsq : 0 < √(1 - q ^ 2) := Real.sqrt_pos.mpr hQ
  have hsq2 : √(1 - q ^ 2) * √(1 - q ^ 2) = 1 - q ^ 2 := Real.mul_self_sqrt hQ.le
  have hsqc := sqrt_one_sub_sq_ge hc
  set K := √(1 - c ^ 2) with hK
  have hKpos : 0 < K := lt_of_lt_of_le hsq hsqc
  have hsqle : ∀ t : ℝ, |t| ≤ q → t ^ 2 < 1 := fun t ht => by
    have : t ^ 2 ≤ q ^ 2 := by rw [← sq_abs]; exact pow_le_pow_left₀ (abs_nonneg t) ht 2
    linarith
  let f : ℝ → ℝ := fun t => (a - t * c) / (√(1 - t ^ 2) * K)
  let f' : ℝ → ℝ := fun t => (a * t - c) / ((1 - t ^ 2) * √(1 - t ^ 2) * K)
  have hfeq : ∀ t, |t| ≤ q → pc a t c = f t := fun t ht => by
    rw [pc_eq_split (hsqle t ht).le]
  have hmem : ∀ t, t ∈ Set.Icc (-q) q ↔ |t| ≤ q := fun t => by
    rw [abs_le]; simp [Set.mem_Icc]
  have hderiv : ∀ t ∈ Set.Icc (-q) q, HasDerivWithinAt f (f' t) (Set.Icc (-q) q) t :=
    fun t ht => (hasDerivAt_pc_b a c K t (hsqle t ((hmem t).mp ht)) hKpos.ne').hasDerivWithinAt
  have hbound : ∀ t ∈ Set.Icc (-q) q, ‖f' t‖ ≤ 2 * q / (1 - q ^ 2) ^ 2 := by
    intro t ht
    have ht' := (hmem t).mp ht
    have hT : 0 < 1 - t ^ 2 := by linarith [hsqle t ht']
    have hT2 : 1 - q ^ 2 ≤ 1 - t ^ 2 := by
      have : t ^ 2 ≤ q ^ 2 := by rw [← sq_abs]; exact pow_le_pow_left₀ (abs_nonneg t) ht' 2
      linarith
    have hst := sqrt_one_sub_sq_ge ht'
    have hnum : |a * t - c| ≤ 2 * q := by
      calc |a * t - c| ≤ |a * t| + |c| := abs_sub _ _
        _ = |a| * |t| + |c| := by rw [abs_mul]
        _ ≤ 1 * q + q := by gcongr
        _ = 2 * q := by ring
    have hden : (1 - q ^ 2) ^ 2 ≤ (1 - t ^ 2) * √(1 - t ^ 2) * K := by
      calc (1 - q ^ 2) ^ 2 = (1 - q ^ 2) * (√(1 - q ^ 2) * √(1 - q ^ 2)) := by rw [hsq2]; ring
        _ ≤ (1 - t ^ 2) * (√(1 - t ^ 2) * K) := by
            apply mul_le_mul hT2 (mul_le_mul hst hsqc hsq.le (Real.sqrt_nonneg _))
              (by positivity) hT.le
        _ = (1 - t ^ 2) * √(1 - t ^ 2) * K := by ring
    have hdpos : 0 < (1 - t ^ 2) * √(1 - t ^ 2) * K := by
      have : 0 < √(1 - t ^ 2) := Real.sqrt_pos.mpr hT
      positivity
    rw [Real.norm_eq_abs, abs_div, abs_of_pos hdpos]
    calc |a * t - c| / ((1 - t ^ 2) * √(1 - t ^ 2) * K)
        ≤ 2 * q / ((1 - t ^ 2) * √(1 - t ^ 2) * K) := by gcongr
      _ ≤ 2 * q / (1 - q ^ 2) ^ 2 := by gcongr
  have key := Convex.norm_image_sub_le_of_norm_hasDerivWithin_le hderiv hbound (convex_Icc (-q) q)
    ((hmem b).mpr hb) ((hmem b').mpr hb')
  rw [hfeq b hb, hfeq b' hb']
  simpa [Real.norm_eq_abs] using key

/-- One-coordinate bound in the first argument: `|g(a',b,c) - g(a,b,c)| ≤ |a' - a|/(1-q²)`. -/
lemma pc_lip_a {a a' b c q : ℝ} (hq : q < 1) (hb : |b| ≤ q) (hc : |c| ≤ q) :
    |pc a' b c - pc a b c| ≤ 1 / (1 - q ^ 2) * |a' - a| := by
  have hq0 : 0 ≤ q := le_trans (abs_nonneg _) hb
  have hQ : 0 < 1 - q ^ 2 := by nlinarith
  have hb2 : b ^ 2 ≤ q ^ 2 := by rw [← sq_abs]; exact pow_le_pow_left₀ (abs_nonneg b) hb 2
  have hc2 : c ^ 2 ≤ q ^ 2 := by rw [← sq_abs]; exact pow_le_pow_left₀ (abs_nonneg c) hc 2
  have hD : 1 - q ^ 2 ≤ √((1 - b ^ 2) * (1 - c ^ 2)) := by
    rw [show 1 - q ^ 2 = √((1 - q ^ 2) ^ 2) from (Real.sqrt_sq hQ.le).symm]
    apply Real.sqrt_le_sqrt
    rw [sq]; exact mul_le_mul (by linarith) (by linarith) hQ.le (by linarith)
  have hDpos : 0 < √((1 - b ^ 2) * (1 - c ^ 2)) := lt_of_lt_of_le hQ hD
  unfold pc
  rw [← sub_div, show a' - b * c - (a - b * c) = a' - a by ring, abs_div,
    abs_of_pos hDpos, one_div_mul_eq_div]
  exact div_le_div_of_nonneg_left (abs_nonneg _) hQ hD

/-- **Lemma 3.4 (Lipschitz bound for the partial correlation).** For `a, a' ∈ [-1,1]` and
`|b|, |b'|, |c|, |c'| ≤ q < 1`:
`|g(a',b',c') - g(a,b,c)| ≤ L(q) · max(|a'-a|, |b'-b|, |c'-c|)`. -/
theorem pc_lipschitz {a b c a' b' c' q : ℝ} (hq : q < 1) (ha : |a| ≤ 1) (_ha' : |a'| ≤ 1)
    (hb : |b| ≤ q) (hb' : |b'| ≤ q) (hc : |c| ≤ q) (hc' : |c'| ≤ q) :
    |pc a' b' c' - pc a b c| ≤ lipL q * max |a' - a| (max |b' - b| |c' - c|) := by
  have hq0 : 0 ≤ q := le_trans (abs_nonneg _) hb
  have hQ : 0 < 1 - q ^ 2 := by nlinarith
  set m := max |a' - a| (max |b' - b| |c' - c|)
  have hma : |a' - a| ≤ m := le_max_left _ _
  have hmb : |b' - b| ≤ m := le_trans (le_max_left _ _) (le_max_right _ _)
  have hmc : |c' - c| ≤ m := le_trans (le_max_right _ _) (le_max_right _ _)
  have s1 := pc_lip_a (a := a) (a' := a') hq hb' hc'
  have s2 := pc_lip_b (c := c') hq ha hb hb' hc'
  have s3 : |pc a b c' - pc a b c| ≤ 2 * q / (1 - q ^ 2) ^ 2 * |c' - c| := by
    rw [pc_symm a b c', pc_symm a b c]; exact pc_lip_b hq ha hc hc' hb
  have k1 : 0 ≤ 1 / (1 - q ^ 2) := by positivity
  have k2 : 0 ≤ 2 * q / (1 - q ^ 2) ^ 2 := by positivity
  have e1 := mul_le_mul_of_nonneg_left hma k1
  have e2 := mul_le_mul_of_nonneg_left hmb k2
  have e3 := mul_le_mul_of_nonneg_left hmc k2
  have tri : |pc a' b' c' - pc a b c| ≤ |pc a' b' c' - pc a b' c'| + |pc a b' c' - pc a b c'| +
      |pc a b c' - pc a b c| := by
    have h1 := abs_sub_le (pc a' b' c') (pc a b' c') (pc a b c)
    have h2 := abs_sub_le (pc a b' c') (pc a b c') (pc a b c)
    linarith
  have hL : lipL q * m = 1 / (1 - q ^ 2) * m + 2 * q / (1 - q ^ 2) ^ 2 * m +
      2 * q / (1 - q ^ 2) ^ 2 * m := by unfold lipL; ring
  rw [hL]
  linarith

/-- **Theorem 3.5, deterministic bracket.** On the event of Theorem 1.5 for the process `(X, Z)`,
each sample correlation is within `δ` of the population one. With
`q' ≥ max(|ρ_uZ| + δ_uZ, |ρ_vZ| + δ_vZ)`, `q' < 1`, and `|r̂_uv|, |ρ_uv| ≤ 1`:
`|π̂_uv - π_uv| ≤ Δ_uv = L(q') max(δ_uv, δ_uZ, δ_vZ)`. -/
theorem thm35_bracket {ruv ruZ rvZ ρuv ρuZ ρvZ δuv δuZ δvZ q : ℝ} (hq : q < 1)
    (hquZ : |ρuZ| + δuZ ≤ q) (hqvZ : |ρvZ| + δvZ ≤ q) (hr : |ruv| ≤ 1) (hρ : |ρuv| ≤ 1)
    (duv : |ruv - ρuv| ≤ δuv) (duZ : |ruZ - ρuZ| ≤ δuZ) (dvZ : |rvZ - ρvZ| ≤ δvZ) :
    |pc ruv ruZ rvZ - pc ρuv ρuZ ρvZ| ≤ lipL q * max δuv (max δuZ δvZ) := by
  have hδuZ : 0 ≤ δuZ := le_trans (abs_nonneg _) duZ
  have hδvZ : 0 ≤ δvZ := le_trans (abs_nonneg _) dvZ
  have a1 : |ruZ| ≤ q := by
    have := abs_sub_abs_le_abs_sub ruZ ρuZ; linarith
  have a2 : |rvZ| ≤ q := by
    have := abs_sub_abs_le_abs_sub rvZ ρvZ; linarith
  have hq0 : 0 ≤ q := le_trans (abs_nonneg _) a1
  have hQ : 0 < 1 - q ^ 2 := by nlinarith
  have hL : 0 ≤ lipL q := by unfold lipL; positivity
  calc |pc ruv ruZ rvZ - pc ρuv ρuZ ρvZ|
      ≤ lipL q * max |ruv - ρuv| (max |ruZ - ρuZ| |rvZ - ρvZ|) :=
        pc_lipschitz hq hρ hr (by linarith) a1 (by linarith) a2
    _ ≤ lipL q * max δuv (max δuZ δvZ) := by
        gcongr

end RelSort
