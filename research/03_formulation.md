# 03 — Formulation (working name: SPIRE)

Status: DRAFT 1, written before any experiment. Nothing here is a result.
Scope: new method, separate from the published SPADE. SPADE is the baseline.
Decision (user): go with directions 1 + 2 from the progress discussion:
(1) one-objective debiasing via a non-convex group penalty, replacing the adaptive/refit/boot pipeline;
(2) a residual-independence term in the loss, so the estimator uses the ANM identifiability signal.

References below are written from memory and are **unverified** until checked in the same way as `01_literature_review.md`.

---

## 1. Why a new objective (what is wrong with the current pipeline)

| Problem | Evidence | What this formulation does about it |
|---|---|---|
| Group-lasso shrinks edge amplitude (calibration slope 0.67 at d=10, 0.48 at d=20) | G1 | Replace ℓ1-type group penalty by group-MCP: zero shrinkage for strong edges, by construction |
| The fix (adaptive → refit → boot) is a chain of hand-set rules with no common objective | review of `spade_variants.py` | One objective; the old adaptive step is shown to be the first LLA step of it (§3.3) |
| λ(d)=min(0.01, 0.2/d), K=15, top-2d are hand constants | `scale_*` scripts | λ from a noise-level rule (§3.2); no top-2d cut |
| Loss is MSE, so the model is "sparse regression + acyclicity"; the ANM signal (residual independent of cause) is never used | GAM-ridge ≈ SPADE (G2) | Add a residual-independence term (§4) |

## 2. Model

Standardized data x ∈ R^{n×d}. Additive-noise SEM:

x_i = b_i + Σ_{j≠i} f_ij(x_j) + ε_i, ε_i ⟂ x_{pa(i)}.

f_ij(t) = B(t)ᵀ θ_ij, cubic B-splines (G=8 grid on [-3,3], K=11 coefficients), centered so E_n[f_ij(x_j)] = 0.

**Whitened coordinates.** Let G_j = E_n[B(x_j)B(x_j)ᵀ] (centered Gram). Define u_ij = G_j^{1/2} θ_ij. Then

W_ij := ‖u_ij‖₂ = ‖f_ij‖_{L2(P_n)},

the standard deviation of the edge's contribution (the quantity already used as edge score). All penalties act on W_ij, so they are invariant to basis choice and have interpretable units: **standard deviations of the target's contribution**.

Residuals: r_i = x_i − b_i − Σ_j f_ij(x_j). DAG constraint on W as before: h(W) = tr(exp(W∘W)) − d = 0.

## 3. Part 1 — one-stage debiasing

### 3.1 Penalty
Group-MCP on each edge (Zhang 2010; Breheny & Huang 2015 for the group form):

p_{λ,γ}(t) = λt − t²/(2γ) for t ≤ γλ;  γλ²/2 for t > γλ,  γ > 1.

p′(t) = (λ − t/γ)₊. For t > γλ the derivative is 0.

### 3.2 Choice of λ (replaces the hand-set schedule)
Group-lasso theory gives λ ≍ σ·sqrt((K + log d)/n) (Lounici et al. 2011). Set

λ = c_λ · σ̂ · sqrt((K + log d)/n),

with σ̂ the residual sd of a pilot unpenalized-ridge fit, and c_λ ∈ {0.5, 1, 2} reported as a sensitivity grid (no tuning to the test benchmark). γ = 3 default (γ ∈ {2, 3, 5} as sensitivity). This is the only place a constant remains, and it has a stated role.

### 3.3 Link to the old heuristic (so the old results remain interpretable)
Local linear approximation (Zou & Li 2008): at current estimate Ŵ, MCP is majorized by a weighted group lasso with weights w_ij = p′(Ŵ_ij)/λ = (1 − Ŵ_ij/(γλ))₊. Strong edges get weight 0 (no shrinkage); small edges keep full penalty. The earlier adaptive group-lasso, w = (‖θ‖+ε)^-γ, is a heuristic version of exactly one such step.

### 3.4 Claim to prove (Proposition 1, sketch)
At any stationary point of the MCP objective with the acyclicity term inactive on that edge (λ_AL-term gradient zero), every edge with Ŵ_ij > γλ satisfies the **unpenalized** first-order condition ∇_{u_ij} (1/2n)Σ‖r_i‖² = 0, i.e. it carries no penalty-induced amplitude bias. Proof: p′ = 0 beyond γλ. The caveat is real: the augmented-Lagrangian term still acts on W, so edges on near-cycles can be shrunk by the constraint; Proposition 1 holds only for edges whose h-gradient contribution vanishes. This must be stated as such and tested (H1 below).

### 3.5 Optimization of the penalized part
Proximal gradient on the smooth part (MSE + augmented Lagrangian + part 2) with the exact group-MCP prox in whitened coordinates (firm thresholding), step η < γ:
- ‖v‖ ≤ ηλ → 0;
- ηλ < ‖v‖ ≤ γλ → (1 − ηλ/‖v‖)₊ v / (1 − η/γ);
- ‖v‖ > γλ → v.

Exact zeros appear, so the support S = {(i,j): u_ij ≠ 0} is well defined and needs no threshold. Convergence to stationary points of non-convex composite objectives: standard under KL conditions (Attouch, Bolte, Svaiter 2013); not re-proved here.

## 4. Part 2 — residual-independence term

### 4.1 Why it is needed, and why naive versions are wrong
- At an unpenalized least-squares fit, r_i is already orthogonal to the span of each parent's basis, so first-moment dependence is zero. The term can only detect **higher-order** dependence (heteroscedasticity, wrong direction, missing non-additive structure). That is exactly the information that distinguishes x→y from y→x under ANM.
- Penalizing dependence between r_i and **all** x_j is wrong: for a child c of i, x_c depends on ε_i, so the true model would be penalized. The term must be restricted to **parents the model currently claims**.

### 4.2 Definition
Normalized HSIC (CKA form, in [0,1], scale-free so the model cannot lower it by inflating residuals):

nHSIC(a,b) = HSIC(a,b) / sqrt(HSIC(a,a)·HSIC(b,b)), RBF kernels, median-heuristic bandwidth (Gretton et al. 2005/2008).

R_ind(θ; S) = Σ_i Σ_{j: (i,j)∈S} nHSIC(r_i, x_j).

Gate = current support S (piecewise constant, no extra constant τ). Gradient flows into θ through r_i only, not into the gate. Edges removed by the prox leave the sum, but dropping an edge is not rewarded by the term itself because the gate is not differentiated.

Full objective:

min_{θ,b} (1/2n) Σ_i ‖r_i‖² + Σ_{i≠j} p_{λ,γ}(W_ij) + μ · R_ind(θ; S)  s.t. h(W) = 0.

For a fixed S this is a single, well-defined objective; S is updated by the prox (outer fixed-point, MM/LLA style). It is therefore *not* globally one function of θ alone; this must be said in the paper.

### 4.3 Cost
Minibatch n_b = 256 per step: O(n_b²) per (i,j) in S. Sparse S → ≈ d·(few parents)·n_b². For d ≥ 100 use random Fourier features, O(n_b·D). Open: whether this is affordable at d = 200.

### 4.4 Schedule
Phase A (epochs 1–T_A): μ = 0, find support. Phase B: μ linearly ramped to its target. Reason: at the start S is dense and includes children, which would penalize the correct model (§4.1). T_A is a required knob; reported with an ablation (T_A ∈ {0, 100, 200}).
μ ∈ {0, 0.1, 0.3, 1, 3} reported in full (μ=0 is the part-1-only ablation).

### 4.5 What can and cannot be claimed
- Motivation: under ANM identifiability (Hoyer 2008; Peters 2014), the true graph with true f makes every r_i ⟂ x_{pa(i)}, so R_ind = 0 at truth, whereas generically no backward model has this property.
- NOT claimed: consistency of the penalized estimator, or that the minimizer is the true graph. That needs a separate proof (open). If the paper has no such proof it must say "motivated by", not "guaranteed by".
- Violations expected: heteroscedastic noise and hidden confounders break the premise; included as stress tests, not as wins.

## 5. Algorithm

```
Input: X (n×d, standardized), c_λ, γ, μ, T_A, T, ρ0
1  Pilot ridge fit → σ̂; λ = c_λ σ̂ sqrt((K+log d)/n)
2  Init u = small random, b = 0, α = 0, ρ = ρ0
3  for epoch t = 1..T:
4     r = residuals(u, b)
5     g = ∇ [ (1/2n)Σ‖r_i‖² + α·h(W) + (ρ/2)h(W)² + μ_t·R_ind(·; S) ]    # S frozen within step
6     u ← prox_{ηλ,γ}( u − η·g )                                           # firm thresholding, per group
7     S ← {(i,j): u_ij ≠ 0};  μ_t ← ramp(t; T_A)
8     every 50 epochs: if h(W) > tol: ρ ← 2ρ;  α ← α + ρ h(W)
9  Return W = ‖u‖, edge functions f_ij, support S
```
Edge score for ranking (AUROC/AUPRC): W_ij (no threshold needed; thresholded graph = S).

## 6. Pre-registered hypotheses and kill criteria

Evaluation protocol is G4's, fixed **before** running. Reporting any hypothesis as failed is part of the plan.

- **H1 (bias).** SPIRE with μ=0 reaches calibration slope ≥ 0.9 on strong-edge curves at d=10,20 (SPADE: 0.67, 0.48), with directed AUROC non-inferior to SPADE (paired difference 95% CI above −0.02).
  Kill: slope < 0.8, or AUROC loss beyond the margin. Secondary comparison: vs SPADE-adaptive. If equal, the contribution is only "principled unification".
- **H2 (orientation).** μ>0 improves **orientation accuracy** (fraction of correctly found skeleton edges oriented correctly) over μ=0.
  Kill: paired Wilcoxon, Holm-corrected, shows no gain over μ=0 beyond the implementation noise floor (0.04–0.06 AUROC) on ≥30 datasets per cell → drop part 2.
- **H3 (separation from regression).** SPIRE beats GAM-ridge (no DAG) on orientation accuracy; skeleton AUROC reported separately.
  Kill: GAM-ridge-like skeleton + any standard orientation heuristic matches SPIRE.
  Note: G2 may have compared directed AUROC for a method with no direction. Re-check how GAM-ridge was scored before using the G2 conclusion.
- **H4 (not a generator artifact).** Gains hold on an iSCM generator (varsortability-controlled) and with var-sortnregress / R²-sortnregress as controls.
  Kill: control ≥ SPIRE → the benchmark, not the method, is the story.

## 7. Experiment design (step 3, after review of this document)

- Generators: iSCM-standardized; mechanisms {GP-sampled, sigmoid, polynomial, spline}; noise {Gaussian, Laplace, uniform}; stress: heteroscedastic. Graphs ER/SF.
- d ∈ {10, 20, 50}, n ∈ {500, 2000}. ≥30 datasets per cell. Validation seeds 0–9 (all tuning), test seeds 1000–1029, disjoint.
- Baselines: SPADE, SPADE-adaptive, GAM-ridge, CAM, SCORE, NoGAM, DAGMA-nonlinear, NOTEARS-MLP, NAM-style additive net, Dagma-DCE if installable, KAN-edge baseline; sortnregress controls.
- Metrics: directed AUROC/AUPRC, skeleton AUROC, orientation accuracy, SID; curve slope, R², SF-F1; runtime; implementation noise floor (repeat runs, threads, float32/64).
- Stats: paired Wilcoxon, Holm across cells, bootstrap CI on paired differences.
- Real data: Sachs (BK tiers), plus any additional dataset with credible ground truth; negative results reported.
- Commit this file and the hypothesis list before the first run, so the plan is time-stamped in git.

## 8. Open issues (honest list)

1. Consistency proof for the estimator with R_ind: none. Possibly out of scope.
2. HSIC on a minibatch is noisy; bandwidth and estimator (biased vs unbiased) need a small ablation.
3. Gate by support only penalizes dependence on selected parents; spurious edges that never enter are unaffected. Not clear this is a problem.
4. T_A and c_λ are knobs, not eliminated; they are now few, named and reported.
5. Whether R_ind adds anything when n is small (HSIC power) is unknown; H2 may fail there.
6. d ≥ 100 requires RFF/HSIC approximation and the order-estimation problem from G3 is untouched by this formulation.

## References (unverified, from memory)
- Zhang (2010), Nearly unbiased variable selection under minimax concave penalty, Ann. Statist. 38.
- Breheny, Huang (2015), Group descent algorithms for nonconvex penalized linear and logistic regression with grouped predictors, Stat. Comput. 25.
- Fan, Li (2001), Variable selection via nonconcave penalized likelihood, JASA 96.
- Zou, Li (2008), One-step sparse estimates in nonconcave penalized likelihood models, Ann. Statist. 36.
- Lounici, Pontil, van de Geer, Tsybakov (2011), Oracle inequalities and optimal inference under group sparsity, Ann. Statist. 39.
- Attouch, Bolte, Svaiter (2013), Convergence of descent methods for semi-algebraic and tame problems, Math. Program. 137.
- Gretton, Bousquet, Smola, Schölkopf (2005), Measuring statistical dependence with Hilbert–Schmidt norms, ALT.
- Gretton et al. (2008), A kernel statistical test of independence, NeurIPS 20.
- Hoyer et al. (2008); Peters et al. (2014); Mooij et al. (2016) as in `01_literature_review.md`.
