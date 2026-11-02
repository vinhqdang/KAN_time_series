# 04 — Bayesian spline additive DAG learner (working name: BASIL)

Status: DRAFT 1, committed BEFORE any code runs. Replaces the SPIRE line (see spire/README_results.md for why it was dropped).
References are from memory / web snippets, unverified unless marked.

## 0. Prior-art warning (checked by web search, not exhaustive)
Bayesian causal discovery for nonlinear additive models already exists: DiBS (Lorch et al. 2021), BayesDAG (Annadani et al., NeurIPS 2023, arXiv:2307.13917), a GP-based structural-marginalisation method (arXiv:2402.14781), order MCMC (Friedman & Koller 2003), order-MCMC with scalable moves (Viinikka et al. 2020, not re-checked), BiDAG package. A spline-basis + closed-form marginal likelihood + order-MCMC combination was NOT found in two searches, but that is absence of evidence. Novelty is therefore **unproven**; the paper's claim would have to rest on (i) edge-function posteriors for interpretation and (ii) a careful evaluation (orientation, calibration), not on the inference idea.

## 1. Model
Standardized x. For node i with parent set pa: x_i = b + Σ_{j∈pa} B_c(x_j) θ_ij + ε, ε~N(0,σ²). B_c = centered B-spline basis (G=8, K≈10 after whitening, as in SPIRE, so ‖θ_ij‖ is the std of the edge contribution).
Prior (Zellner g-prior on the stacked whitened features Z_pa, p=|pa|K): θ | σ² ~ N(0, g σ² (Z_paᵀZ_pa)^-1), p(σ²) ∝ 1/σ². Then the marginal likelihood is closed form (n observations, centered y):

log m(y | pa) = ((n−1−p)/2) log(1+g) − ((n−1)/2) log(1 + g(1−R²_pa)) + const(y),

with R²_pa the R² of OLS of y on Z_pa. No sampler needed for θ, σ².
Prior on structure: modular, P(pa_i) ∝ ρ^{|pa_i|}, ρ = π/(1−π), |pa_i| ≤ k_max (3). Order prior: uniform over permutations.
Posterior over DAGs by **order MCMC** (Friedman & Koller 2003): state = node order; the order score is Π_i Σ_{pa ⊆ pred_i, |pa|≤k} P(pa) m(x_i|pa); moves = swap of two nodes; Metropolis-Hastings. Edge posterior P(j→i | data) = mean over sampled orders of Σ_{pa∋j, pa⊆pred_i(order)} w(pa)/Σ w.

Note (known): order MCMC with uniform order prior over-weights DAGs compatible with many orders; we do NOT correct for it in v1 and report it as a limitation.

## 2. What the Bayesian formulation is supposed to deliver (and why that differs from SPIRE)
- No penalty and no thresholds: sparsity comes from the marginal likelihood (Occam factor) and the structure prior. With g = n the posterior-mean amplitude of an included edge is shrunk by only g/(1+g) ≈ 1 → **no systematic amplitude bias for included edges** (G1), and the model-averaged curve is P(edge)·curve, a calibrated shrinkage rather than a heuristic.
- Soft scores: edge probabilities are continuous and dense (no exact zeros as in SPIRE), so ranking keeps information.
- Uncertainty: posterior edge probabilities (calibration is testable) and credible bands for edge functions (coverage is testable).
- Orientation: the Gaussian-additive score is not score-equivalent once features are nonlinear, so direction information exists; how much is an empirical question (H2).

## 3. Hypotheses and kill criteria (fixed now)
Protocol as in 03 §7: iSCM generator (`research/spire/bench.py::gen`), validation seeds 0–9 for hyperparameters only, test seeds 1000–1029, paired tests, 95% bootstrap CI.
- **H1 (accuracy, non-inferiority).** BASIL edge-probability AUROC ≥ SPADE-adaptive − 0.02 (paired CI lower bound above −0.02) at d=10, n=1000, uniform noise. Kill: CI lower bound below −0.02.
- **H2 (orientation).** Orientation accuracy ≥ SPADE-adaptive − 0.03. Kill: below −0.05. (SPADE-adaptive currently 0.948.)
- **H3 (calibrated curves).** Model-averaged posterior-mean curve slope on true edges in [0.85, 1.10] with no debiasing step; 90% pointwise credible band covers the true curve ≥ 80% on average over true edges. Kill: slope < 0.8 or coverage < 0.6.
- **H4 (calibrated edge probabilities).** Expected calibration error of edge probabilities ≤ 0.10 on all off-diagonal pairs (10 bins), reported next to SPADE's scores min-max scaled (which carry no probability meaning, so no comparison claim).
- **H5 (controls).** var-sortnregress and R²-sortnregress stay ≥ 0.15 AUROC below BASIL (generator not trivial); and a stress cell with heteroscedastic noise is reported, not hidden.
- Secondary (reported, no kill): d=20, n ∈ {500, 2000}, Gaussian/Laplace noise, runtime, order-MCMC convergence (R̂ over 4 chains, effective sample size).
Hyperparameters fixed a priori: g = n, π = 2/(d−1), k_max = 3, 2000 order-MCMC iterations after 500 burn-in, 4 chains. Sensitivity grid on validation only: g ∈ {n, n/4... } {n, 5n}, π × {0.5, 1, 2}.

## 4. Known limitations to state in any write-up
1. k_max=3 restricts in-degree; ER graphs here have mean degree 2, so a few nodes may exceed 3.
2. Gaussian likelihood is misspecified for uniform/Laplace noise (tested).
3. g-prior couples all edges of a node through (ZᵀZ)^-1; concurvity enters the posterior (this is a feature to test, not hide).
4. Order MCMC: prior bias on orders, mixing at d ≥ 50 untested.
5. Sparse prior prior-art overlap (Section 0).
