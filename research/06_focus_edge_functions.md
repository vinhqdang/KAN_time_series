# 06 — New focus: accurate, calibrated edge functions at scale

Status: DRAFT 1, committed BEFORE the new code runs. Supersedes the "beat SPADE on structure" framing of 03/04/05.

## 1. Why the focus changed (evidence from our own runs, all in research/spire/)
- Structure: SPADE-scaled (lambda=0.2/d + tree screening) already reaches AUROC 0.997-0.999 at d=100/200/500 on iSCM data in 1-9 min. Structure at scale is not an open problem for this family; BASIL / BASIL-S do not beat it.
- Function estimates: the same SPADE-scaled run has edge-curve slope 0.44 and curve R² 0.57 at d=100 (4 datasets): edges are found but amplitudes are shrunk by more than half. At d=10-20 the unscaled SPADE had slope 0.65-0.68 (G1, and test_d10 results), at d=20 n=500 0.44. BASIL(-S) has slope 0.99 with ECE <= 0.02, but needs a working sampler (mixing fails at d>=200 from random starts).
- No competitor is evaluated on this: a web search found no benchmark that scores recovered edge functions for amplitude/shape (only edge presence); closest methodological relatives are adaptive group-lasso for DAGs (Xu et al. 2022; and a 2026 ICML paper on independence + adaptive group sparsity for MLP DAG learners, not re-checked), and confidence sets for orderings (Wang, Kolar, Drton), which are linear/additive-noise only.
Unverified: prior art may exist that the searches missed. The paper's claim has to be an empirical one about measurement and repair of edge-function bias, not a new inference idea.

## 2. Problem
Given n i.i.d. samples of d standardized variables from an additive-noise SEM x_i = Σ_{j∈pa(i)} f_ij(x_j) + ε_i, estimate (a) the DAG and (b) every edge function f_ij, such that the returned functions are **accurate in amplitude and shape** and carry **valid uncertainty**, for d up to a few hundred. Structure-only metrics (AUROC/SHD) are insufficient: SPADE-scaled scores 0.998 and still returns curves with less than half the true amplitude.

Primary metric (new, defined here): **edge-function NRMSE** over TRUE edges, NRMSE_ij = ||φ̂_ij − f_ij|| / ||f_ij|| on the data points of x_j (both centered, truth rescaled by the effect's sd, as in G1). A missed edge has φ̂=0 and NRMSE=1. Secondary: calibration slope (median), curve R² (mean), 90% band coverage (where bands exist), edge-probability ECE.

## 3. Methods under comparison
1. **SPADE-scaled** (frozen, lambda=0.2/d, K=15). Structure + shrunk curves.
2. **SPADE-scaled + refit.** Take SPADE-scaled's edge ranking, keep its top k·d edges (k ∈ {1, 1.5, 2}, chosen on validation seeds), refit the same spline model with lambda=0 under the acyclicity constraint on that support. This is the cheap, rule-based baseline; if it matches the Bayesian method on NRMSE, the Bayesian machinery is only worth its uncertainty.
3. **BASIL-W** (the proposal): BASIL-S with (i) candidate parents ranked by SPADE-scaled's edge importance (top 20, ties by marginal R²) instead of marginal R² only, and (ii) the order MCMC **warm-started at the topological order implied by SPADE-scaled** (source-peeling on its weights), 2 chains, 200·d iterations, chain selection by end score. Outputs edge probabilities, model-averaged curves and 90% bands.
4. BASIL-S (cold start, amended config) as a reference for what warm-starting buys.

## 4. Hypotheses and kill criteria (fixed now)
Data: iSCM, ER mean degree 2, uniform noise, n=1000, d ∈ {100, 200}, 5 test datasets each (seeds 1000-1004); d=500 attempted on 1-2 datasets. Validation seeds 0-2 only for refit k and warm-start chain length.
- **F1 (accuracy).** Mean NRMSE of BASIL-W ≤ 0.70 × SPADE-scaled's. Kill: ratio ≥ 0.90.
- **F2 (vs the simple refit).** BASIL-W NRMSE ≤ refit NRMSE + 0.03. Kill: BASIL-W worse than refit by > 0.06. If refit is within ±0.03 of BASIL-W, the conclusion is that the Bayesian part buys uncertainty only; this is reported as such.
- **F3 (structure not sacrificed).** BASIL-W AUROC ≥ SPADE-scaled − 0.01 and orientation ≥ SPADE-scaled − 0.03. Kill: AUROC gap > 0.03 or orientation gap > 0.08.
- **F4 (cost).** Total wall-clock ≤ 2 × SPADE-scaled at d=200; d=500 completes in ≤ 25 min. Kill: > 3×.
- **F5 (uncertainty).** 90% band coverage ≥ 0.80 (kill < 0.60), ECE ≤ 0.10.
- **F6 (control).** var-sortnregress ≥ 0.15 AUROC below BASIL-W (generator not trivial).

## 5. Limitations stated in advance
5 datasets per cell and one generator family (iSCM, 4 mechanism types, uniform noise): no significance tests, wide uncertainty; refit baseline k chosen on only 3 validation datasets; BASIL-W depends on SPADE-scaled for both candidates and starting order, so its structure errors can be inherited; truth for NRMSE is available only in simulation; real-data validation (Sachs etc.) would test structure only; novelty is unproven.
