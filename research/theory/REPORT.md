# Theory candidates A, B, C: novelty, provability, verdicts

Scope: new folder `research/theory/` only (A/, B/, C/ hold code and result files; nothing else in the repo was touched). Python 3.13-style numpy 1.26 / scipy 1.16, `OMP_NUM_THREADS=1`, fixed seeds (A: 0, B: 12345 and 7, C: 2024). All numbers below are copied from `A/results_A.txt`, `A/results_A_repo_validation.txt`, `B/results_B.txt`, `B/results_B_star.txt`, `C/results_C.txt`. Reproduce with `OMP_NUM_THREADS=1 /usr/bin/python3 sim_A.py | sim_B.py | star_check.py | sim_C.py`.

## 0. Verdict table

| | Claim | Literature status | Provable? | Verdict |
|---|---|---|---|---|
| A | Do-effect error <= sum over learned-DAG paths of (path Lipschitz product) x (per-edge error); amplitude shrinkage gives relative error 1 - alpha^m on a length-m path; correct structure with small observational error can still give arbitrary do-error | The path-Lipschitz propagation bound is already stated and proved (Prop. 1) in a 2026 arXiv paper that I opened; the linear case is the resolvent identity | Yes, all of it, elementary | **KNOWN / routine corollary.** Too weak to carry a paper. The 1 - alpha^m law is exact but trivial algebra; it fits the repo's SPADE-scaled numbers well. |
| B | Aggregation turns lagged edges into contemporaneous ones; R2-sortnregress orientation depends on window and graph shape | Aggregation -> contemporaneous edges is known (Gong et al.; Fan et al.). I found no formal result on R2-sortability in the one paper I read in full text (Reisach et al. 2023 is empirical/heuristic) | Yes for the pieces below | **NEW (as far as my search shows) and provable, but elementary and narrow.** Best of the three; a modest-size contribution. |
| C | One-step forecast risk minimisation cannot see contemporaneous edges; hidden shared forcing makes the risk-optimal sparse predictor non-causal; sufficiency when no contemporaneous edges and no hidden variables | This is the classical Granger-causality limitation (instantaneous causality, latent confounding, Eichler; Peters et al. TiMINo) | Yes, elementary | **KNOWN.** The counterexample is a clean illustration, not a theorem worth a paper. |

Honest ranking: B > A > C. None can carry a theory-first paper at a main-track venue (NeurIPS/ICML/ICLR/UAI main). B could support a short theory section or appendix in an empirical benchmark paper, or a workshop/TMLR-type note.

## 1. Literature check (what I opened, what is snippet-only)

Opened and read (text extracted):
- Reisach, Tami, Seiler, Chambaz, Weichwald, "A Scale-Invariant Sorting Criterion to Find a Causal Order in Additive Noise Models", NeurIPS 2023, arXiv 2303.18211 (full PDF, via pdftotext). Definition: R2 is the fraction of a variable's variance explained by all other variables; R2-sortability is the fraction of correctly ordered cause-effect pairs (ties 1/2). The paper has no Theorem/Proposition/Lemma environments (grep over the extracted text); the argument is an informal "high weights make relations more deterministic along chains" plus simulations. The words fork, collider, v-structure, hub, star, in-tree and out-tree do not occur in the text. So the star results in B are not in that paper. (pdftotext extraction may be imperfect.)
- Javidian, "Transferring Information Across Interventions in Causal Bayesian Optimization", arXiv 2606.01457 (PDF text, Appendix N). Proposition 1 (path-Lipschitz nonlinear error propagation): with coordinate-wise Lipschitz mechanisms (sum_i L_ij |u_i - u_i'|) and sup-norm mechanism errors eps_i, the intervention-response error is <= sum_i L^(s)_{i->y} eps_i, with L^(s) the sum of path Lipschitz products in the mutilated graph, proved by the same telescoping recursion I use. This is exactly candidate A's bound. Date 2026, preprint.
- SphUnc, arXiv 2603.01168, Theorem 2 (partially read): finite-sample interventional consistency obtained by telescoping error through forward simulation, constants left implicit. Confirms the propagation argument is routine in the recent literature.
- arXiv abstract page of Fan, Gong, Zhang, "On the Recoverability of Causal Relations from Temporally Aggregated I.I.D. Data" (ICML 2024), abstract only.

Seen only in search snippets (not opened; claims unverified):
- Gong et al. UAI 2017 "Causal Discovery from Temporally Aggregated Time Series" (aggregated VAR is SVAR with instantaneous effects; identifiable under non-Gaussian noise); Gong et al. 2015 subsampled data.
- Eichler, "Causal inference with multiple time series: principles and problems" (Granger limitations, instantaneous causality, latent confounders); Peters, Janzing, Schoelkopf NeurIPS 2013 TiMINo.
- Pellet and Elisseeff JMLR 2008 (Markov blanket = parents, children, spouses); Ernest and Buehlmann (S-mint, interventional effects with known structure).

Not searched, from memory only (unverified): Danks and Plis 2013 undersampled time series; Hyvarinen/Shimizu SVAR-LiNGAM identifiability; Granger 1969; the Maathuis et al. IDA literature. A real novelty claim for B needs these checked, plus a search of 2024-2026 follow-ups of Reisach et al. My searches returned nothing on R2-sortability theory or on R2 with aggregation, but search coverage was limited (a handful of queries).

## 2. Candidate A: do-effect error propagation

### Statements
Setting: ANM X_j = sum_{i in pa(j)} f_ji(X_i) + eps_j on a DAG G; learned model g_ji on a (possibly different) DAG Ĝ, both evaluated under a hard do(X_s = x) with the same noise draw (coupling).

- **A1 (linear, exact; proved).** With B, B̂ the weighted adjacency matrices, T = (I-B)^-1: T̂ - T = T̂ (B̂ - B) T. Entrywise, |T̂ - T|_ki <= [ (I-|B̂|)^-1 |B̂-B| (I-|B|)^-1 ]_ki, i.e. a sum over (learned path to the changed edge) x (true path from the source). Also covers wrong structure (B̂-B has entries for added/missing edges). Textbook resolvent identity.
- **A2 (nonlinear, pathwise; proved).** Suppose each learned edge function ĝ_ji is L_ji-Lipschitz-sum (|ĝ_j(u)-ĝ_j(u')| <= sum_i L_ji|u_i-u_i'|) and sup_{u in U_j}|ĝ_j(u) - f_j(u)| <= eps_j on a set U_j containing the interventional range of the parents. Then for every noise realisation |X̂_k^{do(x)} - X_k^{do(x)}| <= sum_j eps_j sum_{paths j~>k in Ĝ_{-s}} prod L. Identical to Javidian Prop. 1 (known).
- **A3 (shrinkage law; proved, trivial).** If every learned edge is alpha_e times the true one (linear case, or ĝ = alpha f), the effect of a path e_1...e_m is multiplied by prod alpha_e, so its relative error is exactly 1 - prod alpha_e (= 1 - alpha^m for uniform alpha), growing with path length; e.g. alpha = 0.44 gives 0.56, 0.81, 0.91, 0.96 for m = 1..4. Source of alpha for a group-lasso estimator: with orthonormal blocks the estimator is the group soft-threshold (1 - lambda/||b_OLS||)_+ b_OLS (proved, verified numerically), so alpha_e = 1 - lambda/||b_e|| depends on the edge's signal strength: weak edges shrink most.
- **A4 (correct structure, large do-error; proved, trivial).** Equal structure and observational L2 error delta_obs do not bound do-error: take ĝ = f + h with h a bump of height H located outside the observational support. The L2(p_obs) error is O(H * P(bump)^(1/2)) (here 8e-5 H) while the do-error at the bump location is H. A bound requires control under the interventional marginal q, e.g. ||h||_{L2(q)} <= ||dq/dp||_inf^(1/2) ||h||_{L2(p)} (needs bounded density ratio, which a hard do violates when x is outside the support), or sup-norm/Lipschitz control on the intervened range. Also: when direct and indirect paths cancel (true total effect near 0), relative errors are unbounded and signs can flip even with 10% per-edge error.

### Numerical checks (A/sim_A.py)
- A1: max abs error of the identity 3.8e-14 over 2000 random DAG pairs; absolute-path bound violated 0 times (max violation 1e-14); bound/actual median 1.00, 90th percentile 4.6, max 13203 (cancellation).
- A2: 300 random tanh-edge SEMs with sin perturbations, 20 noise draws each: 0 violations; looseness median 3.5, 90th pct 23: valid but loose.
- A3 chain check exact: relative errors equal 1 - alpha^m to 4 decimals. Group soft-threshold matches the numerical minimiser to 5 decimals.
- A4: observational L2 error 8.3e-5 / 8.3e-4 / 8.3e-3 for H = 1, 10, 100 while do-error at x = 6 equals H.
- Attempt to break A3 (T6): with a direct path and a cancelling indirect path and unequal shrinkage, the learned total effect is +0.25 / -0.31 / +0.65 against a true 0: the 1 - alpha^m law holds per path, not for sums with cancellation (not a counterexample to the statement, a limit of its use).
- **Repo validation** (spire/results/do_all.csv, SSC = SPADE-scaled, means per cell; table in `A/results_A_repo_validation.txt`): measured amplitude slope alpha = 0.17-0.45; direct-pair do-NRMSE 0.57-0.75 vs 1-alpha = 0.56-0.83 (uniform noise cells: 0.579 vs 0.562, 0.569 vs 0.559, 0.568 vs 0.558, 0.615 vs 0.555); indirect-pair do-NRMSE 0.82-0.93 lies between 1-alpha^2 (0.80-0.97) and 1-alpha^3 (0.91-0.995), as expected for mean path length 2-3 (path length per pair is not in the CSV, so this is consistent, not a precise test). Limit: for the refit (SSR, slope 0.92-0.97) the law predicts ~0.05 but observed direct do-NRMSE is 0.14-0.39, so shape error, not shrinkage, dominates there; the law explains the SPADE-scaled gap only.

### Verdict A: KNOWN (A2) / trivial (A1, A3, A4). Not a paper.
What would remain is a "diagnostic" use: predict indirect-effect error from measured direct slope. That is an empirical remark. Needed from the repo to make it a validated proposition: per-pair path length in `diag_do.py`, and a test on the stabilised BASIL-W/refit outputs.

## 3. Candidate B: aggregation and R2-sortability

### Setting
Lagged linear SEM with delays, white-noise innovations: X_j(t) = sum_i b_ji X_i(t - tau_ji) + eps_j(t) on a DAG, integer delays tau >= 1 (native step). Block-sum aggregation with window D: Y_s = sum_{u<D} X(sD+u). R2_i = 1 - 1/(Sigma_ii (Sigma^-1)_ii) (R2 of regressing node i on all others; scale-invariant; this is the R2-SortnRegress criterion). Edge i->k is "correctly ordered" if R2_i < R2_k (tie = 1/2, as in Reisach et al.).

### Statements and proof status
- **B0 (R2 closed form; proved).** For a contemporaneous linear SEM with noise variances omega, variances v, children set ch(i): R2_i = 1 - u_i / (1 + omega_i c_i), with u_i = omega_i/v_i and c_i = sum_{k in ch(i)} b_ki^2/omega_k. Proof: Sigma^-1 = (I-B)^T Omega^-1 (I-B), so (Sigma^-1)_ii = 1/omega_i + c_i. Verified to 9e-16 on 500 random SEMs. Consequence: R2_i >= parent-explained fraction, with a child-driven bonus: R2 measures Markov-blanket strength, not direction.
- **B1 (aggregation closed form; proved).** For a single edge X->Z with delay tau < D, white-noise source, b: Cov(Y_X, Y_Z) = b(D - tau)_+ . So the contemporaneous share of the edge is (1 - tau/D) and the lagged share tau/D; contemporaneous correlation is exactly 0 for D <= tau. General: Sigma_Y(D) = sum_{|h|<D} (D-|h|) Gamma(h) (Fejer weights). Verified exactly (7.2000 = 7.2000; 4.8000 = 4.8000).
- **B2 (triviality for small windows; proved).** If D <= min{|h| : Gamma_ij(h) != 0 for some i != j}, the aggregated covariance is diagonal, all R2 = 0, and R2-sortnregress has expected accuracy exactly 1/2 (all ties; noise in finite samples gives random order). For a white-noise delay-SEM the support of Gamma_ij is the set of differences of path-delay sums from common ancestors, so the threshold is the smallest such difference (in particular D <= min delay for pairs that are ancestor-descendant). Simulation at D = 1 on trees: tie fraction 1.0 for in-trees and chains.
- **B3 (limit, rate; proved modulo routine details).** If sum_h |h| |Gamma(h)| < infinity (true for a stable delay-SEM), ||Sigma_Y(D)/D - Sigma_infinity|| <= (1/D) sum_h |h| ||Gamma(h)||, where Sigma_infinity = (I - sum_edges B_tau)^-1 Omega (...)^-T is the covariance of the static "summary SEM" (each edge with its total weight regardless of delay). So by continuity R2-sortability(D) -> R2-sortability of the summary SEM at rate O(1/D) except at ties. This is a statement about the long-run covariance and is routine; it is the reason why a plateau is expected.
- **B4 (fork/collider theorem; proved; the main new-looking statement).** Contemporaneous (or summary) SEM, all b != 0, all omega > 0.
  - (i) Out-star: root 0 with m >= 2 leaf children (leaves have no other parents). Then R2_0 > R2_k for every leaf k, strictly. Every edge is misordered: R2-sortability is exactly 0.
  - (ii) In-star: sink 0 with m >= 2 parent sources. Then R2_k < R2_0 for every parent: R2-sortability is exactly 1.
  - (iii) m = 1: R2_0 = R2_1 exactly (tie): R2-sortability 1/2 on an isolated pair, for every D.
  Proof: write r_k = omega_root b_k^2/omega_k (out-star) or b_k^2 omega_k/omega_0 (in-star). By B0 the hub has 1 - R2 = 1/(1 + sum_k r_k) and every leaf/parent has 1 - R2 = 1/(1 + r_k) (leaf: v_k = omega_k + b_k^2 omega_0, no children; source parent with one child: u = 1, c = b_k^2/omega_0). Since sum_k r_k > r_k for m >= 2, the hub has strictly larger R2 in both orientations. Verified on 20000 random stars each (0 exceptions). Interpretation: R2 ranks nodes by Markov-blanket strength (hub last); direction enters only through whether the hub is a collider.
- **B5 (graph class and window; numerical, partly conjectural).** Exact population computation (B/sim_B.py, d = 10, delays U{1..8}, |b| in 0.3-0.9, 150 graphs per class, SE <= 0.016):

| edge-sortability by window D | 1 | 2 | 4 | 8 | 16 | 32 | 64 | inf (summary SEM) |
|---|---|---|---|---|---|---|---|---|
| in-tree (confluences, river-like) | 0.500 | 0.567 | 0.675 | 0.795 | 0.828 | 0.837 | 0.855 | 0.867 |
| out-tree (divergences) | 0.567 | 0.477 | 0.386 | 0.291 | 0.231 | 0.230 | 0.216 | 0.215 |
| chain | 0.500 | 0.500 | 0.500 | 0.509 | 0.528 | 0.533 | 0.536 | 0.534 |
| ER, mean degree 2 | 0.627 | 0.673 | 0.647 | 0.581 | 0.535 | 0.534 | 0.530 | 0.526 |

Findings: in river-like in-trees sortability rises monotonically from the tie value 1/2 to a plateau near 0.87; in out-trees it falls below 1/2 (anti-sortable, as B4(i) predicts); on chains it stays near 1/2; on ER graphs it is non-monotone (peak at D = 2, +2.9 SE over D = 1, then decays to 0.53). So "aggregation helps R2 orientation" is false in general: it holds for collider-dominated graphs and reverses for fork-dominated ones. Monotonicity in D for in-trees is empirical only (conjecture, no proof). Pair (ancestor-descendant) sortability follows the same pattern.

### Adversarial notes
- B4 is elementary algebra; a referee will call it a lemma. B0 is essentially the Sigma^-1/partial-variance identity and may be folklore (unverified).
- Delay-SEM assumptions (white noise, integer delays, block sums, no self-autocorrelation, DAG, d = 10) are restrictive; river data have autocorrelation, hidden drivers and continuous-time delays. Block averaging vs sampling at stride D gives a different Sigma_Y (not tested).
- ER non-monotonicity may depend on my weight/delay ranges.

### Verdict B: NEW (to my search) and provable, but small. 
What the proofs still need: a mathematician's check of B2 and B3 constants for general delay structures, a literature search for prior statements of B0/B4 (R2 as Markov-blanket strength, "hub last"), and a proof or counterexample for monotonicity of R2-sortability in D on in-trees.

### Repo evidence for validation
stage0 results (research/ts/stage0/results.md): R2-sortnregress orientation accuracy on true contemporaneous edges 0.739 / 0.738 / 0.720 at 6 / 12 / 24 h (flat), AUROC 0.649 / 0.655 / 0.654, versus lag-only VAR at 0.5 by definition. A flat profile over 6-24 h is consistent with a plateau at D large (B3, B5 in-tree row from D = 16), if the native step is 15 min (assumed, unverified). Level 0.72-0.74 vs 0.87 in my simulation is expected to differ (hidden drivers, autocorrelation, mixed forks). Stage0 raw-scale results (var_sortnregress orientation 0.92-0.94) show scale leakage, not related to B. Experiments that would test B: (1) stratify orientation accuracy of R2-sortnregress by whether the child node has >= 2 parents (confluence) vs the parent node has >= 2 children (fork): B4 predicts accuracy near 1 vs near 0; (2) sweep the window finely from native step to 48 h and check the rise from 1/2 (B2) to the plateau; (3) compare block sums against stride sampling.

## 4. Candidate C: forecast loss and causal graph selection

### Statements
Setting: stationary linear SVAR X_t = A0 X_t + A1 X_{t-1} + eps_t (A0 strictly acyclic), optionally with hidden H_t. One-step forecast risk of a lag-1 predictor with support S for target j: R_j(S) = min over coefficients supported on S of E (X_j(t) - c^T X_S(t-1))^2.
- **C1 (invisibility; proved).** R_j(S) depends on the model only through the second moments (Sigma_0, Sigma_1 = Cov(X_t, X_{t-1})), hence only through the reduced-form matrix Phi = (I-A0)^-1 A1 and the innovation covariance. Two different causal structures (e.g. contemporaneous edge i->j versus correlated innovations) with the same reduced form have identical risk for all supports. The Bayes lag-1 support for target j is supp(Phi_j) = {i: exists k with a contemporaneous path k ~> j (or k = j) and A1_ki != 0}, generically (excluding measure-zero cancellation): it contains non-parents. Verified exactly: risk differences 0 for every support in the C1b example; on random SVARs (d = 6) 40% of targets have at least one spurious predictor in supp(Phi), mean 0.79 spurious per target; the true-parents-only model loses 0.052 MSE on average (39% of targets have a strictly positive gap).
- **C2 (hidden forcing; proved by explicit example, verified).** X_0 -> X_2 (lag 1, a = 0.3); H AR(1) rho = 0.9 loads on X_1 and X_2 (loading 1, small observation noise 0.1). Population MSE of X_2 by support from X(t-1): {X_1}: 1.2695; {X_0}: 5.3632; {X_0,X_1}: 1.1795; empty: 5.4532. So under a one-parent budget the risk-optimal parent is the non-causal X_1, and with any per-edge penalty below ~4.18 (MSE reduction from adding X_1 to {X_0}) the risk-penalised optimum contains X_1. A parameter scan (5000 draws, rho in .3-.98) finds the proxy beating the true parent in 60% of draws (wide ranges, not a calibrated frequency). Without hidden forcing (control) the true parent wins (0.100 vs 0.190).
- **C3 (sufficiency; proved).** If A0 = 0 (no instantaneous effects), no hidden confounding, innovations independent of the past: Phi = A1, the unique smallest Bayes-risk support for node j is exactly its lag-1 parent set (Sigma_0 nonsingular, so the Bayes linear predictor is unique; for nonlinear ANM time series E[X_t | X_{t-1}] = f(X_{t-1}) depends on exactly the parents). Penalised risk (BIC/AIC-like) is then consistent. Verified: 0 failures of 3000 targets (exhaustive supports). This is Granger causality equivalence.
- **C4 (necessity; proved up to genericity).** Generic failure iff either some contemporaneous ancestor k of j has a lag parent not already a lag parent of j (nested-parent condition violated), or a hidden variable that is autocorrelated and loads on both j and a proxy node. Near-ties: the risk gap between true and non-causal supports can be arbitrarily small (C4 rows) or zero (C1b), so finite samples cannot separate them.

### Verdict C: KNOWN. 
Cite Granger; Eichler on instantaneous/latent problems; Peters et al. 2013. The statements are elementary; the only extra is the exact characterisation of the Bayes support as supp(Phi) and the explicit numbers. Contribution beyond the literature: none that I could defend.

### Repo evidence for validation
gapB README: on CausalRivers median Spearman(forecast loss, AUROC) 0.0 (lasso 0.00, thresholded VAR -0.08, DYNOTEARS -0.20 significantly negative, PCMCI+ 0.09, VAR-LiNGAM 0.25); forecast-selected AUROC within 0.01 of a random grid draw for 4/5 families; BIC and sparsity-only do as well; synthetic well-specified SVAR gives rho 0.5-0.76. C1/C2 explain the pattern qualitatively (stage0: most river edges are contemporaneous at 24 h; hidden regional rainfall). The author notes the mechanism was not tested directly. A direct test: synthetic SVAR with a controlled share of contemporaneous edges and a controlled hidden-driver loading; C predicts rho collapses toward 0 or negative as these increase, matching the DYNOTEARS negative rho.

## 5. Which, if any, can carry a paper
- B is the only candidate with a statement I could not find in the literature, and it is checkable against stage0 data. But the proofs are one-page algebra, the model is restrictive, and the novelty is unconfirmed beyond a few searches. Realistic placement: a section or appendix in an empirical paper on causal discovery from aggregated river data (workshop/benchmark-track tier, or a TMLR/CLeaR-level short contribution if the stratified experiments in B support B4).
- A is a correct but known proposition; use it as a lemma motivating "report do-effect error, not only AUROC" and the 1 - alpha^m shrinkage diagnostic (empirically supported for SPADE-scaled).
- C is a known textbook point; use it to explain the negative gapB finding in two sentences with citations.
- A paper "whose strength is new theory" is not realistic from these three: together they are an elementary-theory section that supports an empirical story (structure is easy, effects and orientation are not, forecast loss does not select graphs). Mid-tier at best if the empirical validation experiments listed above are added.

## 6. What proofs still need
- A: nothing beyond citing Javidian 2026 (and checking earlier sources such as IDA-type perturbation results for linear effects).
- B: independent check of B0/B4 (likely folklore), the general-delay version of B2/B3, a proof or counterexample of in-tree monotonicity in D, and extension to block average vs sampling and to autocorrelated nodes (AR self-loops).
- C: only a citation pass; formal "generic" statements need a measure-zero argument written out.
- All: a mathematician's check; all novelty claims rest on limited search (see section 1 for unopened items).
