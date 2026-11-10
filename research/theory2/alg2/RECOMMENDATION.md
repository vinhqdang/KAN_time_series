# Recommendation

Date: 2026-11-10. Candidates and evidence: `CANDIDATES.md`. References: `refs.bib`, checks in `REFS_VERIFY.md`. Pilot: `PILOT_PLAN.md` (DRAFT, not run).

## 1. Verdict

**Nothing passes the bar as it stands.** Of nine candidates, six are done or empty, two are incremental, and one (C5) is a new combination whose first sanity check did not show a gain over a trivial physical baseline.

What I recommend:

1. **Do not add a learning method to the current paper.** The seed idea that fits its theory best (S1, the transient as identifying signal) is closed by Proposition N in `CANDIDATES.md`: with time labels the start carries no information about the mechanism, and without replicate tracking the problem is already published.
2. **If one method project is started, it should be C5, frontier structure learning for nonnegative additive flows, as a separate paper, and only after the pilot in `PILOT_PLAN.md` passes its kill criterion.** The pilot costs under 3 CPU-hours.
3. Proposition N can go into the current paper as one remark (optional). It says that the fixed start is an artefact for label-free methods only.

My estimate that C5 survives: about 0.3 that the pilot passes; given a pass, about 0.4 that ICML reviewers accept the novelty. Overall about 0.1 to 0.15.

The rest of this file specifies C5 so that the decision can be made on a concrete object.

## 2. The problem

A network transports a nonnegative quantity. Each node receives what its parents send, with delays, and adds a local input. The local inputs of all nodes are driven by common latent causes. Examples: discharge at river gauges (rain), vehicle counts (time of day), sewer and water networks, goods in a supply chain.

Standard discovery fails here for a reason documented in this repository on CausalRivers: non-adjacent stations correlate at 0.64 to 0.66, every pair tests dependent, the driver has rank above one and cannot be conditioned away (`CANDIDATES.md`, Section 1). The best of PCMCI+, VAR, VAR-LiNGAM and DYNOTEARS reaches a directed AUROC of 0.69 at 24 h, the same as plain correlation.

## 3. Model

For nodes j = 1, …, p and times t:

  X_j(t) = Σ_{i ≠ j} Σ_{ℓ = 0}^{L} K_ij[ℓ] X_i(t − ℓ) + U_j(t),  K_ij[ℓ] ≥ 0,  U_j(t) ≥ 0.

- K_ij is the routing kernel of the edge i → j (travel-time distribution times the fraction of the flow of i that reaches j). The graph has the edge i → j iff K_ij ≠ 0.
- U = (U_1, …, U_p) is the vector of local inputs. It is a stationary ergodic process with **arbitrary dependence between nodes and over time**. No independence, no distributional family, no causal sufficiency: unobserved upstream nodes are part of U.
- Lag 0 is allowed. Acyclicity is not needed for the theory below. In lossless networks it holds automatically: if Σ_ℓ K_ij[ℓ] ≥ 1 on an edge and E U_j > 0, then E X_i < E X_j, so the order of the means is a causal order. In such networks sortability by size is a law of the system, not a simulation artefact. This is the link to the paper's theme.

Write Z_j(t) for the vector of all regressors (X_i(t − ℓ)), i ≠ j, 0 ≤ ℓ ≤ L, and k*_j for the vector of true kernel values, so X_j(t) = k*_jᵀ Z_j(t) + U_j(t).

## 4. Objective and optimiser

**Frontier LP (primary).** For each target j:

  k̂_j = argmax_k  Z̄_jᵀ k  subject to  k ≥ 0  and  X_j(t) − kᵀ Z_j(t) ≥ 0 for all t,

where Z̄_j is the sample mean of Z_j. Maximising the mean explained flow is the same as minimising the sum of the residuals. It is the maximum-likelihood estimator when U_j is exponential (Aigner and Chu 1968; Feigin and Resnick 1994 for autoregressions). It is a linear programme with (p − 1)(L + 1) variables and one constraint per time step.

**Noisy data.** Replace the hard constraints by the pinball loss at a small fixed level τ, which is the asymmetric-Laplace likelihood (Koenker and Bassett 1978) and still a linear programme. A second option is the stochastic-frontier likelihood, residual = exponential + Gaussian (Aigner, Lovell, Schmidt 1977), fitted by gradient methods. Theorem 1 does not cover either; see Section 6.

**Coupling across targets (optional).** The constraint Σ_j Σ_ℓ K_ij[ℓ] ≤ 1 (a unit of flow leaves a node once) couples the p programmes into one LP.

**Output.** Edge score s_ij = (Σ_ℓ k̂_ij[ℓ]) X̄_i / X̄_j, the share of the mean flow of j attributed to i.

## 5. Theorems

Fix a target j and drop the index. (Z, U) is stationary and ergodic with E‖Z‖ < ∞.

- **A1 (no exact collinearity).** There is no δ ≠ 0 with δᵀZ = 0 almost surely.
- **A2 (directional tightness).** For every open cone C with P(Z ∈ C) > 0 and every ε > 0: P(Z ∈ C and U ≤ ε‖Z‖) > 0.

A2 says: in whatever direction the regressors point, the local input is sometimes negligible against them. It replaces independence. It allows U to depend on Z in any other way.

> **Theorem 1 (identification from the support).** Under A1 and A2, k* is the unique solution of the population programme: maximise E[Z]ᵀk over F = {k ≥ 0 : X − kᵀZ ≥ 0 a.s.}. So the kernels and the graph are functions of the joint support of (X(t − L), …, X(t)), for any dependence among the local inputs.

Proof. Take k ∈ F and δ = k − k*. Feasibility means U ≥ δᵀZ a.s. Suppose P(δᵀZ > 0) > 0. Then for some η > 0 the open cone C = {z : δᵀz > η‖z‖} has P(Z ∈ C) > 0. By A2 with ε = η/2, with positive probability Z ∈ C and U ≤ (η/2)‖Z‖ < δᵀZ, a contradiction. So δᵀZ ≤ 0 a.s., hence E[Z]ᵀδ ≤ 0, with equality only if δᵀZ = 0 a.s., that is δ = 0 by A1. Since k* ∈ F, it is the unique maximiser. ∎

> **Theorem 2 (strong consistency).** Under A1, A2 and P(Z_a > 0) > 0 for every coordinate a, every sequence of solutions k̂_n of the sample programme converges to k* almost surely.

Proof idea. The sample feasible set F_n contains F, so Z̄_nᵀk̂_n ≥ Z̄_nᵀk*. F_n is bounded once every coordinate of Z has been positive at some time (k_a ≤ X(t)/Z_a(t)). Let k̄ be a limit point. If k̄ violated the constraint at a support point of (Z, X), it would violate it on a neighbourhood of positive probability; by ergodicity the process visits that neighbourhood at a finite time t₀, the constraint of t₀ holds for all k̂_n with n ≥ t₀, and constraints are closed, a contradiction. So k̄ ∈ F. By the ergodic theorem E[Z]ᵀk̄ ≥ E[Z]ᵀk*, and Theorem 1 gives k̄ = k*. ∎

No rate is claimed. For i.i.d. inputs with a regularly varying lower tail of index α the classical rate is n^(−1/α), faster than √n for α < 2 (Davis and McCormick 1989; Feigin and Resnick 1994). Under dependence the rate is open.

> **Proposition 3 (what happens without A2).** For every feasible k, (k − k*)ᵀZ(t) ≤ U(t) for all t almost surely. So the mean flow share attributed in excess to node j is at most E U_j / E X_j, and it is zero in every direction in which A2 holds.

Proof. Feasibility. ∎ The errors of the method are confined to nodes whose own local input is a large share of their flow, and to candidate parents that are small against that input.

**What could fail.**

1. **A2 on real rivers.** Base flow never vanishes, so a small unrelated creek can always be subtracted from a large river. Proposition 3 bounds the damage in flow share, not in edge count. The sanity check shows exactly this: 0.94 to 0.96 instead of 1 in a noise-free generator with base flow.
2. **Measurement error.** With two-sided noise the boundary is blurred. The pinball version at level τ estimates a τ-quantile surface, which under confounding is biased; the bias goes to zero with τ only under A2-type conditions and at the price of variance. I have no theorem for the stochastic-frontier version with dependent inputs. This is the main open theoretical point.
3. **Ancestors.** If the local input of a parent is negligible, the parent and the grandparent carry the same flow and A1 nearly fails. The method then cannot say which of the two is adjacent; nor can any other method.
4. **Losses and abstraction.** They are negative inputs. Small ones act like measurement error.

## 6. Closest prior work and the delta

| Work | What it has | Delta |
|---|---|---|
| Tran, Buck, Klüppelberg (JRSS-B 2024), QTree | Directed tree of a river network from extremes; max-linear model, independent noise; pairwise quantile scores plus spanning tree; consistency | Sum-linear model with delays; whole distribution, not extremes; DAG, not tree; no independence; one LP per node |
| Rayas, Anguluri, Dasarathy (NeurIPS 2022); Mada and Anguluri (2025) | Structure learning under a conservation law, ℓ1-penalised Gaussian likelihood | Undirected and static there; injections with known statistics; no latent confounding |
| Feigin and Resnick (1994); Davis and McCormick (1989); Aigner and Chu (1968) | LP estimators for positive-innovation autoregressions and for frontiers | The estimator is theirs. New: use for structure learning, identification under dependent inputs (A2), multivariate lagged graph |
| Donoho and Stodden (2003); Fu et al. (2019) | Identifiability of nonnegative factorisations from the data cone | Same geometry. New: regression form with observed regressors, dynamics, causal reading |
| Wang and Drton (JMLR 2023); Mesters et al. (2026) | Discovery with dependent errors via non-Gaussian moments or mean independence | Different assumption (boundary instead of moments); no restriction of the confounding pattern |

A reviewer can fairly say: "LP frontier estimation is from 1968, the max-linear river papers already use the inequality X_j ≥ cX_i, and the domain is narrow." What is defensible as new: Theorem 1 as an identification principle for causal discovery under unrestricted confounding of the inputs, and a real benchmark on which it can be tested.

## 7. Data and baselines

- **Primary: CausalRivers** (Stein et al. 2025), East Germany and Bavaria, raw discharge, 15-min series aggregated by block means. Ground truth: the river graph. Obtainable here: `../DATA.md`, Section 1; the stage0 caches are at the scratch location named there, or are rebuilt by `../reproduce.sh fetch` and `../empirical/build_cr_cache.py`. Subgraph sampler: `../empirical/sample.py` with new seeds.
- **Secondary (not for the pilot):** the Upper Danube and Lower Colorado data of the QTree paper, to compare with QTree on its own ground. Availability here was not checked.
- **Not suitable:** Causal Chambers (bipartite graphs, signed sensor signals, no conserved quantity).
- **Baselines.** |corr| at the best lag; correlation times mean order (the trivial physical baseline that matched the method in the sanity check); nonnegative least squares VAR; VAR; PCMCI+; VAR-LiNGAM; DYNOTEARS (implementations of `../../ts/stage0/`); the naive physical baseline of the CausalRivers benchmark; QTree if its code runs here.

## 8. Fit

- **Current paper:** no. The paper is about benchmark generators and sortability; this is a method for a physical model class. Two sentences could point to it: on rivers the size order is physical, and saturation by a latent driver (Section on finite samples) is what a frontier method avoids.
- **Own paper:** yes, if the pilot passes. Title direction: causal discovery in conserved flow networks under unrestricted confounding of the inputs. It would need a second data domain (traffic or a water network) to answer the narrow-domain objection.

## 9. Probabilities, stated plainly

| Event | My estimate |
|---|---|
| The pilot passes its kill criterion | 0.3 |
| Given a pass, the novelty survives ICML review | 0.4 |
| Given a pass, a reviewer finds a hydrology paper that does the same (journals not searched) | 0.2 |
| Overall: an ICML-level method paper comes out of C5 | 0.10 to 0.15 |
