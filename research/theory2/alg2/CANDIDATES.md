# Candidate learning-method directions outside the unlabelled-columns setting

Date: 2026-11-10. Scope: one new learning method for the causal-discovery project, outside the setting of `../alg/` (TiDAL, rejected by the author).
Nothing in `../icml_paper/` or in `manuscript/` was changed.
References: every work named here is in `refs.bib`; the check made for each one is in `REFS_VERIFY.md`. Three entries could not be checked against an API record and are marked there.
Computation: three tiny synthetic sanity checks, under 2 CPU-minutes in total (`sanity/`, outputs in `sanity/OUTPUT.txt`). No real data were read.

## 0. Result in short

- **No candidate clears the full bar today** (new method, principled objective, theory, and measured headroom against strong baselines).
- The seed idea S1 ("the transient identifies what stationary data cannot") is closed: for time-labelled trajectories it gives nothing (Proposition N below, a three-line proof), and for snapshots it is already published (Wang et al. 2023; Guan et al. 2026).
- One candidate is worth a cheap, pre-registered pilot: **C5, frontier structure learning for nonnegative additive flows**. Its identification principle (the lower support boundary instead of independence) was not found in the causal-discovery literature, and it targets the one failure that this repository has documented on real data. The sanity check did not show a gain over a trivial physical baseline, so the prior that the pilot passes is low. Details: `RECOMMENDATION.md`, `PILOT_PLAN.md`.

| # | Candidate | Novelty | Main kill risk |
|---|---|---|---|
| C1 | S1: covariance trajectory from a known start as identifying signal | **Done / empty** | Proposition N: the law of a labelled trajectory depends on (B, A, D) only through the reduced form (M, Ω) |
| C2 | S1 variant: heteroskedasticity created by the transient (state-dependent noise) | Done as theory, application only | Lewis (2021) covers arbitrary time-varying volatility; no evidence that the available data have state-dependent noise |
| C3 | S2: start-robust objective (conditional likelihood, free initial law) | **Done** | Classical (Friedman et al. 1998; Binder et al. 2005); no headroom (PCMCI+ at 0.02 with labels) |
| C4 | S3: factor-adjusted sparse SVAR for a pervasive latent driver (rivers) | Done (FNETS), application only | Small subgraphs (d = 5, 10) have no factor/idiosyncratic separation; the repository's own conditioning remedy already failed |
| C5 | S3 / own: frontier structure learning for nonnegative additive flows | **New combination** (between incremental and clearly new) | A trivial baseline (correlation times mean order) matched it in the sanity check; base flow breaks the support condition; domain is narrow |
| C6 | S4: power-aware / decision-theoretic structure learning from the detectability oracle | Incremental | It is a Bayes or FDR post-processing step, not a new learner; a threshold rule in disguise |
| C7 | Own: Gaussian second-moment identifiability of a subsampled SVAR with diagonal innovations | Incremental (small theorem) | Finite multiplicity is known since Palm and Nijman (1984); ill-conditioned; no headroom number |
| C8 | Own: mean transient under step inputs with known targets | **Done** | Local projections, external-instrument SVAR, backShift |
| C9 | Own: amortised discovery trained with randomised start and burn-in | Incremental | Domain randomisation; needs GPU-scale training; Montagna et al. (2025) bound what it can achieve |

## 1. Evidence from the repository used below

Headroom (methods far from the ceiling on real data with ground truth):

- CausalRivers, 24 h block means, 100 test subgraphs, standardised (`../../ts/stage0/README.md`): directed AUROC |corr| 0.684, PCMCI+ 0.689, DYNOTEARS 0.648, VAR-LiNGAM 0.614, lag-only VAR 0.531. Orientation accuracy on contemporaneous edges: PCMCI+ 0.532, VAR-LiNGAM 0.598, DYNOTEARS 0.462.
- Same data, oracle-tuned over each method's grid (`../../ts/gapB/results_test.md`): at most 0.767 (PCMCI+). Forecast loss does not select good graphs (median Spearman 0.0 over families).
- Literature (`../../ts/01_literature.md`): CausalRivers paper, random 5-node subgraphs: VAR 0.80, naive physical baseline 0.75, PCMCI 0.65, DYNOTEARS 0.61. SC3D follow-up at d = 10: best 0.610.
- Magnitude alone orients river edges: R²-sortnregress on the raw scale reaches 0.92 to 0.94 on contemporaneous edges (`../../ts/stage0/README.md`, gate G3). On rivers the size order is physics, not a simulation artefact.

Why standard methods fail there (`../README.md`, `../README_part2.md`):

- Station pairs that are not river relatives have mean correlation 0.64 (12 h) and 0.66 (24 h), against 0.73 and 0.74 for relatives. With 700 to 3,600 steps every pair tests dependent (saturation, F2).
- The driver has rank above one (rain reaches each gauge through its own catchment filter). A scalar proxy cannot remove it (Prop 3.6); the pre-registered proxy conditioning failed; conditioning on 10 principal components raised confluence accuracy only to 0.58 (post hoc).

Where there is no headroom:

- Time-unrolled benchmark with time labels: PCMCI+ at SHD/|E| = 0.02 under both starts (`../alg/NOVELTY.md`, Section 2.9). Any method for labelled short panels starts from a solved problem.
- Causal Chambers: the published graphs are bipartite except one (`../icml_exp/README.md`); inputs are designed and independent. There is little structure left to learn among measured variables.

Negative results not to repeat: forecast-loss model selection (Gap B), instantaneous-effect machinery on aggregated rivers (stage 0, H1 failed), relatives counting on rivers (Phase 2), scalar or PCA proxy conditioning (Phase E).

## 2. Candidates

### C1. S1: the covariance trajectory from a known start as identifying signal

**Problem.** Replicated short series that start from a known or reset state (perturb-and-relax, step inputs, panels). Stationary Gaussian data do not identify the instantaneous matrix B of an SVAR. Hope: the trajectory Σ_t, which follows a Lyapunov recursion with fixed mechanisms, does.

**Method sketch.** Joint Gaussian likelihood of the N trajectories over (B, A, D) with an acyclicity penalty; continuous optimisation.

**Candidate theorem, and why it is false.** Model: X_t = B X_t + A X_{t−1} + e_t, e_t i.i.d. with covariance D (diagonal), I − B invertible, X_0 ~ ν independent of the noise (ν may be a point mass). Put M = (I − B)⁻¹A, η_t = (I − B)⁻¹e_t, Ω = Cov(η_t).

> **Proposition N (no gain from the transient).** The law of (X_0, …, X_T) is a function of (ν, M, law of η) only. So two structures with the same reduced form are observationally equivalent for every start law, every T and every number of replicates. In the Gaussian case the equivalence class of (B, A, D) is {(B′, A′, D′): (I − B′)⁻¹D′(I − B′)⁻ᵀ = Ω, A′ = (I − B′)M}, the same class as for one stationary series. If Σ_t is nonsingular for some t ≤ T − 1, then (M, Ω) is identified, and nothing more is.

Proof. X_t = M X_{t−1} + η_t is a Markov chain; its transition kernel depends on (M, law of η) and its initial law is ν. For the last sentence, M = Cov(X_{t+1}, X_t) Σ_t⁻¹ and Ω = Σ_{t+1} − M Σ_t Mᵀ. ∎

The trajectory Σ_t = Mᵗ Σ_0 (Mᵗ)ᵀ + Σ_{j<t} Mʲ Ω (Mʲ)ᵀ is itself a function of (Σ_0, M, Ω). Identification through heteroskedasticity (Rigobon 2003; Lanne and Lütkepohl 2008) needs the *innovation* covariance to change between regimes. In a transient with fixed mechanisms it does not change; only the state covariance does.

**Snapshot version** (each replicate observed at one time only, so only the marginals are seen). Then the marginals carry the information about M, and identification from them is published: Wang et al. (2023) give necessary and sufficient conditions for identifying the generator of a linear SDE from the law of the solution with a fixed initial state; Guan et al. (2026) identify drift, diffusion and causal graph from temporal marginals and give an algorithm (APPEX). Both identify the reduced form (drift and GGᵀ), in agreement with Proposition N. Stationary-only versions: Varando and Hansen (2020), Dettling et al. (2023), Lorch et al. (2024).

**Closest prior work.** Rigobon (2003); Lanne and Lütkepohl (2008); Wang et al. (2023); Guan et al. (2026); Huang et al. (2020, CD-NOD); Hyvärinen and Morioka (2016); Bellot et al. (2022); Saggioro et al. (2020).

**Novelty.** Done (snapshots) or empty (labelled trajectories).
**Who has the problem.** Perturbation time courses, pharmacology, reset experiments. The problem is real; the transient is not the missing ingredient.
**Headroom.** None measured. With labels, PCMCI+ is at 0.02 in the benchmark.
**Feasibility.** Trivial. **Kill risk.** Realised: Proposition N.
**By-product for the current paper.** Proposition N can be one remark: the fixed start changes what a label-free method can exploit, and it adds no information about the mechanism for a method that has the labels.

### C2. S1 variant: heteroskedasticity created by the transient

**Problem.** If the noise scale depends on the state (shot noise, chemical Langevin noise, multiplicative noise), the moving mean of a relaxation changes D_t, so Ω_t = (I − B)⁻¹D_t(I − B)⁻ᵀ changes and B becomes identifiable in Rigobon's sense, with time points as regimes.

**Method sketch.** Gaussian likelihood with D_t = diag(d_i · v(μ_{i,t})) or a free D_t per time point; joint diagonalisation of the Ω̂_t as initialiser.

**Candidate theorem.** B is identified up to column order and scale if the ratios D_t/D_s separate the components for some pair (t, s).

**Closest prior work.** Rigobon (2003); Lanne and Lütkepohl (2008); Sentana and Fiorentini (2001); Lewis (2021), who proves identification from time-varying volatility of arbitrary form; Immer et al. (2023) for location-scale noise models in causal discovery; Komorowski et al. (2009) for inference under the linear noise approximation.

**Novelty.** Done as theory. It would be an application of Lewis (2021) with replicates.
**Who has the problem.** Systems biology and sensor physics, in principle.
**Headroom.** No data set in `../DATA.md` is known to have state-dependent noise together with instantaneous structure among measured variables. Not checked.
**Feasibility.** High. **Kill risk.** Reviewer: "Rigobon with time as regime". No data.

### C3. S2: start-robust objective

**Problem.** A learner whose output does not depend on the initial law.
**Method sketch.** Conditional likelihood given X_0, or a free initial covariance as nuisance.
**Theorem.** Consistency for (M, Ω) under any start law (this is Proposition N read as a positive statement).
**Closest prior work.** Friedman, Murphy, Russell (1998) (prior network plus transition network); Anderson and Hsiao (1982), Hsiao et al. (2002), Binder et al. (2005) (initial-conditions problem in short panels); already rejected as alternative (A) in `../alg/NOVELTY.md`.
**Novelty.** Done. **Headroom.** None. **Kill risk.** Realised.

### C4. S3: factor-adjusted sparse SVAR for a pervasive latent driver

**Problem.** A common driver with node-specific filters (rain) makes every pair dependent; see Section 1.
**Method sketch.** X_t = common component (generalised dynamic factor model) + idiosyncratic sparse VAR; estimate both, read the graph from the VAR.
**Theorem.** Consistency as p → ∞ with pervasive factors.
**Closest prior work.** Forni et al. (2000); Barigozzi, Cho, Owens (2024, FNETS), which is this model with consistency theory and software; Basu, Li, Michailidis (2019); Chandrasekaran, Parrilo, Willsky (2012); Frot, Nandy, Maathuis (2019); Agrawal et al. (2023); Gerhardus and Runge (2020, LPCMCI).
**Novelty.** Done; applying FNETS to CausalRivers is a benchmark entry, not a method.
**Who has the problem.** Hydrology, climate, finance, neuroscience.
**Headroom.** Large on paper (0.69 against 1). But the theory needs many series to separate factors from idiosyncratic parts, and the benchmark task is on d = 5 and 10. In a river network the common component also travels along the edges, so "common" and "network" are not separable by rank alone. The repository's PCA conditioning with 10 components from 429 to 615 leave-out stations reached only 0.58 confluence accuracy.
**Feasibility.** A day. **Kill risk.** Not new; the closest thing already failed here.

### C5. S3 / own: frontier structure learning for nonnegative additive flows

**Problem.** Networks that transport a nonnegative quantity additively (river discharge, traffic counts, sewer and water networks, goods, packets). The local inputs of all nodes share latent drivers, so independence-based and conditional-independence-based discovery fails (Section 1).

**Idea.** In X_j(t) = Σ_i Σ_ℓ K_ij[ℓ] X_i(t − ℓ) + U_j(t) with K ≥ 0 and U ≥ 0, confounding biases the centre of the conditional distribution of X_j, but not its lower support boundary, provided the local input U_j can become negligible in every direction of the regressors. The mechanism is then the *frontier* of the data, and it is identified with arbitrary dependence among the U_j.

**Method sketch.** Per target j, maximise the mean explained flow Σ_i Σ_ℓ K_ij[ℓ] X̄_i subject to K ≥ 0 and nonnegative residuals. This is a linear programme and the maximum-likelihood estimator under exponential local inputs (Aigner and Chu 1968; Feigin and Resnick 1994). With measurement error: the stochastic-frontier likelihood, residual = one-sided + Gaussian (Aigner, Lovell, Schmidt 1977), by continuous optimisation. Edge score: the share of the mean flow of j attributed to i.

**Candidate theorems.** T1: identification of K from the joint support, without independence. T2: strong consistency of the LP estimator for stationary ergodic data. T3: an error bound when the support condition fails (base flow). Stated in `RECOMMENDATION.md`.

**Closest prior work.**
- Tran, Buck, Klüppelberg (2024, QTree): directed trees for river networks from extremes, max-linear model with independent noise, pairwise scores plus a spanning-tree step; near-perfect on the Upper Danube. Same data type and the same inequality X_j ≥ c X_i in the upper tail. Delta: sum-linear with lags, all of the data instead of extremes, general DAG, no independence, LP likelihood.
- Gnecco et al. (2021); Gissibl and Klüppelberg (2018); Bodik et al. (2023); Li and Castro-Camilo (2026); Routh (2026): causal discovery from heavy tails. Independent noise; tails only.
- Rayas, Anguluri, Dasarathy (2022); Mada and Anguluri (2025); Jayadev, Narasimhan, Bhatt (2019); Rajeswaran and Narasimhan (2015); Deka, Backhaus, Chertkov (2015): structure learning under conservation laws. Undirected or steady-state, Gaussian or exact balances, no latent confounding of the injections.
- Feigin and Resnick (1994); Davis and McCormick (1989): LP estimators for autoregressions with positive innovations (i.i.d. innovations, no graph). Aigner and Chu (1968); Aigner, Lovell, Schmidt (1977): frontier estimation.
- Donoho and Stodden (2003); Arora et al. (2012); Huang et al. (2014); Fu et al. (2019); Plumbley (2003): identifiability of nonnegative factorisations from the geometry of the data cone. Same geometric idea, no graph and no dynamics.
- Wang and Drton (2023); Mesters et al. (2026): discovery with dependent errors, by non-Gaussian moments or mean independence. Different assumption.
- Misiakos, Wendler, Püschel (2023): DAGs from data with few root causes (sparse inputs that percolate through the DAG). Sparse instead of one-sided inputs; no confounding.

**Novelty.** New combination; the identification principle was not found in causal discovery (searches listed in `REFS_VERIFY.md`, Section 3). Every ingredient is classical. Rated between incremental and clearly new.
**Who has the problem.** River topology is known from maps, so rivers are a benchmark, not a user. Users: karst and groundwater connectivity, sewer inflow, water and power distribution networks with unknown switching state, traffic and logistics flows.
**Headroom.** Section 1: best method 0.69 directed AUROC at 24 h, oracle-tuned 0.77.
**Sanity check** (`sanity/sanity_flow.py`, synthetic in-forest, p = 8, multi-filter latent driver, 8 seeds): directed AUROC with local events 0.927 (frontier LP), 0.951 (correlation times mean order), 0.880 (correlation), 0.852 (VAR); without local events 0.777, 0.859, 0.703, 0.555. Noise-free: 0.958 against 0.958. **The frontier LP beat the standard methods and did not beat the trivial physical baseline.** The generator has a strictly positive base flow, so it violates the support condition of T1; this is also what real rivers do.
**Feasibility.** High: LPs with at most 45 variables and about 7,300 rows; the pilot needs under 3 CPU-hours.
**Kill risk.** (1) The mean-order baseline takes the orientation gain, and the frontier adds nothing to the skeleton. (2) Base flow and rating-curve errors break the support condition. (3) Reviewers see a hydrology method.

### C6. S4: power-aware or decision-theoretic structure learning

**Problem.** Under saturation every test rejects; without it, power differs strongly between edges. The paper's detectability oracle predicts the detection probability of each pair from n_eff and the population correlation.
**Method sketch.** (a) Report the graph that minimises posterior expected structural Hamming distance given edge posteriors. (b) Test against an empirical null estimated from the bulk of all pairwise statistics, so that the driver-induced correlation is the null.
**Theorem.** Bayes optimality of (a); FDR control of (b) under Efron's assumptions.
**Closest prior work.** Carvalho and Lawrence (2008); Li and Wang (2009); Strobl, Spirtes, Visweswaran (2019); Efron (2004); Runge (2020).
**Novelty.** Incremental. (a) is a known estimator; (b) is a known large-scale-inference device and needs many pairs.
**Headroom.** Unknown. The oracle is post hoc: it needs the population correlations.
**Kill risk.** It is a decision rule on top of an existing learner, which the author excluded.

### C7. Own: second-moment identifiability of a subsampled SVAR with diagonal innovations

**Problem.** Data sampled more slowly than the causal time scale. Gong et al. (2015) state that under Gaussianity "the solutions are generally not unique" and use non-Gaussianity; so do Tank et al. (2019) and Gong et al. (2017) for aggregation.
**Observation.** With no instantaneous effects and a diagonal D at the causal scale, the subsampled process identifies P = Mᵏ and Ω_k = Σ_{j<k} Mʲ D (Mʲ)ᵀ. The roots of P are finitely many, and each root must also solve p(p + 1)/2 linear equations in the p unknowns of D. Sanity check (`sanity/sanity_subsample.py`, p = 4, k = 2): in 190 of 200 random draws exactly two of the 16 roots survive, M and −M (the other 10 draws failed numerically).
**Method sketch.** Gaussian likelihood in (M, D) with the exact subsampled moments, sparsity penalty, continuous optimisation.
**Candidate theorem.** Generic identifiability of M up to a finite group from second moments.
**Closest prior work.** Palm and Nijman (1984); Phillips (1973), Hansen and Sargent (1983), McCrorie (2003) (aliasing in continuous time, including covariance restrictions); Marcellino (1999); Gong et al. (2015, 2017); Tank et al. (2019); Hyttinen et al. (2016); Fan, Gong, Zhang (2024); Wang, Jennings, Gong (2023, SCOTCH) and Bellot et al. (2022) for continuous-time learners.
**Novelty.** Incremental. Finite multiplicity is the classical aliasing result; whether the diagonal restriction is stated in that literature was not checked. A small theorem, not a method.
**Who has the problem.** fMRI, economics, climate. **Headroom.** No number in the repository.
**Feasibility.** High. **Kill risk.** Known in econometrics; roots of a noisy P̂ are ill-conditioned; the start plays no role (the subsampled process is Markov too).

### C8. Own: mean transient under step inputs with known targets

**Problem.** Step inputs (Causal Chambers) move the mean; μ_t follows μ_t = Mμ_{t−1} + (I − B)⁻¹c.
**Theorem.** A step on a known target identifies one column of (I − B)⁻¹.
**Closest prior work.** Jordà (2005); Stock and Watson (2018); Mertens and Ravn (2013); Rothenhäusler et al. (2015, backShift); Peters et al. (2016); Pfister et al. (2019); Mooij et al. (2020); Wagenmaker and Jamieson (2020) for the design of inputs.
**Novelty.** Done. **Headroom.** None: with a known target the response is read off. **Kill risk.** Realised.

### C9. Own: amortised discovery trained with randomised start and burn-in

**Problem.** Pretrained discovery networks learn generator artefacts; Causal Pretraining reached 0.65 on CausalRivers.
**Method sketch.** Train on simulators whose start law, burn-in and scale are randomised, with the burn-in bound B ≥ Δ of the paper as the design rule.
**Closest prior work.** Lorch et al. (2022, AVICI); Ke et al. (2022); Stein et al. (2024); Montagna et al. (2025).
**Novelty.** Incremental (domain randomisation). **Feasibility.** Not on one CPU core. **Kill risk.** Engineering paper; bound by identifiability (Montagna et al.).

## 3. What the scan did not cover

- arXiv API, Crossref API and 11 web searches (queries in `REFS_VERIFY.md`, Section 3) on 2026-11-10. No citation crawl. The hydrology and water-resources journals were not searched systematically; a topology-inference method based on mass balance may exist there under another name.
- Abstracts were read; full text only for Gong et al. (2015), Sections 1 and 3.
- The claim that the frontier principle is absent from causal discovery rests on these searches only.
