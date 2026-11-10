# Peer Review Report: Reviewer 1 (Methodology)

## Manuscript Information
- **Title**: Rel-Sortability of Time-Unrolled Causal Graphs (Q.-V. Dang), Machine Learning (Springer), first-round review of a new submission
- **Review date**: 2026-11-10
- **Round**: 1
- **Seat**: Peer Reviewer 1, methodology (research design, statistics, theory soundness, reproducibility). I did not read the other seats' files or earlier rounds.

## Recommendation: MAJOR REVISION

**Confidence**: 4/5. Graph theory, Gaussian time-series statistics and design of the simulation experiments are within my competence. I only spot-checked the long Erdős–Rényi probability proof and did not rebuild the Lean project.

### Ratings (1 to 5)
| Dimension | Score | One-line basis |
|---|---|---|
| Significance | 3 | The core graph results are correct and useful to benchmark designers. They are short consequences of a walk-length lemma, as the paper itself says. The attribution of benchmark effects to them is not established (W1). |
| Originality | 3 | An exact per-edge value, a tight-in-the-worst-case burn-in and the no-label identifiability layer are new. The finite-sample layer combines textbook tools. |
| Soundness | 3 | The deterministic theory is right (I re-derived it independently). The empirical inference and several headline phrasings are weaker than the evidence (W1–W5). |
| Clarity | 3 | The paper is honest and carefully hedged, but 76 pages and five contributions in one paper obscure the main thread. Status labels are used inconsistently across sections. |
| Reproducibility | 4 | Pinned environment, checksummed data, plan commits and a reproduction script. The full pipelines were not re-run end to end by the author (REPRODUCE.md, "What was verified"). |

---

## Summary Assessment

The paper completes the unrolled-graph analysis of Reisach et al. (2026). It gives the exact rel-sortability of every edge of a time-unrolled DAG for any summary graph with self-loops and any horizon (Thm 1). It gives the limit and a Δ-form rate (Thm 2), a necessary and sufficient condition (Cor. 3) and a stationary value after a latent burn-in B ≥ Δ (Thm 5, Cor. 6). Simulations (B1, B1-NL, a learner extension) and four small real-data studies argue that the fixed start inflates rel- and var-sortability for label-free learners. A finite-sample layer (effective sample size, saturation by a latent driver) and a no-label identifiability layer (Props 9–12, Conj. 13) are added, together with partial Lean 4 verification.

From the methodology seat, the deterministic graph theory is correct and well supported. The empirical and inferential layers are less secure. The simulated "start effect" is not separated from an ordinary correlation-strength transient. The real-data inference treats segments of a handful of runs as independent units. Several headline phrases ("cannot be shortened", "exact values predict the sign", "machine-checked") are broader than what is proved. The identifiability results depend on a faithfulness assumption that the authors themselves say may fail for tied parameters. The paper is unusually candid about failures, AI authorship and the limits of its time-stamped plans. I would like to see the paper again after the revisions below.

---

## Strengths

**S1. Core exact results independently reproduced.** Thm 1/2/5 and Cor. 6 hold in my independent brute-force implementation (see Checks C1). The Lean statements match the prose (C6). Anchor: `theorem: Thm 1, Thm 2, Thm 5, Cor. 6, Sec. 3.2, App. B`.

**S2. Lean verification is real and the fidelity is good where I checked it.** The BUILD_LOG and AXIOMS.txt show 163 declarations, each depending only on propext, Classical.choice and Quot.sound, with no sorry/admit/axiom. Definitions (`IsRel`, `sRho`, `sBurn`, `maxFinDist`, `unrolled`) follow the paper, and the unrolled DAG is defined from scratch. The paper states clearly what is not formalised. Anchor: `file: lean/BUILD_LOG.txt, lean/AXIOMS.txt, lean/RelSort/{Basic,Unrolled,Limit,Stationary}.lean`.

**S3. Candid process disclosure.** The paper calls its plans "time-stamped analysis plans", not pre-registrations, with the reasons (no external registry, client-set commit times, six hours of agent work, prior exposure to CausalRivers). It reports failed predictions (P2–P4, P6, P9, Q9, the HX2 prediction) and discounts weak positives (the B1-NL swap). Git history is consistent with the stated plan-before-results order (e.g. B1 plan at 2026-11-08 22:31, results at 2026-11-09 01:26). Anchor: `section: Sec. 5.2 Protocol; App. J, Q, S`.

**S4. Reproducibility infrastructure.** Pinned versions, SHA-256/MD5 manifests, `python3 -I` isolation, a staged reproduce.sh, and a data-location abstraction. Anchor: `file: REPRODUCE.md, requirements.txt, DATA.md, data_sha256/`.

**S5. Appropriate clustering in the primary simulation inference.** For B1 the unit of analysis is the summary graph (H1: mean over T, Wilcoxon over 60 graphs per family). Bootstrap CIs resample whole graphs, and the paper explains why data-set bootstraps would be too narrow. Anchor: `file: b1/analyse_b1.py lines 51-66; Sec. 5.1; App. S "Re-analysis"`.

---

## Major Weaknesses

### W1. The headline benchmark effect is not attributed to the structural mechanism, and "predicts the sign" has little content
**Problem.** The abstract says the exact values "predict the sign and rank the conditions" and that a post hoc calculation "reproduces the size". The paper's own analysis shows three things:
- The detectability oracle (R²=0.97, slope 1.03) is the analytic expectation of the very algorithm run on the same covariances. It includes every way the start changes pairwise correlation strength, so it validates the simulation and not the theorem (App. S, "What this shows and does not show").
- In a joint regression the exact gap adds nothing (coefficient −0.05).
- All 42 cells have a positive gap and a positive gain (Fig. Q1 caption), so "predicts the sign" cannot discriminate. The Spearman 0.80 across cells is dominated by between-family and T structure, because gap and gain both fall with T and differ by family. The unit-level value is 0.62.

The decomposition into "pairs that are relatives under one start only" (the graph mechanism) versus "relatives under both starts with a different correlation strength" was explicitly not done.

**Evidence anchor**: `section: Sec. 5.1 (Rel-sortability paragraph, Fig. 3); App. S "Detectability calculation"; abstract sentences 3–4`.

**Why it matters.** The contribution that links the graph theory to benchmarks, "the fixed start inflates rel-sortability", is supported only as an association. Tying the observed gain to Theorems 2/5 is the paper's main interpretive claim.

**Suggestion.** Add a counterfactual start that keeps the relatives sets of G1 but equalises marginal variances and correlation strengths with the stationary start. One option is an independent first slice with stationary marginal variances, with the transient decay otherwise unchanged. The reverse manipulation, a start with G2's relatives but G1-like correlations, is also possible. Report the gain under each. Report within-family partial correlations (gap vs gain controlling for family and T), and a mixed model with a graph random effect. Replace "predict the sign" by a statement that survives the all-positive observation, or drop it. Decompose the oracle's predicted gain into the two parts. Also report a bootstrap CI for the 0.53 slope at graph level.

**Severity**: Major. **Confidence**: 4/5 (core expertise: experimental design for simulation studies).

### W2. Pseudo-replication in the real-data, E1 and B3 inference; units are segments of very few runs
**Problem.** The "units" in several tests are segments cut from a few long trajectories of the same device:
- E1: 10 segments of 2,000 steps from the single 100,000-step citris_1 series ("10/10 units, Holm p = 0.0078" is 2⁻⁷, the sign-flip floor). Glin was also chosen after study-3 observations on the same device.
- B3 H1: "30 units of 3,000" come from 3 experiments × 30,000 steps. The p ≈ 10⁻⁵ sign-flip floor with "30/30 units" treats consecutive segments of three trajectories as exchangeable replicates. The effective number of independent units is about 3, for which the smallest attainable sign-flip p is 1/8.
- CausalRivers: 120 "subgraphs" (seeds 7000–7029, d ∈ {5,10}) cover 900 station slots but only 457 distinct stations, with a maximum reuse of 8 (my count from `empirical/subgraphs_7000.json`). All stations lie in two regions and share precipitation, which the paper itself identifies as the confounding driver. Subgraph-level cluster bootstrap CIs (`boot_ci` over subgraphs) therefore understate variance.
- The 12 h and 24 h columns of Table 4 are the same records at two resolutions, not replications.

**Evidence anchor**: `table: Table 4 and Table K1; App. T "Data"; App. K "Metric and tests"; empirical/subgraphs_7000.json`.

**Why it matters.** The effects reported (Δ ≈ −0.5, 99.8% agreement) are so large that the qualitative conclusion is probably safe. The reported p-values and "k/k units" are not valid evidence of generalisation across runs or devices, and they make a positive demonstration look stronger than it is. The paper's own limitation, "one device per chamber", is the right one, but the statistics do not reflect it.

**Suggestion.** Define the unit as the independent experiment or run (3 for B3 H1, 1 for E1 citris_1, 15 runs for wind). Report exact permutation p-values that respect that unit, or drop p-values and report effect sizes with run-level ranges. For CausalRivers use a cluster that groups subgraphs sharing stations, or a region/station-level block bootstrap, or a design with station-disjoint subgraphs. State that the two resolutions are not independent.

**Severity**: Major. **Confidence**: 4/5.

### W3. "Tight" and "cannot be shortened" are proved only for Δ = 1
**Problem.** The abstract says the burn-in "cannot be shortened"; the Introduction says "B ≥ Δ cannot be lowered (Remark 21)". Remark 21 and the Lean theorem `edgeLoops_burn_sharp` establish this only for the single edge with Δ = 1 (B = 0 gives ≥ 2/3 versus 1/2). The only other support is an enumeration in which B = Δ−1 failed on 445 of 726 graphs (App. Q). The same enumeration shows it did not fail on the other 281, so tightness is a worst-case statement about the family and not a property of each graph.

**Evidence anchor**: `quote: Abstract "this burn-in cannot be shortened"; Remark 21; App. Q "Exact curves and the Δ bound"; lean/RelSort/Extras.lean edgeLoops_burn_sharp`.

**Why it matters.** "Tight burn-in condition" is one of the four stated novelties (Sec. 2.6). For general Δ it rests on enumeration of random graphs. A reader would otherwise take "cannot be shortened" as true for every Δ and every graph.

**Suggestion.** Add a short proof, and ideally a Lean lemma, for a family with Δ arbitrary. My check shows the directed chain with self-loops works: for Δ = 1,…,5, B = Δ−1 gives window value strictly above 1/2 (0.75, 0.65, 0.607, 0.583, 0.568 at T = 3), while B = Δ gives 1/2 (C1). Rephrase to "B ≥ Δ is sufficient and, for every Δ, cannot be lowered in the worst case over summary graphs" and say that individual graphs may need less.

**Severity**: Major (a headline claim stronger than the proof); easily repairable. **Confidence**: 5/5.

### W4. The identifiability layer (Props 9–12, Conj. 13) rests on an unproved and possibly vacuous faithfulness hypothesis, and its proofs are unread
**Problem.** Prop. 11 assumes the Gaussian column distribution is faithful to the unrolled DAG. The paper says that tied parameters form a null set and cites Boege et al. (2025) for models with no faithful distribution. No example or proof of generic faithfulness is given for any family. All of Props 8–12 are agent-written paper proofs with no human reader (Sec. 6.4). Conj. 13 covers the burn-in window and has been checked by hand only for a → b at T = 3. The conclusion "removing the labels does not hide the time order in the population under the fixed start" (Sec. 6.1, Sec. 3.5 Reading) is therefore conditional on an assumption whose truth is unknown. In addition, the "population" identifiability says nothing about the n = 1000 behaviour where FLOP is far from the labelled reference.

I checked the statements I could. The T = 2 non-identifiability example of Prop. 12(a) is correct (C2). The Markov-class size T^{m0} of Prop. 10 is correct (C3). The proof of Prop. 11(b) is correct (the ratio argument v₃−v₂ = a²(v₂−v₁)). I found no structural unfaithfulness in six small tied-parameter models (C4), but in 2 of 12 draws from a coarse rational grid a conditional independence held by coincidence (ring p=3, collider p=3, T=3). This is exactly the situation a genericity theorem would have to exclude.

**Evidence anchor**: `theorem: Prop. 11 (and its proof, App. U), Sec. 3.5 "Reading", Conj. 13; Sec. 6.2`.

**Why it matters.** The identifiability results are presented as a contribution (the paper's contribution 5 and Section 6.1 recommendation). In their current form they are unconditional-sounding corollaries of an assumption that is not shown to hold for any non-trivial family.

**Suggestion.** Either (a) prove generic faithfulness (outside a proper algebraic subset of tied parameters) for at least the chain, out-tree and ring families, or (b) restate Props 9–11 explicitly as "conditional on faithfulness, which we verified symbolically for the following graphs", with an exact-arithmetic check of the d-separation/CI correspondence for those graphs (as in C4). Consider moving Sec. 3.4–3.5 to a separate paper. Do not use the identifiability statements to qualify the benchmark recommendations until faithfulness is addressed.

**Severity**: Major. **Confidence**: 4/5.

### W5. Finite-sample theory: the non-asymptotic bound is vacuous for the test actually used, and the code deviates from the theorem
**Problem.**
- Prop. 31 shows that for Bonferroni Fisher-z thresholds the failure probability that Theorem 27 can certify tends to d(d−1)e^{−z_c²/(2B)} > 0 (0.089 for i.i.d. data, d = 5, α = 0.01). The theorem therefore gives no non-asymptotic guarantee for the test applied to real data. The paper discloses this, but Sec. 4.1 (F1) presents "a non-asymptotic exact-recovery bound for the relatives set" as an addition.
- Prop. 30 needs K_n → ∞ with K_n = o(n). The code uses fixed K (20–100). The authors show a size of 0.113 against 0.05 at AR 0.99, K = 50. The K-sensitivity check was run only on B3 data, "not on the river data, which are the most persistent, or on study 3 and E1". Those are the studies where F1 and F2 are used quantitatively.
- Asymptotics are for fixed d with n → ∞, but real data have n_eff ≈ 100–230 (App. H) and the saturation argument for rivers (n_{1/2} ≈ 20–27) is asymptotic.
- The Gaussian assumption is used in the proofs, while the real records (rivers, Causal Chambers) are not Gaussian. The non-Gaussian null-variance claim (Remark 32) is cited, not re-proved.

My own simulation (C5) agrees with Prop. 14 at n = 1000: for independent AR(1) pairs with coefficients (0.8, 0.8), nominal rejection was 0.371 against the predicted 0.358 (R = 1500, SE ≈ 0.012). The plug-in test with K = 50 gave 0.061 against the target 0.05, so there is a small finite-sample excess even at moderate persistence.

**Evidence anchor**: `theorem: Prop. 14, Thm 27, Prop. 30, Prop. 31; section: Sec. 4.1 "F1"; App. E "Fixed K in practice"`.

**Why it matters.** The diagnostic recommended in Sec. 6.1 (compare n_eff with 3 + (z_c/atanh κ̂)²) is built on these results, and the post hoc "F2 explains the CausalRivers failure" depends on them.

**Suggestion.** Run the K-sensitivity check on the river data, study 3 and E1, since that is the stated gap. Add a simulation for a non-Gaussian, heavy-tailed or nonlinear case. Move the sentence "non-asymptotic exact-recovery bound" later in Sec. 4.1 so the vacuity for Bonferroni thresholds appears next to it. Report finite-sample size of the plug-in test at the actual n_eff of each real-data study.

**Severity**: Major (affects interpretation of real-data results). **Confidence**: 4/5.

### W6. Multiplicity and claim-level inference are constructed post hoc and not applied to the real-data tests
**Problem.** Holm is applied "across all 21 confirmatory claims" of B1, B1-NL, B3 and the learner extension, but each claim-level p is defined after the fact: the k-th smallest Holm p for "k of m families" rules, the best swap for swap rules, the cluster permutation for tracking, the plan's Holm p for B3 (App. S, "Multiplicity"). The permutation p-values sit at their floor (10⁻⁴), and B3's at 10⁻⁵. "The best swap" as a claim-level p is selection over many tests. The real-data studies have about 15 graded criteria (P1–P9, Q7–Q10, E1a–c, R7–R9), judged by point rules with no multiplicity treatment. The statement that Holm "keeps every claim except one" is therefore less informative than it sounds. The "294 of 595 interactions" is reported as significance counts with no effect-size distribution beyond selected examples.

**Evidence anchor**: `section: App. S "Re-analysis of B1 and B1-NL (post hoc)" Multiplicity paragraph; Table 4`.

**Why it matters.** The abstract says "All failed tests are reported", which is good, but the reader cannot tell which positive results survive a defensible family definition.

**Suggestion.** State the family definition and the claim-level p rule in the plans' language (if they were fixed ex ante) or label them post hoc in the main text. Report for each real-data criterion the effect size with a run-level interval, and say that pass/fail by point rule is a descriptive device. For the swap claims, report the number of tests, and an FDR or permutation-based family-wise p for "any swap".

**Severity**: Minor to Major (affects confidence in 'supported' verdicts, not the direction of results). **Confidence**: 3/5.

### W7. Learner extension: ranking changes among competitive methods rest on default settings, one parametrisation and a post hoc attribution
**Problem.** DAGMA, NOTEARS and GOLEM were run with default hyperparameters and a 0.3 threshold, on one linear-Gaussian parametrisation, T ≤ 5, and GOLEM at p = 5 only. In-tree is the only family with Holm-significant reversals against FLOP. The abstract says "DAGMA and NOTEARS beat FLOP on in-trees but lost to it after a burn-in", which is accurate but omits that there were two reversals out of 64 tests, both on one family. The attribution to var-sortability is called a hypothesis, and the variance-matched control has not been run. In a standardised run DAGMA and GOLEM are below chance (order accuracy 0.31–0.42), so they cannot show whether a scale-free effect exists. The G2 truth also differs from G1 by a latent first-slice confounder, so a G1-vs-G2 comparison mixes the start effect with causal insufficiency (the paper handles this with "without first-slice pairs", and the reversal for DAGMA persists).

**Evidence anchor**: `table: Table 3; section: Sec. 5.1 "Competitive learners"; App. S "Predictions of the plan and what the attribution rests on"`.

**Why it matters.** The result is presented as a practical warning for benchmark readers ("rankings among label-free learners can change with the start"). Its generality depends on hyperparameters that continuous-optimisation methods are known to be sensitive to.

**Suggestion.** Add the variance-matching control the authors describe (rescale each G1 data set to G2's marginal variances and vice versa). Add a small hyperparameter sweep (λ, threshold) for DAGMA/NOTEARS on in-trees, and at least one more parametrisation (noise, weight range). Keep the claim in the abstract at the level supported: "on in-trees under one parametrisation".

**Severity**: Major for the abstract's claim, Minor for the theory. **Confidence**: 4/5.

---

## Minor Issues

**M1. "Machine-checked" and counts.** The paper reports 163 declarations (33 headline, 53 graph lemmas, 65 algebraic, 12 examples). The Lean sources contain 259 theorem/lemma declarations (my grep), and the axiom report lists 163, so 96 are outside the report though they are covered by the "no sorry/admit/axiom" grep. No `example` keyword occurs in the sources, although the paper says "12 examples". Say which declarations make up the 163 and how "examples" were counted. Anchor: `lean/AXIOMS.txt; Sec. 3.6; App. G`.

**M2. Lean modelling of finite-sample files.** Random variables are vectors of an inner-product space and independence is "vanishing inner products". The statements labelled "algebra only" in App. G are accordingly algebra only. Sec. 4.1 says "The algebra of F2 and F3 is machine-checked", which is fine, but the abstract's "machine-checked in Lean 4" is placed near the whole finite-sample story. Keep the abstract's sentence tied to Theorems 1–6 explicitly.

**M3. Lean–paper statement fidelity is the agents' own reading.** The paper says so. The cheapest fix is for a human to read the 8–10 headline Lean statements (Thm 1, 2, 5, Cor. 3, 6, Remark 21, Table 1 values) against the paper. I did this for edge_score_dist, sRho_unrolled_exact/bounds_maxFinDist, sBurn_eq_maxFinDist and edgeLoops_burn_sharp and found them faithful. A one-page table in App. G listing the Lean type signature beside each paper statement would help the next reader.

**M4. App. O.1 (random graph limits).** Five pages of probability proof, written by agents and unread by a probabilist, with a "standard coupling" lemma written concisely. My independent numerics (C7) match the limit formulas (L, S) to within sampling error for c = 0.5, 1.5, 2, 3, and the quoted peaks (0.754 at c ≈ 1.07; 0.665 at c ≈ 1.09). So the statement is very likely right. Ask a probabilist to review it, or state Prop. 7 as "proved on paper; verified numerically".

**M5. Prop. 17 genericity.** The proof uses a "connected open set of stable parameters with the given support". Nonzero entries split into orthants, so the set is not connected. The argument works if the support is read as "allowed support" (coordinates may be zero), a star-shaped set, with a.e. non-vanishing of a non-identically-zero analytic function. Say so.

**M6. Wording of the Introduction contributions.** "(1) and (2) are machine-checked in Lean 4" is correct for the exact values, but the same paragraph lists the ER limit (paper only). Keep the status label in the sentence, not only in Sec. 3.3.

**M7. Table 2 and abstract effect size.** The abstract's "+0.13" is the pooled value over seven families (including two strongly connected ones); H1's own pooled estimate is +0.125 over five non-strongly-connected families. Give both with CIs in the abstract or main text.

**M8. Oracle R² = 0.97 and slope 1.03.** These are cell-level (42 cells). The unit-level R² is 0.61 (0.34 for the gap). Say "cell-level" where R² = 0.97 appears in the abstract summary and Sec. 5.1.

**M9. Time-series generator survey.** The survey covers code reading but no mixing time; the "exposed" classification is partly speculative. The paper says so. Consider moving the survey to the supplement and keeping one sentence in the main text.

**M10. Length and focus.** Five contributions (graph theory, finite samples, aggregation, benchmarks and real data, identifiability) in 76 pages. Sec. 4.2 (aggregation) is not used by any experiment except the real-data estimand. Splitting into a theory paper and an empirical paper would make each easier to evaluate and review.

**M11. Notation.** B_ab (with absolute values) and W_ab (without) are both used for "the Bartlett sum"; Prop. 14 states the limit with W_ab and Thm 27 uses B_ab. A single sentence at first use stating which one governs size and which governs the bound would help.

**M12. Plug-in factor floor.** B̂ = max(1, Ŵ) makes the test conservative when W < 1 (e.g. opposite-sign AR pair, size 0). Say that power is lost in that case and quantify it.

---

## Checks I ran (scripts in my scratchpad, not in the repo)

| # | Check | Method | Outcome |
|---|---|---|---|
| C1 | Thm 1, Thm 2 (both bounds), Thm 5 (B = Δ), tightness | Independent bitset implementation of relatives on G_T and on burn-in windows; 1,500 random summary graphs with self-loops (p = 2–5) × T ∈ {2,3,5,7}; Thm 5 at T = 3 for B = Δ and B = Δ−1; chains of length Δ = 1–5 | 6,000 cases, 0 failures for Thm 1/2; Thm 5 holds at B = Δ in all graphs; B = Δ−1 differed from the stationary value in 958 cases; chain family confirms tightness for Δ = 1–5 (B = Δ−1 gives 0.75, 0.65, 0.607, 0.583, 0.568; B = Δ gives 0.5). The ring has sρ = 1 for all T ≤ p (p = 3, 5, 6) and decays after. |
| C2 | Prop. 12(a): T = 2, free start covariance, a → b with loops | Exact rational arithmetic: build the 4-variable covariance, then the alternative labelling {a₁,b₂},{a₂,b₁}; check α′ = α, conditional independence of a₂ and b₂ given a₁, orthogonality of the new residuals | Verified. The alternative is a different column DAG with the same number of edges (3 vs 3), all coefficients non-zero, diagonal innovations. The counterexample is correct. It needs a non-diagonal first slice, consistent with Prop. 11 for the fixed start. |
| C3 | Prop. 10: Markov class size T^{m₀} | Brute force over all orientations of the skeleton with identical v-structures; 8 graphs (a→b, chains, rings, trees, 2-cycle, isolated units) × T = 2, 3, 4 | Class sizes equal T^{m₀} everywhere (e.g. in-tree with m₀ = 2: 4, 9, 16; ring: 1). The flattened PDF text "T m0" is T^{m₀} in the source; correct. |
| C4 | Faithfulness of tied-parameter unrolled models | Exact rational covariance for tied weights/variances; all pairs and conditioning sets of size ≤ 4, compared with d-separation; a→b (T = 3, 4), chain3, ring3, collider, 2-cycle, 2 seeds each | No Markov violations. No structural unfaithfulness in 10 of 12 runs. Two runs (ring3 and collider, seed 1, coarse rational grid) had 21 and 44 coincidental conditional independences, so genericity needs an explicit argument. This is a small sample. |
| C5 | Prop. 14 / plug-in size | Pure-Python simulation, independent AR(1) pairs, n = 1000, R = 1,500 | Nominal rejection 0.371 against predicted 0.358 for (0.8, 0.8) and 0.000 for (0.8, −0.8); plug-in K = 50 gave 0.061 against 0.05. Consistent with the theory and with the fixed-K caveat. |
| C6 | Lean names and fidelity (grep level) | All ~45 Lean names cited in the paper's text checked for existence in `lean/RelSort`; statements of edge_score_dist, sRho_unrolled_exact, sRho_unrolled_bounds_maxFinDist, sBurn_eq_maxFinDist, edgeLoops_burn_sharp, definitions of unrolled, dist, maxFinDist, sBurn read; AXIOMS.txt and BUILD_LOG.txt inspected; sorry/admit/axiom grep | All names exist. Statements match the prose (T = n+1 slices, window = last T slices, relatives from the full graph). 163 axiom lines all clean; no sorry/admit/axiom; 259 theorem/lemma declarations in total. I did not rebuild Lean. |
| C7 | Prop. 7 ER limits | Independent sparse-ER simulator (p = 20,000, 3 seeds) with SCC and source-set bitsets; fixed-point formulas for x and d | L, S (sim vs formula): c = 0.5: .6671/.5661 vs .6667/.5656; c = 1.5: .6975/.6161 vs .6981/.6163; c = 2: .6231/.5635 vs .6217/.5632; c = 3: .5431/.5216 vs .5433/.5214. Peaks: L = 0.7542 at c = 1.073, S = 0.6650 at c = 1.09, as in the paper. |
| C8 | Prop. 16 algebra and numeric examples | Re-derived the partial correlation formula, r_crit and n_{1/2} examples by hand and numerically | Formula, r_crit = 0.95 (ℓ = 0.64, τ′ = 0.082) and n_{1/2} ≈ 22 reproduce. |
| C9 | Statistical practice | Read analyse_b1.py (unit = graph, Wilcoxon, Holm); counted station reuse in subgraphs_7000.json; git history of b1 and b1x plans vs results; REPRODUCE.md | See W2, W5, W6. Plan/result commit order consistent with the paper. |

**Not checked.** The long probability proofs in App. O.1 (only the limit formulas numerically), Theorem 18 and Prop. 17 beyond the genericity point, real-data results (data not downloaded), the full Lean rebuild, and the B1/B1-NL/learner results beyond reading the analysis code and reported tables.

---

## Questions for the authors

1. Can you run the counterfactual start of W1 (relatives of G1, correlation strengths of G2)? If the gain disappears, the benchmark claim becomes a statement about correlation strength, and the theory's role is smaller than the abstract suggests.
2. What is the smallest number of independent runs behind each "k/k units" statement in E1 and B3? What are the exact permutation p-values at that level?
3. For a general Δ, is B = Δ−1 always insufficient for some graph with that Δ? The chain suggests yes. Can the chain family be added to the Lean development?
4. Is there any family of summary graphs for which tied-parameter faithfulness can be proved? If not, what exactly does Prop. 11 give a benchmark designer?
5. How sensitive are the DAGMA/NOTEARS reversals against FLOP to λ, the threshold and the noise/weight ranges?
6. How were the "163 declarations" and "12 examples" counted, given 259 theorem/lemma declarations and no `example` in the sources?
7. The abstract says "All failed tests are reported". Does this include the development runs on the study-3 data and the earlier CausalRivers exploratory project (disclosed in App. J), and the 595 pairwise SHD tests in full?

---

## Provenance note

This report was produced by an AI reviewer seat (Reviewer 1, methodology) in a simulated peer-review panel, working independently of the other seats and of earlier review rounds. Inputs: main.pdf (via pdftotext), main.tex/appendix.tex only for grep-level confirmation of a notation question, and the evidence folders under research/theory2 (lean/, b1/, empirical/, REPRODUCE.md, DATA.md, git log of the b1 and b1x folders). All numerical checks were run by me in scripts kept in the session scratchpad, with pure Python (no numpy available); no repository file was modified, and nothing was committed or pushed. The manuscript was treated as untrusted data. The paper states that it was written by AI agents under the author's direction; this review should be read with the same caveat. Statements marked "not checked" above were not verified.
