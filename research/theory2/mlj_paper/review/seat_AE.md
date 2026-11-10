# Peer Review Report (Seat: Journal-Fit Reviewer / Action Editor)

## Manuscript Information
- **Title**: Rel-Sortability of Time-Unrolled Causal Graphs
- **Journal**: Machine Learning (Springer), first-round review, new submission
- **Review Date**: 2026-11-10
- **Review Round**: Round 1
- **Reviewer role**: Journal-Fit Reviewer / Action Editor (bird's-eye view: fit, originality, significance, presentation, contribution relative to what a journal paper must be)
- **Confidence**: 4/5. I am close to the area (benchmark artefacts in causal discovery) but did not re-derive the paper-only proofs (Appendix O.1, U). I checked headline numbers against result files in research/theory2.

## Summary Assessment
The paper gives a closed-form per-edge value of rel-sortability (Reisach et al. 2026, arXiv 2605.06288) for time-unrolled DAGs built from any summary graph with self-loops (Thm 1), its limit and rate (Thm 2), a necessary and sufficient condition (Cor 3), and a "stationary window" value after a latent burn-in B >= Delta (Thm 5, Cor 6). The graph results are formalised in Lean 4. Around this core the paper adds:
- a simulation study (B1, B1-NL, learner extension) of how the fixed start inflates rel- and var-sortability;
- finite-sample results on estimating relatives from one autocorrelated record;
- aggregation results;
- "no-label" identifiability propositions;
- real-data tests on CausalRivers and Causal Chambers, with time-stamped analysis plans and failures reported.

The paper is unusually candid. It states that the graph results are "short consequences of a walk-length lemma", that no new algorithm is proposed, that most real-data tests failed or only partly passed, and that everything was agent-generated and not human-checked. The core combinatorial result is correct as far as I can tell, and the supporting numbers I checked match the result files. As a journal contribution, however, it is a bundle of one small sharp theorem, several loosely connected auxiliary results, and a reporting exercise. The central claim that the fixed start is an artefact that changes what benchmark users conclude rests on a post hoc mechanism check and a failed prediction. The headline theory (rel-sortability) is only weakly linked to the headline empirical finding (learner rank changes). Reproducibility is good in spirit but not demonstrated end to end. My recommendation is major revision, mainly for focus, framing of significance, and verification status.

## Strengths

### S1: Exact, complete, falsifiable characterisation of an open question
Thm 1 gives the strict-sort condition for every edge and every T; Thm 2 and Cor 3 give the limit and necessity. This answers Q1 and Q2 of the introduction cleanly, and the ring example (s_rho = 1 for all T <= p) is a concrete, checkable counterexample to the "strong connectivity protects" reading at benchmark horizons.
**Evidence Anchor**: text: Sec. 3.2, Theorem 1 "has s_rho = 1 if some u in V satisfies d(u,j) <= t <= d(u,i), and s_rho = 1/2 otherwise"

### S2: Machine-checked core with honest scope statement
The Lean development (4,159 lines in 16 files, 163 declarations; I confirmed 16 files / 4,159 lines under lean/RelSort and 163 lines in lean/AXIOMS.txt, no sorry found) covers Thm 1, 2, 5, Cor 3, 6 and Prop 4. The paper states what is not covered and that nobody has compared Lean statements with paper statements. This is rare and valuable, and it is how a theory paper should signal trust.
**Evidence Anchor**: text: Sec. 3.6 "It does not certify the ER limits, the loop-free limit and periodicity, the lag-weighted extension"

### S3: Reporting discipline on failed predictions
Table 4 reports 9 failed graded criteria. The plan's failed prediction (no rank change against FLOP) is stated in the abstract. The post hoc oracle is labelled post hoc and its limitation is stated ("does not separate the structural mechanism ... from a general change of correlation strength").
**Evidence Anchor**: table: Table 4 (P2, P3, P4, P6, P9, Q9 fail; E2 "not met")

### S4: Numbers I checked match the logs
Table 3 entries (DAGMA aoa +0.197 [0.177, 0.216]; NOTEARS +0.156; GOLEM-NV +0.232; std DAGMA -0.064; std NOTEARS +0.036) match b1x/results/analysis.txt. The B1 pooled gain (+0.1249 [0.1161, 0.1338]; Spearman 0.798 over 42 cells; unit-level 0.616; slope 0.530) matches b1/results/analysis.txt. The exact values for rings (1.0 for T <= p) match b1/results/exact_predictions.txt.
**Evidence Anchor**: table: Table 2 and Table 3 vs b1/results/analysis.txt and b1x/results/analysis.txt

### S5: Useful practical message for benchmark designers
The recommendation to start from the stationary distribution or to use a burn-in checked against mixing, plus the generator survey, is actionable for a benchmark community.
**Evidence Anchor**: text: Sec. 6.1 "Start trajectories from the stationary distribution"

## Major Weaknesses

### W1: Significance of the core theorem is modest and the paper's own framing concedes this, yet the title and abstract present it as the main contribution
**Problem**: The graph results are short consequences of Reisach et al.'s Thm 1 / Lemma 1 plus a walk-length lemma (Lemma 19). Cor 3's "if" half is the earlier Thm 4. Prop 4 and Thm 5 are counting arguments. The self-assessment in the introduction ("not as a new theory of sortability") is honest, but the abstract, Sec. 2.6 ("a burn-in condition that is both sufficient and tight") and conclusions are phrased more strongly than the self-assessment. A reader finishing Sec. 3.2 will find a clean but small result; the remaining weight must come from the empirical and finite-sample layers, which are weaker (see W2-W5). Moreover, the whole contribution is a completion of a single preprint (arXiv 2605.06288), which the paper says shows "only sufficiency ... in the arXiv version". A revised version of that preprint could pre-empt the novelty, and the paper provides no sense of how the community regards rel-sortability beyond that one source (it is a 2026 arXiv criterion, not yet a standard benchmark property).
**Evidence Anchor**: text: Sec. 1 "The graph results of (1) are short consequences of a walk-length lemma and of Reisach et al. (2026, Lemma 1)"
**Why it matters**: For Machine Learning the question is whether the paper is a complete, significant contribution on its own. At present the main theorem would fit a short note.
**Suggestion**: (i) Reframe title/abstract around the artefact diagnosis (what a fixed start does to generators and learners) and present Thm 1-5 as the enabling tool. (ii) State explicitly the dependence on the preprint version and record in the cover letter whether the authors of the preprint have a v2 containing these results. (iii) Add a section showing the value to a designer beyond the exact formula, for example a procedure that, given a generator spec, returns the minimal burn-in (see W2).
**Severity**: Major
**Confidence**: 4 - adjacent field, familiar with sortability literature

### W2: "Tight" and "cannot be shortened" are overstated; tightness is shown only as an existence statement
**Problem**: The abstract says the burn-in "cannot be shortened". Remark 21 shows B >= Delta cannot be lowered for the single edge a -> b (Delta = 1), and enumeration shows that B = Delta - 1 failed to give the stationary value in 445 of 726 random graphs, i.e. it succeeded in 281. So Delta is a worst-case sufficient threshold, tight on a subclass, not a characterisation of the minimal burn-in for a given graph. The exact per-graph minimal burn-in is not given, although Thm 1's machinery (the windowed analogue of the distance condition) could yield it.
**Evidence Anchor**: text: Abstract "this burn-in cannot be shortened"; Remark 21 "B = Delta - 1 did not in 445"
**Why it matters**: A benchmark designer reading the abstract will take Delta as necessary. It is necessary only in the worst case, and the paper recommends B >= Delta tau_max as if it were the operative rule.
**Suggestion**: Either prove a per-graph characterisation of the minimal B (an equivalence analogous to Thm 1) or rewrite the abstract/Sec. 2.6/Sec. 6.1 as "sufficient, and not improvable uniformly over graphs with a given Delta".
**Severity**: Major
**Confidence**: 5 - checked against Remark 21 and Appendix B

### W3: The central empirical claim is not isolated from alternative mechanisms; the theory predicts sign/rank but not size, and the oracle that does predict size does not need the theory
**Problem**: Sec. 5.1 reports that exact gap vs observed gain has slope 0.53 and Spearman 0.80, while the post hoc detectability oracle has R2 0.97, slope 1.03, and "in a joint regression the exact gap adds nothing to it (coefficient -0.05)". The paper itself says the oracle does not separate the structural mechanism of Sec. 3.2 from "a general change of correlation strength during the transient". So the evidence that the theorem's mechanism (pairs that are relatives under one start only) drives the gain is correlational at the cell level (42 cells, graph-structured design with p, T, family confounded). The relatives-sort baseline is also close to tautological: its order accuracy is "essentially the estimated rel-sortability" (the paper's wording), so the experiment tests whether a plug-in estimator tracks its target.
**Evidence Anchor**: text: Sec. 5.1 "In a joint regression the exact gap adds nothing to it (coefficient -0.05)"
**Why it matters**: The claim that "the fixed start inflates rel-sortability" is a population statement already proven. What the experiments are meant to add, impact on benchmark conclusions, rests on this mechanism-versus-correlation question being unresolved.
**Suggestion**: Add an intervention that holds correlation strength fixed while changing only the relatives structure (e.g. generator variants with matched pairwise correlation magnitudes, or a per-pair analysis: gain on pairs that are relatives only under G1 vs pairs that are relatives under both, at matched |corr|). The authors already have the machinery (the oracle gives per-pair detection probabilities), so this is feasible at low cost.
**Severity**: Major
**Confidence**: 4 - adjacent field

### W4: The learner-ranking result, the most consequential finding for ML readers, is a failed prediction with an untested explanation and a narrow design
**Problem**: DAGMA/NOTEARS beat FLOP on in-trees under G1 and lose under G2. The plan predicted no rank change; the explanation (var-sortability) is "a hypothesis ... the evidence for it is weak", the matching control "has not been run", and standardised DAGMA/GOLEM are below chance (0.31-0.42 order accuracy) so they cannot test it. The design is one linear-Gaussian parametrisation, T <= 5, default hyperparameters, GOLEM at p = 5 only, 96 graphs, and the rank change is localised in one family (in-trees); per b1x/results/analysis.txt, the SHD rank change is significant only for DAGMA-raw and NOTEARS-raw on in-trees (2 of 64 tests), and on SID the effects are small (G1 -0.069 and -0.057 vs FLOP in-trees). The abstract nevertheless presents "a ranking change" as a headline result, and this finding is not covered by the theory that the title promises.
**Evidence Anchor**: text: Sec. 5.1 "A control that matches marginal variances across the two starts has not been run"
**Why it matters**: Machine Learning readers will care most about whether benchmark rankings are altered. The paper supplies an observation, not an explanation, and cannot tell designers which learners are affected in general.
**Suggestion**: Run the variance-matched control (same G1 start with per-column rescaling to G2 marginal variances before the learner, or an initial-covariance start that matches variances but not relatives), tune DAGMA/NOTEARS regularisation, and extend to T up to 10 and p = 10, 20. Otherwise move this finding out of the abstract's headline and into the limitations as an unexplained observation.
**Severity**: Major
**Confidence**: 4 - adjacent field

### W5: Verification status is thin for a paper whose selling points include "machine-checked" and "finite-sample theory"
**Problem**: (a) Lean statements are unchecked against paper statements by any human (acknowledged, Sec. 3.6 and 6.4). The map in Appendix G is "the agents' reading". (b) All paper proofs, including Prop 7 (ER limits, a probabilistic result), Prop 14-16, 17, Thm 18, 27, 30 and 31 and Prop 8-12, are not machine-checked and "no human has read line by line". (c) Prop 11 depends on faithfulness, which the paper itself says is not generic for tied parameters ("form a null set among all linear Gaussian models"), so the headline claim that the "time order is identifiable ... up to explicit exceptions" holds only for a model class where the assumption is unjustified, and Conjecture 13 is open. (d) The identifiability and aggregation sections have no empirical or even simulated confirmation in the main text. A journal cannot publish theorem statements whose only validation is the same agent family that wrote them; the "simulated pre-submission reviews" (Sec. 6.4) are explicitly not independent.
**Evidence Anchor**: text: Sec. 6.4 "What no human has yet read line by line: the paper-only proofs"
**Why it matters**: Soundness of roughly half of the main text is unverified by anyone independent. This is the single biggest risk to the correctness of the record.
**Suggestion**: Before resubmission, have at least one human probabilist/statistician verify Appendix O.1 and Props 14-16, 30-31, and one human compare Lean statements with the paper (a half-day task for a short table). Drop or demote to a remark any proposition that cannot be verified. For Prop 11, either prove faithfulness holds generically within the tied model class (even for a subclass), or restate the headline in the abstract with the assumption visible.
**Severity**: Major
**Confidence**: 4 - general standards

### W6: Scope and coherence: five loosely coupled contributions in one paper, with the main text acting as an index to a 50-page appendix
**Problem**: The paper has main text of about 26 pages followed by about 50 pages of appendices (76 pp total, Appendices A-U). Sections 3.4-3.5 (labelled identifiability), 4.1 (finite samples), 4.2 (aggregation) and 5.2 (real data) each could be a short standalone paper, and each is connected to the exact-rel-sortability core only through the same unrolled-DAG setting. Sec. 4.2 "claims only the sortability consequences" of known results (Gong et al. 2017; Herman et al. 2025 App. D). Sec. 3.4's Prop 8 is, by the paper's own words, "a short consequence of the Markov property". Contributions (4) and (5) in the introduction do not feed back into the narrative of (1)-(3). Many statements in the main text are made with forward pointers (e.g. "Appendix S", "Appendix K", "Appendix E") and the key evidence for several claims (the Table 4 footnotes, E1's weak-component equivalence, the K-sensitivity check) lives in the appendices. A reader cannot assess the claims from the main text alone.
**Evidence Anchor**: text: Sec. 1 "Organisation. Section 2 places the work among studies of benchmark regularities"; Appendix A "Appendix B gives full proofs ... Appendix U the proofs of Sections 3.4 and 3.5"
**Why it matters**: Machine Learning expects a coherent, complete paper. Breadth here dilutes the single strong result and makes verification by a referee near impossible (see W5).
**Suggestion**: Cut to a focused paper: Sec. 3.2-3.3, Lean, Sec. 5.1 (with W3/W4 controls) and the generator survey. Move 3.4-3.5, 4.1, 4.2 and 5.2 into a second paper or a short supplement clearly labelled as secondary. If the authors keep them, add a single-paragraph "how the pieces fit" argument and a table of claim, status (machine-checked / proved on paper / simulated / empirical), and where the evidence is.
**Severity**: Major
**Confidence**: 4 - editorial judgement

### W7: Real-data evidence does not support the claim of "partial support" for the theory in a way a reader can interpret
**Problem**: On CausalRivers, the one natural-graph test of rel-sortability (in-forests with confluences), the estimator carries no signal (P2 0.501; 98% of non-relative pairs dependent) and the conditioning remedy fails. On Causal Chambers, the positive results are either (a) designed independent sources on a bipartite graph where the prediction "coincides with 'sensors have more relatives than actuators'" (wind), (b) on a graph where, on the primary graph Glin, the prediction "coincides with the rule 'tie iff same weak component, else compare component sizes'", (c) a replication with i.i.d. inputs fixed in the plan as not counting, or (d) B3 on the same device by the same author without external registration. P9 and Q9 (graded agreement) fail in every study where they are run. The abstract's "a physical testbed gave partial support" is accurate but the real-data layer, as it stands, shows only that population relatives can be detected when sources are strong and independent, which is the content of F1, not of the rel-sortability theory. The analysis plans were "produced by AI agents within about six hours of one day", with push times controlled by the client, so the "pre-registered-style" label offers little protection against analyst degrees of freedom (the paper says as much).
**Evidence Anchor**: table: Table 4 (P2 0.501, P3 -0.018, P9 0.267, Q9 0.243, E2 "not met"); text: Sec. 5.2 "the prediction coincides with the rule"
**Why it matters**: The real-data section occupies a large share of the paper and its abstract; it is mostly negative or non-discriminating. Presenting it as evidence layer on a par with the theorems overstates what it shows.
**Suggestion**: Reframe Sec. 5.2 as a feasibility study with explicit negative result (relatives counts are unusable on persistent, driver-confounded records, which is a useful finding in itself), and drop the language of "partial support". If a second positive dataset with independent sources and no weak-component equivalence is available (multi-source real SCMs), add it; otherwise say clearly that no real-data test discriminates the theorem from a weak-component rule beyond B3's single device.
**Severity**: Major
**Confidence**: 4 - adjacent field

### W8: Reproducibility is promised more strongly than it is demonstrated
**Problem**: REPRODUCE.md states the full stages "were not re-run, because of the CPU budget", that `b1x/` and `finite2/` were added to `reproduce.sh` on 2026-11-10 (the day of this review) and "not re-run end to end", and that several outputs "are not reproducible by one command" because arguments or seed counts were not recorded: x1_er tables (the number of seeds is not recorded), x2_enum random modes, x2_limit.py, and the x1proof C++ sweeps. Those x1_er / x1proof outputs are the numerical support for Prop 7 (Fig. 2: "exact computation on seeded graphs, p = 2e4 to 1e5"). The Lean build was not timed. The paper's Supplementary information says reproduce.sh "accompanies the article" with stages for all simulations, which a reader would take to mean end-to-end reproducibility.
**Evidence Anchor**: text: REPRODUCE.md "The full stages were not re-run, because of the CPU budget"; "not reproducible by one command"
**Why it matters**: Machine Learning's reproducibility standard requires that results can be regenerated or that limits are stated. The limits are stated only in a repository file, not in the paper.
**Suggestion**: Run `./reproduce.sh all` once and report the outcome (identical/DIFFERS per stage) in an appendix. Record seeds and arguments for the x1_er and x2 scripts and regenerate Fig. 2 from a scripted call. Mention the unreproducible items in the paper's Supplementary information. Archive code and data checksums at a DOI (Zenodo) rather than a GitHub path.
**Severity**: Major
**Confidence**: 5 - read REPRODUCE.md

## Minor Issues

### W9: Authorship and AI-use statement is candid but creates an accountability question the journal must decide
**Problem**: Sec. 6.4 says the proofs, Lean code, experiments, analysis plans, simulated reviews and manuscript text "were generated by these agents", and "The author takes responsibility for all content", yet the same section says nobody checked the Lean/paper match or the paper proofs. Springer's policy permits disclosed AI assistance but expects the human author to have verified the content. Responsibility without verification is not the same thing.
**Evidence Anchor**: text: Sec. 6.4 "The paper proofs ... and the manuscript text were generated by these agents"
**Suggestion**: The editor should ask the author to confirm which statements he has personally verified. Add to the abstract/introduction one line that the proofs outside Lean are unreviewed.
**Severity**: Minor
**Confidence**: 3 - policy judgement

### W10: Abstract is overloaded and hard to parse
**Problem**: The abstract (about 230 words) lists nine claims in dense clauses, e.g. "on raw data DAGMA and NOTEARS beat FLOP on in-trees but lost to it after a burn-in, a ranking change we had not predicted". "Rel-sortability" is never defined, "FLOP" is not introduced, and "source strongly connected components" is unexplained. Several terms ("fixed start", "burn-in", "latent burn-in") are not defined where first used.
**Evidence Anchor**: text: Abstract "the exact values predict the sign and rank the conditions, and a post hoc detectability calculation reproduces the size"
**Suggestion**: Rewrite for a general ML audience in about 150 words with three claims: the exact value and the burn-in condition; the artefact in simulation; what does not carry to real data.
**Severity**: Minor
**Confidence**: 5

### W11: Table 3 is partly unreadable
**Problem**: The in-tree rows ("in-trees: DAGMA / NOTEARS / FLOP .13 / .76; .30 / .56") are typeset as a fragment with no column alignment, and the PCMCI+ row reports order accuracy 1.0 / 1.0 and SHD/|E| .02 / .02 with no CI. The in-tree DAGMA row ".08 / .87" is not labelled with its metric.
**Evidence Anchor**: table: Table 3 lower block
**Suggestion**: Make the in-tree block a proper sub-table with method rows and metric columns for G1/G2 and CI.
**Severity**: Minor
**Confidence**: 5

### W12: Prop 14 and the text around it mix sloppy notation
**Problem**: In Sec. 4.1, B_ab is defined with the absolute-value summand and W_ab as the Bartlett sum, then Prop 14 reuses W_ab as the sum of gamma products with a different normalisation, and "neff = n(1 - rho^2)^2 / W_ab" appears with a delta-method remark ("W the delta-method Bartlett variance") that conflates two quantities. The text states the plug-in uses B-hat, not W-hat, so the "effective" test is conservative, which is not stated where the claim "asymptotic size at most alpha/m" is made.
**Evidence Anchor**: text: Sec. 4.1 "neff = n(1 - rho^2)^2/W_ab (W the delta-method Bartlett variance)"
**Suggestion**: Use one symbol per quantity and state the conservativeness explicitly.
**Severity**: Minor
**Confidence**: 3

### W13: Terminology "pre-registered-style" / "time-stamped plans" is used inconsistently
**Problem**: The paper correctly says these are not pre-registrations, but headings such as "pre-registered", "kill rules", "confirmatory claims" and "Holm across all 21 confirmatory claims" invite the reading that the evidential standard is preregistration. The same author (agents) writes the plan, runs the analysis, and sets the commit times.
**Evidence Anchor**: text: Sec. 5.2 "commit times are set by the client and no external registry was used"
**Suggestion**: Replace "confirmatory" with "pre-specified in-repository"; reserve "kill rule" for a rule that was actually independently timed. Or register new studies externally (OSF) before running them.
**Severity**: Minor
**Confidence**: 4

### W14: Related-work coverage and positioning
**Problem**: Closest competitors are cited. But several are 2026 arXiv items (Reisach et al., Wienöbst et al. for FLOP, Guan et al., Tan and Syrgkanis), which I could not verify from the manuscript, and the identifiability section (3.5) is not positioned against the dynamic Bayesian network/structure-learning-without-time-labels literature or against permutation/ordering learning for exchangeable time series. The new generator survey is deferred to an appendix although it is the paper's most direct benchmark-relevant deliverable.
**Evidence Anchor**: text: Sec. 2.2 "We read the source code of the simulators in common use (Appendix S)"
**Suggestion**: Check that all 2026 references have stable identifiers and versions; move the survey table into the main text.
**Severity**: Minor
**Confidence**: 3

### W15: Self-citation and data-dependence of narrative
**Problem**: Sec. 2.1 devotes a paragraph to the author's SPADE paper, whose connection to rel-sortability is tenuous ("an instance of the setting"). It reads as positioning rather than as related work.
**Evidence Anchor**: text: Sec. 2.1 "Our own SPADE paper (Dang, Le, & Dinh, 2026) is an instance of the setting"
**Suggestion**: Shorten to one sentence or remove.
**Severity**: Minor
**Confidence**: 3

## Questions to Authors
1. Is a per-graph minimal burn-in computable in closed form (an equivalence analogous to Thm 1), and does it differ from Delta on the 281 of 726 graphs where B = Delta - 1 sufficed?
2. At matched pairwise correlation magnitude, does the G1-G2 gain in relatives-sort accuracy still differ between pairs that are relatives only under G1 and pairs that are relatives under both?
3. Does the DAGMA/NOTEARS vs FLOP rank change on in-trees survive a start that matches marginal variances across G1 and G2, longer horizons (T = 10) and tuned regularisation?
4. Which of the paper-only propositions (8-12, 14-18, 27, 30, 31, O.1) has a human other than the AI agents read? If none, which would the author withdraw if an error were found?
5. Has the authorship of arXiv 2605.06288 seen this work, and does a later version of that preprint include any of Theorems 1-5?
6. Can Prop 11 be given a generic-faithfulness justification inside the tied-parameter model class, or is it conditional on an assumption that holds on a null set?
7. Can the full `reproduce.sh all` run be reported with per-stage identical/DIFFERS, including `b1x/` and `finite2/`?
8. For real data, is there any dataset in which the theorem's prediction differs from the weak-component rule on a graph with several independent sources and is not the same device as B3?

## Ratings (1-5)
| Criterion | Score | Note |
|---|---|---|
| Significance | 3 | Clean, correct, practically relevant diagnosis for benchmark designers; main theorem small; downstream ML impact (ranking changes) unexplained. |
| Originality | 3 | Completes an open question from one 2026 preprint; later sections largely apply standard tools and known results. |
| Soundness | 3 | Core graph results are Lean-checked and numerically confirmed; large share of the paper rests on unreviewed paper proofs, a non-generic faithfulness assumption, and post hoc mechanism checks. |
| Clarity | 2 | Honest and precise sentence by sentence, but overloaded abstract, bundle of five contributions, key evidence in appendices, and a 76-page length. |
| Reproducibility | 3 | Pinned environment, checksums, seeds and a reproduction script are strong; end-to-end run not performed, some outputs not regenerable by one command, analysis-plan timing not externally verifiable. |

## Recommendation: **Major Revision**

**Justification.** The topic fits Machine Learning (benchmark validity, theory of data-generating processes, empirical study), and the core exact result is correct, useful and unusually well verified (Lean). The author's honesty about limits is a real strength. The manuscript is not yet a complete, focused journal paper, for three reasons.
1. The headline theorem is small, and the empirical finding that would give it consequence (learner ranking change) is an untested hypothesis from a failed prediction (W1, W4).
2. The mechanism link between theorem and experiment is not isolated (W3).
3. A large part of the claimed theory is unverified by any independent reader and partly rests on an assumption the paper itself says is non-generic (W5), while the real-data layer is mostly negative or non-discriminating (W7).

I do not recommend rejection: the problems are fixable by narrowing the paper, running two controls (W3, W4), obtaining human verification, and reporting an end-to-end reproduction. I would expect a revised, focused version (core theory plus Lean plus the artefact experiments with controls plus the generator survey, with the rest moved out) to be publishable. Re-review is required.

## Provenance note
This report was produced by a fresh subagent instance, same model family as the author-side agents that wrote the manuscript, code and analysis plans. It is therefore not independent evidence of correctness. It read the manuscript text (pdftotext of main.pdf, main text through Appendix E in detail; later appendices only skimmed or not read) and spot-checked numbers against research/theory2 result files (b1/results, b1x/results, lean file counts, REPRODUCE.md). It did not re-run any experiment, compile Lean, or re-derive the paper-only proofs, and did not read other review files or earlier review rounds.
