# Peer Review Report (Reviewer 3, Cross-Disciplinary Perspective)

## Manuscript Information
- **Title**: Rel-Sortability of Time-Unrolled Causal Graphs
- **Journal**: Machine Learning (Springer), first-round review of a new submission
- **Review Round**: 1

## Reviewer Information
- **Role**: Peer Reviewer 3 (Perspective)
- **Identity**: Formal methods and verified mathematics (Lean 4, Mathlib); time-series econometrics (SVAR, initial conditions, stationarity, identification through heteroskedasticity); metascience and research integrity (pre-registration, post hoc analysis, AI-assisted research); practical impact for benchmark builders.
- **Review focus**: (1) what is machine-checked and what is only claimed; (2) the research-integrity story and the AI-use disclosure; (3) accuracy of the econometric framing and missing SVAR/DSGE prior art; (4) whether a benchmark builder would change practice; (5) overclaiming and confusion for an informed outsider.
- **Calibration status**: `NOT_CALIBRATED`

## Overall Assessment

### Recommendation
**Major Revision.** The graph-theoretic core is correct as far as I could check and is unusually well verified. The paper is also candid about its limits. But the manuscript is a 76-page bundle of at least four separable papers, and its headline empirical claims are weaker than the abstract suggests. The practical recommendation rests on one linear-Gaussian simulation family and on a threat model that applies to a narrow construction. The integrity apparatus is described honestly, but it is called "time-stamped plans" while the plan, the code and the results were all produced by the same agents. That gives much less protection than the vocabulary suggests.

### Confidence Score
4. I am at home with the Lean, econometrics and metascience aspects. I did not re-derive the probabilistic appendices (O.1, E, U), and I did not run the simulations.

### Summary Assessment
The paper computes the exact rel-sortability of every edge of a time-unrolled DAG. It shows that a deterministic start inflates the criterion, and that a latent burn-in of Δ slices removes the structural excess. It adds finite-sample theory for estimating relatives from one record, identifiability statements for hidden time labels, simulation experiments with raw and standardised data, and real-data studies on CausalRivers and Causal Chambers.

I compared 20 or so Lean statements against the prose. The core theorems (Theorem 1, Theorem 2, Theorem 5 and the sharpness example) are formalised faithfully, with no dropped hypotheses. The "machine-checked" claim is honest for that core. It is inflated in the 163 headline count and in the abstract's wording about tightness. The econometric framing is correct but ignores standard SVAR/DSGE initial-condition practice, and the notion of "burn-in" used in the title is a graph-level notion that a practitioner could misread as a stationarity guarantee.

The strongest empirical finding, that the fixed start inflates rel-sortability, is nearly definitional. The paper's own regression shows the exact gap adds nothing beyond a detectability oracle. The real-data section tests a different question (can relatives be estimated from one record?), and the tests that bear on the theory mostly failed. The recommendation is cheap to follow and probably correct, but its evidential base is thin relative to the length and the claims.

## Strengths

### S1: The Lean core is faithful and checks out
The Lean statements for the central results match the paper. `edge_score_dist` states the per-edge score with the ℕ∞ distance form of Theorem 1. `sRho_unrolled_bounds_maxFinDist` states Theorem 2 with min(Δ, T−1) and the inner-edge count. `sBurn_eq_maxFinDist` states the burn-in result for B ≥ Δ with m ≥ 1, i.e. T ≥ 2. All 158 `\lname{}` names cited in the paper exist in the sources and, except for two non-theorem names, in the axiom report. The sources contain no `sorry` or `axiom`, and the build log shows a clean rebuild. Relatives, s_ρ and G_T are defined from scratch (`IsRel`, `sOf`, `unrolled`), not imported from an unverified model.
**Evidence Anchor**: dataset: lean/RelSort/Unrolled.lean `edge_score_dist`; Limit.lean `sRho_unrolled_bounds_maxFinDist`; Stationary.lean `sBurn_eq_maxFinDist`; lean/BUILD_LOG.txt (clean rebuild, 9,064 jobs); lean/AXIOMS.txt (163 lines, standard axioms only)

### S2: Unusually candid about what is not verified
The AI-use section, the Lean section and Appendix G state that no human has checked that the Lean statements match the prose, and that the paper-only proofs have not been read line by line. The paper separates "checked", "proved", "asymptotic", "sketched" and "enumerated". It reports that formalisation exposed four missing nondegeneracy hypotheses in earlier finite-sample statements (e.g. b_u b_v ≠ 0).
**Evidence Anchor**: text: §6.4 "No human has checked that the Lean statements say what the paper statements say" ; Appendix G "Findings"

### S3: Negative results and failed predictions are reported
The paper lists failed predictions (P2, P3, P4, P6, P9, Q9, the "no rank change" prediction) and the kill-rule triggers. It discloses prior exposure to CausalRivers, the post hoc status of the detectability oracle, and that E1 coincides with a weak-component-size rule. It also notes that Holm correction removes one B1-NL claim. This is above the norm for the field.
**Evidence Anchor**: table: Table 4 (verdict column); text: §5.2 "so our account of that failure is explanatory, not predictive"; Appendix J "Prior exposure to CausalRivers (disclosure)"

### S4: The recommendation to benchmark builders is cheap and actionable
Start from the stationary distribution (exact Lyapunov or Cholesky start) or use a mixing-checked burn-in, standardise columns, and score with first-slice pairs removed. Each item is a few lines of code. The paper also correctly separates the graph artefact (burn-in ≥ Δτ_max) from the distributional transient (spectral radius).
**Evidence Anchor**: text: §6.1 "Start trajectories from the stationary distribution ... or use a burn-in that is checked for distributional mixing"

### S5: A reproducibility package that is more complete than usual
The package has pinned versions, a Mathlib commit hash, SHA-256 data checksums, stage-wise `reproduce.sh` with identical/DIFFERS comparison, and `python -I` isolation. Per-plan commits exist in the history, and I confirmed from `git log` that plan commits precede result commits for B1, B1-NL, B3 and the learner extension.
**Evidence Anchor**: dataset: REPRODUCE.md "Stages"; `git log` for research/theory2/b1, b1nl, b3, b1x

## Weaknesses

### W1: "Machine-checked" is used in a way that overstates both the count and the tightness
**Problem**: (a) The abstract says "These results are machine-checked in Lean 4" right after "this burn-in cannot be shortened". The Lean sharpness result (`edgeLoops_burn_sharp`) covers one graph (one edge, Δ = 1, B = 0). Appendix C/Remark 21 admits that B = Δ−1 failed to give the stationary value in only 445 of 726 random graphs, so a smaller B suffices for the other 281. "Tight" therefore means "not improvable uniformly in the graph", which is a statement about a worst case, not a per-graph characterisation. The abstract and contribution list drop "in general". (b) "163 declarations" is the number of lines of the axiom report. The sources contain 259 theorem/lemma declarations. Of the 163, 12 are examples and 65 are "algebraic facts, many elementary" (e.g. |ŝ − 1/2| ≤ 1/2; constant counts give s = 1/2). Prop. 15's headline "population rel-sortability is exactly 1/2" is, in Lean, `sOf_const`/`sOf_saturated`, which is a three-line identity: when every node is a relative of every other, counts are constant. The probabilistic content of F1–F3 is not formalised, and the formal "correlation" lives in an abstract inner-product space with independence encoded as orthogonality. The number 163 is therefore an unreliable signal of verified content; about 29 theorems carry the verified graph theory. (c) The statement-to-prose map is by the same agents that wrote both sides. No human or independent check of statement faithfulness exists, so the Lean evidence certifies the Lean statements, not the paper's.
**Evidence Anchor**: text: Abstract "a latent burn-in ... and this burn-in cannot be shortened. These results are machine-checked in Lean 4." ; text: Remark 21 "B = Δ − 1 did not in 445" (726 graphs); dataset: lean/RelSort/FiniteCorr.lean `sOf_const`, `sOf_saturated`
**Why it matters**: A reader who sees "163 machine-checked declarations" and "tight" will assume per-graph tightness and verified finite-sample theory. The first is false, and the second holds only for deterministic algebra. Overclaiming of this kind is exactly what formal verification is supposed to prevent.
**Suggestion**: (i) Replace "cannot be shortened" in the abstract and Contribution (2) with "cannot be shortened in general (sharp for Δ = 1; smaller B suffices for some graphs)", or add a theorem characterising the minimal B per graph, or at least a family with arbitrary Δ for which B = Δ−1 fails (a path or ring should work), and formalise it. (ii) Report the count as "29 headline theorems that formalise numbered statements of Section 3.2, plus 134 supporting declarations", and list the non-trivial ones. (iii) Have one person who did not write the paper compare the 29 headline Lean statements against the prose, using a short checklist (definitions, hypotheses, quantifiers), and report the result. I performed this check on a subset and found no mismatch; the author should do it on all.
**Severity**: Major
**Confidence**: 5, core expertise: Lean/Mathlib

### W2: The econometric framing is accurate but ignores the standard literature, and "burn-in" is a graph notion that practitioners will misread
**Problem**: The paper's "fixed start" is the zero (or noise-only) initial condition that every SVAR/DSGE simulation text discusses, and its remedy (draw X_0 from the unconditional distribution, or discard a burn-in) is the textbook remedy (e.g. Hamilton 1994 on exact vs conditional likelihood and stationary initial draws; Lütkepohl 2005 and Kilian & Lütkepohl 2017 on simulation and presample values; Dynare's `drop` and exact-initial options for DSGE simulation). None of this is cited; `refs.bib` has no entry of this kind (checked `main.tex`, `refs.bib`). The initial-condition literature in unit-root testing (Elliott 1999; Müller & Elliott 2003) also shows that the start matters for inference. Second, the central "burn-in B ≥ Δ" result is a statement about population relatives under faithfulness, i.e. about graph connectivity. It does not make the process stationary: after Δ slices the covariance is still transient at rate ρ(A)^{2B}. The paper says this (B1 needed 100 steps), but the title-level message "tight burn-in" and the phrase "latent burn-in as long as the largest finite distance" could lead a practitioner to think that Δ steps give a stationary start. Third, the stationary-start results assume a stable VAR with unit lags and no contemporaneous edges; this excludes B ≠ 0 SVARs, which the Section 3.4 model allows but the theorems do not cover. Fourth, near-unit-root and levels data, where no stationary start exists and where the paper's own motivating example (SPADE on levels) lives, are not discussed. Fifth, Prop. 8 conditions on a start law ν that does not depend on the parameters; the stationary start depends on (M, Ω), and while the conclusion still holds (the stationary law is a function of the reduced form), the exact-likelihood literature treats this case separately and one sentence would suffice.
**Evidence Anchor**: absence: Sections 2.2, 2.3, 3.4 and refs.bib — expected citations to SVAR/DSGE simulation and initial-condition practice (Hamilton; Lütkepohl; Kilian & Lütkepohl; Dynare drop/initial-state options; Müller & Elliott on initial conditions); checked main.tex, refs.bib, appendix.tex, Appendix S generator survey
**Why it matters**: Without these references the work reads as a new discovery about simulation starts, when the underlying fact (transients contaminate samples from a zero start) is classical. What is new is the translation into relatives and rel-sortability. The paper should claim that, and state precisely which part of the stationarity problem its Δ addresses.
**Suggestion**: Add a short subsection "Initial conditions in SVAR and DSGE simulation" with these references. State, in the abstract and next to Theorem 5, that B ≥ Δ equalises relatives and does not give stationary covariances. Give the practitioner rule in one line: use an exact stationary draw when ρ(A) < 1, otherwise there is no stationary benchmark and rel-sortability will be start-dependent. Add one sentence in Prop. 8 for ν = ν(M, Ω). State whether B1 weights were constrained to ρ(A) < 1 and what the spectral radii were.
**Severity**: Major
**Confidence**: 4, core expertise: econometrics; did not audit every B1 parameter

### W3: The headline empirical claim is close to a restatement, and the paper's own regression shows the theory adds nothing beyond detectability
**Problem**: The abstract says "the exact values predict the sign and rank the conditions, and a post hoc detectability calculation reproduces the size". But rel-SortnRegress order accuracy is essentially the estimated rel-sortability (the paper says so: "this order accuracy is essentially the estimated rel-sortability"), and G1 versus G2 differ exactly in whether the fixed start makes more pairs relatives. A positive sign is close to guaranteed. The ranking across 42 cells (Spearman 0.80, slope 0.53) pools family, p and T, which are strongly confounded with detectability. In the joint regression "the exact gap adds nothing to it (coefficient −0.05)" next to the detectability oracle (R² = 0.97), and the oracle "does not separate the structural mechanism ... from a general change of correlation strength during the transient". So the data cannot distinguish the paper's mechanism (relatives that exist only under one start) from the ordinary fact that a transient changes correlations. For the learner experiments the paper concedes "Theorems 2 and 5 do not predict these ranking changes", and the attribution to var-sortability is a hypothesis with no variance-matched control. Table 3 also shows FLOP's own accuracy falls from .89 to .77 and SHD/|E| rises from .19 to .52 under G2 but reports no ΔOA or CI for FLOP, so the "ranking change" is a difference of two shifts of similar size, tested only as an interaction.
**Evidence Anchor**: text: §5.1 "In a joint regression the exact gap adds nothing to it (coefficient −0.05)" ; table: Table 3 FLOP row OA .89 / .77, ΔOA "–"; text: §5.1 "A control that matches marginal variances across the two starts has not been run."
**Why it matters**: The paper's claimed contribution beyond Reisach et al. is "change what a benchmark designer should conclude". The evidence that the start changes learner rankings is one linear-Gaussian family, n = 1000, T ≤ 5, default hyperparameters, in-trees only for the reversal, and the mechanism is unidentified.
**Suggestion**: (i) Run the variance-matched control (rescale the G1 data per column to the G2 marginal variances, or compare standardised versions with R²-matching) and report whether the in-tree reversal survives. (ii) Report FLOP's own G1−G2 shift with a CI and test the ranking change as a contrast with a cluster CI. (iii) Add a negative-control experiment in which the start changes correlations but not the relatives sets (e.g. a strongly connected graph with ρ(A) near 1 and a start chosen so the transient is long), to test whether the structural mechanism predicts anything the detectability oracle does not. (iv) Reword the abstract: "the fixed start raises order accuracy of relatives sorting, as expected from the exact values; the exact values do not explain its size beyond detectability".
**Severity**: Major
**Confidence**: 4, adjacent expertise: causal-discovery benchmarking, core expertise: time-series simulation

### W4: The research-integrity apparatus is honestly described but gives little protection, and its vocabulary drifts
**Problem**: (a) The plans, code and results were all produced by the same AI agents within hours. The paper itself says "push order cannot show that nothing was computed before a plan". Hosting push records were retrieved for only three of nine plans; the B1 plan and the four earlier plans have only client-set timestamps. (b) The repository calls these artefacts "pre-registered": the commit messages for b1x and b1nl say "pre-registered plan", and `b3/OSF_PREREG.md` is an OSF-formatted text. The paper says "not pre-registrations" and that no registry was used. An OSF registration costs nothing and could have been posted before the B3 held-out access; it was not. (c) The study sequence is adaptive: B1 → post hoc re-analysis → B1-NL → learner extension → real-data E1 motivated by study-3 observations on the same device. The Holm correction covers 21 claims but not the forking paths of the sequence. (d) The CausalRivers plan was written by agents that already knew the negative exploratory result; the paper discloses this, but the P1–P3 predictions then amount to a test that the agents expected to fail or at least could not expect to pass. (e) Interpretation thresholds were fixed by the same agents that chose them as achievable; P9's CI contains the threshold, yet the point rule declares failure, which is fine, but E1's pass on the primary graph G_lin depends on removing six real edges chosen by the same agents.
**Evidence Anchor**: text: Appendix J "Push records order pushes, not computations ... no plan was posted to an external registry" ; dataset: git log messages for research/theory2/b1x (5ac1bd2 "pre-registered plan") and b3/OSF_PREREG.md; text: §5.2 "the graph refinement used by E1 was motivated by study-3 observations on the same device"
**Why it matters**: A reader may take "time-stamped plans with all failures reported" as equivalent to pre-registration. It is an internal discipline that makes honest reporting easier, not an external constraint. Mixed vocabulary between repository and paper makes that easier to misread.
**Suggestion**: (i) Retrieve and tabulate hosting push records (GitHub events or `git reflog` on the server) for every plan, and archive the repository state at submission in Software Heritage or Zenodo with a DOI so history cannot be rewritten. (ii) Align vocabulary: replace "pre-registered" in repository files, or add a header saying these are internal plans. (iii) For any follow-up study, post a plan on OSF before data access. (iv) Add a "garden of forking paths" paragraph listing every analysis choice made after seeing results (G_lin, the weak-component comparator, the oracle, graph-cluster inference) and state which claims are confirmatory and which exploratory in a single table.
**Severity**: Major
**Confidence**: 4, core expertise: metascience; Springer's exact policy wording not re-checked

### W5: The AI-use disclosure is thorough but does not meet the intent of authorial accountability
**Problem**: Springer's policy (Nature Portfolio family) requires that LLMs are not authors, that their use beyond copy-editing is documented, and that authors take responsibility for the content. Here the manuscript text, all proofs, all code, all plans and the simulated reviews were produced by agents, and the author "takes responsibility for all content" while stating that no human has read the paper proofs (Appendix O.1, Props. 8–18, Prop. 30–31, Thm 27) or compared Lean statements with the prose. These two statements are in tension: responsibility for unread proofs is formal, not substantive. The disclosure also lacks the model name and version, dates, the agent framework, prompt or transcript availability, and how the agents' outputs were accepted or rejected. The "simulated pre-submission reviews that motivated revisions" are a further undocumented influence (their text is not in the supplement as far as I could see).
**Evidence Anchor**: text: §6.4 "The author takes responsibility for all content" together with "What no human has yet read line by line: the paper-only proofs"; absence: §6.4 and Statements — expected model name/version, agent tooling, transcript or log availability; checked main.tex, appendix.tex, Statements and Declarations
**Why it matters**: For this journal, reviewers will be the first human readers of 76 pages of unread proofs. The disclosure is honest, which is the right instinct, but the editors need to decide whether unchecked AI-written proofs are acceptable in a research article, and the paper gives them no basis for that decision beyond the labels.
**Suggestion**: Add model identifiers and dates, the agent harness, and a pointer to the logs. Say what the author personally checked (for example, which theorems he read and verified by hand). Before resubmission, have a human (the author or a named colleague) read and sign off the proofs that carry the main claims (Props. 11, 14, 15, 16, 17 and Appendix O.1), or move unread material to a clearly labelled supplement or a separate preprint. Include the simulated review texts in the supplement.
**Severity**: Major
**Confidence**: 4, adjacent expertise: publication policy; policy text paraphrased from memory

### W6: Scope sprawl and a weak link between the benchmark claim and the real-data section
**Problem**: The paper has at least four separable contributions: (i) exact rel-sortability and burn-in with Lean; (ii) finite-sample estimation of relatives and saturation by a latent driver; (iii) identifiability with hidden time labels; (iv) a benchmark experiment on the start artefact. The real-data studies (CausalRivers, Causal Chambers) test (ii) and the stationary value s̃, not the start artefact of (iv). Their outcome is mostly negative: P2, P3, P4, P6, P9 and Q9 fail; E1 passes mainly through ties and a single-source graph that coincides with a weak-component rule; B3 uses the same device. The abstract's "on a river benchmark the count was saturated; a physical testbed gave partial support" does not connect to the title question. The identifiability section (Props. 9–12, Conjecture 13) is unrelated to the benchmark recommendation, relies on faithfulness the paper itself says does not generalise to tied parameters, and is paper-only. Several reviewers will find the 76 pages difficult to evaluate.
**Evidence Anchor**: table: Table 4 verdict column (P2, P3, P4, P6, P9, Q9 fail); text: §3.5 "The statements below are population statements with paper proofs ... none is machine-checked"
**Why it matters**: A reader cannot tell which claim carries the paper. The Lean-verified graph theory and the start-artefact simulation form a coherent 25-page paper. The rest dilutes it and raises the review burden.
**Suggestion**: Split into (A) graph theory + Lean + start artefact + recommendations, and (B) finite-sample relatives estimation + real data (+ identifiability as a short note or appendix). If the author keeps one paper, move Sections 3.4–3.5, 4.2 and the real-data details to supplementary material and give the main text a single storyline, with each claim labelled confirmed / unconfirmed / hypothesis in one table.
**Severity**: Major
**Confidence**: 4, adjacent expertise: editorial judgment

### W7: The practical threat model is narrow, and exposure of deployed generators was not measured
**Problem**: The artefact bites only when the slices of independent trajectories are given to a method that ignores time order, which is the construction proposed in Reisach et al. (2026). The paper's own generator survey concludes that every other generator "produce[s] one record for time-series methods, so neither artefact applies to them as shipped"; the reference implementation of the unrolled construction is not public, so even there "only the graph-level start is verified". The variance artefact needs mixing times that "we did not compute for any deployed generator". The CauseMe documentation could not be opened, and no claim is made. Causal-discovery benchmark builders working on real time-series methods use time order and one record, so their practice would not change. The finite-horizon rel-sortability results also exclude contemporaneous (lag-0) edges, which are common in CausalTime-style and tigramite generators, and require a self-loop at every node.
**Evidence Anchor**: text: Appendix S "so neither artefact applies to them as shipped" ; text: §3.2 "Contemporaneous (lag-0) edges are not covered."
**Why it matters**: The recommendation is proportionate for people who build unrolled benchmarks for static learners, but "which generators are exposed" is answered with essentially "none, as shipped". The audience is small, and the paper should say so up front, while also saying who gains.
**Suggestion**: State the threat model in the abstract's first lines. Add a minimal lag-0 extension (an SVAR with B ≠ 0 acyclic, which gives a layered DAG in each slice) or state that it is open and give a counter-example check by enumeration. Run the mixing-time computation (spectral radius of typical generated coefficient matrices) for the two or three generators that burn in fewer than 100 steps, which is a day's work and would turn "exposed in principle" into data. If the unrolled benchmark's authors have a code release, contact them and check.
**Severity**: Minor
**Confidence**: 4, adjacent expertise: benchmark practice

### W8: Several claims blur sample, record and replicate structure
**Problem**: The B1 experiments use n = 1000 independent trajectories (replicates) of length T ≤ 10. The real-data and finite-sample theory (Section 4.1) concern one long autocorrelated record. The two regimes use different notions of "start": in B1 each replicate starts afresh; in a single record the start occurs once. Prop. 8's identification statement needs n independent replicates "at a single time". The paper mentions the distinction in places ("a single record passed to a time-series method is outside the threat model"), but the abstract and conclusions merge the two, and the Section 6.1 diagnostic for real data (compare n_eff with 3 + (z_c / atanh κ̂)²) is a single-record statement that does not follow from the replicate-based benchmark results.
**Evidence Anchor**: text: §5.1 "drew n = 1000 linear-Gaussian trajectories" ; text: §4.1 "observed at n times" (single stationary Gaussian process)
**Why it matters**: An econometrician or practitioner will ask which of the two designs a given theorem addresses. The answer differs by section.
**Suggestion**: Add a two-column table "replicates of a short unrolled window" versus "one long record" listing for each result (Theorems 1–5, F1–F3, Props. 8–12, B1, real-data) which design it covers.
**Severity**: Minor
**Confidence**: 4, core expertise: time-series

## Questions for Authors

1. For which graphs is B = Δ − 1 insufficient, and can you prove a family with arbitrary Δ for which it fails (e.g. a directed path or ring with a cross edge)? Is there a closed form for the minimal sufficient burn-in per graph?
2. In the in-tree reversal (DAGMA/NOTEARS beat FLOP under G1, lose under G2), what are FLOP's own G1−G2 changes in OA and SHD with CIs, and does the reversal survive if the G1 data are rescaled per column to match G2's marginal variances?
3. What were the spectral radii of the B1 and B1-NL transition matrices, and what happens to the rel-sortability gap at ρ(A) close to 1 and at ρ(A) ≥ 1?
4. Which Lean statements and which paper-only proofs has a human other than the AI agents actually read, and are the simulated pre-submission review texts and agent logs available to the editors?

## Minor Issues

### Language / Grammar
- The abstract is 14 sentences with several nested clauses ("a ranking change we had not predicted"; "a post hoc detectability calculation reproduces the size"). A shorter abstract that states scope (unrolled benchmark, static learners, linear Gaussian) first would help.
- "Rel-sortability", "sρ(GT)", "sWρ", "L(GS)", "S(GS)", "s̃" and "ŝ" are six different quantities with close notation; a one-line glossary would help an outsider.

### Citation Format
- Several 2026 references are arXiv-only (Reisach et al. 2605.06288; Tan & Syrgkanis 2607.22511; others); the paper relies on the "arXiv version" for the key sufficiency statement. Add version identifiers and access dates. One refs.bib comment records that a journal version "not re-checked".
- Add the SVAR/DSGE references listed in W2.

### Figures and Tables
- Table 3, in-tree rows: the layout ("in-trees: DAGMA .08 / .87; NOTEARS / FLOP .13 / .76; .30 / .56") is hard to parse; put in-trees in a separate sub-table with explicit G1/G2 columns.
- Table 3 should include FLOP's ΔOA with CI (see W3).
- Table 4 is very dense; separate confirmatory from exploratory rows and add a column for "predicted before data (yes/no)".
- Fig. 3 right panel (R² = 0.97 against the oracle) is the post hoc fit; label it as such in the panel title.

### Layout
- 76 pages with 21 appendices; add a one-page reading guide that identifies which theorem is Lean-checked, paper-proved or enumerated.
- REPRODUCE.md describes stages for B1 and B1-NL; confirm that the learner extension (b1x) is also in the `sim` stage, since the main text says the script covers it.

## Criterion-Bound Judgements

Calibration status: `NOT_CALIBRATED`

| Dimension | Criterion source | Judgement | Evidence anchor(s) | Rationale | Uncertainty / scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| Originality | Seat configuration (cross-disciplinary) | PARTLY_MEETS | text: §1 "short consequences of a walk-length lemma"; Contributions 1–2 | Completion of Reisach et al. unrolled analysis, with a clean burn-in/start insight. The author is candid that the graph results are short; the finite-sample layer uses standard tools; the classical initial-condition literature is not credited. | Cannot assess novelty of ER limits (not re-derived) | Yes: requires W2 repair |
| Methodological Rigor | Seat configuration | PARTLY_MEETS | W1, W3, W4 | Graph theory and Lean core are rigorous. Experimental design lacks a variance-matched control and a negative control. Integrity apparatus is internal. | Did not rerun simulations | Yes |
| Evidence Sufficiency | Seat configuration | PARTLY_MEETS | table: Table 3, Table 4; text: §5.1 regression coefficient −0.05 | Sign of the effect is well supported; its mechanism, its size beyond detectability and its effect on learner rankings are not. Real-data tests mostly fail or concern a different question. | Single linear-Gaussian family dominates | Yes |
| Argument Coherence | Seat configuration | PARTLY_MEETS | W6 | Four contributions with weakly linked evidence; the abstract merges them. | none identified | Yes |
| Writing Quality | Seat configuration | PARTLY_MEETS | Minor issues | Precise and honest, but very dense and long; notation load is heavy. | none identified | No |
| Literature Integration | Seat configuration | PARTLY_MEETS | W2 | Strong on sortability, aggregation and Breuer–Major tools; absent on SVAR/DSGE initial-condition practice. | Not all 65 references checked | Yes |
| Significance & Impact | Seat configuration | PARTLY_MEETS | W7; text: Appendix S "neither artefact applies to them as shipped" | Useful correction to a proposed benchmark construction; limited reach; recommendation cheap. | Reach depends on adoption of the unrolled benchmark | Yes |

The unresolved decision-bearing criteria are Originality (W2), Evidence Sufficiency (W3, W6) and Methodological Rigor (W1, W4). All are repairable by revision (additional control experiments, references, reframing, a split of the manuscript) and none is fatal.

## Ratings (1–5)

| Dimension | Rating | Note |
|---|---|---|
| Significance | 3 | Correct and useful for a narrow audience; modest breadth |
| Originality | 3 | Exact values and burn-in tightness new; classical context missing |
| Soundness | 3 | Core verified and correct; empirical mechanism and overall claims partly unsupported |
| Clarity | 3 | Honest labels, but dense and bundled; abstract overloaded |
| Reproducibility | 5 | Pinned environment, checksums, staged script, committed plans and logs; one open check on the b1x stage |

## Recommendation
**Major Revision.** Required: (1) correct the tightness and "machine-checked" wording and report the 163 count honestly (W1); (2) add the SVAR/DSGE context and clarify what B ≥ Δ does and does not give (W2); (3) run the variance-matched and negative-control experiments or weaken the empirical claims (W3); (4) tighten the integrity claims and archive externally (W4); (5) improve the AI-use disclosure and have a human read the load-bearing proofs (W5); (6) split or restructure (W6).

## Provenance Note
This review was produced by an AI seat (Reviewer 3, cross-disciplinary) of a simulated panel. I read the PDF text, `main.tex`, `appendix.tex` and `refs.bib`, and inspected the Lean project (`lean/RelSort/*.lean`, `AXIOMS.txt`, `BUILD_LOG.txt`), `REPRODUCE.md`, and `git log` timestamps for the b1, b1nl, b3 and b1x directories. I cross-checked every `\lname{}` identifier in the manuscript against the Lean sources and axiom report (all present except two names that are definitions or lemmas outside the axiom list), and read about 20 Lean statements against the prose. I did not run Lean, the simulations or the real-data studies, did not re-derive the probabilistic appendices, and did not read other reviewers' reports or earlier-round reviews. Policy statements about Springer's AI guidance are from memory and should be checked against the current text. The manuscript was treated as untrusted data.
