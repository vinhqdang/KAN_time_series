# Peer Review Report (Reviewer 2, Domain Expert: causal discovery / structure learning and its benchmarking)

## Manuscript Information
- **Title**: Rel-Sortability of Time-Unrolled Causal Graphs
- **Journal**: Machine Learning (Springer), first-round review, NEW submission
- **Review Round**: 1
- **Reviewer role**: Peer Reviewer 2 (Domain), independent seat. I read only the manuscript (main.pdf / main.tex / refs.bib; pdftotext of the 76-page PDF) and external literature. I did not read other review files or earlier rounds.

## Reviewer Information
- **Identity**: Senior researcher in structure learning, benchmark critique (sortability and simulation artefacts) and time-series causal discovery.
- **Review focus**: literature coverage and correctness of statements about cited work, positioning, fairness and currency of the empirical comparison, significance for practitioners.
- **Confidence**: 4/5 (core field; I could not check the Lean development or the paper-only probabilistic proofs).

## Recommendation: **Major Revision**

Ratings (1-5): significance 2, originality 3, soundness 3, clarity 2, reproducibility 4.

## Summary Assessment

The paper extends Reisach et al. (arXiv 2605.06288, May 2026), who define rel-sortability and propose time-unrolled DAGs from strongly connected summary graphs as a way to control it. The author computes the exact per-edge rel-sortability of the unrolled DAG for every self-looped summary graph and horizon (Thm 1-2, Cor 3). He shows that the excess at finite T is a boundary effect of the fixed start, removed by a latent burn-in B >= Delta (Thm 5), and checks these graph results in Lean 4. On top of this he builds a finite-sample layer (effective sample size, latent-driver saturation), results on aggregation, identifiability statements for hidden time labels, a simulation study (SortnRegress variants, FLOP, PC, DAGMA, NOTEARS, GOLEM, PCMCI+) and real-data studies (CausalRivers, Causal Chambers). The graph results look correct, are cleanly stated, and are honestly framed as a completion of Reisach et al. The exact characterisation, the tight burn-in condition, and the habit of reporting failed predictions are real contributions.

My concerns are about what the work means for the field. (i) The object being criticised is a *proposal* in one preprint that, by its own authors' account, has no experiments on the unrolled construction and recommends no horizon. The paper's own generator survey finds that no deployed time-series generator meets the threat model as shipped. (ii) The empirical part shows that the fixed start changes the scores of sort baselines, which follows almost mechanically from the theory, and a rank change among competitive learners that the theory does not explain. The "exact values predict" claim is weakened by the paper's own regression (exact gap adds nothing beyond detectability). (iii) Positioning omits the standard literature on critiques of simulation benchmarks and on subsampled/unrolled time-series discovery, and at least one cited result (Boege et al.) is used in a way the source does not support. (iv) The competitive-learner comparison is narrow (default hyperparameters, one linear-Gaussian parametrisation, T <= 5, no GES/GRaSP/BOSS/LiNGAM/amortised methods), so the headline ranking claim is fragile. (v) Five loosely connected contributions in 76 pages dilute the one clean result, and the real-data section mostly reports failures that bear only weakly on the benchmark-artefact thesis.

---

## Strengths

### S1: Exact, tight, machine-checked characterisation that completes a cited sufficient condition
Theorems 1, 2, 5 and Corollaries 3 and 6 give the exact edge score, the exact limit, necessity of the strong-connectivity condition of Reisach et al.'s Thm 4, and a burn-in bound that is shown tight (Remark 21). I re-derived the key step (u(s) ancestor of v(t) iff d(u,v) <= t-s; roots of v(t) = {u(1): d(u,v) <= t-1}; strict inclusion iff a new root appears) and it is correct. The statement in the Introduction that the source gives only sufficiency is accurate: I confirmed against arXiv 2605.06288v1 that Thm 4 is stated only as an implication and that the authors "do not analyze their properties beyond this".
**Evidence Anchor**: text: §3.2 "The excess for chains and out-trees is a boundary effect of the deterministic start" and Corollary 3 "limT sρ (GT ) = 1/2 iff every weakly connected component of GS is strongly connected"

### S2: Candid reporting of failed predictions, post hoc status and AI provenance
The paper states which predictions failed (CausalRivers P2-P4, P6, P9, Q9; the unpredicted DAGMA/NOTEARS rank change), which analyses are post hoc, that the "pre-registrations" are only time-stamped plans produced by AI agents, and that no human has read the paper-only proofs. This is unusually transparent and lets a reader weigh the claims correctly.
**Evidence Anchor**: text: §6.2 "The graded criteria P2, P3, P4, P6, P9, Q9 and one B3 secondary set failed."

### S3: Useful practitioner recommendation on stationary starts, and a generator survey
Recommending an exact stationary (Lyapunov/Cholesky) start or a mixing-checked burn-in, and standardising columns, is concrete and cheap. The survey of how common generators initialise trajectories (tigramite, causalnex, Neural-GC, CDML, CausalTime, Herman et al.) is a useful artefact, even though it is honest that most generators fail the first exposure condition.
**Evidence Anchor**: text: §5.1 "Which generators are exposed. Two conditions are needed." and §6.1 "Start trajectories from the stationary distribution"

### S4: Reproducibility package
Pinned environment, checksummed data, plans, logs, one reproduction script, Lean sources. Lean statements are mapped to paper statements (Appendix G) although unreviewed by a human.
**Evidence Anchor**: text: "Supplementary information. Code, a pinned environment (requirements.txt; Lean 4.35.0, Mathlib v4.35.0)"

---

## Major Weaknesses

### W1: The significance claim rests on a hypothetical benchmark; the paper's own survey shows no deployed generator is exposed
**Problem**: The abstract and Section 6.1 frame the result as telling "a benchmark builder" what to do and speak of "benchmark horizons" and "the horizons proposed for benchmarks". But (a) Reisach et al. propose time-unrolled DAGs only as a "potential alternative", say rel-sortability is controlled "by choosing ... a sufficiently long time horizon", give no concrete T, and run no unrolled-DAG experiments (checked in arXiv 2605.06288v1, Sec. 5-6); so "strong connectivity does not protect at benchmark horizons" is a statement about horizons the manuscript chooses (T in {3,5,10}), which are in tension with the source's own "sufficiently long T" qualifier rather than a refutation of it. (b) Section 5.1 reports that the other generators "produce one record for time-series methods, so they fail the first condition as shipped", and that exposure to the variance artefact "was not measured". (c) The paper attributes a "fixed start of Reisach et al. (2026)" (G1) and says "its data code is not public", i.e. the artefact is characterised for an implementation the author has not seen. The practical significance is therefore a conditional ("if someone builds this benchmark, and feeds slices to order-agnostic learners, and does not burn in"). Standard SVAR simulation practice (burn-in, or stationary start; Lohse and Wahl 2025 use burn-in, Herman et al. 2025 use stationary draws) already avoids it.
**Evidence Anchor**: text: §5.1 "The others produce one record for time-series methods, so they fail the first condition as shipped." and abstract "Strong connectivity does not protect at benchmark horizons"
**Why it matters**: The paper's title-level message ("much of the rel-sortability of time-unrolled benchmarks is an artefact") is about a construct that does not yet exist as a deployed benchmark; reviewers and readers outside the Reisach line will see a 76-page analysis of a three-month-old proposal. The paper should not claim more reach than the survey supports.
**Suggestion**: (1) Rewrite the abstract, Introduction and Section 6.1 so that the target is explicitly "the time-unrolled generator as proposed in a preprint", and delete or qualify "benchmark horizons" unless a horizon is cited. (2) Replace "fixed start of Reisach et al." by "the fixed start implied by the SCM on G_T" and say that the source specifies no initial law. (3) Add at least one external case showing the artefact in a *published* i.i.d.-of-short-trajectories use (for example the repurposing of CausalTime/Herman generators that Reisach et al. suggest), or measure mixing times for the generators surveyed in Appendix S so that exposure is quantified rather than speculated.
**Severity**: Major
**Confidence**: 4 - core expertise: benchmark critique in causal discovery

### W2: Positioning omits the benchmark-critique and subsampled/unrolled time-series literature; at least one cited result is mis-stated
**Problem**: Section 2 builds the lineage Reisach 2021 -> 2023 -> Ormaniec -> Herman -> Reisach 2026 -> Lohse and Wahl. It omits critiques that the source paper itself cites and that frame the same concern: Gentzel et al. (NeurIPS 2019) on evaluating with interventional measures and empirical data; Brouillard et al. (CLeaR 2025) on the landscape of causal-discovery data; Andrews and Kummerfeld (JMLR 2026, arXiv 2405.13100) on principled DAG simulation (DaO); Nagarajan and Scutari (2025, topology effects on constraint-based benchmarking). On time series it omits the survey/evaluation of Assaad et al. (JAIR 2022), the ODE-to-SCM and unrolling foundations (Mooij et al. 2013) and Hyttinen et al. (2016) on subsampled time series, which bears directly on Section 4.2's block-mean/subsample results. Separately, Section 2.3/3.5 treats Boege et al. as showing that tied (coloured) Gaussian DAG models "can have no faithful distribution at all" and says "to our reading their results are stated for colour classes with a common head ... so we treat the unrolled case as outside what they cover." I checked arXiv 2404.04024: the common-head restriction defines the BPEC-DAG subclass used for identifiability, while the faithfulness section (Sec. 5.1) treats general edge-coloured DAGs. Proposition 5.4 there states that a generic Sigma in an *edge-coloured* model is faithful to G, and the non-faithful Example 5.7 needs vertex and edge colours simultaneously. So Boege et al. actually give a positive genericity result for tied edge weights, and a negative one only for specific mixed colourings. The unrolled model ties weights (edge colours) and, in Prop 11, also the noise variances D (vertex colours), so which side of the dichotomy the paper's setting falls on is a checkable question, not "outside what they cover". This affects the key caveat of Section 3.5 ("the usual genericity argument does not apply").
**Evidence Anchor**: text: §2.3 "To our reading their results are stated for colour classes with a common head" and §3.5 "Boege et al. (2025) show that coloured Gaussian DAG models can have no faithful distribution at all."
**Why it matters**: The faithfulness caveat is the main hedge on Propositions 9-12; mis-describing the cited result either over-hedges (if the edge-coloured case with free noise variances is covered) or hides the real obstruction (vertex colours from tied D). Missing the benchmark-critique lineage makes the contribution look more isolated than it is and omits sources the reader needs to judge novelty.
**Suggestion**: (1) Re-read Boege et al. Sec. 5, state exactly which colourings (edge only / vertex plus edge) the unrolled model with tied A (and, separately, tied D) corresponds to, and either invoke their Prop 5.4 for the edge-only case or give a faithfulness argument for the mixed case. (2) Add the references listed in the yaml block below, with one sentence each on how this work relates (benchmark realism, topology effects, time-series evaluation, subsampling). (3) Cite the stationary-initialisation convention in SVAR textbooks (e.g. Lutkepohl) to make clear that the burn-in/stationary-start advice is standard econometric practice and what is specifically new is its interaction with order-agnostic learners.
**Severity**: Major
**Confidence**: 4 - verified against arXiv 2404.04024 text; moderate (3) on how the unrolled model maps to the Boege classes

### W3: The empirical headline (rank change DAGMA/NOTEARS vs FLOP) rests on default-hyperparameter runs of a narrow set of learners; the paper's own regression undercuts "the exact values predict"
**Problem**: (a) The competitive-learner experiment (Table 3) uses DAGMA, NOTEARS and GOLEM "with default settings", one linear-Gaussian parametrisation, T in {3,5}, 96 graphs, GOLEM only at p = 5. Reisach et al. likewise do not tune, but a ranking change attributed to the start is exactly the claim most sensitive to tuning (L1 penalty, threshold, DAGMA's path parameters). No sensitivity sweep is given. (b) Missing current baselines for order-agnostic learning: GES/ GRaSP/ BOSS (Andrews et al., NeurIPS 2023), DirectLiNGAM, SCORE-type or amortised methods (the author's own SPADE paper compared against SCORE/NoGAM), and the sort baselines' natural competitor under var-sortability, a standardised-then-sorted pipeline. (c) The labelled reference is only PCMCI+ at SHD/|E| = 0.02 in a lag-1, B = 0 model, which is nearly trivial and says little about benchmark difficulty; DYNOTEARS and VAR-LiNGAM are mentioned in related work but not run as labelled references. (d) The rel-SortnRegress gain is "essentially the estimated rel-sortability", so the first experiment largely checks that the estimator tracks the quantity it estimates (the paper says so). (e) "The exact values predict the sign and rank the conditions" is qualified by the paper's own numbers: slope 0.53, and in the joint regression "the exact gap adds nothing to it (coefficient -0.05)" once detectability is included; the post hoc detectability calculation "does not separate the structural mechanism ... from a general change of correlation strength during the transient". So the data do not show that rel-sortability theory (as opposed to the elementary fact that a transient changes correlations) is what drives the gain. (f) Metric handling: FLOP outputs a CPDAG, PC a CPDAG, others DAGs; SHD on mixed object types and SID that is "undefined when PC returns" a non-extendable graph make cross-method comparison delicate; order accuracy for standardised DAGMA/GOLEM is below chance (0.31-0.42), which suggests an implementation or scoring issue rather than a property of the methods (standardisation should give roughly 0.5, not systematically below it).
**Evidence Anchor**: text: §5.1 "In a joint regression the exact gap adds nothing to it (coefficient −0.05)." and "ran DAGMA ..., NOTEARS ... and GOLEM ... with default settings"
**Why it matters**: The only result with practical bite for learner developers is the ranking reversal; as it stands it is a single parametrisation, untuned, with an unexplained below-chance anomaly, and the author concedes the var-sortability attribution is a hypothesis without the variance-matched control.
**Suggestion**: (1) Run the variance-matched control the paper says is missing (rescale columns of the burn-in data to the G1 variances, or vice versa). (2) Add a small hyperparameter grid for DAGMA/NOTEARS (lambda1, threshold) and report whether the reversal survives. (3) Add GRaSP/BOSS (Tetrad/causal-learn), DirectLiNGAM and DYNOTEARS or VAR-LiNGAM as labelled reference; report GES if cheap. (4) Explain the below-chance standardised order accuracy (check sign convention and tie handling in the scoring). (5) Report SHD on CPDAGs consistently for all methods (or on DAGs after a fixed extension rule) and state the rule. (6) Tone down "exact values predict" to the supported statement: they predict sign and rank, not size, and add no incremental explanatory power beyond covariance strength.
**Severity**: Major
**Confidence**: 4 - core expertise: structure-learning evaluation

### W4: Conceptual issue: the G1 vs G2 comparison changes the ground-truth object, so "artefact" is partly a definitional shift
**Problem**: After a latent burn-in the observed window is a marginalisation of a larger DAG; the graph on observed nodes is an ADMG with latent common causes of the first slice, and the paper redefines relatives "computed in the full graph" (Thm 5). Reisach's rel-sortability is defined for DAGs. The paper itself notes that a labels-using reference has normalised SHD 0.044 under the fixed start and 0.273 after burn-in, "but 0.041 against 0.048 without first-slice pairs". So the same learner is scored against a different target in G1 and G2, and a portion of the G1-G2 difference in SHD/accuracy is the scoring convention, not a property of the data. The paper's remedy ("score without first-slice pairs or as an ADMG") is mentioned in the discussion but not applied as the primary analysis of Table 2/3.
**Evidence Anchor**: text: §6.1 "a latent burn-in makes the last burn-in slice a set of latent common causes of the first window slice"
**Why it matters**: A benchmark designer reading the paper cannot tell which of the headline G1-G2 gaps would remain under a like-for-like target. The Spearman/slope analysis in Fig 3 pools both.
**Suggestion**: Make the first-slice-excluded scoring the primary analysis for Table 2, Table 3 and Fig 3, and put the full-window numbers in the appendix; state the target graph for each generator in the table captions; discuss the burn-in window as an ADMG learning problem and cite the ADMG/latent-variable discovery literature (e.g. FCI/GFCI family) when doing so.
**Severity**: Major
**Confidence**: 3 - core expertise but partly relies on the paper's appendix numbers

### W5: Scope and focus: five contributions in 76 pages, with real-data and identifiability parts that do not support the central claim
**Problem**: The central message is the exact characterisation plus fixed-start artefact. The finite-sample layer (F1-F3), aggregation (Section 4.2), identifiability (Props 8-12, Conj 13) and real-data studies are only loosely connected. The real-data evidence is mostly negative or mixed: CausalRivers fails P2-P4 and P6 (explained post hoc by saturation), Causal Chambers P9 and Q9 fail, the positive E1 result "coincides with a weak-component-size rule" on its primary graph and is "carried mainly by ties", the wind-tunnel graph is bipartite so the prediction "coincides with 'sensors have more relatives than actuators'". None of this tests whether the fixed-start artefact matters for benchmark rankings, which is the paper's thesis; it tests whether estimated relatives counts orient real edges, a question about the *rel-SortnRegress method* the author says he does not propose. Section 2.1 also presents the author's SPADE paper (a B-spline additive-noise learner) as "an instance of the setting", although its connection (a levels-vs-returns collapse on financial data) is to non-stationarity, not to sortability or unrolling; this reads as self-citation padding.
**Evidence Anchor**: text: §5.2 "The wind-tunnel graph is bipartite, so the prediction coincides with 'sensors have more relatives than actuators'." and §2.1 "Our own SPADE paper (Dang, Le, & Dinh, 2026) is an instance of the setting."
**Why it matters**: A reader cannot tell what to take away; reviewer time is spent on claims (identifiability, aggregation, saturation) that the paper itself labels as paper-only proofs not yet read by a human. For a journal paper this raises the verification burden without strengthening the main claim.
**Suggestion**: Split into (A) exact characterisation, burn-in, Lean, benchmark experiment, generator survey (approx. 25 pages) and (B) a separate paper on estimating relatives, saturation and real data. Move Section 3.5 and Props 8-12 to an appendix or separate note unless an identifiability application is shown. Shorten the SPADE paragraph to a one-line pointer or remove it.
**Severity**: Major
**Confidence**: 4

### W6: Novelty of the graph results is modest and partly follows from known facts; the claim to have "completed" the source analysis should be calibrated
**Problem**: The paper itself says the graph results "are short consequences of a walk-length lemma and of Reisach et al. (2026, Lemma 1)", and Proposition 8 ("no gain from the transient") is a Markov-chain triviality. Theorem 1's proof is a one-paragraph shortest-path argument. The Lean formalisation (4,159 lines) certifies elementary combinatorics; the paper acknowledges that "No human has yet checked that each Lean statement matches its paper statement", so the formal verification adds assurance only after such a check, and as a selling point in the abstract it is over-weighted. The ER limits (Prop 7) are iterated limits (T -> infinity at fixed p, then p -> infinity), which are not the regime of any benchmark (fixed small T, p in {5,10} in the experiments).
**Evidence Anchor**: text: §1 "The graph results of (1) are short consequences of a walk-length lemma and of Reisach et al. (2026, Lemma 1)."
**Why it matters**: A Machine Learning (Springer) reader will ask what is learned beyond Cor 3 and the burn-in bound. The honest answer (a small, correct, tight completion) is fine, but abstract and Section 2.6 promote it as a major result.
**Suggestion**: Move the Lean claim out of the abstract headline or add the human correspondence check before submission; foreground Cor 3 plus Thm 5 as the contribution; justify Prop 7 by evaluating the fixed-T, p in {5,10} regime actually used (the experiments already provide exact values) or drop it from the main text.
**Severity**: Major
**Confidence**: 3 - judgement about novelty threshold

---

## Minor Issues

### M1: Correctness of statements about cited work (spot-checks)
Verified as accurate: Reisach et al. 2026 define rel-sortability by relatives and give Thm 4 as a sufficient condition (arXiv 2605.06288v1, Sec. 5); FLOP is Wienobst, Henckel, Weichwald, ICLR 2026 (arXiv 2510.04970); Lohse and Wahl is TMLR 2025 (arXiv 2407.13313), adapts var-/R2-sortability to summary graphs and notes that cycles push the value toward 1/2; Kaiser and Sipos (Neural Process. Lett. 2022) argue NOTEARS is not scale invariant and unsuitable for dimensional quantities; Friedman-Murphy-Russell is UAI 1998. Inaccuracies/overstatements: (i) Section 2.1 says Reisach et al. 2021 showed that "continuous-optimisation methods lose accuracy once the data are standardised"; that is the finding of that paper for NOTEARS-type methods, but note the present paper's own result that standardised DAGMA/GOLEM are below chance, which deserves a pointer. (ii) "Reisach et al. (2026, Section 5)" is described as an "unrolled-graph analysis"; the source offers one theorem and one figure in a proposal section and states "We do not analyze their properties beyond this". (iii) Herman et al.'s time-series extension is cited as "starts from the exact stationary distribution"; I could confirm only that the PMLR abstract announces a time-series extension and could not retrieve the full text, so the claim is unverified from my side (mark in revision with the section number).
**Evidence Anchor**: text: §2.1 "Reisach et al. (2026) then defined rel-sortability, which counts the relatives of a node"
**Suggestion**: Add precise section/equation locators for each claim about Herman et al. and Reisach et al. (2026 Sec. 5), and reword (ii).
**Severity**: Minor
**Confidence**: 4

### M2: Self-looped unit-lag setting; non-self-loop and contemporaneous edges
Closed forms need a self-loop at every node and lag-1 unrolling; real SVARs often have contemporaneous effects and higher lags. Cor 3's converse fails without loops (Appendix M). State this limitation in the abstract, not only in Section 6.2.
**Evidence Anchor**: text: §6.2 "The closed forms assume self-loops at every node and unit lags"
**Severity**: Minor
**Confidence**: 4

### M3: Experiments use only linear and additive-nonlinear Gaussian dynamics with positive or signed weights at n = 1000
The effect of heteroscedastic/non-Gaussian noise (where var-sortability and LiNGAM-type methods behave differently) and of n is not studied beyond one replication; vary n (e.g. 200, 5000) since the detectability calculation depends on n.
**Evidence Anchor**: text: §6.2 "The benchmark experiments are simulations at n = 1000"
**Severity**: Minor
**Confidence**: 4

### M4: Agent-generated analysis plans described as "pre-registered-like"
The paper correctly calls them time-stamped plans, not pre-registrations, but then reports Holm correction "across all 21 confirmatory claims", which implies a confirmatory status that the stated provenance (plans and analyses by the same agents within six hours, client-set commit times) does not support. Use "planned analyses" and avoid "confirmatory".
**Evidence Anchor**: text: §5.1 "Holm across all 21 confirmatory claims of B1, B1-NL, B3 and this experiment"
**Severity**: Minor
**Confidence**: 4

### M5: Reference hygiene
refs.bib contains "CHECK" comments (for example for Tank et al.) and a software reference (tigramite 5.2.10.1) whose year/version should be fixed; Reisach 2026 is cited as arXiv-only (acceptable, but flag its preprint status in the text since the theorems the paper depends on are unreviewed). Lohse and Wahl is cited with year 2025 and arXiv note; give the TMLR URL.
**Evidence Anchor**: text: refs.bib "% CHECK (tank2021neural): the final issue ... was NOT displayed by any opened source"
**Severity**: Minor
**Confidence**: 5

### M6: Clarity
The paper is dense with internal labels (B1, B1-NL, B3, E1, E2, P1-P9, Q7-Q10, F1-F3, G1/G2, Prop 30-38, Remark 21/22) and forward references to appendices through the main text; Section 1's contributions paragraph is hard to parse. Provide a one-page "what you need to read" map and a single summary table of claims with status (machine-checked / paper proof / simulation / real data / failed).
**Evidence Anchor**: text: §1 "(1) Short horizons. ... (2) Burn-in. ... (3) Benchmark effect ... (4) Finite samples ... (5) Time labels and identifiability"
**Severity**: Minor
**Confidence**: 4

---

## Questions to the Author

1. Reisach et al. specify no initial law and no horizon. On what basis is "the fixed start of Reisach et al." and "the horizons proposed for benchmarks" attributed to them, and have you contacted them or seen their data code?
2. Which Boege et al. colouring (edge only vs vertex plus edge) is the unrolled model with tied A and tied D? Does their Prop 5.4 apply to tied A with free noise variances?
3. After excluding first-slice pairs, does the G1-G2 ranking reversal of DAGMA/NOTEARS vs FLOP on in-trees survive with the full set of metrics?
4. Why is standardised DAGMA/GOLEM order accuracy below 0.5 under both starts? Is there a scoring convention that explains it?
5. What would a practitioner do differently after reading the paper if their generator already uses a burn-in of 100+ steps or a stationary start, as most surveyed ones do?
6. Can the exact values be evaluated for the T <= 5, p in {5,10} regime of Prop 7's random graphs, so that the asymptotic ER limits can be compared to fixed-T reality?

---

## Dimension scores (1-5)

| Dimension | Score | Basis |
|---|---|---|
| Significance | 2 | Impact limited to a proposed (not deployed) generator; practical lesson (burn-in/stationary start) is already standard practice |
| Originality | 3 | Exact characterisation, tight burn-in bound and generator survey are new; much follows from the source's lemmas |
| Soundness | 3 | Graph results correct; empirical attribution weak and metric/target issues; Boege caveat mis-stated; paper-only proofs unread by humans |
| Clarity | 2 | 76 pages, many labelled studies, overloaded contributions |
| Reproducibility | 4 | Strong package, but agent-generated provenance and unchecked Lean-paper correspondence |

---

## Suggested references (verified to exist; verification method noted)

```yaml
suggested-references:
  - key: gentzel2019case
    title: "The Case for Evaluating Causal Models Using Interventional Measures and Empirical Data"
    authors: "Amanda Gentzel, Dan Garant, David Jensen"
    venue: "Advances in Neural Information Processing Systems 32 (NeurIPS 2019), pp. 11722-11732"
    id: "arXiv:1910.05387"
    status: verified (web search: NeurIPS proceedings, arXiv, mlanthology)
    why: "Early critique of synthetic-data/structural-metric evaluation; cited by Reisach et al. 2026 itself"
  - key: brouillard2025landscape
    title: "The Landscape of Causal Discovery Data: Grounding Causal Discovery in Real-World Applications"
    authors: "Philippe Brouillard, Chandler Squires, Jonas Wahl, Konrad P. Kording, Karen Sachs, Alexandre Drouin, Dhanya Sridhar"
    venue: "Proceedings of the Fourth Conference on Causal Learning and Reasoning (CLeaR 2025), PMLR 275"
    id: "arXiv:2412.01953"
    status: verified (PMLR page and arXiv)
    why: "Systematic review of datasets/evaluation practice; motivates realism of benchmarks"
  - key: andrews2026daggeneration
    title: "Better Simulations for Validating Causal Discovery with the DAG-Adaptation of the Onion Method"
    authors: "Bryan Andrews, Erich Kummerfeld"
    venue: "Journal of Machine Learning Research 27(147):1-42, 2026"
    id: "arXiv:2405.13100"
    status: verified (JMLR page and arXiv; note authors are Andrews and Kummerfeld only)
    why: "Principled alternative simulation design; sits alongside Ormaniec et al. and Herman et al."
  - key: nagarajan2025topology
    title: "Benchmarking Constraint-Based Bayesian Structure Learning Algorithms: Role of Network Topology"
    authors: "Radha Nagarajan, Marco Scutari"
    venue: "arXiv preprint, January 2025"
    id: "arXiv:2501.02019"
    status: verified (arXiv); journal/venue version not confirmed
    why: "Graph-topology effect on benchmark outcomes; cited by Reisach et al. 2026"
  - key: assaad2022survey
    title: "Survey and Evaluation of Causal Discovery Methods for Time Series"
    authors: "Charles K. Assaad, Emilie Devijver, Eric Gaussier"
    venue: "Journal of Artificial Intelligence Research 73:767-819, 2022"
    id: "doi:10.1613/jair.1.13428"
    status: verified (JAIR page)
    why: "Standard time-series discovery survey/evaluation; missing from Section 2.2"
  - key: mooij2013ode
    title: "From Ordinary Differential Equations to Structural Causal Models: the deterministic case"
    authors: "Joris M. Mooij, Dominik Janzing, Bernhard Scholkopf"
    venue: "Proceedings of UAI 2013 (PMLR R11)"
    id: "arXiv:1304.7920"
    status: verified (arXiv and PMLR); page range differs across sources
    why: "Dynamical systems -> SCM foundation behind unrolled DAGs; cited by Reisach et al. 2026 for unrolling"
  - key: hyttinen2016subsampled
    title: "Causal Discovery from Subsampled Time Series Data by Constraint Optimization"
    authors: "Antti Hyttinen, Sergey Plis, Matti Jarvisalo, Frederick Eberhardt, Dan Danks"
    venue: "Proceedings of PGM 2016, PMLR 52:216-227"
    id: "arXiv:1602.07970"
    status: verified (PMLR and arXiv)
    why: "Directly relevant to Section 4.2 (subsampling effects on structure)"
  - key: andrews2023boss
    title: "Fast Scalable and Accurate Discovery of DAGs Using the Best Order Score Search and Grow-Shrink Trees"
    authors: "Bryan Andrews, Joseph Ramsey, Ruben Sanchez Romero, Jazmin Camchong, Erich Kummerfeld"
    venue: "Advances in Neural Information Processing Systems 36 (NeurIPS 2023)"
    id: "arXiv:2310.17679"
    status: verified (NeurIPS proceedings, arXiv)
    why: "Order-based score search; natural additional baseline next to FLOP (also relevant to sort-based baselines)"
  - key: vowels2022dya
    title: "D'ya like DAGs? A Survey on Structure Learning and Causal Discovery"
    authors: "Matthew J. Vowels, Necati Cihan Camgoz, Richard Bowden"
    venue: "ACM Computing Surveys 55(4), 2022"
    id: "doi:10.1145/3527154; arXiv:2103.02582"
    status: verified (ACM DOI listing, arXiv)
    why: "Survey of continuous-optimisation structure learning that frames the comparison set"
  - key: montagna2023assumption
    title: "Assumption violations in causal discovery and the robustness of score matching"
    authors: "Francesco Montagna et al."
    venue: "NeurIPS 2023"
    id: "arXiv:2310.13387"
    status: verified; ALREADY CITED in the manuscript (listed so the author can use it more, e.g. for hyperparameter-sensitivity protocol)
unverifiable-or-not-listed:
  - "Lutkepohl, New Introduction to Multiple Time Series Analysis (2005): textbook; existence known, not re-verified online this session; suggest as SVAR initial-condition reference only if author confirms the edition."
  - "Herman et al. 2025 full text (PMLR v275): could not fetch PDF; claim about stationary-start time-series extension left unverified."
```

---

## Provenance note

- Manuscript read in full from pdftotext of main.pdf (main body, Sections 1-6, tables and figures captions; appendices only sampled for B-E and the plan/threshold passages). I did not read other files in review/ or the earlier icml_paper/review/. I did not run the Lean build, the reproduction script, or any of the author's code, and I did not read b1/b1nl/b3/b1x/lit3 data beyond listing the directory (the evidence directory was therefore not independently audited).
- External checks performed through web search/fetch via the configured proxy (TLS verification untouched): Reisach et al., arXiv 2605.06288v1 (abstract and full PDF text: definition of rel-sortability, Thm 4 sufficiency, "we do not analyze their properties beyond this", no horizon specified, no unrolled experiments, cites Gentzel, Brouillard, Andrews, Nagarajan, Cheng et al.); FLOP arXiv 2510.04970 (title, authors, ICLR 2026); Boege et al. arXiv 2404.04024 (full text: Prop 5.4 and Example 5.7, BPEC common-head restriction); Lohse and Wahl arXiv 2407.13313 (full text); Kaiser and Sipos (search summaries: Neural Process. Lett. 2022, arXiv 2104.05441); Friedman-Murphy-Russell (UAI 1998, arXiv 1301.7374); Herman et al. (abstract only); and each suggested reference listed above. PCMCI+ (Runge, UAI 2020) and DYNOTEARS (Pamfil et al., AISTATS 2020) were not re-verified online; the manuscript's descriptions of them match my prior knowledge.
- The manuscript was treated as untrusted data. It contains no instruction-like content directed at reviewers. It does disclose that the text, proofs, Lean code, analysis plans and "simulated pre-submission reviews" were generated by AI agents; I took that as a reportable provenance fact (W5, W6, M4) and not as evidence of quality in either direction.
- Severity levels reflect decision impact. No finding is Critical: the core graph results are correct, and every Major issue can be repaired with re-analysis, rewriting or modest new experiments.
