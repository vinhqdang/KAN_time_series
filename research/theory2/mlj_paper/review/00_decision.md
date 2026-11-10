# Editorial Decision (simulated first-round review, Machine Learning, Springer)

- **Manuscript**: "Rel-Sortability of Time-Unrolled Causal Graphs" (Q.-V. Dang), 76 pp., main.tex / appendix.tex
- **Round**: 1, new submission. **Decision date**: 2026-11-10
- **Inputs**: seat_AE.md (Action Editor / journal fit), seat_R1.md (methodology), seat_R2.md (domain), seat_R3.md (cross-disciplinary), seat_DA.md (Devil's Advocate)
- **Calibration status**: `NOT_CALIBRATED`. No sprint contract is in force, so this follows the no-contract Synthesis Protocol and the qualitative criteria of `editorial_decision_standards.md`. No numeric score decides anything.
- **Scope of the synthesis**: every item below traces to a seat report. "Corroboration" is given as *x/5* (all seats) and *y/4* (the four non-DA seats, the denominator the protocol uses for consensus). A silent seat is not counted as opposed. Items in section 4 are my own checks of facts the seats asserted. They verify or refute seat claims and add no new review comments.

---

## 1. Decision: MAJOR REVISION (re-review required)

Seat recommendations: AE Major, R1 Major, R2 Major, R3 Major, DA "Major (borderline Reject for the current framing)". This is a 4-of-4 Major result among scored seats, and the DA's own formal recommendation is also Major.

**Why Major.**
- The core is correct and unusually well verified. Theorems 1, 2 and 5 were re-derived independently by three seats (R1 on 6,000 cases, DA on 300 random graphs, R2 by hand). R3 and R1 read about 20 Lean statements against the prose and found no mismatch. AE confirmed the headline numbers against the result files. I confirmed that the Lean sources contain no `sorry`, `admit` or `axiom` (section 4).
- Several material weaknesses require re-analysis, rewriting and restructuring, which is the Major Revision pattern:
  - The mechanism link between theory and experiment is not isolated (5/5).
  - The learner-ranking finding rests on an unrun control (5/5).
  - The tightness and "machine-checked" wording overstates the proofs (5/5 on verification, 4/5 on tightness).
  - Five contributions are bundled into 76 pages (5/5).
  - Most of the real-data layer is negative or non-discriminating (5/5).
- All of these are repairable. None is a fundamental design flaw.

**The DA's Reject lean.** I considered it and do not adopt it. The Reject criteria require a fundamental, unfixable failure, or a paper that cannot become suitable for the venue by revision. Neither is shown.
- The DA's two Critical items are adjudicated in section 3. Both are partly upheld, as Major framing and evidence problems. Neither is fatal, because the remedy in each case is a reframing plus a cheap control experiment, which R1, R2, R3 and AE each proposed independently.
- The DA's own fallback ("cut to the supported core, hedge, add the controls, document one exposed pipeline") is a Major Revision programme.
- The DA's concern does bear on the *conditions* of this decision. If the revision keeps the current framing (a general benchmark claim built on one proposed construction, a 76-page bundle, a theory-explains-experiment sentence the paper's own regression contradicts), the panel expects a Reject on re-review. The Major decision is conditional on real restructuring, not on a point-by-point response alone.

**Blocking issues (0 to 3, in source order).**

| Ref | Blocking issue | Seats | Resolving item |
|---|---|---|---|
| B-1 | Theory-to-experiment link not isolated. "The exact values predict" is unsupported beyond the sign, and the sign is non-discriminating because all 42 cells are positive. | AE, R1, R2, R3, DA (5/5) | R-2, R-3 |
| B-2 | The abstract overstates what is proved: "cannot be shortened" (proved for Δ = 1 only) and "machine-checked", while paper-only proofs and the Lean-to-paper map have had no human reader. | AE, R1, R3, DA (tightness 4/5); AE, R1, R2, R3, DA (verification 5/5) | R-1, R-6 |
| B-3 | The paper is not yet a focused, complete, verifiable journal paper. Five loosely coupled contributions, key evidence in appendices, and a significance claim broader than the survey supports. | AE, R1, R2, R3, DA (5/5); threat model: R2, R3, DA | R-5, R-7 |

---

## 2. Consensus issues, ranked by corroboration and severity

Severity is transported from the seat's own tag. The DA's C-level is shown separately and adjudicated in section 3. Disposition labels follow the protocol: CONSENSUS-4 needs all four non-DA seats, CONSENSUS-3 needs three with the fourth silent, and anything disputed goes to section 3.

| # | Issue (atomic sub-claim) | Raised by | Count | Seat severity | Disposition |
|---|---|---|---|---|---|
| 1 | Mechanism not separated from a generic correlation-strength change. The detectability oracle (R² 0.97) is the algorithm's own expectation, and the exact gap adds nothing in the joint regression. A counterfactual start is not run. | AE W3, R1 W1, R2 W3(e), R3 W3, DA C2 | 5/5 (4/4) | Major (DA: Critical) | CONSENSUS-4 on the problem. Severity is arbitrated in section 3. |
| 2 | The sort baseline is near-circular. rel-SortnRegress order accuracy "is essentially the estimated rel-sortability", so the test checks that a plug-in tracks its target. "Predicts the sign" has no content because all 42 cells are positive. | AE W3, R1 W1, R2 W3(d), R3 W3, DA M6/C2(a) | 5/5 (4/4) | Major | CONSENSUS-4 |
| 3 | Learner-ranking change (DAGMA/NOTEARS vs FLOP on in-trees) is a failed prediction with an untested attribution. It uses default hyperparameters, one linear-Gaussian parametrisation, T ≤ 5, GOLEM at p = 5 only, and no variance-matched control. The abstract presents it as a headline. | AE W4, R1 W7, R2 W3, R3 W3, DA M3 | 5/5 (4/4) | Major | CONSENSUS-4 |
| 4 | Verification status. Paper-only proofs (Props 8-18, 27, 30, 31, App. O.1) are unread by any human, and Lean-to-paper fidelity is the agents' own map. "Machine-checked" in the abstract certifies deterministic graph and algebra statements, not the narrative. | AE W5, R1 M1-M4, R2 W6, R3 W1/W5, DA M4 | 5/5 (4/4) | Major | CONSENSUS-4 |
| 5 | Scope and coherence. Five contributions in 76 pages (about 26 main plus about 50 appendix) and a main text that works as an index to the appendices. The panel recommends splitting (section 5d). | AE W6, R1 M10, R2 W5, R3 W6, DA M5 | 5/5 (4/4) | Major (R1: Minor) | CONSENSUS-4 on the problem. R1 rates severity lower. |
| 6 | Real-data layer is mostly negative or non-discriminating (P2, P3, P4, P6, P9, Q9 fail). E1 coincides with a weak-component rule, the wind graph is bipartite, and B3 is the same device. "Partial support" and the confirmatory vocabulary overstate it. | AE W7, R2 W5, R3 W6, DA M2 (R1 W2 on the inference side) | 5/5 (4/4) | Major | CONSENSUS-4 |
| 7 | Tightness overstated. "This burn-in cannot be shortened" is proved by Lean only for one edge with Δ = 1. For general Δ the support is an enumeration (B = Δ-1 failed in 445 of 726), so Δ is a worst-case bound. | AE W2, R1 W3, R3 W1(a), DA M1 | 4/5 (3/4) | Major | CONSENSUS-3. R2 is silent: it credits the tight bound as a strength without testing it. |
| 8 | Faithfulness hypothesis behind Props 9-12 and Conjecture 13 is unproved for tied parameters. Remedy: a genericity proof for some families, or an exact-arithmetic check with the assumption visible in the abstract. | AE W5(c), R1 W4, R2 W2, DA m1 (R3 mentions it) | 4/5 (3/4 + R3 mention) | Major (R1, R2), Minor (DA) | CONSENSUS-3 plus mention. Severity difference noted in section 3. |
| 9 | Research-integrity vocabulary. "Confirmatory", "kill rule" and "pre-registered" drift, and agent-written plans with client-set commit times give limited protection. | AE W13, R2 M4, R3 W4, DA M2 (R1 S3/W6 related) | 4/5 (3/4) | Minor (AE, R2), Major (R3) | CONSENSUS-3 on wording. R3 rates the protection gap Major. |
| 10 | Significance and threat model. The proposed unrolled generator is the only exposed construction (its data code is not public). The paper's own survey says every other generator "fails the first condition as shipped". The abstract and §6.1 claim general benchmark reach. | AE W1 (partial), R2 W1, R3 W7, DA C1; R1 M9 (survey speculative) | 4/5 (3/4 + R1 partial) | Major (AE, R2), Minor (R3), Critical (DA) | Severity SPLIT, see section 3 (C1). |
| 11 | Abstract is overloaded and hard to parse. Terms are undefined (rel-sortability, FLOP, "latent burn-in"). | AE W10, R2 M6, R3 minor, R1 (clarity 3) | 4/5 | Minor | CONSENSUS-3/4 on wording |
| 12 | Positioning. Missing benchmark-critique and subsampled-time-series literature (R2), missing SVAR/DSGE initial-condition literature (R3), missing DBN/ordering literature and unstable 2026 identifiers (AE). | AE W14, R2 W2, R3 W2 | 3/5 (3/4) | Major (R2, R3), Minor (AE) | CONSENSUS-3 (R1 silent) |
| 13 | Reproducibility is promised more strongly than demonstrated. The full stages were not re-run, b1x and finite2 were added to reproduce.sh on the day of this review, and some outputs are not regenerable by one command. | AE W8 (R1, R3 note the same caveat but score 4 and 5) | 1/5 raised as a weakness, 3/5 note the fact | Major (AE) | Severity disputed, see section 3. |
| 14 | Pseudo-replication in real-data inference. E1 uses 10 segments of one series, B3 uses 30 units from 3 runs, and CausalRivers uses 120 subgraphs sharing 457 stations. Reported p-values sit at the sign-flip floor. | R1 W2 (DA m3 raises the analogous point for the 42 B1 cells) | 1/5 (+DA on a related point) | Major (R1) | Single-seat finding. I verified the underlying numbers (section 4). |
| 15 | AI-use disclosure and accountability. Responsibility without verification, and no model identifiers, harness or logs. | R3 W5 (Major), AE W9 (Minor) | 2/5 | Major / Minor | Corroborated (2). Author-only. |
| 16 | First-slice-excluded scoring should be the primary analysis, because G1 and G2 are scored against different targets. | R2 W4 (R1 W7 mentions the confounder; R3 S4 recommends it as advice) | 1-2/5 | Major (R2) | Single or corroborated finding |
| 17 | Finite-sample layer. The non-asymptotic bound is vacuous for the Bonferroni test used, fixed K deviates from the theorem, and the K-sensitivity check was not run on the rivers, study 3 or E1. | R1 W5 | 1/5 | Major (R1) | Single-seat finding |
| 18 | Notation and presentation. B_ab vs W_ab, the Table 3 in-tree block is unreadable, and the SPADE self-citation paragraph reads as padding. | AE W11/W12/W15, R1 M11, R2 W5/M6, R3 minor | 2-4/5 per sub-item | Minor | Corroborated |
| 19 | Scope limits stated late: self-loops and lag-1 only, no contemporaneous edges. B ≥ Δ equalises relatives but does not give stationary covariances. Replicates vs one record are blurred. | R2 M2, R3 W2/W7/W8 | 2/5 (R3 only for the last two) | Minor to Major (R3 W2) | Corroborated / single |
| 20 | Multiplicity and claim-level p-values are constructed post hoc. Holm over 21 claims, but the claim-level p rule was set after the fact. | R1 W6 (DA m3 related) | 1/5 | Minor to Major | Single-seat finding |
| 21 | Preprint dependence. The paper rests on one 2026 arXiv version (2605.06288v1). State the version and date and check for a v2. | AE W1/Q5, DA m5, R2 M1/M5 | 3/5 | Minor | Corroborated |

---

## 3. Disputed points and arbitration

### 3.1 R2 "no Critical" vs DA "two Critical"

The panel's other three seats rated the same underlying problems Major, not Critical. R2 states that no finding is Critical because every Major issue is repairable by re-analysis, rewriting or modest experiments. I adjudicate each DA Critical on the manuscript evidence, not on how forcefully it is worded.

**DA C1: practical importance of the artefact; no exposed generator.**

- **DA argument.** The paper's own survey says every surveyed generator fails condition 1 (a static learner receives unlabelled slices). The only exposed construction is Reisach et al.'s unrolled benchmark, whose data code is not public. So the general benchmark claim is unsupported unless the authors exhibit a released benchmark where the artefact occurs and changes a conclusion, or retitle.
- **Corroboration.** R2 W1 (Major) and R3 W7 (Minor) independently make the same factual point. R2 also verified in arXiv 2605.06288v1 that Reisach et al. specify no horizon and run no unrolled experiments. AE W1 raises the related overclaim. R1 M9 calls the survey's exposure classification speculative. AE S5 and R3 S4 value the practical recommendation, which is cheap and probably correct.
- **My check.** Confirmed at main.tex lines 1160-1170: "its data code is not public", "The others produce one record for time-series methods, so they fail the first condition as shipped", and the mixing times "we did not compute for any deployed generator". The DA's factual base is accurate. The paper does state the threat model in the main text. The problem is that the abstract and §6.1 are phrased more generally than that text supports.
- **Verdict: PARTLY UPHELD (as Major, not Critical).**
  - Upheld: the reach claim exceeds the evidence, and the abstract and §6.1 must state the threat model first.
  - Rejected: Critical severity and the DA's specific cure. The paper is a correct and Lean-checked analysis of a construction proposed in a preprint. It also contains a threat-model-independent graph fact that three seats singled out as the most useful single sentence, namely that rings have s_ρ = 1 for T ≤ p. The deficiency is repaired by reframing (R2's suggestion 1-2) and by quantifying exposure with a mixing-time computation for generators that burn in fewer than 100 steps (R3, R2). It does not require finding a released benchmark. If the authors cannot show any, retitling or reframing to "a proposed construction" is an acceptable outcome.
- **Required author response.** Yes. The abstract, introduction and §6.1 must name the threat model, and the authors must either show exposure data or scope the claim.

**DA C2: theory does not explain the experiments; the oracle is circular.**

- **DA argument.**
  - (a) The sign follows from any story in which the start changes correlation strength.
  - (b) The size is off by about 2x (slope 0.53).
  - (c) The exact gap has coefficient -0.05 once detectability is in the regression.
  - (d) The R² = 0.97 oracle is the algorithm's own expectation under the exact covariances, so it validates the simulator, not the theory.
  - (e) H1 is nearly guaranteed once any positive gap exists.
  - (f) Per App. Q, only the rel-vs-R² swaps follow the mechanism of Theorems 2 and 5. The other sign flips are Var-SortnRegress vs FLOP/PC, i.e. var-sortability.
- **Corroboration.** All four non-DA seats make points (a) to (d) (AE W3, R1 W1, R2 W3(e), R3 W3), each as Major. R1 adds that all 42 cells are positive. R3 adds that the ranking pools family, p and T, and that FLOP's own G1-G2 shift is not reported with a CI.
- **My checks.**
  - main.tex line 937/1055 contains the sentence "the exact gap adds nothing to it (coefficient -0.05)" and the concession "it does not separate the structural mechanism ... from a general change of correlation strength".
  - b1x/results/oracle_b3.txt reports a joint OLS gap coefficient of -0.063 at cell level (R² 0.977 full vs 0.974 oracle-only) and -0.039 at unit level (R² 0.611 vs 0.611). So the non-incrementality is confirmed. The paper's -0.05 differs from the file's -0.063 (README: -0.06), a minor numerical drift for the authors to reconcile.
  - appendix.tex line 1932 confirms point (f): only the rel vs R²-SortnRegress swaps "follow the mechanism". The DA's counts for (e) and (f) beyond that sentence are the DA's own and I did not reproduce them.
- **Verdict: PARTLY UPHELD (as Major, not Critical).**
  - Upheld: the abstract sentence "the exact values predict the sign and rank the conditions" is unsupported as phrased. A positive sign is non-discriminating, and the oracle cannot separate the theorem's mechanism from correlation strength.
  - Rejected: Critical, because the theorems themselves are proven, Lean-checked and numerically verified. What fails is the empirical claim that links them to learner behaviour. That claim is repairable with a wording change (R3's replacement abstract sentence) plus a matched-correlation counterfactual. The DA's required control is the same experiment that AE (W3), R1 (W1) and R3 (W3(iii), a negative control) each independently proposed.
- **Required author response.** Yes. Either run the counterfactual or remove "predict" and demote the experiment to a consistency check.

Both adjudications are recorded visibly. If this synthesis were run under a contract, the DA lines would read `C1=PARTLY UPHELD`, `C2=PARTLY UPHELD`, each with the rationale above. Neither blocks a Major decision, and both block Accept.

### 3.2 Other disputes

| Point | Positions | Arbitration |
|---|---|---|
| Overall: Reject vs Major | DA "borderline Reject"; AE, R1, R2, R3 Major | **Major.** See section 1. |
| Reproducibility severity | AE Major (W8, rated 3); R1 rated 4 and R3 rated 5 (R3 notes only "one open check on the b1x stage") | **Should-fix, not blocking.** REPRODUCE.md says the full stages were not re-run, which AE, R1 and R3 all read. The disagreement is about how much weight the gap carries. Cheapest remedy: run `./reproduce.sh all` once, report identical/DIFFERS per stage, put the limits in the paper's Supplementary information, and record the missing seeds and arguments (x1_er, x2). |
| Faithfulness: severity and direction | R1 Major, R2 Major (mis-description), AE Major sub-point; DA Minor, with a computation (about 34,000 tests, 0 violations) suggesting the statement is "probably not vacuous". R1's own run found 2 of 12 coarse-grid draws with coincidental CIs. | **Not a real conflict on existence.** Both sides want the caveat replaced by evidence. Arbitration: Major, because Props 9-12 and the abstract carry the assumption. The remedy is either an exact-arithmetic check for chain, out-tree and ring, or a short genericity argument, with the assumption visible in the abstract. R2's Boege point (section 4) shows the paper's hedge is mis-described, which strengthens the Major rating. |
| Tightness: R2 credits it as a strength | R2 S1 says the bound is "shown tight (Remark 21)". AE, R1, R3 and DA say it is shown only for Δ = 1 plus an enumeration. | **Evidence first: AE, R1, R3, DA are right.** I checked Remark 21 and appendix line 1947 (section 4). R2 did not test the claim, so this is silence, not a dissent. |
| Severity of the integrity vocabulary | AE, R2: Minor. R3: Major. | **Should-fix with a Major-sized remedy for the follow-up studies.** The paper text is already candid. Wording in the repository and in the labels ("confirmatory", "kill rule") is the live problem, and external registration is the only real fix for any new study. Author-only. |
| Threat-model severity (see C1) | R2 Major, R3 Minor, AE Major (partial), DA Critical | **Major**, via C1. |
| Real-data: drop "partial support" vs keep | AE: reframe as a feasibility study with a clear negative result. R2, R3: the studies test a different question. R1: effects are large, so the qualitative conclusion is probably safe, but the p-values are invalid. | **Reframe as feasibility, and re-do inference at the run level.** Not a conflict. These are compatible. |

---

## 4. Point-by-point verification of claims that could embarrass the authors

| # | Claim in dispute | Result of my check | Status |
|---|---|---|---|
| V1 | **Boege et al. (arXiv 2404.04024) mis-described** (R2 W2). The paper says Boege et al. "show that coloured Gaussian DAG models can have no faithful distribution at all", and that "their results are stated for colour classes with a common head ... so we treat the unrolled case as outside what they cover" (main.tex lines 326-336, 764-770). | I fetched the arXiv v3 PDF and read the text. **R2 is correct.** §5.1 is general. Prop. 5.3 (Wu-Drton): a generic Σ of a vertex-colored DAG is faithful. **Prop. 5.4: "Let (G,c) be an edge-colored DAG. A generic Σ ∈ M(G,c) is faithful to G."** Example 5.7 (non-faithful) needs "vertex and edge colors ... simultaneously". Question 5.8 (which colored DAGs admit faithful distributions) is left open. The common-head condition belongs to the BPEC-DAG subfamily (Def. 5.14) used for structural identifiability and the GECS algorithm, not to the faithfulness results. So the paper's "outside what they cover" rationale is wrong. The paper's sentence that such models "can have no faithful distribution" is true only for mixed colourings, and omits the positive edge-only and vertex-only results. Tied A is an edge colouring and tied D a vertex colouring. R2 reads the unrolled model as the mixed case with both ties, so the faithfulness question is open there and not covered. I did not independently re-trace how Prop. 11's model maps onto Boege's classes (R2 itself gives this a confidence of 3). | **Mis-description confirmed.** Fix the text. Whether the unrolled model is edge-only (positive result, Prop. 5.4 applies) or mixed (open) must be stated. |
| V2 | **Lean counts: "163 declarations" vs 259 theorem/lemma** (R1 M1, R3 W1, AE S2). | I counted. `lean/RelSort` has 16 files and 4,159 lines. `lean/AXIOMS.txt` has 163 lines (162 distinct `RelSort.` names). The sources contain **259 theorem/lemma declarations**, 76 defs and **0 `example` keywords**. 97 theorem/lemma names are absent from the axiom report. `grep sorry|admit|^axiom` over the sources returned nothing. Examples.lean holds 13 theorem/lemma declarations, which are presumably the paper's "12 examples" (concrete graphs, the sharpness lemma, one numeric instance). The paper's sentence is literally accurate ("the axiom report lists 163 declarations", main.tex 790) but the 33+53+65+12 breakdown reads as the full inventory. I did not rebuild Lean, so "clean rebuild" rests on BUILD_LOG as read by R1 and R3. | **Not false, but incomplete.** State that 163 of 259 were audited via `#print axioms`, say what the other 96-97 are, and say what "examples" means. R3's point that 65 of the 163 are elementary algebra and about 29 headline theorems carry the verified graph theory is consistent with the paper's own breakdown. |
| V3 | **"Tight" proved only for Δ = 1; 445/726 enumeration** (AE W2, R1 W3, R3 W1, DA M1). | Confirmed. appendix.tex lines 151-156 (Remark 21): the single edge 0→1 with self-loops, Δ = 1; B = 0 gives at least 2/3 for every T ≥ 2 (`edgeLoops_burn_sharp`); enumeration: B = Δ gave the stationary value on 726/726, B = Δ-1 did not in 445. So B = Δ-1 sufficed in 281 (about 39%). Line 1947 hedges it correctly ("cannot be lowered in general"), but the abstract (main.tex line 85) and the Introduction (line 151) do not. R1's independent check found the directed chain family tight for Δ = 1-5 (window value 0.75, 0.65, 0.607, 0.583, 0.568 at T = 3 for B = Δ-1, against 0.5 at B = Δ). DA found the minimal sufficient B below Δ in 24% of all random graphs and in 51% of those with Δ ≥ 2. | **Verified.** "Sufficient, and not improvable uniformly over graphs with a given Δ" is supportable. The chain family would close the gap for every Δ if proved (and ideally formalised). |
| V4 | **"Pre-registered" wording drift between repository and paper** (R3 W4, AE W13). | Confirmed. Repository commit messages: 5ac1bd2 "B'1 pre-registered plan"; 2d19c9b "pre-registered plan for B1-NL"; 827179d "OSF-formatted preregistration text". `pre-regist` also occurs in b3/PLAN.md, b3/README.md, b1x/README.md, README.md and PLAN2.md. The paper (main.tex 1187) says "time-stamped analysis plans, not pre-registrations", yet uses "confirmatory" (line 1129: "Holm across all 21 confirmatory claims") and "kill rule" (lines 1221, 1248). Commit order is consistent with plan-before-results for B1 (6e51eb1 before c2d804e), B1-NL (2d19c9b before c96f018), B3 (827179d before 7071855) and b1x (5ac1bd2 before 40d9ee7). | **Verified.** The paper text is candid. The drift is on the repository side, and in two labels in the paper. |
| V5 | **Push records for 3 of 9 plans** (R3 W4). | Appendix lines 866-870 retrieve push records for three plans: B1-NL, B3 and the learner extension. They state explicitly that none was retrieved "for the B1 plan or the four earlier plans". By the paper's own enumeration that is 3 of 8 plans (four real-data, B1, B1-NL, B3, b1x). I could not reproduce R3's total of nine. The paper also says "no plan was posted to an external registry". | **3 verified; the denominator is 8, not 9, by the paper's own list.** The authors should reconcile the count and retrieve the remaining push records, which agents can do. |
| V6 | **Pseudo-replication in E1, B3 and CausalRivers** (R1 W2). | Confirmed from the files. E1: 10 segments of the single `citris_1` series (appendix lines 912, 938, 944). A sign-flip test over 10 units has a floor of 1/1024, which with Holm over 8 competitors gives the reported p = 0.0078. B3: "3x30,000 steps, 30 units of 3,000" (appendix line 2187), so about 3 independent runs and a sign-flip floor of 1/8 at that level. CausalRivers: `empirical/subgraphs_7000.json` has 120 subgraphs, 900 node slots, **457 distinct stations, maximum reuse 8**, in two regions (60 east_germany, 60 bavaria). Subgraph-level bootstrap CIs therefore understate variance. The 12 h and 24 h columns of Table 4 are the same records at two resolutions. | **Verified.** The large effects (Δ ≈ -0.5) probably survive. The p-values and "k/k units" statements do not. |
| V7 | Preprint sufficiency claim (paper: "only sufficiency is shown in the arXiv version"). | Not re-fetched by me. R2 reports confirming it against arXiv 2605.06288v1, namely that Thm 4 is an implication only and that the authors "do not analyze their properties beyond this". | **Verified by R2.** State the version and date; check for a v2. |
| V8 | Headline numbers vs logs (AE S4, R1). | AE matched Table 2 and 3 entries and the B1 pooled gain (+0.1249 [0.1161, 0.1338]; Spearman 0.798 over 42 cells; unit-level 0.616; slope 0.530) to analysis.txt files. I confirmed the unit-level oracle values (R² 0.611) and the coefficient discrepancy noted in 3.1 (-0.05 in the paper vs -0.063 and -0.039 in oracle_b3.txt). | **Verified, with one small numerical drift to fix.** |

---

## 5. Revision Roadmap

Each item carries its seat sources and corroboration. "Plan first" means a new analysis plan must be committed before the run (see also the integrity points above). Obligation: **M** = must fix, **S** = should fix.

### (a) Text-only fixes (AI agents can do now)

| # | Item | Seats | Ob. |
|---|---|---|---|
| a1 | Rewrite the abstract. Lead with the threat model (the unrolled construction fed to static learners). Define rel-sortability, FLOP, "latent burn-in". Make three claims at about 150 words. | AE, R2, R3, R1 (4/5) | M |
| a2 | Tightness wording: replace "cannot be shortened" with "sufficient, and not improvable uniformly over graphs with a given Δ" in the abstract, contribution (2), §2.6 and §6.1. | AE, R1, R3, DA (4/5) | M |
| a3 | "Machine-checked": scope the abstract to named theorems (Thms 1-6); keep the status label in the contribution sentence; correct the Lean count text (V2); say which items were audited and what "examples" are. | AE, R1, R2, R3, DA (5/5) | M |
| a4 | Delete "the exact values predict the sign and rank the conditions". Use R3's replacement ("raises order accuracy as expected from the exact values; the exact values do not explain its size beyond detectability"). Report both pooled gains (+0.133 over seven families, +0.125 over five) and mark R² 0.97 as cell-level and post hoc. Reconcile -0.05 with -0.063. | AE, R1, R2, R3, DA (5/5) | M |
| a5 | Fix the Boege description (§2.3, §3.5) per V1. State which colouring the unrolled model is, cite Prop. 5.3 and 5.4, and say that faithfulness for the mixed case is open (their Question 5.8). | R2 (verified by me) | M |
| a6 | Reframe the real-data section as a feasibility study with a clear negative result. Drop "partial support" and the confirmatory vocabulary ("confirmatory", "kill rule"); replace with "pre-specified in-repository". Add a "predicted before data? yes/no" column and a forking-paths paragraph. Align the repository files ("pre-registered") with the paper. | AE, R2, R3, DA (4/5) | M |
| a7 | State scope limits in the abstract: self-loops at every node, lag-1 only, no contemporaneous edges, n = 1000 simulations. State that B ≥ Δ equalises relatives and does not give stationary covariances; give the one-line practitioner rule. | R2, R3 | S |
| a8 | Positioning: add the R2 reference set (Gentzel 2019, Brouillard 2025, Andrews & Kummerfeld 2026, Nagarajan & Scutari 2025, Assaad 2022, Mooij 2013, Hyttinen 2016, BOSS) and the R3 SVAR/DSGE initial-condition references (Hamilton, Lütkepohl, Kilian & Lütkepohl, Müller & Elliott), after checking identifiers and editions. Clean the `CHECK` comments in refs.bib. State the Reisach version and date. Shorten or remove the SPADE paragraph. | AE, R2, R3, DA (3-4/5) | S |
| a9 | Add a claim-status table (machine-checked, paper proof, enumerated, simulated, real data, failed) and a "replicates of a short window vs one long record" table. Move the generator survey into the main text or a single summary line. | AE, R2, R3, R1 | S |
| a10 | Prop. 8: add the sentence on ν = ν(M, Ω). Prop. 17: read the support as the "allowed support" (star-shaped set). Theorem 1: say that unreachable u satisfy the upper inequality trivially. | R3, R1 M5, DA m2 | S |
| a11 | Notation: one symbol each for B_ab and W_ab, and state conservativeness of the plug-in test where the size claim is made. Make the Table 3 in-tree block a proper sub-table with CIs. | AE, R1 (2/5) | S |
| a12 | Put the REPRODUCE.md limits into the paper's Supplementary information (full stages not re-run, outputs not regenerable by one command). Reconcile the plan count in the appendix. Re-word "All failed tests are reported" to say exactly what is covered (R1 Q7). | AE, R3, R1 | S |
| a13 | Reword "Reisach et al.'s fixed start" and "the horizons proposed for benchmarks". The source specifies neither an initial law nor a horizon. | R2 | M |

### (b) Analyses and experiments agents can do, with new plans committed first

| # | Item | Seats | Ob. |
|---|---|---|---|
| b1 | **Matched-correlation counterfactual start**. Keep G1's relatives sets but equalise marginal variances and correlation strengths. Include the reverse manipulation and a negative-control graph (strongly connected, ρ(A) near 1). Report within-family partial associations, a graph-level mixed model and a bootstrap CI for the 0.53 slope. Also decompose the oracle's predicted gain into relatives-only and correlation parts. | AE, R1, R3, DA (4/5; R2 supports via W3(e)) | M |
| b2 | **Variance-matched control** (rescale G1 columns to G2 marginal variances and vice versa), a small DAGMA/NOTEARS grid (λ, threshold), one more parametrisation, T up to 10, p = 10 and 20. Report FLOP's own G1-G2 shift with a CI and test the ranking change as a contrast. Explain the below-chance standardised order accuracy (0.31-0.42) by checking sign and tie handling in the scoring. | AE, R1, R2, R3, DA (5/5) | M |
| b3 | **First-slice-excluded scoring as the primary analysis** for Tables 2-3 and Fig. 3, with the full-window numbers in the appendix, and a stated target graph per generator. Discuss the burn-in window as an ADMG problem. | R2 (R3, R1 related) | M |
| b4 | **Per-graph minimal burn-in or chain-family tightness for general Δ**: either an equivalence analogous to Thm 1 (computable from the distance matrix) or a proved family attaining B = Δ-1 failure for every Δ (the directed chain with self-loops, per R1), ideally with a Lean lemma. | AE, R1, R3, DA (4/5) | M |
| b5 | **Cluster-level inference for E1, B3 and CausalRivers.** Define the unit as the independent run (3 for B3 H1, 1 for E1, 15 runs for wind), use exact permutation p-values at that level, and use station- or region-blocked bootstrap for the rivers (or station-disjoint subgraphs). | R1 (verified by me) | M |
| b6 | **Missing baselines**: GES, GRaSP, BOSS, DirectLiNGAM, DYNOTEARS and VAR-LiNGAM as labelled references. Report SHD on CPDAGs consistently. | R2 | S |
| b7 | **SVAR spectral radii**: report ρ(A) for the B1 and B1-NL transition matrices and the gap at ρ(A) close to 1; compute mixing times for the generators that burn in fewer than 100 steps, so exposure is measured. | R3, R2 | S |
| b8 | K-sensitivity of the plug-in test on the river data, study 3 and E1, and finite-sample size at each real study's n_eff. One non-Gaussian simulation. | R1 | S |
| b9 | Faithfulness: an exact-arithmetic d-separation vs CI check for chain, out-tree and ring (R1's C4 protocol), reported as "verified symbolically for these graphs". | R1, DA, R2 | S |
| b10 | Run `./reproduce.sh all` once and report identical/DIFFERS per stage. Record seeds and arguments for x1_er and x2, and regenerate Fig. 2 from a scripted call. | AE | S |
| b11 | Retrieve hosting push records for the remaining plans and tabulate them. | R3 | S |
| b12 | Optional: a lag-0 (B ≠ 0 acyclic) extension by enumeration, n-variation (200, 5000), and fixed-T ER values for p in {5, 10}. | R3, R2 | consider |

### (c) Author-only items

| # | Item | Seats |
|---|---|---|
| c1 | A human reads the load-bearing proofs (Appendix O.1, Props 11, 14-17, 30, 31) and signs off, or the unread propositions are demoted to remarks or withdrawn. State in the paper what the author personally verified. | AE, R1, R2, R3, DA |
| c2 | A human compares the 8-10 headline Lean statements (Thm 1, 2, 5, Cor. 3, 6, Remark 21, Table 1 values) with the paper, using a short checklist (definitions, hypotheses, quantifiers). Publish the one-page signature table in App. G. R1 and R3 already did a partial check and found it faithful. | AE, R1, R3, DA |
| c3 | External registration (OSF) of any new study before data access, and an archive of the repository at submission with a DOI (Zenodo or Software Heritage). | R3, AE |
| c4 | AI-use statement: model names and versions, dates, agent harness, pointers to logs, how agent outputs were accepted or rejected, and the simulated reviews included in the supplement. The identifiers must come from the author's records. Springer's current policy wording must be checked by the author, since R3 paraphrased it from memory. | R3, AE |
| c5 | The decision to split the paper (see 5d). | AE, R1, R2, R3, DA |
| c6 | Contact the Reisach et al. authors: is there a later version containing any of Theorems 1-5, and is the unrolled data code available? Say so in the cover letter. | AE, R2, DA |

### (d) Recommendation on splitting the paper

**Recommendation: split. Submit a focused Paper A to Machine Learning, and prepare Paper B separately.** Three seats raised splitting by name (AE W6, R2 W5, R3 W6), R1 M10 recommends it, and the DA's M5 proposes the same 15-20 page core. This is the strongest agreement in the panel on any remedy.

**Paper A (the Machine Learning submission, about 25-30 pages).**
- Contents:
  - The exact per-edge value, limit and rate, the necessity of the condition, and the stationary window (Thms 1, 2, 5, Cors 3 and 6), with the walk-length lemma.
  - The Lean development, scoped to the headline statements, with the human-checked statement map (c2).
  - The generator survey, moved into the main text and quantified with mixing times (b7).
  - The B1 and B1-NL start-artefact experiment, with the matched-correlation counterfactual (b1), the variance-matched control and tuned learners (b2), and first-slice-excluded scoring as primary (b3).
  - Per-graph minimal burn-in or the chain-family tightness result (b4).
  - The practitioner recommendations (stationary start, burn-in checked against mixing).
  - One paragraph and a table of the open pieces, pointing to Paper B.
- Why this fits: every claim is checkable by a referee, the storyline is one thing (what a fixed start does to unrolled benchmarks and what to do about it), and the verification burden shrinks to a handful of statements.

**Moves to Paper B (a separate estimation and real-data paper).**
- Finite-sample estimation of relatives, the F1-F3 results, saturation by a latent driver, the aggregation results (§4.2), and the CausalRivers and Causal Chambers studies.
- These test a different question: can relatives be estimated from one autocorrelated record? Real-data inference must be redone at the run level (b5) and the K-sensitivity checks added (b8).

**Cut, or shrink to a short note.**
- Sections 3.4-3.5 (hidden-label identifiability, Props 8-12, Conjecture 13). They rest on an unproved faithfulness assumption (item 8), are unread paper proofs, and do not feed the benchmark recommendation. Keep them out of Paper A, and do not use them to qualify Paper A's recommendations. They return only as a clearly conditional note, after b9 or a genericity argument, and after Boege's mixed-case question is addressed (V1).
- The ER limits (Prop 7), as a remark with the numerical verification, unless the fixed-T values for p in {5, 10} are shown (b12).
- The SPADE paragraph in §2.1.

**Effect on the MLJ fit.**
- Fit improves. As submitted, the paper fails the "complete, focused and verifiable" test (items 4 and 5), and the main text reads as an index to a 50-page appendix.
- Paper A's significance is modest but honest: a small, correct, Lean-checked exact result, plus an artefact diagnosis for a proposed benchmark construction, plus a concrete recommendation. What turns it into a Machine Learning paper rather than a note is the experimental layer with its controls (b1, b2, b3), so those are not optional.
- Risk to state plainly: if the matched-correlation counterfactual shows that the gain is a correlation-strength effect, the theory's role shrinks and Paper A becomes a short note on the exact values plus a cautionary result. The authors should decide in advance how they will report that outcome (a plan committed first, with the interpretation fixed).

---

## 6. Verified-OK items (authors should not touch these)

- **Graph theory core.** Theorems 1, 2 and 5 and Cor. 6 are correct. Independently re-derived by R1 (1,500 summary graphs x 4 horizons, 0 failures), DA (300 graphs, 0 violations), R2 (key step) and AE (ring values vs exact_predictions.txt).
- **Prop. 10** (Markov class size T^m0): checked by brute force by R1 (8 graphs) and DA (7 graphs). **Prop. 12(a)** (T = 2 non-identifiability) checked in exact rational arithmetic by R1. **Prop. 11(b)** proof checked by R1. Table 1 values consistent with the theorem (DA).
- **The ring result** (s_ρ = 1 for all T ≤ p) was confirmed by AE, R1 and DA, and the DA called it the most useful single sentence in the paper.
- **ER limit formulas** (Prop. 7) match independent simulation at c = 0.5, 1.5, 2, 3 and the quoted peaks (R1). The proof itself is still unread (c1).
- **Prop. 14** agrees with a simulation at n = 1000 (R1: 0.371 vs 0.358 predicted).
- **Lean.** No `sorry`, `admit` or `axiom` in the sources (my grep; R1, R3, DA). Definitions of relatives, s_ρ and G_T are built from scratch. The statements for edge_score_dist, sRho_unrolled_exact, sRho_unrolled_bounds_maxFinDist, sBurn_eq_maxFinDist and edgeLoops_burn_sharp match the prose (R1, R3). All Lean names cited in the paper exist in the sources (R1, R3).
- **Reported numbers.** Tables 2 and 3 and the B1 pooled gain match the result files (AE).
- **Plan-before-results order** is consistent in git history for B1, B1-NL, B3 and b1x (R1, R3; confirmed in the log).
- **Inference design.** The B1 primary inference uses the summary graph as the unit with graph-level bootstrap CIs (R1 S5).
- **Cited-work accuracy.** Reisach et al. 2026 (definition, Thm 4 as sufficiency only), FLOP (ICLR 2026), Lohse and Wahl (TMLR 2025), Kaiser and Sipos, and Friedman-Murphy-Russell checked by R2.
- **Candour.** The reporting of failed predictions, the post hoc labels, the AI authorship and the "not pre-registrations" framing were praised by all five seats. Keep this register. Only the labels that drift from it need fixing.
- **Reproducibility infrastructure.** Pinned environment, checksummed data, staged reproduce.sh and `python -I` isolation (R1, R3, AE).

---

## 7. Provenance note

- **Panel.** Five fresh subagent seats (AE, R1, R2, R3, DA) wrote their reports independently, without seeing each other's output or earlier review rounds. All five are the same model family as the author-side agents that wrote the manuscript, code, Lean files and analysis plans. They are **not independent error processes**. Agreement among them is evidence of a shared reading, not of independent confirmation. Seat-run numerical checks (R1's 6,000 cases, DA's 300 graphs and tied-parameter tests, R1's ER simulation) add real computational evidence, but each was written by the same family of agent. I did not re-run any of them.
- **Synthesizer's own checks.** Done directly: the Lean file, line, declaration and `sorry`/`axiom` counts; the Boege arXiv v3 text (Prop. 5.3, 5.4, Example 5.7, the BPEC-DAG definition); the manuscript passages quoted in section 4; the git commit sequence and messages; the CausalRivers station-reuse count from `empirical/subgraphs_7000.json`; the oracle regression output in b1x/results/oracle_b3.txt. Not done: no Lean rebuild, no simulation re-run, no re-derivation of the paper-only proofs, no re-fetch of the Reisach preprint (relies on R2), no independent check of the DA's counts for C2(e) and C2(f) beyond the appendix sentence.
- **Process.** The `editorial_synthesizer_agent.md` Synthesis Protocol (no-contract path), the decision template and the decision standards were followed as far as practical. **No sprint-contract checker scripts were run.** No Schema 6/7 artefacts, `review-panel-provenance` artefact or replay validator exist for this panel. The panel-provenance block of the template is therefore replaced by this note, and its six axes are left `unknown` rather than inferred. Only role separation is observed here, and the model-family axis is known to be non-distinct. The cross-model blind check (Step 4b) was not run.
- **Limits of this decision.** It is a simulated first-round editorial synthesis for the authors' internal use. It is not an actual Machine Learning decision and carries no editorial authority. Some seat statements about external facts (Springer's AI policy, Herman et al. full text) were made from memory or from abstracts only and are flagged as such in the seat reports.
