# Round-3 per-seat reports

Simulated ICML 2027 re-review of research/theory2/icml_paper/ at commit e6f4c54 (title: "How the Start of a Simulation Makes Time-Unrolled Causal-Discovery Benchmarks Sortable"). Run 2026-11-09. Decision, traceability matrix and Round-4 roadmap: `04_round3_decision.md`. Line numbers refer to main.tex / appendix.tex at e6f4c54.

**Provenance.**
- All five seats were run inline by one model (Claude family, the same family as the agents that wrote and revised the paper), in one context, sequentially.
- They reuse the Round-1/Round-2 reviewer cards and score yardsticks. They are not independent: they share one evidence pass, and a later seat could read an earlier seat's text.
- The three-gate re-review contract was approximated, not executed: no checker, no sidecars, no cross-model pass. The details are in the decision file's judge record.
- Read the scores as one model's structured self-critique from five angles, not as five referees.

---

## Area Chair (journal fit) — Overall 5 → 6 (weak accept, borderline); Soundness 3, Presentation 3, Contribution 2 → 3; confidence 4

**Summary of the revision.** The revision:
- retitles the paper around benchmark generators;
- leads the abstract with the short-horizon result for strongly connected graphs and with burn-in Δ (now Lean-checked in the Δ form);
- moves aggregation to the appendix and adds a B1 figure (fig:oracle);
- runs the requested competitive-learner experiment (DAGMA, NOTEARS, GOLEM, PCMCI+);
- adds a generator survey.

**Round-2 items (this seat).**
- W1/DA1 (thin core advance): PARTLY. B′1 gives the benchmark story a competitive-method consequence; the theory is still described as "short consequences" (l.179-182).
- W4/C11 (focus): PARTLY. Title, abstract order and Sec. 4 are done. Finite-sample theory and real data remain in the main text, about 2.5 pages, and abstract (iv) pitches them.
- N3 (H1 named as tracking; FLOP status): RESOLVED (l.419-422, l.170).
- N4 bookkeeping: RESOLVED (overview, pooled labels, "four of the six", Impact Statement).

**Strengths.**
1. B′1 is the result the panel asked for. It was planned and pushed before any score (server push 08:20:31, results 12:44:27). It shows a Holm-significant rank change of DAGMA and NOTEARS against FLOP on in-trees, on SHD, SID and order accuracy, and robust to excluding first-slice pairs.
2. Strongly connected summary graphs are not protected at benchmark horizons (rings: s_ρ(G_T)=1 for T ≤ p). This now leads the abstract and contributions, where it belongs.
3. The Discussion recommendations (l.808-823) are concrete and correct: a stationary or mixing-checked start, standardisation, and scoring without first-slice pairs. The normalised-SHD cost (0.044 → 0.273, but 0.041 vs 0.048 without slice 1) is now in the main text.
4. Disclosure remains exemplary: process limits, AI use, unread proofs, failed graded criteria.

**Weaknesses.**
- **W1 (major, framing; NEW-3).** The title covers Secs. 3–4 only. Secs. 5 (finite samples), 6 (Lean) and 7 (real data, with a full-width table) are not about the start. The paper still reads as two papers, and the second costs the first its tables: tab:b1, tab:examples, fig:sat, Thm F1 and the estimand definition are quoted from the main text but live in the appendix (l.354, l.404, l.558, l.679, l.746).
  - Fix: move F2/F3 detail and tab:real to the appendix; keep 6–8 lines of real-data summary plus the saturation diagnostic; bring back a compact tab:b1; cut abstract (iv) to one sentence.
- **W2 (major; NEW-1).** The abstract says "a detectability oracle predicts the size" (l.106-108). The analysis is post hoc and is a computation of the algorithm's expectation. The main text's "not another mechanism" (l.425-426) goes beyond what it shows.
  - Fix: "a post hoc detectability analysis reproduces the size"; drop "not another mechanism".
- **W3 (major; NEW-4).** The survey (l.520-534) grades generators by B ≥ Δτ_max, but the Discussion (l.812-818) says that criterion removes only the structural artefact, while the var-sortability transient, which drives the CO result, needs mixing.
  - Fix: regrade by mixing and by threat model.
- **W4 (minor; NEW-5, NEW-9).**
  - The abstract groups GOLEM with DAGMA/NOTEARS as beating FLOP on in-trees. It did not.
  - The main text ends exactly at the bottom of p. 8 in icml2026.sty, so the icml2027 swap is a risk.

**Questions.**
1. Is the intended reader a benchmark designer (then cut Secs. 5 and 7) or a theorist (then the title is wrong)?
2. Does Reisach et al.'s data generator actually start at t=1 with no burn-in, or is that inferred from the graph construction (app:b1x says the code is not public)?

**Score justification.**
- Contribution 2 → 3: the competitive-learner rank change gives the benchmark story practical consequence, even though it runs through var-sortability.
- Overall 5 → 6.
- Not higher: the refocus is incomplete, and two new overclaims (oracle, survey) replace the Round-2 ones.

**Top 3.** Finish the refocus; reword the oracle claim and add its decomposition; regrade the survey.

---

## Reviewer 1 (probability and statistics) — Overall 5 → 6 (weak accept, borderline); Soundness 3, Presentation 3, Contribution 2; confidence 4

**Round-2 items (this seat).**
- **Plug-in n_eff: RESOLVED on paper.**
  - prop:plugin (app l.497) gives consistency of B̂ for K_n → ∞, K_n = o(n), and asymptotic size ≤ α/m per pair. The main text states the fixed-K caveat honestly (l.587-592).
  - Size 0.113 vs 0.05 at AR 0.99, K=50: matches finite2/FINITE2.md B, row λ=0.99, n=64,000.
  - B3 K-sensitivity, 0 of 2,730 decisions change: matches k_sensitivity_b3_results.txt.
- **F1 scope: RESOLVED.** prop:f1scope (app l.517) states the certified failure probability d(d−1)e^{−z_c²/(2B)} > 0 for the Bonferroni Fisher-z threshold, and recovery is routed through prop:plugin (main l.565-570).
- **Fourth cumulants: RESOLVED.** rem:nongauss: for independent pairs the null variance is unchanged; for relatives power changes (l.592-596).
- **B vs W: RESOLVED.** B_ab is defined with absolute values; W_ab ≤ B_ab with equality iff all products are nonnegative (l.546-551).
- **ER c=1: RESOLVED.** Convergence in p is reported (app l.1552-1560).
- **Multiplicity: RESOLVED.**
  - Holm over 21 claim-level p (app:b1x). Only B1-NL H3-SGN fails (p 0.108). The closest survivors are at 0.0079 and 0.033.
  - The claim-level p construction (k-th smallest Holm p) was chosen post hoc, but it is reasonable and stated.
- **Graph-cluster CIs, unit-level tracking: RESOLVED.** tab:b1 now uses graph-cluster CIs (narrower, because T is a fixed factor; correctly explained). Unit-level Spearman 0.62 and cluster permutation p ≤ 1e-4 are in the main text (l.423-425).
- **B3 inference unit: NOT ADDRESSED (residual of R-N6).** The 30 units are consecutive segments of three experiments on one device. The sign-flip p of 1e-5 treats segments as exchangeable, and no per-experiment effects are given.
- **Human read: NOT DONE (author-only, B′4).**

**Strengths.**
1. The finite-sample layer now covers the estimator actually used, with the correct asymptotic statement and an honest fixed-K caveat checked by simulation.
2. The statistical reporting of B1/B1-NL is now at the right unit (the summary graph), with interaction effects in place of swap counting. This turned the "did not replicate" story into a correct one: interactions replicate, sign flips do not, because level differences are larger in B1-NL.
3. All numbers I checked against result files match (decision file, judge record).

**Weaknesses.**
- **W1 (major; NEW-1): the oracle is not independent evidence.**
  - The detection-probability oracle (app:b1x) replaces each sample correlation by P(|r| > ε) at n=1000 under Fisher z. It then computes P(c_i < c_j) + ½P(tie), treating indicators as independent. That is the expectation of rel-SortnRegress's order accuracy under the true covariance. Agreement with the Monte Carlo mean (cell R² 0.97, slope 1.03) checks the simulation against its own expectation, plus the independence approximation. It does not test a mechanism.
  - It shows that the shortfall from the graph value is a detection effect at n=1000, which is useful. But the oracle uses the full G1 and G2 covariance matrices, so it cannot separate (a) pairs that are relatives under G2 but not under G1 (the paper's structural mechanism) from (b) pairs that are relatives under both with different correlation magnitudes (a generic transient).
  - The joint regression, where the graph gap gets −0.05, says the graph quantity adds nothing once covariances are known.
  - Fix: decompose the predicted gain into (a) and (b) (a population computation, no new data). Reword l.425-426 and the abstract, and label the analysis post hoc in the abstract.
- **W2 (moderate; NEW-6): the B′1 inference is thinner than the main text suggests.**
  - HX2 rests on 24 graphs per family, with Holm p 0.012/0.0011 (G1) and 1.4e-5/0.020 (G2); the claim-level p 0.012 becomes 0.033 after Holm over 21 claims.
  - The "3 of 8 cells" best-method count includes chain p=5 at 0.09 vs 0.10, untested.
  - The plan predicted no rank change and "reduced, not removed" for HX3. Both predictions failed. The paper should say so; it raises credibility.
- **W3 (minor; NEW-10).** Fixed-K size control is unchecked on the rivers, the most persistent series (K = 50/100). It cannot rescue a positive claim, but "controls size only away from a unit root" (Limitations l.846-847) should name the rivers as untested.
- **W4 (minor).** The independence approximation in the oracle (shared nodes make the count indicators dependent) should be quantified once, for example by Monte Carlo on one cell from the population covariance.

**Score justification.**
- Every technical item I raised in Rounds 1–2 that an agent can fix has been fixed on paper, so Overall goes 5 → 6.
- Soundness stays 3: no human has read app:erproof or the new App. D proofs; the oracle is overinterpreted; B3 inference is at the wrong unit.
- Contribution stays 2.

**Top 3.** B′4 human read (author-only); oracle decomposition and rewording; per-experiment B3 effects, plus the fixed-K check on the rivers.

---

## Reviewer 2 (causal discovery) — Overall 5 → 5 (borderline); Soundness 3, Presentation 3, Contribution 2; confidence 4

**Round-2 items (this seat).**
- **N1 (overclaim, contribution 5): RESOLVED** (l.422).
- **N2 (selective B1-NL reporting): RESOLVED** (l.463-470).
- **N3 (pooled labels): RESOLVED.**
- **N4 (Discussion contradiction): RESOLVED** (l.808-823), but see W3.
- **N5 (ground truth under G2): PARTLY.** Scoring without first-slice pairs is recommended and reported (l.430-433, l.818-821); orientation semantics against a PAG are still not discussed.
- **N6 (B3 D* coincidence): RESOLVED** (l.784-788).
- **N7 (citations): RESOLVED.**
- **N8 (AI statement): RESOLVED.**
- **N9 (DAGMA/NOTEARS omission): RESOLVED by B′1.**
- **DA6 (is burn-in standard?): RESOLVED as a survey,** but with the wrong criterion (W3).
- **Threat model:** now explicit (l.405-412, l.821-823).

**Strengths.**
1. B′1 is well designed:
   - authors' code or a gradient-checked port; defaults; raw and standardised data; graph-level units;
   - an empty-graph reference; PCMCI+ as a labels-using reference;
   - a size rule fixed before the run; one run, no deviations.
2. PCMCI+ recovering the window under both generators (SHD/|E| 0.02) makes the threat model concrete: the artefact matters only for methods that ignore time order.
3. The citations requested in Rounds 1–2 are in, with tools credited (FLOP, causal-learn, gadjid, flopsearch via the paper, tigramite).

**Weaknesses.**
- **W1 (major; NEW-2): the var-sortability attribution is plausible but not demonstrated, and one piece of evidence contradicts the paper's sentence.**
  - (a) Standardised DAGMA and GOLEM have ancestral order accuracy below 1/2 under both generators: DAGMA-std 0.35/0.41; GOLEM-EV/NV-std 0.33/0.42 and 0.31/0.41 (b1x README, Levels). They are not working estimators after standardisation, so "standardisation removes the gain" cannot distinguish "no scale-free effect" from "methods non-functional on standardised windows".
  - (b) NOTEARS-std, the one standardised CO method near 1/2, keeps a significant start effect (OA +0.036 [0.022, 0.049], SHD/|E| −0.077). That is a scale-free start effect on a competitive method. It contradicts "the scale-free artefact of Thms 2 and 5 moves the sort baselines" (l.488-490).
  - (c) R²-sortability is also scale-free, rises under G1 (+0.08 in B1) and is outside the theory; it is not considered.
  - (d) In B1 the var-sortability gain is the same on rings (+0.224) and chains (+0.231), yet the CO interaction against FLOP is reversed on rings and strongly favourable on chains. The README's levels argument (Var-SnR 0.95–0.96 vs 0.77) was formulated after seeing the result.
  - Fix: a variance-matched control. Rescale G2 windows column-wise to G1's population marginal variances, and the converse; rerun DAGMA/NOTEARS on chains and in-trees. Until then write "consistent with var-sortability", and report NOTEARS-std's residual and R² as alternatives.
  - Note on the "convenient escape" question: attributing the effect to var-sortability does not rescue the theory; it concedes that the theory does not explain the most consequential result. The problem is the reverse: title and abstract take credit for an effect that the theory disowns, on a weak test.
- **W2 (major; contribution).**
  - That continuous-optimisation methods exploit var-sortability on raw data is Reisach et al. (2021). That a fixed start creates a variance ramp across slices is elementary.
  - What is new is the observation that time-unrolled benchmarks with a fixed start hand this artefact to i.i.d. learners and change a ranking against FLOP on in-trees. That is useful but narrow:
    - one linear-Gaussian parametrisation, T ≤ 5, default hyperparameters, GOLEM at p=5 only;
    - in-trees only for the rank change;
    - the generator that the theory says is affected (Reisach et al. 2026) has no public data code.
  - The paper's own mechanism (rel-sortability) moves only baselines whose SHD/|E| is 1.70 (rel-SnR) under G2.
- **W3 (major; NEW-4).** The survey (l.520-534) concludes that most generators "meet B ≥ Δτ_max" and are therefore not, or only weakly, affected. By the paper's own Discussion (l.812-818) and B1's mixing check (burn-in raised from 50 to 100), the relevant criterion for the var artefact is mixing time, about log(tolerance)/log(spectral radius), not Δτ_max. Tigramite's ⌊0.2T⌋ for short series, CDML's 2τ_max and CausalTime's 20-step rollouts should be graded against mixing. A second axis matters too: most deployed generators produce one long record for a time-series method, which is outside the threat model. Say so.
- **W4 (minor; NEW-5, NEW-6).**
  - The abstract groups GOLEM with DAGMA/NOTEARS as beating FLOP on in-trees (GOLEM: G1 −0.011 and −0.042, not significant).
  - tab:b1x omits GOLEM-EV raw (SHD/|E| 0.44/0.93) and all standardised GOLEM rows (EV-std 0.97/1.30).
- **W5 (minor).** Orientation semantics under G2: excluding first-slice pairs fixes adjacency, not the latent-confounded v-structure issue (Round-2 N5). One sentence would do.

**Score justification.**
- Soundness 3 and Presentation 3, both improved in substance. Contribution stays 2.
- My Round-2 route to Contribution 3 was B′1. B′1 delivered a ranking change, but through a known artefact, and the attribution test is weak.
- If the variance-matched control confirms the attribution and the survey is regraded, I would move to 6. If the control instead shows a scale-free component in the CO effect, the theory gains a competitive-method consequence, and I would consider Contribution 3.
- Overall stays 5.

**Top 3.** Variance-matched control and rewording; regrade the survey by mixing and threat model; fix the GOLEM claim and complete tab:b1x.

---

## Reviewer 3 (formal methods and metascience) — Overall 6 → 6 (weak accept); Soundness 3, Presentation 3 → 4, Contribution 2; confidence 4

**Round-2 items (this seat).**
- **N1 (Δ forms under "checked"): RESOLVED.**
  - Verified: `rhoBurn_eq_maxFinDist` (Stationary.lean l.137) and `sBurn_eq_maxFinDist` (l.148) take `hB : maxFinDist E ≤ B`. `maxFinDist` (Walk.lean l.242) is the sup of finite distances, matching Δ in Sec. 2.
  - `sRho_unrolled_bounds_maxFinDist` covers Thm 2. `edgeLoops_burn_sharp` covers rem:sharp. The appendix proofs use Δ.
  - The lag extension B ≥ Δτ_max is labelled paper-only (l.316-317; app l.1444-1445).
- **Lean counts: VERIFIED.**
  - 163 axiom-report entries, all propext/Classical.choice/Quot.sound.
  - No sorry/admit/axiom; 4,159 lines in 16 files.
  - All 158 cited declaration names exist.
  - BUILD_LOG reports a clean rebuild of the final state (2026-11-09T08:06Z, 9,064 jobs, exit 0). My own re-run of the axiom file timed out, so the build itself is not re-verified here.
- **N2 (app:prereg out of date): PARTLY.** All plans are now listed (four real-data plus B1, B1-NL, B3, B′1; l.1018-1043), but with client commit times only.
  - I retrieved the server-side push events (GitHub activity API): B1-NL plan 04:10:47, B3 plan 04:21:12 → results 04:24:37, B′1 plan 08:20:31 → results 12:44:27. Each is 2–28 s after the client time.
  - The ordering claims are therefore true and verifiable; the paper should say so with the times (NEW-8).
- **N3: RESOLVED.**
- **N4 (clean rebuild): RESOLVED.**
- **C3 AI statement: RESOLVED in form.** No "(draft …)" heading. It names the Δ forms and states that no human has checked Lean-statement fidelity (l.874-876). Its unread-proofs list includes the new App. D results. Authors must still check it against the ICML 2027 policy.
- **C10 abstract Lean wording: RESOLVED** ("The exact and stationary results are machine-checked", l.101-102).
- **C14/A9 reproducibility: PARTLY** (NEW-7).
  - /tmp paths and exec loading are gone. There is a pinned requirements.txt, DATA.md with SHA-256 files, and reproduce.sh with stages.
  - But b1x/, finite2/ and lit3/ are not in reproduce.sh (REPRODUCE.md l.58: "under active development … Add them when they are final"), and requirements.txt omits dagma 1.1.1 and tigramite 5.2.10.1, which tab:b1x uses.
  - The Artefact Statement's "one-command reproduction … simulations" therefore does not cover a main-text table and figure.

**Strengths.**
1. The Lean side now matches the paper statement for statement on the exact and stationary results, including the Δ forms. The coverage paragraph (Sec. 6, l.636-657) says precisely what is not certified.
2. The process record is unusually auditable: every plan was pushed before its results (server-verified), deviations are logged, and B′1 had a size rule fixed before the run.
3. Presentation 3 → 4 for this seat: labels, status markers and disclosures are now consistent across the abstract, the body, the appendix and the AI statement.

**Weaknesses.**
- **W1 (moderate; NEW-8): push records order pushes, not computations.**
  - A plan pushed 3 min 25 s before results (B3), with a 2-minute run inside that window, is consistent with an honest process. It cannot exclude computation before the plan.
  - The whole Round-2 roadmap was executed by agents in about 5 h (07:44–12:57 on 9 Nov): new Lean proofs, new finite-sample propositions, a 4.2 h experiment, a code survey of about nine packages, and the rewrite. The paper discloses agent execution. Readers should still know that "planned in advance" here means minutes to hours, inside one agent session.
  - Only external registration by a human (B′5) changes this.
- **W2 (moderate, author-only).** Human checks:
  - No human has checked that the 33 headline Lean statements say what the paper says. For a paper whose main verifiable asset is the formalisation, this is the cheapest high-value check left, a few hours for someone fluent in Lean.
  - Likewise, no human has read app:erproof.
- **W3 (minor; NEW-7).** Add b1x and finite2 to reproduce.sh; add dagma and tigramite to requirements.txt.
- **W4 (minor).** The B′1 commit message calls the plan "pre-registered" (5ac1bd2), while the paper correctly avoids the word. Keep the supplement consistent (b1x/README.md also says "pre-registered").

**Score justification.**
- My three Round-2 blockers are fixed or nearly fixed: Δ labels, the disclosure layer and A9.
- Overall stays 6, not 7. Contribution 2 is unchanged (I agree with R2 that the consequential effect runs through var-sortability). The remaining process limits are author-only: no human check of Lean fidelity or proofs, no external registration.

**Top 3.** Human Lean-fidelity check and human proof read (author-only); server push times in app:prereg and consistent "time-stamped" wording in the supplement; complete reproduce.sh/requirements.txt.

---

## Devil's Advocate — argues reject for the ICML main track (borderline reject; weaker than Round 2)

**Strongest counter-argument.**

The revision answers the panel's "so what" question, and the answer undercuts the paper's thesis.
- The thesis, in Sec. 3 and in its Lean-certified core, is about rel-sortability.
- Asked whether the start artefact matters for competitive methods, the authors ran DAGMA, NOTEARS and GOLEM. They found the ranking against FLOP changes on in-trees, and then attributed the change to var-sortability "which our theory does not cover" (l.487-488).
- Var-sortability exploitation by these methods is Reisach et al. (2021). Variance growing from a noise-only first slice is a one-line observation.
- The rel-sortability effect, the paper's own, still moves only rel-SortnRegress. That baseline has SHD/|E| 1.70 under G2 and is worse than the empty graph in 88% of data sets (app:b1x).

The new detectability oracle then shows that, once the exact window covariances are known, the graph quantity has no explanatory weight (joint-regression coefficient −0.05). The theory is reduced to predicting the sign and the rank order of a baseline's gain.

The generator survey finds that mainstream generators burn in for 100–1000 steps or use stationary starts. The clearly affected construction is the time-unrolled proposal of Reisach et al. (2026), whose data code is not public.

The new title, "How the Start of a Simulation Makes … Benchmarks Sortable", is general enough to absorb the var-sortability result. The theorems it advertises explain the part of the effect that does not matter for competitive methods.

What remains:
- a correct, short, machine-checked completion of Reisach et al. §5;
- a clean benchmark-hygiene experiment on one proposed construction;
- 2.5 pages of finite-sample and real-data material that the title no longer covers.

That is a strong TMLR paper or workshop paper, not an ICML main-track contribution.

**Issues.**

CRITICAL
- **DA-C1 (significance / centre of gravity).** As argued above: the consequential effect lies outside the theory, the theory-covered effect moves only non-competitive baselines, the oracle makes the graph quantity redundant for size, and the exposed generator population is narrow. Location: abstract (iii), l.105-110; l.484-491; l.520-534; app:b1x (oracle joint regression, survey).
  - Fix: either (a) present the paper as benchmark hygiene, with the exact theory as the reason the burn-in rule is Δ and tight, and drop the implication that rel-sortability drives the consequential effect; or (b) show a scale-free component in a competitive method (the variance-matched control could show this; NOTEARS-std's +0.036 hints at it).
  - *Adjudicated in the decision file: upheld in part; caps Contribution; not a soundness defect.*

MAJOR
- **DA-M1 (NEW-1).** "The missing half is detectability, not another mechanism" (l.425-426). The oracle uses the full start-dependent covariances, so it cannot exclude a generic correlation-magnitude transient, my Round-2 alternative. It absorbs it. Decompose structural versus magnitude contributions, or withdraw the sentence.
- **DA-M2 (NEW-2).** The var attribution's key test (standardisation) uses methods that are below chance after standardisation. NOTEARS-std's residual gain is a scale-free effect the text then denies (l.488-490).
- **DA-M3 (NEW-4).** The survey's exposure criterion (B ≥ Δτ_max) is the structural one. The var artefact, which is what moves competitive methods, needs mixing. Under the right criterion, more short-series generators (tigramite ⌊0.2T⌋, CDML 2τ_max, CausalTime 20 steps) are exposed. That helps the paper's relevance, but the current text understates it for the wrong reason.
- **DA-M4 (NEW-3).** Title versus content: Secs. 5–7 are not about the start. Real data still occupies a full-width table in an 8-page paper whose title is about simulation.

MINOR
- **DA-m1.** The abstract's GOLEM claim (NEW-5).
- **DA-m2.** HX2/HX3 predictions failed and are not reported as such (NEW-6).
- **DA-m3.** "Planned in advance" (l.170-172) covers plans pushed minutes to hours before results, inside agent sessions. Say "time-stamped" consistently, including in the supplement's "pre-registered" (b1x README, commit 5ac1bd2).
- **DA-m4.** Reisach et al.'s data-level start is inferred, not verified (app:b1x). The main text calls it "the clearest affected case" (l.526-529) without that caveat.
- **DA-m5.** "A continuous method was the best label-free method in 3 of 8 cells" includes an untested 0.09 vs 0.10 cell.

**Alternative explanations.**
- B1 rel gain: a correlation-magnitude transient that the oracle cannot separate from loss of relatives (DA-M1).
- CO rank change: var-sortability (paper), R²-sortability, or the CO methods' sensitivity to the noise-only first slice's scale, which interacts with their sparsity penalty and fixed threshold 0.3 on raw data. A variance-matched control separates all three.

**So what.** For a narrow audience, yes:
- Benchmark designers who use time-unrolled i.i.d. windows should start from the stationary distribution, standardise, and not treat ring-type graphs as protected at T ≤ p.
- Practitioners using single-record time-series generators are mostly unaffected.
- Methods that use time labels are unaffected.

**Non-defects (conceded).**
- R-N1 is genuinely fixed in Lean.
- The re-analysis turned my R-N5 objection into a reported, correct statement.
- B′1 is honest: it reports a rank change that hurts the paper's theory-centric framing.
- Every number I checked matches its result file.
- The disclosures remain the best I have seen in this series.

**Recommendation.** Reject for the ICML main track in current form; weaker than Round 2. If DA-M1 to DA-M3 are fixed and the paper is reframed as benchmark hygiene (DA-C1, option a), I would not oppose a weak accept.
