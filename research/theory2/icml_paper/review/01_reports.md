# Per-seat reports (condensed)

Condensed by the orchestrating session from the five full reports returned on 2026-11-08. Every weakness, its location and fix, the scores, and the top-3 fixes are kept; prose is shortened. Line numbers refer to main.tex / appendix.tex at commit 0e841cd.

## Area Chair (journal fit) — Overall 4 (borderline reject); S3 P2 C2; confidence 4

Strengths: graph core correct and complete (Thm 1-2, Cor 3; checked against Reisach App. A.3); stationary-start result is a real design insight (Prop 4, Thm 5, Cor 6, Table 1); unusually strong verification (Lean, 46,724-case enumeration, four missing hypotheses found); informative loop-free counterexamples; honest real-data reporting (Table 2, out-of-sample Table 3, n_eff diagnostic l.878-880); candid credit to classical work.

Weaknesses:
- W1 Core advance thin; paper presents itself as an addendum (l.197-208). Thm 1 is ~5 lines from Lemma A.1 + Reisach Lemma 1; necessity is a one-line witness. Thm 2's bound is vacuous for T <= p, exactly the short-trajectory regime Reisach et al. propose. Fix: reframe around short horizons and the start artefact; drop the "a referee should read..." sentence.
- W2 Main text and appendix disagree on ER status (abstract l.103, l.170, Prop 8 l.412, Fig. 3 vs appendix l.429, 958-985, 1019). L(c) is close to a corollary of known giant-SCC edge fractions. Fix: full proof in Appendix L or relabel consistently; credit prior work for L(c).
- W3 Benchmark significance argued, not demonstrated; no algorithm run on finite vs burn-in generators. Burn-in makes burn-in nodes latent common causes (MAG/PAG ground truth); not discussed. Fix: one simulation table at T in {3,5,10}; paragraph on burn-in ground truth.
- W4 Seven contributions, two competing storylines (benchmark artefacts vs estimating summary relatives from one record); abstract ~330 words with undefined codes (P9, E2). Fix: pick the benchmark story; move Sec. 4, F1/F3, E1/E2, no-loop to appendix; abstract ~180 words.
- W5 Pre-registration weaker than worded: ~5.9 h one day, agent-driven, client-set timestamps, plan 4 3 min 36 s before results; F1-F3 developed after study 1 failed; G_lin informed by study-3 observations on the same device. Fix: rename "pre-specified analyses (same-day, internally time-stamped)"; disclose; supply push records.
- W6 E1 favours the theorem through ties (1,365 of 1,662 predicted 1/2; competitors cannot output 1/2); constant-1/2 only post hoc (81.5%; wind tunnel 0.311 vs 0.363). Fix: lead with the 297 strict instances and constant-1/2 / "tie iff same weak component" baselines, or move E1 to appendix.
- W7 Smaller: (a) "only criterion" overclaim l.503-504; (b) assumed joint CLT should be in theorem headers; (c) state the s-tilde formula actually used; (d) garbled limit-order wording l.887-888; (e) addressing "a referee" l.205; (f) no AI-use disclosure.

Questions: short-T values for ER c in {1,2,4}, T in {2..10}; intended ground truth under burn-in; any ranking change between generators; complete proof of d; timing of the G_lin decision; var/R2 under stationary start.

Top 3 fixes: benchmark-impact table + burn-in ground-truth paragraph; consistent ER status labels; refocus, rename pre-registration, disclose G_lin motivation and AI use.

## Reviewer 1 (probability and statistics) — Overall 4 (borderline reject); S3 P2 C2; confidence 4

Strengths: exact graph results correct (Table 1 re-derived); off-paper ER proof (X1PROOF §3-7) is serious work, reduction of E_up to a radius-(2K+1) local functional essentially correct in spot-check; independent numerics agree (p=20,000, 3 seeds: c=1 L 0.7498 vs 0.7500, S 0.6579 vs 0.6580; c=2 L 0.6214 vs 0.6217, S 0.5634 vs 0.5632; peaks 0.7542 at c=1.073, 0.6650 at c=1.09); F1 Thm 1.5 correct line by line; Cor 1.8 reproduced in simulation (0.286 predicted vs 0.288); candid reporting.

Weaknesses:
- W1 CRITICAL: ER result labelled "proved" but proof not in the submission and appendix contradicts the label (l.103, 170, 203, 412 vs appendix l.429, 958-1019). Fix: move X1PROOF §2-7 into app:er; delete obsolete sketch text.
- W2 MAJOR: abbreviated steps. (a) Lemma 4 is stated for isomorphism-invariant bounded functionals but Case I-b uses a non-invariant "smallest label" parent and an unbounded sum (fix: union over parents, truncate in-degree, uniform integrability); (b) forced-edge claim asserted, conditional-independence checks not written out; (c) Case II-1 (B2) informal (fix: expectation on high-probability event + Cauchy-Schwarz); (d) iterated limit misstated; sharpen p-1 to max finite distance Delta (O(log p) for ER) in Thm 2 and for burn-in in Thm 4. Timeline: PLAN6 21:26, full proof 21:54; recommends a human probabilist check.
- W3 MAJOR: F-theory does not cover the estimator used (plug-in n_eff from truncated ACFs; Fisher-z thresholds keep the F1 exponent bounded, so F1 cannot certify recovery); the joint CLT is not an assumption but follows from multivariate Breuer-Major (Arcones 1994); "inconsistency" is a misnomer (size distortion); Gaussianity vs heavy-tailed river data. Fix: cite Arcones; size result or explicit absence for plug-in n_eff; rename; note non-Gaussian 4th-cumulant terms.
- W4 MAJOR: pre-registration credibility; undisclosed earlier exposure (stage0 hidden-driver diagnostic 0.51 known before plan 1); abstract "fails as the theory predicts" reads as a prediction; plans and results within ~6 h, client-set timestamps (plan 3 09:13:26 / results 09:17:07; plan 4 10:05:48 / 10:09:24). Fix: disclose; third-party registry for new studies; reword abstract.
- W5 MAJOR: on G_lin the theorem's prediction equals the trivial weak-component-size rule on all 1,662 instances; competitors cannot tie; constant-1/2 beats the theorem on G_D; G_lin depends on unverifiable B2; 10 units of one series; Holm p at its floor. Only scm_5 (i.i.d. collider) and parts of buchholz/scm_2 discriminate. Fix: new held-out pre-registered E1 with component-size, constant-1/2 and G_D competitors on data with several source SCCs per component.
- W6 MAJOR: P9/Q9 underpowered (redraw moves P9 0.267 -> 0.338), arbitrary 0.3, clustering ignored; no family-wise accounting over ~20 criteria; many post hoc analyses. Fix: power/simulation calibration, cluster-level inference, state which claims survive family-wise correction.
- W7 MINOR: in-sample F3 check (0.743 vs 0.741) near-tautological; Table 3 designed after study 3.
- W8 MINOR: overloaded abstract, inline Lean names; "l = 0.64 at n ≈ 25" vs n_1/2 ≈ 22; D vs D_gamma; failure-probability statement omits prefactor; proved vs sketched in theorem headers.

Proof spot-check log: Thm 1/2/Cor 3 correct; Thm 4/Cor 5 correct (burn-in sharpenable to Delta); X1 Lemma 1 constants correct; forced-edge variant correct in principle, write-out missing; Lemma 2 correct; Cor 3 correct given Lemmas 1, 2, 4; Lemma 4 acceptable but applied beyond its statement; Prop 5 correct; Prop 6 (fixed point d) correct, matches numerics; Prop 7 correct in substance, W2a/W2c need tightening; Props 9/10 correct; F1 Lemmas 1.1-1.4 and Thm 1.5 correct; F2 constants correct; F3 r_crit correct; app:er inconsistent with main text.

Top 3 fixes: full ER proof in appendix + W2a-c + Delta sharpening + human probabilist; disclose stage0 and third-party timestamps; rerun E1 with proper baselines, Breuer-Major for the CLT, size check of plug-in n_eff.

## Reviewer 2 (causal discovery) — Overall 4 (weak reject); S3 P2 C2; confidence 4

Strengths: necessity closes a gap Reisach et al. leave open; boundary-effect diagnosis useful to benchmark designers; classical credit now correct; failures reported honestly; out-of-sample F check and n_eff diagnostic practical.

Weaknesses:
- W1 CRITICAL: no connection to any algorithm (rel-SortnRegress, FLOP, R2-SortnRegress, DYNOTEARS) and no MEC analysis; every edge of G_T goes from slice t to t+1, so any method seeing time stamps orients cross-time edges for free; rel-sortability of G_T is a shortcut only for time-agnostic methods. Fix: experiment with time labels hidden (two generators x {strongly connected, chain/in-tree, ER c≈1} x T in {3,5,10}); state the threat model.
- W2 MAJOR: recommended generator has side effects: latent burn-in breaks causal sufficiency; var/R2-sortability of G_T under each start not analysed (compare Lohse-Wahl); finite-T regime neglected (bound vacuous for T <= p). Fix: three analyses or at least the sufficiency trade-off and a small-T curve.
- W3 MAJOR: real-data estimand (lag-0 correlations between p series along one record, self-loops dropped, ergodicity needed) is not rel-sortability of G_T. Fix: define it formally, prove its relation to s^W, relate to Lohse-Wahl; retitle claims.
- W4 MAJOR: E1 tests ties and the refined graph; 297 strict instances come from a handful of distinct edges; theorem is deterministic so E1 tests faithfulness of the assumed graph. Fix: constant-1/2 and naive rel_S-count competitors; per-edge-type results; separate tie vs strict accuracy; reframe.
- W5 MAJOR: CausalRivers framing overclaims (pre-registered prediction was success, failed; F2 post hoc; F3 explanation in-sample). Fix: "failed; post hoc F2/F3 analysis consistent".
- W6 MAJOR: novelty overclaims: Thm agg(a) is textbook Cesàro/Fejér (Brockwell-Davis 1991 Thm 7.1.1; Hamilton 1994); aggregation-induced instantaneous dependence classical (Weiss 1984; Breitung-Swanson 2002); saturation is the "crud factor"; three-indicator identity is Spearman/Bollen 1989 — "did not trace its source" is not acceptable.
- W7 MAJOR: lag-1-only unrolling; community full-time graphs allow lags 1..tau_max and contemporaneous edges; internally inconsistent with Sec. agg. Fix: state VAR(1) restriction or generalise to lag-weighted distances; burn-in >= (p-1) tau_max.
- W8 MINOR: E2 reinvents classical cross-correlation independence tests (Haugh 1976 prewhitening; Hong 1996).
- W9 MINOR: Reisach 2026 adds nuance to Ormaniec on iSCM R2-sortability; mention it.
- W10 MINOR: presentation (abstract, internal labels, rebuttal-like phrasing, inline Lean names).
- W11 MINOR: ER limits tangential; Reisach propose symmetrised ER (strongly connected); justify directed G(p,c/p).

Missing references (partly from memory; verify): Weichwald et al. 2020; FLOP (Wienöbst, Henckel, Weichwald, ICLR 2026); Brouillard et al. 2025; CausalTime; CauseMe (Runge et al. 2019 Nat. Commun.); DYNOTEARS; PCMCI+; Montagna et al. 2023; Andrews-Kummerfeld 2024; Seng et al. 2024; Kaiser-Sipos 2022; Gentzel et al. 2019; Reisach-Suárez-Weichwald-Chambaz; Peters-Janzing-Schölkopf 2013; Assaad et al. 2022; Plis et al. 2015; Hyttinen et al. 2016; Breitung-Swanson 2002; Weiss 1984; Brockwell-Davis 1991; Haugh 1976; Granger-Newbold 1974; Yule 1926; Bollen 1989; Malinsky-Spirtes 2018; Schultheiss-Ulmer-Bühlmann 2025. (Reviewer confirmed by web search only Weichwald 2020, FLOP and Schultheiss et al.)

Top 3 fixes: algorithmic relevance experiment + s_rho-vs-T curve + threat model + sufficiency cost; define the estimand and reword claims (CausalRivers, E1, agg(a), three-indicator identity); complete literature and state/generalise the lag-1 restriction.

## Reviewer 3 (formal methods and metascience) — Overall 5 (borderline); S3 P3 C2; confidence 4

Verified directly: lake build exit 0 (9063 jobs); AXIOMS.txt 138 entries, standard axioms only; no sorry/admit/axiom; 12 statements compared (audit log in the full report): Thm 1, Thm 2 bounds, Cor 3, Prop 4, Thm 5, Cor 6, loop-free edge formula and accuracy reading match; F1 (Thm 7) and F2 (Prop 9) deterministic parts only; F3 (Prop 10) population part matches; ER (Prop 8) lemmas only.

Strengths: Lean/paper boundary clearly marked; Section 3 definitions faithful (IsRel = desc(anc), unrolled graph, dist = loop-free distance, burn-in window); honest hypotheses (self-loop hypothesis exactly where assumed; four missing hypotheses surfaced); failed tests reported; careful E1 design check (generator replay, SHA-256 hashes).

Weaknesses:
- W1 CRITICAL (for the pre-registration claim; fixable): research/ts/stage0 (commits 3cd4581-814af68, 2026-11-03) ran correlation/VAR/PCMCI/DYNOTEARS on the same CausalRivers regions and resolutions and found unrelated stations correlate as strongly as siblings; PLAN.md reuses its loaders; not disclosed; Protocol sentence inaccurate. Fix: disclose in App. H; reword; present F2 account as explanatory.
- W2 MAJOR: no AI-use disclosure; readers need it to calibrate trust in paper-only proofs no human has read.
- W3 MAJOR: "proved" for ER not supported at the implied level (agent-written in under 30 minutes; Appendix L an outline; only witness lemmas in Lean).
- W4 MAJOR: Lean covers easy steps; F1-F3 Lean content is elementary algebra (e.g. abs_sOf_sub_half_le is s in [0,1]); risky steps unformalised; "all graph results" overclaims (ER, loop-free limit/periodicity, loop-free stationary value are paper-only). Fix: reword coverage; one sentence that no probability bound is certified; breakdown of the 138.
- W5 MAJOR: the theorem-to-experiment bridge (s-tilde over induced non-loop edges with latent nodes) is not formalised; Lean's statValue averages over all edges of the fully observed graph. Fix: Lean lemma on sign preservation, or state s-tilde is paper-defined.
- W6 MAJOR: pre-registration evidence weak in an agent-driven single-day pipeline; no third-party timestamp; agent could read earlier analyses. Fix: server-side push records or scrubbed mirror (git metadata contains a real identity); disclose workflow; "time-stamped analysis plans"; OSF for future studies.
- W7 MINOR: reproducibility (hardcoded /tmp/claude-0 paths; no environment file; exec-based code loading; no CausalRivers SHA-256).
- W8 MINOR: inconsistencies: app:lean says 133 main theorems vs 138; Lean README still says 133 in X2 section; Src restatement should be noted in main text.
- W9 MINOR: quote Reisach et al.'s s_rho definition verbatim (edge-averaged vs pair-averaged).

Top 3 fixes: disclose process (AI use, stage0 exposure, push records); relabel claims to match verification (Lean coverage, ER "derived", 133/138, breakdown); close theorem-to-experiment gap and make code reproducible.

## Devil's Advocate

Strongest counter-argument (condensed): the theory is a short extension of Reisach et al.; their App. A.3 already has the absorption step, and crossing edges plus absorption give Thm 2, Cor 3 and the stationary theorem (rel-sortability of the condensation with self-loops as ties). The real-data layer never tests the time-unrolled object: it estimates contemporaneous summary-level correlations from one long record. Six pre-registered graded criteria failed (P2, P3, P4, P6, P9, Q9); the one clean pass (E1) came last, with the most knowledge, on a graph refined using earlier same-device evidence, against competitors that cannot express ties; on the published graph the theorem loses. The programme ran in about 6 hours by agents with client-set timestamps, CausalRivers had already been analysed, and there is no AI disclosure. No evidence that any evaluation practice should change.

Issues: (1) CRITICAL derivative novelty; (2) CRITICAL construct validity of real-data tests (Prop 7 makes them tests of summary-graph rel-sortability; the finite-T i.i.d.-trajectory setting is never evaluated); (3) CRITICAL E1 rests on ties, refined graph, weak competitors; 10 units from one series, Holm p at floor; (4) MAJOR process credibility (stage0 exposure, minutes-long plan-to-result gaps, no registry); (5) MAJOR reframing of failed tests (abstract wording; F3 in-sample; P9 redraw softens a fail); (6) MAJOR no "so what" (no algorithm run; burn-in already standard practice in DoTime/TimeGraph/CauseMe); (7) MAJOR overgeneralisation from one device; (8) MAJOR Lean adds confidence where least needed (toy theorems inflate the count; risky parts paper-only); (9) MAJOR undisclosed AI production; (10) MAJOR finite-sample theory textbook; (11) MINOR ER limits of limited relevance (order of limits, L(c) from known x^2); (12) MINOR "only criterion" overclaim; (13) MINOR presentation (title, abstract, default impact statement).

Ignored alternatives: E1 as structure recovery (any relatives count on G_lin wins); ties from component structure ("same component means tie" rule absent); CausalRivers failure from known hydrological shared forcing; chamber passes from block structure of independent actuators; P9 variability means little power either way. Paths not taken: test the finite-T i.i.d.-trajectory setting with real algorithms; nonparametric/lagged dependence measures; validate B2 on lt_malus_v1.

Missing stakeholders: benchmark maintainers (does any deployed generator use a fixed start?); method developers; domain scientists; Reisach et al. (camera-ready may contain necessity); metascience community; formal-methods reviewers.

Conceded merits: candour; correct, machine-checked graph results with counterexamples; the finite-start inflation point is clean and useful; ER curves match exact computation; the n_eff diagnostic is practical; the anti-sorted prediction (99.1%) is a risky prediction that came true; careful documentation.

"So what?" verdict: careful and honest completion of Reisach et al., but the theorems are a few lines from their Lemma 1 and Thm 4 proof, the real-data tests check static summary-graph relatives, and the one clean positive depends on ties and a refined graph; it does not show any practice should change, so it does not clear the ICML bar.
