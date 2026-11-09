# Simulated ICML 2027 re-review (Round 2): editorial decision and remaining roadmap

Manuscript: research/theory2/icml_paper/ at commit fb4a1d6. Round 1: `00_editorial_decision.md`, `01_reports.md` (commit 0e841cd). Round-2 reports: `03_rereview_reports.md`. Run 2026-11-09.

## Panel provenance (read before the scores)

- The same five role-separated seats as Round 1 (cards reused, not re-derived): Area Chair, Reviewer 1 (probability/statistics), Reviewer 2 (causal discovery), Reviewer 3 (formal methods/metascience), Devil's Advocate. Each ran in a fresh subagent context, read its own Round-1 report and the revised manuscript, and did not see the other Round-2 reports.
- All seats ran on the same model family and provider as the session that produced the paper, the proofs, the experiments and the revision. Persona separation is not independence; shared blind spots are likely. This is a pre-submission stress test, not a substitute for human referees or for a probabilist checking the proofs.
- The skill's #576 three-gate re-review contract (revision-blind criteria commitment, persuasion-blind evidence verdict, checker scripts, roadmap/adjudication sidecars) and the advisory attachments (acronym, suggested-reference, reviewer-claim checks) were NOT run. Seats verified claims directly against the manuscript, result files and git history. Reviewer 3 also read GitHub server-side push records.
- The orchestrating session spot-checked two reviewer claims and confirmed both (see R-N1 and R-A9 below).

## Scores

| Seat | Soundness | Presentation | Contribution | Overall R1 → R2 | Confidence |
|---|---|---|---|---|---|
| Area Chair | 3 | 2 → 3 | 2 | 4 → 5 (borderline) | 4 |
| Reviewer 1 | 3 | 2 → 3 | 2 | 4 → 5 (borderline) | 4 |
| Reviewer 2 | 3 | 2 → 3 | 2 | 4 → 5 (borderline) | 4 |
| Reviewer 3 | 3 | 3 | 2 | 5 → 6 (weak accept) | 4 |
| Devil's Advocate | — | — | — | argues reject for ICML | — |

Mean overall 4.25 → 5.25.

## Decision

**Borderline (leaning reject) for the ICML main track; clearly improved.** Every seat raised its overall score by one point. Presentation went up for three seats because the labelling and process overclaims of Round 1 are now fixed or disclosed. No seat raised Contribution above 2. The shared reason: B1 and B1-NL show that the fixed start shifts sortability and sort-based baselines in the order the exact values predict, robustly across parametrisations, but the best label-free method (FLOP) never changes and the ranking reversals were not detected in the replication. The graph theory is still a short step from Reisach et al. Lemma 1. Realistic outcome as submitted: borderline, depending on reviewers who value careful negative results, formal verification and benchmark hygiene. Better fit: TMLR/JMLR-style journal, a strong workshop, or ICML with a sharper benchmark story (see Tier B').

## Round-1 Devil's Advocate CRITICALs (adjudicated)

1. **Derivative novelty: PARTLY RESOLVED, still blocking a clear accept.** All seats agree. B1 adds evidence of practical relevance at the level of baselines. The most practice-relevant new result, that strongly connected summary graphs give s_rho(G_T) near 1 at the short horizons Reisach et al. propose (rings exactly 1 for T ≤ p), is under-sold: it is not in the abstract or the contributions (AC, R2, DA).
2. **Construct validity / estimand: RESOLVED as labelling** (AC, R2, R3; DA: "partly", because the finite-T unrolled object is tested only in simulation while the title still says "Real-Data Tests"). s-tilde is defined, its one-directional sign relation is Lean-checked, and the text says it is not the rel-sortability of any unrolled graph.
3. **E1 ties: LARGELY RESOLVED by B3, with caveats.** B3 beats constant-1/2 and the weak-component rule on held-out data with several sources per component. Caveats raised by R1, R2 and DA: on the primary set D* the eight naive rules and the theorem on G_D agree with the theorem, so the primary test discriminates only against the tie rules; separation from the naive rules rests on secondary S1, narrowly against G_D (21/30 units, Holm p 0.021 at α 0.025); same device; low-risk by the authors' own account; 30 units come from 3 experiments; no external registration; plan to results under 4 minutes.

## Round-1 consensus findings: status

| # | Status | Note |
|---|---|---|
| C1 ER labels | Resolved (labels); B4 human read pending | R1 re-derived the key steps and found no error. R3: keep "proved" conditional until a human reads it. |
| C2 stage0 exposure | Resolved | |
| C3 AI use | Partly | Heading still "(draft for the authors)"; omits Δ forms; "by hand" label; no statement on human check of Lean-statement fidelity. |
| C4 benchmark impact | Partly | Shown on baselines; no DAGMA/NOTEARS/GOLEM; no labels-using time-series method. |
| C5 E1 baselines | Largely resolved (B3) | See CRITICAL 3. |
| C6 estimand | Resolved | |
| C7 "pre-registered" | Resolved as wording; irreducible process limits remain | app:prereg not updated to seven plans. |
| C8 short horizon / Δ | Resolved mathematically; see R-N1 labelling | |
| C9 classical credits | Resolved | Plug-in n_eff consistency and F1 scope still open (R1). |
| C10 Lean coverage | Partly | Abstract still says "the graph results are machine-checked". |
| C11 focus | Partly | Title still promises finite-sample and real-data tests; Sec. 4 in main text. |
| C12 lags | Resolved (minor Δτ_max vs (p−1)τ_max inconsistency) | |
| C13 "only criterion" | Resolved | |
| C14 reproducibility (A9) | **Unresolved; the Tier A claim was inaccurate** | Confirmed: icml_exp/test4.py l.18-19 hardcodes /tmp paths; NOTES.md l.35 says no code was changed. |
| C15 Lemma 4 etc. | Resolved in writing; human check pending | |

## New findings in Round 2 (seats raising each)

| # | Finding | Seats | Verified by the session? |
|---|---|---|---|
| R-N1 | Δ forms sit under "checked" labels: Thm stat assumes B ≥ Δ but Lean `rhoBurn_eq`/`sBurn_eq` assume B ≥ p−1, with no caveat; Lean map says "checked"; appendix proofs and Prop. lags(d) still use p−1 while main text says Δ and Δτ_max | R3 (MAJOR), DA (minor) | Yes: Stationary.lean l.118, 129, 135 |
| R-N2 | Contribution (5) "by amounts the exact values predict" overclaims: observed gain ≈ half the gap (slope 0.53; NL 0.34) | AC, R1, R2, R3, DA | Numbers match analysis.txt |
| R-N3 | B1-NL "did not replicate" vs plan verdict H3-NL SUPPORTED (1 swap); SGN SID/orientation swaps of the same type as B1's largest; lower power (25/10 graphs, no PC); B1 used SID/orientation as support, B1-NL only SHD | R1, R2, R3 | — |
| R-N4 | B1 H1 is near-tautological: rel-SortnRegress order accuracy equals estimated rel-sortability in every family; it is estimator tracking, not algorithm impact | AC, R2 | — |
| R-N5 | Reversals are among non-competitive methods (SHD/|E| > 1 for rel-, R²- and random-SortnRegress, worse than an empty graph); add empty-graph reference | DA | — |
| R-N6 | Statistics: tab:b1 CIs treat 3 horizons per graph as independent (chain [.199,.235] vs graph-level [.209,.226]); tracking p treats 42 cells as independent (unit-level 0.616 unreported); no multiplicity statement across B1/B1-NL/B3; B3 replication is 3 experiments | R1 | — |
| R-N7 | Discussion recommendation: "SC graphs or burn-in ≥ Δ" contradicts the next clause; B ≥ Δ is graph absorption, distributional mixing needs longer (D1 raised 50 → 100); recommend stationary initialisation or mixing-checked burn-in, standardisation, window/ADMG scoring; move 0.044/0.273 cost to main text | R2 | — |
| R-N8 | B3 main paragraph omits that on D* the naive rules agree with the theorem, and omits the G_lin refinement failure in pol_1_90 | R2, DA, R1 | — |
| R-N9 | FLOP, PC, causal-learn, flopsearch, gadjid uncited; Weichwald 2020, CauseMe, DYNOTEARS, PCMCI+, CausalTime, Montagna absent; no survey of how deployed generators initialise | R2 | refs.bib to check |
| R-N10 | Bookkeeping: "+0.08 to +0.22 in most cells with T ≥ p" is 4/6; two unlabelled pooled values (+0.125 non-SC vs +0.133 all); appendix overview omits new appendices; "earlier drafts" sentence; default Impact Statement; B3 timing "minutes" (exact: 3-4 min); BUILD_LOG clean rebuild predates Observed.lean | AC, R1, R2, R3, DA | — |
| R-N11 | Finite-sample gaps: plug-in n_eff consistency; F1 cannot certify recovery for the Fisher-z thresholds used; 4th-cumulant terms; ER table c=1 S 1.8 s.d. off; B_ab absolute values | R1 | — |
| R-N12 | Mechanism alternative for B1: generic deterministic-start transient and detection horizon of thresholded correlations; propose an oracle analysis with exact window covariances | DA | — |

## Remaining roadmap (before about 22 Jan 2027)

Tier A' (writing and bookkeeping, days)
- A'1 (R-N1) Lift the Lean hypotheses to Δ (`hB : maxFinDist ≤ B`; same for `sRho_unrolled_bounds`), about one day; otherwise label Δ forms "paper + enumerated" everywhere. Rewrite appendix proofs and Prop. lags(d) with Δ.
- A'2 (R-N2, R-N3, R-N4, R-N5, R-N8, R-N10) Calibrate claims: "track, at about half the size"; state the B1-NL plan verdict and why it is discounted; "not detected at the replication's power"; name H1 as estimated-vs-exact tracking; empty-graph reference; B3 D* coincidence and G_lin failure in the main text; fix the listed bookkeeping.
- A'3 (C11, CRITICAL 1) Refocus: retitle around benchmark generators; lead the abstract and contributions with (i) strongly connected graphs are not protected at benchmark horizons, (ii) burn-in Δ is needed and tight, (iii) the fixed start inflates var- and rel-sortability predictably, (iv) the best method is unchanged; move Sec. 4 to the appendix; give B1 a figure; rewrite the Discussion recommendation (R-N7).
- A'4 (C3, C10) Final AI-use statement (Δ forms, "by hand", Lean-fidelity check); abstract "the exact and stationary results are machine-checked"; app:prereg updated to seven plans with server push times.
- A'5 (C14) Actually do A9: pinned environment, configurable paths, remove exec, CausalRivers SHA-256, one-command reproduction, clean Lean rebuild of the final state.
- A'6 (R-N9) Cite FLOP, PC, tools and missing literature, each verified before adding.

Tier B' (new analysis, weeks; the panel's route to Contribution 3)
- B'1 (C4, R-N5) Add DAGMA/NOTEARS/GOLEM (raw and standardised) and one labels-using time-series method as reference to B1; short survey of how deployed generators initialise.
- B'2 (R-N6, R-N3) Statistical hygiene: graph-cluster CIs, unit-level tracking, interaction-effect analysis with CIs instead of swap counting, one multiplicity statement.
- B'3 (R-N12) Detectability-oracle analysis of the B1 mechanism.
- B'4 (B4) Human probabilist reads app:erproof; plug-in Ŵ consistency lemma; F1 scope corrected.
- B'5 (CRITICAL 3) Any further confirmatory test: a human posts the plan to OSF before the agent touches the data, ideally another device or dataset.

## Honest bottom line

The revision moved the panel from 4.25 to 5.25: from "reject" to "borderline". The gains come from honesty and evidence quality (disclosures, a planned and replicated benchmark experiment, a discriminating held-out test), not from a larger contribution. Two problems were introduced or left by the revision process itself and must be fixed before submission: the Δ forms under "checked" labels (R-N1) and the incorrect claim that A9 was done. The single change most likely to move Contribution from 2 to 3 is B'1 plus the refocus A'3: show what happens to competitive continuous-optimisation and time-series methods, and lead with the short-horizon result for strongly connected graphs.
