# Simulated ICML 2027 review: editorial decision and revision roadmap

Manuscript: research/theory2/icml_paper/ (main.tex, appendix.tex at commit 0e841cd). Review run 2026-11-08.

## Panel provenance (read before the scores)

- Five role-separated seats: Area Chair (journal fit), Reviewer 1 (probability and statistics), Reviewer 2 (causal discovery), Reviewer 3 (formal methods and metascience), Devil's Advocate. Reviewer cards were confirmed by the user before the panel ran.
- Each seat ran in a fresh subagent context and did not see the other reports before committing its own.
- All five seats ran on the same model family and provider as the session. Persona separation is not independence: shared blind spots are likely. This is a pre-submission stress test, not a substitute for human referees or for a probabilist checking the proofs.
- The ARS sprint-contract machinery (paper-blind Phase 1, checker scripts) and the advisory attachments (acronym check, suggested-reference check, reviewer claim check) were not run. Suggested references in Reviewer 2's report were partly written from memory and must be verified before use.
- Per-seat reports (condensed, all findings and scores kept) are in `01_reports.md`.

## Scores

| Seat | Soundness | Presentation | Contribution | Overall | Confidence |
|---|---|---|---|---|---|
| Area Chair | 3 | 2 | 2 | 4 (borderline reject) | 4 |
| Reviewer 1 (probability/statistics) | 3 | 2 | 2 | 4 (borderline reject) | 4 |
| Reviewer 2 (causal discovery) | 3 | 2 | 2 | 4 (weak reject) | 4 |
| Reviewer 3 (formal methods/metascience) | 3 | 3 | 2 | 5 (borderline) | 4 |
| Devil's Advocate | — | — | — | argues reject | — |

## Decision

**Reject in current form for ICML main track (borderline; mean overall 4.25).** All four scored seats agree that the Section 3 graph theory is correct and well verified, and that the paper is unusually candid about failures. All four also agree the contribution is an incremental completion of Reisach et al. Section 5 whose impact on causal-discovery practice is asserted rather than shown, and that several labels and process statements overclaim. Reviewer 1 states that fixing W1-W5 could move the paper to weak accept; no seat saw a path to a clear accept without new evidence of benchmark impact.

## Devil's Advocate CRITICAL issues (adjudicated)

1. **Derivative novelty (main results a few lines from Reisach et al. Lemma 1 / App. A.3).** Validated: Area Chair W1, Reviewer 2 contribution score and justification, Reviewer 3 contribution score all concur; the manuscript itself says so (main.tex l.197-208). Not fatal for a smaller venue, but blocks an ICML accept unless the paper adds a result that changes practice (roadmap item B1).
2. **Construct validity: the real-data studies test a contemporaneous summary-level relatives sort, not rel-sortability of the unrolled DAG.** Validated: Reviewer 2 W3, Reviewer 3 W5. Fixable by defining the estimand and proving its relation to the stationary value (roadmap A6).
3. **E1's pass rests on ties, a graph refinement informed by the same device, and competitors that cannot express ties.** Validated: Area Chair W6, Reviewer 1 W5 (the theorem's prediction on G_lin coincides with a trivial weak-component-size rule on every instance), Reviewer 2 W4. Fixable only by a new pre-registered test with the trivial baselines on data with several source SCCs per component (roadmap B3).

## Consensus findings (number of seats raising each)

| # | Finding | Seats | Verified by the session? |
|---|---|---|---|
| C1 | ER limits labelled "proved" in main text, but appendix still says "sketch", "derived heuristically and not proved"; Lean map says "sketch + numerics"; full proof only in an internal file | AC W2, R1 W1 (CRITICAL), R3 W3 | Yes: appendix.tex l.429, 958, 972, 985, 1019 |
| C2 | CausalRivers was analysed (correlations, hidden-driver diagnostic 0.51 vs 0.51) in research/ts/stage0 the day before "pre-registration"; not disclosed; Protocol sentence inaccurate | R1 W4, R3 W1 (CRITICAL), DA 4 | Yes: stage0 commits 3cd4581-814af68 precede theory2 PLAN.md |
| C3 | No AI-use disclosure although proofs, Lean, experiments and draft were agent-produced | AC W7f, R3 W2, DA 9 | Yes (NOTES.md decision 2) |
| C4 | No experiment shows any algorithm's benchmark result changes (finite vs stationary generator); time labels orient every cross-time edge, so the threat model must be stated; latent burn-in breaks causal sufficiency | AC W3, R2 W1 (CRITICAL), R2 W2, DA 6 | Not tested |
| C5 | E1 lacks constant-1/2 and weak-component-size baselines; tie-dominated; 10 units from one series | AC W6, R1 W5, R2 W4, DA 3 | — |
| C6 | Real-data estimand (s-tilde: observed subset, latent nodes, loops dropped) differs from the theorems' object; sign preservation unproved | R2 W3, R3 W5, DA 2 | — |
| C7 | "Pre-registered" overstates: same-day, client-set timestamps, plan 3 and 4 committed 3.5-4 min before results, development runs on study-3 data, no third-party registry | AC W5, R1 W4, R3 W6, DA 4 | Yes (git log) |
| C8 | Short-horizon regime ignored: Thm 2 bound vacuous for T <= p, but Reisach et al. propose short trajectories; R1: p-1 can be sharpened to the max finite distance Delta (O(log p) for ER), also for burn-in | AC W1, R1 W2d, R2 W2 | — |
| C9 | Classical results claimed or left unsourced: Fejer bound (Brockwell-Davis), three-indicator identity (Bollen 1989), joint CLT is a theorem (Breuer-Major / Arcones 1994), "inconsistency" should be "size distortion" | R1 W3, R2 W6, DA 10 | Citations to verify |
| C10 | Lean coverage overclaimed ("all graph results"); 133 vs 138 inconsistency; give breakdown headline/lemma/identity | R3 W4, R3 W8, DA 8 | 133 vs 138: to check |
| C11 | Too many contributions; abstract overloaded with study codes; refocus on the benchmark story | AC W4, R2 W10, R1 W8, DA 13 | — |
| C12 | Lag-1-only unrolling restricts Thm 1/2 (aggregation section assumes general lags) | R2 W7 | — |
| C13 | Overclaim "rel-sortability is the only criterion that is graph-only and observation-invariant" (l.503) | AC W7a, DA 12 | — |
| C14 | Reproducibility: hardcoded /tmp paths, no environment file, no CausalRivers hashes | R3 W7 | — |
| C15 | Lemma 4 applied beyond its statement (non-invariant parent choice, unbounded sum); forced-edge check not written out; Case II-1 informal | R1 W2a-c | — |

## Revision roadmap (before about 22 Jan 2027)

Tier A: disclosure, relabelling and writing (days, no new research)

- A1 (C1) Replace Appendix L with the full X1PROOF argument (sections 2-7) or relabel the ER limits "derived; agrees with exact computation" everywhere (abstract, contributions, Prop. title, Fig. 3, Lean map, Limitations). Fix the iterated-limit wording.
- A2 (C2, C7) Disclose the stage0 exposure in App. H and reword the Protocol sentence; reclassify the CausalRivers F2 account as explanatory; rename "pre-registered" to "time-stamped analysis plans" unless a third-party registry exists; supply server-side push logs or an anonymised mirror.
- A3 (C3) Add an AI-use statement: which artefacts were AI-generated, which were human-verified and how, which paper-only proofs no human has read. (Authors' decision; check the ICML 2027 policy.)
- A4 (C9) Credit Brockwell-Davis for the Fejer bound, Bollen for the three-indicator identity; replace the assumed joint CLT with the Breuer-Major/Arcones theorem; rename "inconsistency" to "size distortion". Verify every citation before adding.
- A5 (C10, C13) Reword Lean coverage claims; add a breakdown of the 138 declarations; fix 133/138; drop the "only criterion" overclaim; quote Reisach et al.'s s_rho definition verbatim.
- A6 (C6) Define the summary-level estimand s-tilde formally and prove (paper, ideally Lean) its sign relation to the stationary value.
- A7 (C12) State that Thms 1-2 are for VAR(1)-shaped unrollings, or generalise to lag-weighted distances.
- A8 (C11) Cut the abstract to about 180 words without study codes; move aggregation details, F1/F3 details, no-loop section, E1/E2 to the appendix; reword "fails as the theory predicts" to "failed; consistent post hoc with F2/F3".
- A9 (C14) Pinned environment, configurable data paths, CausalRivers SHA-256, one-command reproduction.

Tier B: new research (weeks; decides whether ICML is realistic)

- B1 (C4) Benchmark-impact experiment: finite vs stationary unrolled generators on strongly connected, chain/in-tree, river-like and ER(c) summary graphs at T in {3, 5, 10}; report s_rho, var- and R2-sortability, and rel-SortnRegress, R2-SortnRegress, FLOP and a random baseline with time labels hidden (SHD/SID). State the time-order threat model and the sufficiency cost of latent burn-in (or stationary initialisation instead).
- B2 (C8) Exact s_rho(G_T) vs T curve for small T from Thm 1; sharpen p-1 to Delta in Thms 2 and 4.
- B3 (C5) New pre-registered E1 on held-out data with several source SCCs per component, with constant-1/2, weak-component-size and G_D-theorem competitors; register externally (OSF) before data access; foreground the scm_5 collider.
- B4 (C15, C1) Write out Lemma 4 for the quantities used, the forced-edge conditional-independence checks and Case II-1; have a human probabilist read X1PROOF.

## Honest bottom line

Tier A removes the overclaims and process problems the panel would otherwise punish, but by itself it does not change the contribution score. The panel's consensus is that ICML acceptance hinges on Tier B1: showing that the finite-start artefact changes how algorithms rank on benchmarks. Without B1 the realistic venues remain a workshop, a journal (Machine Learning / ECML-PKDD journal track), or arXiv plus contact with Reisach et al.
