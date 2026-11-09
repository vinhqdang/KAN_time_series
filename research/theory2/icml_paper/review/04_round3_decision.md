# Simulated ICML 2027 re-review (Round 3): editorial decision, traceability matrix and Round-4 roadmap

Manuscript: research/theory2/icml_paper/ at commit e6f4c54 (main.tex, appendix.tex, refs.bib, main.pdf, 40 pages; main text ends on page 8). Round 1: `00_editorial_decision.md`, `01_reports.md`. Round 2: `02_rereview_decision.md`, `03_rereview_reports.md`. Round-3 per-seat reports: `05_round3_reports.md`. Run 2026-11-09. Line numbers refer to main.tex / appendix.tex at e6f4c54.

## Judge record and provenance (read before the scores)

- **Verification judge:** one model (Claude family) running all five seats inline in one context, one after another. The seats are role-separated personas, not independent reviewers. They share one context, one evidence pass and one set of blind spots. Unlike Rounds 1 and 2, no seat ran in a fresh subagent context. This round is therefore less independent than Round 2.
- **Same family as the authoring agents.** This verification round ran on the same model family that drove the revisions; over-optimization to this judge's latent biases is possible (Ren et al. 2026, arXiv:2607.13104 §8.1.2).
- **Blind cross-model pass:** not_configured. The single-family disclosure above applies to every row.
- **Contract:** the skill's re-review protocol (`references/re_review_mode_protocol.md`, three-gate contract) was approximated, not executed. Phase 0 was not rerun. Phase 1 (criteria commitment) was approximated by taking the Round-2 roadmap text and the Round-2 per-seat fixes as the acceptance criteria before the revised manuscript was opened. Phase 2A (evidence verdicts) was done against the manuscript, the result files and git/GitHub records. `icml_paper/NOTES.md`, which plays the role of the response letter, was read after the manuscript and the result files but before the verdicts were written down. Persuasion-blindness is therefore not guaranteed. No precommitment hash, no sidecars, and `scripts/check_re_review_synthesis.py` was not run.
- **Reviewer configuration:** round1_cards_reused. The seats and personas are the same as in Rounds 1 and 2: Area Chair / journal fit, R1 probability and statistics (methodology), R2 causal discovery (domain), R3 formal methods and metascience (perspective), Devil's Advocate. The Round-1/Round-2 scores are the yardstick.
- **Routing:** card_mapped. Each item was verified by the seat that raised it in Round 2.
- **Evidence seen:** the revised manuscript (.tex and pdftotext of main.pdf); the Round-2 manuscript, through the Round-2 reports and the git history; b1/, b1nl/, b1x/ (PLAN, README, results/analysis.txt, b2_reanalysis, oracle, multiplicity); b3/; finite2/FINITE2.md and the K-sensitivity output; lit3/; lean/ (sources, AXIOMS.txt, BUILD_LOG.txt); REPRODUCE.md, requirements.txt, DATA.md; NOTES.md; and GitHub server-side push events for the repository.
- **What the session verified itself:**
  - All 158 Lean declaration names cited by `\lname{}` in main.tex/appendix.tex exist in `lean/RelSort/*.lean`. The two remaining tokens are file names.
  - AXIOMS.txt has 163 "depends on axioms" entries, all on the three standard axioms.
  - No `sorry`, `admit` or `axiom` in the sources.
  - The sources have 4,159 lines in 16 files.
  - `rhoBurn_eq_maxFinDist` and `sBurn_eq_maxFinDist` take `hB : maxFinDist E ≤ B` (Stationary.lean l.137, l.148).
  - `lake env lean AxiomsDelta.lean` was started but timed out after 5 min. The build is therefore not re-run here; the session relies on BUILD_LOG.txt (clean rebuild 2026-11-09T08:06Z, exit 0).
  - Numbers spot-checked against the result files: tab:b1x, the B1 per-cell gains, "4 of 6 cells", +0.125/+0.133, Var-SnR +0.202/+0.002, G2 vs exact start 0.013, 11/1,190 swaps, 294/595 interactions, SHD/|E| 1.39–2.04, oracle R² 0.97 and slope 1.03, ring +0.094/+0.079, slice-1 exclusion +0.089, Holm over 21 claims (B1-NL H3-SGN p 0.108), K=50 size 0.113, B3 K-sensitivity 0/2,730, the Lean counts. All match, with one exception: the abstract's GOLEM claim (NEW-5).
  - Server push times (GitHub activity API) confirm that every plan was pushed before its results: B3 plan 04:21:12 → results 04:24:37; B'1 plan 08:20:31 → results 12:44:27. Client commit times are 2 to 28 s earlier than the push times.
- This is a pre-submission stress test. It does not replace human referees, a human probabilist reading the proofs, or a human checking that the Lean statements match the paper.

## Scores

| Seat | Soundness | Presentation | Contribution | Overall R1 → R2 → R3 | Confidence |
|---|---|---|---|---|---|
| Area Chair | 3 | 3 | 2 → 3 | 4 → 5 → **6** (weak accept, borderline) | 4 |
| Reviewer 1 (stats) | 3 | 3 | 2 | 4 → 5 → **6** (weak accept, borderline) | 4 |
| Reviewer 2 (causal discovery) | 3 | 3 | 2 | 4 → 5 → **5** (borderline) | 4 |
| Reviewer 3 (formal/metascience) | 3 | 3 → 4 | 2 | 5 → 6 → **6** (weak accept) | 4 |
| Devil's Advocate | — | — | — | argues reject for ICML main track (weaker than Round 2) | — |

The mean overall score moves from 5.25 to **5.75** (+0.5). Two seats moved up by one point (AC, R1). Two did not move (R2, R3). Contribution moved for one seat only: AC, from 2 to 3. R2, whose Round-2 top fix was B'1, held at 5. The reason: B'1 did change a competitive ranking, but the paper itself attributes that change to var-sortability. Var-sortability is a known artefact (Reisach et al. 2021) and lies outside the paper's theory. The theory-covered artefact still moves only the sort baselines, which are worse than the empty graph on SHD under the burn-in.

## Decision

**Protocol state: Major Revision** (`reject_recommended: false`). Rule B3 of the decision derivation fires because three new issues introduced by this revision are rated major (NEW-1, NEW-2, NEW-4). The must_fix A′ items are FULLY or PARTLY addressed, each PARTLY with a should_fix residual. The should_fix B′ items are addressed at 4/5 = 80%, so B5 alone would have given Minor.

**ICML-style recommendation: borderline, leaning weak accept (mean 5.75); not yet safe.**
- The revision fixed every Round-2 labelling defect (R-N1 in Lean, R-N2 to R-N5, R-N7 to R-N10).
- It added the competitive-learner experiment the panel asked for.
- On the evidence it is now a careful, honest, well-verified paper.

Three things keep it short of a clear accept:
- (a) The consequential benchmark effect (DAGMA/NOTEARS overtaking FLOP on in-trees) is attributed to var-sortability, outside the theory. The attribution itself rests on a weak test (NEW-2).
- (b) The detectability oracle is presented as mechanism evidence ("not another mechanism"). It is an analytic expectation of the algorithm and cannot separate the paper's structural mechanism from a correlation-magnitude transient (NEW-1). It also shows that the graph quantity adds nothing once covariances are known.
- (c) The paper is still two papers in eight pages. The new title fits Sections 3–4 but not Sections 5–7, which take about 2.5 pages (NEW-3).

Realistic outcome if submitted as is: a borderline score spread (5–6), decided by whether the reviewers value benchmark hygiene and formal verification over theoretical novelty. TMLR remains an excellent fit.

## Devil's Advocate CRITICAL (adjudicated)

**DA-C1 (significance / centre of gravity).** The finding that matters for practice (a continuous-optimisation method beats FLOP under the fixed start and loses after burn-in) comes from var-sortability created by a non-stationary start. Variance growing from a noise-only first slice is elementary, and continuous-optimisation methods exploiting var-sortability is Reisach et al. (2021). The paper's own contribution, the exact rel-sortability theory, explains the sign and rank of a gain in rel-SortnRegress only. That baseline has SHD/|E| 1.70 under G2 (main l.459-462). Once the exact covariances are known, the graph quantity is redundant (joint-regression coefficient −0.05, app:b1x). The generator survey (main l.520-534) finds that mainstream generators burn in. The clearly affected construction is Reisach et al.'s time-unrolled proposal, whose data code is not public. So the paper is a theorem-backed note on one proposed benchmark construction, retitled to sound general.

**Adjudication: UPHELD in part. It caps Contribution, but it is not a soundness defect and not a reject-level finding on its own.**
- Upheld:
  - (i) The theory-covered effect does not move any competitive method. The paper says so itself (main l.487-491, Limitations l.851-856).
  - (ii) The oracle demotes the graph gap to a sign-and-rank predictor.
  - (iii) The practically exposed population of generators is narrow. Rated by distributional mixing (NEW-4), it is narrower still.
- Not upheld:
  - (i) "Only one construction" understates the result. Theorems 1–5 hold for every unrolled generator that hands slices to a time-agnostic method. The short-horizon result (rings perfectly rel-sortable for T ≤ p) qualifies a published benchmark proposal directly.
  - (ii) The paper does not hide the var-sortability attribution. It concedes it in the main text, the table and the Limitations. That is a concession, not an escape.
  - (iii) The B'1 rank change is a real, planned, Holm-significant result on SHD, SID and order accuracy. It also survives excluding first-slice pairs.
- Net effect: AC raises Contribution to 3 for the B'1 result. R1, R2 and R3 keep Contribution at 2. No seat lowers Soundness because of DA-C1. The fix is framing, not new mathematics: Round-4 items 1–4.

## Traceability matrix: Round-2 findings

| # | Round-2 finding | Verdict | Anchor (revised) | Residual / note |
|---|---|---|---|---|
| R-N1 | Δ forms under "checked" labels | **RESOLVED** | main l.258-267 (Thm limit, `sRho_unrolled_bounds_maxFinDist`), l.284-296 (Thm stat, `*_maxFinDist`), l.311-318 (lags: paper proof, not checked); app l.130-167, l.696-698, l.1444-1470; Stationary.lean l.137, l.148 | Lean now proves the Δ forms; the p−1 forms are corollaries. The lag extension stays paper-only and is labelled. |
| R-N2 | "by amounts the exact values predict" | **RESOLVED** | main l.422-425 ("track, at about half the size") | The abstract now claims that the oracle "predicts the size" without "post hoc" (NEW-1). |
| R-N3 | B1-NL selective reporting | **RESOLVED** | main l.463-470; app:b1x "Re-analysis" | The plan verdict is stated and discounted; the interactions replicate on SHD/SID/orientation. |
| R-N4 | H1 is estimator tracking | **RESOLVED** | main l.419-422 | The oracle reintroduces the same tautology one level up (NEW-1). |
| R-N5 | Empty-graph reference | **RESOLVED** | main l.459-462; app:b1x | Raw CO methods are worse than empty on chains under G2 (1.14–1.25); this is stated in the appendix only. |
| R-N6 | Statistical hygiene | **PARTLY** | tab:b1 graph-cluster CIs (app l.2042); main l.423-425 (cluster permutation, unit-level 0.62); main l.492-494 and app:b1x (Holm over 21 claims) | Not done: B3 inference is still over 30 segments of 3 experiments on one device. No per-experiment effects, and the sign-flip p 1e-5 treats segments as independent. Residual: should_fix. |
| R-N7 | Discussion recommendation | **RESOLVED** | main l.808-823 | It now contradicts the survey paragraph (NEW-4). |
| R-N8 | B3 D* coincidence; G_lin failure | **RESOLVED** | main l.784-793 | |
| R-N9 | Missing citations | **RESOLVED** | main l.197-209, l.405-415, l.474-476, l.520-534; refs.bib (52 entries, 0 undefined) | `tank2021neural` volume/pages unverified (NOTES.md). |
| R-N10 | Bookkeeping | **RESOLVED** | main l.436-440 ("four of the six"), l.414-417 (+0.125 vs +0.133 labelled); app overview l.3-19; Impact Statement l.857-861; app:b3 "3–4 minutes"; BUILD_LOG clean rebuild 2026-11-09 | |
| R-N11 | Finite-sample gaps | **RESOLVED (on paper)** | main l.553-570 (F1 scope), l.581-597 (plug-in, fixed K, non-Gaussian); app prop:plugin l.497, prop:f1scope l.517, rem:nongauss l.535; app l.1552-1560 (ER c=1) | Fixed-K size control checked only on B3, not on the rivers (most persistent) or studies 3/E1 (NEW-10). No human has read the proofs (B′4). |
| R-N12 | Mechanism alternative; oracle | **PARTLY** | main l.425-433, fig:oracle; app:b1x "Detectability oracle" | The oracle was done, but it is post hoc and does not discriminate (NEW-1). |

Summary: 10 RESOLVED, 2 PARTLY, 0 NOT.

## Traceability matrix: Round-2 roadmap

| Item | Verdict | Anchor | Residual |
|---|---|---|---|
| A′1 Lean Δ hypotheses | **RESOLVED** | as R-N1 | — |
| A′2 Calibrate claims | **RESOLVED** | as R-N2 to R-N5, R-N8, R-N10 | New wording slips: GOLEM in the abstract (NEW-5); oracle (NEW-1). |
| A′3 Refocus (title, abstract, Sec. 4 out, B1 figure, Discussion) | **PARTLY** | title l.75-76; abstract l.88-116; aggregation moved to app:agg; fig:oracle l.443-454; Discussion l.805-823 | Finite-sample theory (Sec. 5, about 1 page) and real data (Sec. 7, about 1.7 pages with a full-width table) remain in the main text, and abstract (iv) still pitches them. Table `tab:b1`, Fig. `fig:sat`, Table `tab:examples`, Thm F1 and the estimand definition, all quoted in the main text, were moved to the appendix to make room (NEW-3, NEW-9). Residual: should_fix. |
| A′4 AI statement; abstract Lean wording; prereg with server push times | **PARTLY** | AI statement l.863-882; abstract l.101-102; app:prereg second timeline l.1020-1043 | The timeline gives client commit times and says the push records "can be supplied". The push records exist and match (checked here); include them (NEW-8). Residual: should_fix (minor). |
| A′5 Reproducibility | **PARTLY** | Artefact Statement l.884-891; REPRODUCE.md; requirements.txt; datapaths.py | /tmp paths and exec are gone (grep: none). Pinned environment, SHA-256 and reproduce.sh are present. But b1x/, finite2/ and lit3/ are not in reproduce.sh (REPRODUCE.md l.58), and requirements.txt omits dagma 1.1.1 and tigramite 5.2.10.1. So tab:b1x and fig:oracle, both main text, are not covered by the "one-command reproduction" the Artefact Statement claims (NEW-7). Residual: should_fix. |
| A′6 Citations | **RESOLVED** | as R-N9 | — |
| B′1 DAGMA/NOTEARS/GOLEM + labels-using reference; generator survey | **RESOLVED (as executed)** | main l.471-494, tab:b1x l.495-518; l.520-534; app:b1x | Scope: one linear-Gaussian parametrisation, T ≤ 5, default hyperparameters, GOLEM at p=5 only. The attribution test is weak (NEW-2). The survey uses the wrong exposure criterion (NEW-4). |
| B′2 Statistical hygiene | **RESOLVED** | as R-N6 | B3 per-experiment effects still missing (see R-N6). |
| B′3 Detectability oracle | **PARTLY** | as R-N12 | NEW-1 |
| B′4 Human probabilist read; plug-in lemma; F1 scope | **PARTLY — AUTHOR-ONLY part open** | prop:plugin, prop:f1scope | The agent-side lemma and scope work is done. The human read of app:erproof and the new App. D propositions is **not done (author-only)**, and no human has checked Lean-statement fidelity (AI statement l.874-876). |
| B′5 OSF posting before data access | **NOT — AUTHOR-ONLY** | app:b3 "Process"; app:prereg l.1022-1024 | No external registration; the B′1 plan was pushed to GitHub only. |

Summary: 5 RESOLVED (A′1, A′2, A′6, B′1, B′2), 5 PARTLY (A′3, A′4, A′5, B′3, B′4), 1 NOT (B′5, author-only).

## New issues introduced or exposed by this revision (frozen list)

| # | Attribution | Severity | Location | Issue | Raised by |
|---|---|---|---|---|---|
| NEW-1 | regression | **major** | abstract l.105-108; main l.422-433; app:b1x "Detectability oracle" | The oracle is the analytic expectation of rel-SortnRegress given the exact covariances, so R² 0.97 checks the simulation against its own expectation, much like R-N4. It uses the full G1/G2 covariances, so it absorbs any mechanism that acts through pairwise correlations, including a generic correlation-magnitude transient. "The missing half is detectability, not another mechanism" is not supported. The abstract states it without "post hoc". In the joint regression the graph gap gets −0.05: the theory is a sign-and-rank predictor only. | R1, DA, AC |
| NEW-2 | regression | **major** | main l.477-491; tab:b1x; app:b1x | The attribution of the CO rank changes to var-sortability rests on standardisation removing the gains. But standardised DAGMA/GOLEM have order accuracy below 1/2 under both generators (0.35/0.41; 0.31–0.33/0.41–0.42), so they cannot show a scale-free effect either way. NOTEARS-std keeps +0.036 OA and −0.077 SHD/\|E\|, a scale-free start effect on a competitive method, which contradicts "the scale-free artefact … moves the sort baselines" (l.488-490). R²-sortability (+0.08, scale-free, outside the theory) is not considered. In B1 the var-OA gain is the same on rings (+0.224) and chains (+0.231), yet the CO interaction has opposite signs; the "levels 0.95 vs 0.77" argument is post hoc. | R2, DA |
| NEW-3 | regression | major (framing) | title l.75-76; abstract (iv) l.111-116; Secs. 5–7 | The new title covers Secs. 3–4. Secs. 5–7 (finite samples, Lean, real data; about 2.5 of 8 pages) are not about the start. The rel-sortability theory is now secondary in impact to var-sortability, yet the title claims both under "the start makes … sortable". | AC, DA |
| NEW-4 | regression | **major** | main l.520-534 vs l.812-818 | The generator survey grades exposure by B ≥ Δτ_max ("for usual lengths they meet B ≥ Δτ_max"). The paper's own Discussion says B ≥ Δ removes only the structural artefact, not the distributional transient that drives var-sortability and hence the CO effect; B1 needed 100 steps. Exposure should be graded by mixing time and by whether slices are handed to an i.i.d. method. Under that criterion the affected population is mainly the time-unrolled construction, whose data code is not public (app:b1x). | R2, DA |
| NEW-5 | regression | minor | abstract l.108-110 | "DAGMA, NOTEARS and GOLEM … beat FLOP on in-trees": GOLEM did not (G1 differences −0.011 and −0.042, Holm 1 and 0.46; b1x/results/analysis.txt l.63-64). | R2 |
| NEW-6 | regression | minor | main l.471-490 | The B′1 plan predicted no rank change (HX2) and "reduced, not removed" (HX3). Both predictions failed, and the paper does not say so. Reporting it would strengthen the paper. tab:b1x shows GOLEM-NV but not GOLEM-EV (worse), and no standardised GOLEM row (SHD/\|E\| up to 1.30). The 3/8 "best label-free method" count includes an untested 0.09 vs 0.10 cell. | R1, R2 |
| NEW-7 | regression | minor | Artefact Statement l.884-891 | The one-command reproduction claim does not cover b1x/finite2/lit3; requirements.txt lacks dagma and tigramite. | R3 |
| NEW-8 | regression | minor | app:prereg l.946-952, l.1020-1024 | Only client commit times are given. The server push records exist and match within 30 s. Push records order pushes, not computations: a 3–4 min plan-to-results gap cannot rule out prior computation; only external registration can (B′5). | R3 |
| NEW-9 | regression | minor | main l.354, l.404, l.558, l.679, l.746 | The main text quotes appendix-only objects: tab:examples, tab:b1, thm:f1, def:estimand, fig:sat. Main text ends exactly at the bottom of p. 8 in icml2026.sty; the icml2027 swap may overflow. | AC |
| NEW-10 | previously_missed | minor | main l.587-592 | Fixed-K size control is unchecked on the rivers (K = 50/100, near-unit-root persistence), study 3 and E1. It cannot rescue a positive claim, but the "size controlled" statement is unverified there. | R1 |

Attribution note: NEW-1 to NEW-9 concern text or analyses added in this revision (regression). NEW-10 concerns a caveat that applies to the code used in Round 1 (previously_missed; decision-inert).

## Decision rationale

- **Rule B3 (Major) fires** on NEW-1, NEW-2 and NEW-4: major issues introduced by the revision.
- All three are analysis or wording problems, not errors in theorems or numbers. Every number spot-checked matches its result file, except the GOLEM phrase (NEW-5).
- No regression is critical. No must_fix item is NOT_ADDRESSED or MADE_WORSE. So neither B1 nor B2 applies, and `reject_recommended` is false.
- The score movement (+0.5) is smaller than in Round 2 (+1.0), although more roadmap items were executed. The panel's reading: B′1 answered the "so what" question in a way that strengthens the benchmark story and weakens the theory's role. The paper is now less a rel-sortability paper than its Section 3 suggests.

## Round-4 roadmap (only items that would move scores)

Tier A″: agent-doable, days.
1. **Oracle: reword and decompose** (NEW-1; moves R1, DA, AC).
   - Abstract: "a post hoc detectability analysis reproduces the size".
   - Main text: drop "not another mechanism".
   - Add a decomposition: the oracle with G1's relative support and G2-like magnitudes, and vice versa. Report how much of the predicted gain comes from pairs whose relative status differs between G1 and G2 (the paper's mechanism) versus pairs that are relatives under both but differ in correlation strength (generic transient). Population computation only, about an hour.
2. **Variance-matched control for the CO attribution** (NEW-2; moves R2, DA; the most likely route to Contribution 3 for R2).
   - Rescale G2 data column-wise to G1's population marginal variances, and G1 data to G2's.
   - Rerun DAGMA/NOTEARS on chains and in-trees (both p, T ∈ {3,5}).
   - If the rank change follows the variances, the attribution is shown. If not, part of it is scale-free, which helps the theory.
   - Plan first, pushed before any score (and see item 8).
   - Until then, write "consistent with var-sortability". State NOTEARS-std's residual gain and R²-sortability as alternatives.
3. **Regrade the generator survey** (NEW-4; moves R2, DA, AC). Grade exposure by mixing (spectral radius × burn-in) and by threat model (slices handed to an i.i.d. method versus one record handed to a time-series method). Say plainly that standard single-record time-series benchmarks are outside the threat model.
4. **Finish the refocus** (NEW-3, NEW-9, A′3; moves AC, R2).
   - Move F2/F3 detail and tab:real to the appendix; keep a 6–8-line real-data summary with the saturation diagnostic.
   - Bring a compact tab:b1 back, plus the one-line sharpness example.
   - Use the freed space for item 2.
   - Abstract (iv) shrinks to one sentence.
5. **Wording and reporting** (NEW-5, NEW-6, NEW-8; moves R3, AC).
   - Fix the GOLEM claim.
   - State the HX2/HX3 predictions and that they failed.
   - Add GOLEM-EV and standardised GOLEM rows, or a footnote.
   - Add the verified server push times to app:prereg, as a table for the AC if anonymity requires.
6. **Reproduction** (NEW-7, A′5; moves R3). Add b1x/finite2 stages to reproduce.sh, and dagma 1.1.1 and tigramite 5.2.10.1 to requirements.txt. Optionally add the B3 per-experiment effects (R-N6 residual) and the fixed-K check on the rivers (NEW-10).

Tier B″: author-only.

7. **Human reads** (B′4; moves R1 and R3 Soundness from 3 towards 4). A probabilist reads app:erproof and prop:plugin/prop:f1scope. Someone fluent in Lean checks that the 33 headline Lean statements match the paper (a few hours). Then update the AI statement.
8. **External registration** (B′5; moves R3, DA). A human posts the item-2 plan to OSF before the agent runs it. This would be the paper's first externally registered test.
9. **Submission hygiene** (no score effect, but blocking): icml2027.sty swap and 8-page re-check; final AI statement against the ICML 2027 policy; decide on the committer of f435a7c; de-anonymisation items.

Not worth doing for scores: more real-data studies, more Lean coverage, more B1 parametrisations.

## Honest bottom line

Round 3 moved the panel from 5.25 to 5.75: from "borderline, leaning reject" to "borderline, leaning weak accept". Every Round-2 labelling and bookkeeping defect is fixed, and the Lean Δ forms are now real theorems. The new experiment answers the panel's main question. Its answer is that the start artefact changes competitive rankings through var-sortability, which is not what the paper's theory is about.

The remaining work is mostly to stop overclaiming in three places: the oracle, the var attribution and the survey's exposure criterion. After that, the paper should decide whether it is a benchmark-hygiene paper backed by exact theory (recommended, and what the title already says) and cut the rest to fit. Two author-only actions would strengthen the paper's credibility more than any further agent work: a human read of the proofs and Lean statements, and one externally registered test.
