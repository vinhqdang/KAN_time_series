# Journal version for *Machine Learning* (Springer): notes

Date: 2026-11-10. Source: `../icml_paper/` at commit 1932c2a (left untouched as the archived
conference version). `main.tex` and `appendix.tex` here are the sources from now on.
Build: `make` (pdflatex + bibtex). State: 73 pages (21 body, 45 appendix, 7 references),
no undefined references or citations, no overfull boxes, 58 references.

## Template and journal requirements

- **Template.** Official Springer Nature LaTeX package, version 3.1 (December 2024), as supplied
  by the user; the same `sn-jnl.cls` (identical SHA-256) was also
  downloaded from resource-cms.springernature.com. Only `sn-jnl.cls` and the one style file in
  use, `sn-apacite.bst` (class option `sn-apa`), are copied here.
  - sn-jnl.cls SHA-256: 36d0c3273a59d48dc6a9c7b080dfa1ec50dc10229d8751568d1f2e490ffa5ecc
- **Verified on the journal's submission-guidelines page** (link.springer.com/journal/10994,
  read with a headless browser on 2026-11-10):
  - abstract of 150 to 250 words (ours: 249);
  - 4 to 6 keywords (ours: 6);
  - citations by name and year; reference list alphabetical, examples in APA form, DOIs as
    full links. Hence `sn-apa`. The page does not name a .bst; `sn-basic` and `sn-mathphys-ay`
    are the other author-year styles in the package if the editorial office asks for one.
  - the Springer Nature LaTeX template is encouraged;
  - a "Statements and Declarations" section and a data availability statement are required;
  - LLMs cannot be authors, and their use must be documented in the Methods section or a
    suitable alternative part. Here: Section 7.1 "Use of AI tools", with a pointer in the
    Declarations.
  - the reference list should contain only published or accepted works.
- **Not verified:** whether the journal is single- or double-anonymous (the page did not say;
  the manuscript is not anonymised); any page limit (none found); whether arXiv preprints in the
  reference list are acceptable in practice (several are cited, e.g. Reisach et al. 2026).

## What changed against the ICML version

- **Title:** "Rel-Sortability of Time-Unrolled Causal Graphs" (shortened at the user's request; the earlier
  longer title listed all four parts of the paper).
- **Front matter:** single author Quang-Vinh Dang, British University Vietnam, with the
  corresponding e-mail, as given by the user. New abstract (249 words) and keywords.
- **Brought back into the body:** the 3-node examples table, the ER figure, the aggregation
  section (now Section 5, proofs in Appendix D), Propositions F2 and F3 (Section 6), and the
  full real-data account with its table, the saturation figure and the F-theory table
  (Section 8). The B1 table in Section 4 was already the full seven-family table.
- **Still in appendices:** all long proofs, the Lean map, E1/E2/B1/B1-NL/learner
  extension/B3 details, the plan timeline, data preprocessing.
- **Removed:** ICML style, the ICML impact statement, page-limit spacing, the anonymity sentence
  in the plan timeline.
- **Layout:** the four widest tables are rotated (`sidewaystable`); tables use the class font.
- **All Round-3/4 caveats are kept:** oracle is post hoc; var-sortability attribution is a
  hypothesis; the failed "no rank change" prediction; the fixed-K caveat; push-time caveat.

## New content (no new experiments)

1. **Section 4.2, Proposition 8 (no gain from the transient).** From
   `../alg2/CANDIDATES.md`, Proposition N. Proof is three lines and is in the main text. Not
   Lean-checked. Cites Rigobon (2003), Wang et al. (2023, arXiv) and Guan et al. (JMLR 2026);
   all three are in `../alg2/REFS_VERIFY.md`.
2. **Section 4.3, identifiability without labels.** From `../alg/DESIGN.md`. I re-read each
   proof before including it:
   - included with complete proofs (Appendix U): T1 (slices = height; Prop. 9), T2 (Markov
     equivalence class, T^{m0} members; Prop. 10), T3 (fixed start, given faithfulness;
     Prop. 11), T4 (three non-identifiability results; Prop. 12);
   - included as an open conjecture: C5 (burn-in window), with the partial reduction;
   - excluded: T6 (consistency of the penalised MLE), the TiDAL method and its optimisation,
     the conjecture that the reversed innovation covariance is generically non-diagonal (it
     appears only as a hypothesis of the conjecture).
   None of this is Lean-checked; the text says so. Faithfulness is an assumption and is not
   shown to be generic; the text says so and cites Boege et al. (2025).
   The sentence about Reisach et al.'s closing question is a paraphrase taken from
   `../alg/NOVELTY.md`; I did not re-open their paper to check the wording.
3. **SPADE citation** (related work, one sentence): Dang, Le, Dinh, *Array* 32, 101335, 2026,
   doi 10.1016/j.array.2026.101335, exactly as supplied by the user. The description (B-spline
   learner; compared with DAGMA, GraN-DAG, GOLEM, NOTEARS, SCORE, NoGAM, PCMCI, VAR-LiNGAM on
   simulated nonlinear DAGs; mixed results against NoGAM; financial-panel graph collapsed on
   stationary returns) follows the summary of its abstract given to me. Please check it against
   the published abstract.

## For the author to do or confirm

- **Competing interests:** "no competing interests" (stated by the user).
- **Code availability:** GitHub URL given; before submission make sure the repository is public and consider a Zenodo archive with a DOI (Springer prefers a persistent identifier).
- **Acknowledgements:** section removed (none supplied); add one if wanted.
- **Funding and author contributions** are written from what the user stated (no funding; sole
  author). Confirm the wording.
- **AI-use statement** (Section 7.1): confirm against Springer's current policy.
- **Human checks not done:** nobody has read the paper-only proofs line by line (ER limits,
  finite-sample propositions, the new Propositions 8 to 12), and nobody has checked that the
  Lean statements match the paper statements.
- **OSF:** no plan is registered externally.
- **Cover letter** and suggested reviewers: not written.
- **Commit f435a7c** on main has committer "Claude": decide whether to rewrite history.
- Figures made with matplotlib embed Type 3 fonts; some production systems ask for Type 1.

## Open scientific items (unchanged)

Variance-matched control for the learner ranking change; population decomposition of the
detectability calculation; B3 clustering at the level of 3 experiments (R-N6); fixed-K size
check on the river data.

## Restructure to six sections and full Related Work (user request)

The user found the Related Work too short (a list of citations, ICML style) and asked for six sections.
Structure now (appendices excluded):

1. Introduction (contributions, new "Organisation" paragraph)
2. Related Work and Background (about 3.7 pages; 2.1 sortability and evaluation of continuous-optimisation learners, with SPADE; 2.2 time-series causal discovery and generators; 2.3 tied unrolled models, labels and identification; 2.4 finite-sample behaviour of dependence tests; 2.5 formal verification; 2.6 positioning)
3. Time-Unrolled DAGs: Exact Values, Burn-in and Identifiability (3.1 setting, 3.2 exact values, 3.3 random summary graphs, 3.4 start identifies nothing more with labels, 3.5 identifiability without labels, 3.6 machine-checked proofs)
4. Finite Samples and Temporal Aggregation (4.1 estimating relatives, 4.2 aggregation versus subsampling)
5. Experiments and Real-Data Evidence (5.1 start artefact in simulated benchmarks, 5.2 real-data evidence)
6. Discussion, Limitations and Conclusions (6.1 implications, 6.2 limitations, 6.3 new short conclusions, 6.4 use of AI tools)

Only headings, two or three bridging sentences per new section, the Conclusions subsection and three
cross-reference wordings ("above" and "this section") changed; no theorem, label, number or caveat was altered.
The old Related-work paragraph in the Introduction was removed (its content is in Section 2).

References added to refs.bib, all verified: kaiser2022unsuitability (Crossref DOI 10.1007/s11063-021-10694-5, arXiv 2104.05441 abstract read),
granger1969investigating (Crossref: Econometrica 37(3), 424 confirmed; the end page 438 is from memory),
lanne2008identifying and lewis2021identifying (copied from alg2/refs.bib; Crossref-verified there, see ../alg2/REFS_VERIFY.md).
Everything else cited in Section 2 was already in refs.bib.

Not verified: how Section 2 describes works was written from their titles, abstracts and the earlier text of this paper, not from the full papers.
In particular the statements on Boege et al. (results for colour classes with a common head, from alg/NOVELTY.md), on FLOP (only "discrete search", from the title)
and on Lohse and Wahl (from earlier text) should be read against the originals before submission. The new Conclusions subsection restates claims made elsewhere in the paper.

## Unified-story revision (one paper, no split)

The author decided against splitting the paper (the review panel had recommended it). The paper
now follows one object, the relatives footprint of the time-unrolled DAG, through four questions
(Q1 footprint and identification without labels; Q2 effect of the start; Q3 what an estimator can
see; Q4 effect on learners and real data). Thesis sentence in the Introduction; every section
opens with its question and closes with a takeaway (Sections 3, 4, 5; Section 2 closes with a
takeaway in the Positioning subsection). The six sections are unchanged in number. Title kept.
No experiment was run, no number changed, no verdict changed.

**Review roadmap items (a), what was done**
- a1/a7 abstract rewritten (246 words by whitespace count): thesis, four questions, scope limits
  (lag one, self-loops everywhere, n = 1000), no deployed generator found exposed, no new algorithm.
- a2 tightness: "cannot be shortened/lowered" removed everywhere; the claim is sufficiency, sharpness
  proved (Lean) for the one-edge graph only, enumeration 445 of 726 fail at Delta-1 (so 281 do not),
  no closed form for the minimal burn-in. New paragraph after Theorem 5, expanded Remark 21 in the
  appendix, claim-status table row.
- a3 "machine-checked": scoped to the Lean theorems as formalised; count sentence corrected (163 in the
  axiom report, 259 theorem/lemma declarations in 16 files / 4,159 lines, 97 not in the report, no
  sorry/admit/axiom, statement map not human-checked). Section 3.5, AI statement, appendix Lean map.
- a4 "predicts the sign" removed; sign is non-discriminating (all 42 cells have positive gap and gain,
  checked in b1x/results/oracle_b3.json); coefficient drift fixed to -0.063 (cells) and -0.039 (units)
  as in oracle_b3.txt. Both pooled values (+0.125 over five, +0.133 over seven) stay.
- a5 Boege et al.: corrected in Related Work and Section 3.4 per review/00_decision.md V1
  (generic faithfulness for edge-only and vertex-only colourings, Props 5.3/5.4; mixed Example 5.7;
  open Question 5.8; common-head condition belongs to BPEC-DAGs/GECS). The text says that on our
  reading the unrolled model is the mixed case and that this reading is not established.
  I did not re-trace the mapping from Prop. 11's model to their classes.
- a6 real data reframed as a different question (estimating relatives from one record), mostly
  negative or non-discriminating; "kill rule" -> "stop condition" and "confirmatory" -> "claim-level"
  in paper and appendix; new Table 5 (units behind each study); pseudo-replication stated in text
  (E1 10 segments of one series; B3 30 units from 3 runs; rivers 900 slots / 457 stations / max reuse 8;
  12 h and 24 h are the same records). "Not pre-registrations" wording kept.
- a8 references: added gentzel2019case, brouillard2025landscape, andrews2026daggeneration,
  nagarajan2025topology, assaad2022survey, mooij2013ode, hyttinen2016subsampled, vowels2022dya,
  hamilton1994time, lutkepohl2005new, kilian2017structural, muller2003tests (andrews2023boss is in
  refs.bib but not cited). Existence checked on 2026-11-10: Crossref for Assaad, Vowels, Muller-Elliott
  (DOI 10.1111/1468-0262.00447), Kilian-Lutkepohl, Lutkepohl, Hamilton; arXiv API for the arXiv ones.
  Venue and page details of the arXiv ones (NeurIPS 2019, CLeaR 2025/PMLR 275, JMLR 27(147), UAI 2013,
  PGM 2016 PMLR 52:216-227) come from the Round-1 domain reviewer's web checks, not from my own.
  The sentences describing them are written from titles, abstracts and that report, not full texts.
  Dynare initial-state options were not cited (not verified). SPADE paragraph shortened.
- a9 claim-status table (Table 1, end of the Introduction) and the replicates-versus-single-record
  table (Table 5). The generator survey stays in Section 5.1 (it was already in the main text).
- a12 REPRODUCE limits added to Supplementary information; plan count (eight plans, push records for
  three) stated in Section 5.2 and the appendix timeline.
- a13 "fixed start of Reisach et al." and "horizons proposed for benchmarks" reworded: the preprint
  fixes neither an initial law nor a horizon.
- Demoted to the appendix: the ER proposition and figure (Remark in Section 3.2 keeps the formula, the
  peaks and the c = 1 convergence sentence); the block-mean/subsampling statements (Section 4.2 is a
  short summary; statements in Appendix "Block Means and Subsamples: Statements").
- SVAR spectral radii: not computed (this environment has no numpy and the check would be a new
  analysis). The paper says the weights were rescaled to a spectral radius of at most 0.8 by
  construction (common.py RHO_CAP) and that realised radii were not tabulated. Still open: roadmap b7.

**Not done in this revision**
- Roadmap (b): matched-correlation counterfactual start, variance-matched control, first-slice-excluded
  scoring as primary, per-graph burn-in or chain-family tightness, run-level cluster inference for
  E1/B3/rivers, missing baselines, mixing times, K-sensitivity on real data, symbolic faithfulness
  check, a full `reproduce.sh all` run, remaining push records.
- Roadmap (c), author only: human reading of the load-bearing proofs and of the Lean statement map,
  OSF registration and a DOI archive of the repository, AI-statement wording with model identifiers,
  contact with the Reisach authors.
- The repository still says "pre-registered" in b1x/README.md, b3/PLAN.md, b3/README.md and some commit
  messages; those files were not touched here (the paper says "time-stamped plan").
- The body is now about 28 pages (appendix from page 29, 83 pages in all).


## b1y results integrated (roadmap b1-b6, b8 of review/00_decision.md)

Source: research/theory2/b1y (plan 5c9f942, Addendum 1 583ff30 pushed before the first scored run,
Addendum 2 c768a7d, results 365c91f, 013c82f, e51698b, 8ad59cd, beca56f). Every number was re-read from
b1y/README.md and results/analysis_*.txt; no verdict or threshold was changed.

- Q6 (exact burn-in): chains p = 2..8, T = 2..5: B_min = Delta; rings ceil(Delta/2); 396 random graphs
  (792 cases) B_min <= Delta always, = Delta in 51%. Added Proposition (chainsharp) with a paper proof that
  the relatives at B = Delta-1 differ from the stationary ones for every p (re-read from b1y/PLAN.md section 8,
  checked by hand: ancestor distance argument); it is NOT Lean-checked, and it does not by itself show that
  the edge-level s_rho differs (that is computed for p <= 8, T <= 5). Abstract, intro, Sec. 3, Sec. 6,
  claim-status table and appendix remark updated; "sharp only for Delta = 1" removed everywhere.
- Q1: new Section 5.2 "Controlled tests ..." (matched-correlation counterfactual, dose, learner reversal,
  first-slice scoring, further baselines) with takeaway; Section 4 and the abstract state V4/V5 and the
  U-A reading (exact gap = n -> infinity ceiling, realised effect = detectable image; Q(25000) = 0.76, not 1).
  The earlier hedge "the exact gap adds nothing" is kept only as the description of the post hoc oracle.
- Q2: tuned in-tree reversal for DAGMA, NOTEARS, GOLEM-NV; GOLEM-EV only ties under G1 (HY2 fails for it);
  HY1 variance channel for all four with negative R_VM (over-correction, not interpreted); injection 0.43/0.49
  (< 0.60) so variance is involved but not sufficient; no reversal on rings and ER2; chains lose to FLOP under
  both starts. The text no longer says that the variance-matched control was not run.
- Q3: Table "x1" (full vs first-slice-excluded). B1 and the learner extension keep their full-window numbers
  as the primary results of their own plans; x1 numbers are shown beside them (the b1y plan makes x1 primary
  only for the new studies).
- Q4: 457/8 corrected to 445/7 (the 457/8 belong to subgraphs_7000.json, the P4-P6 analysis uses
  subgraphs_8000.json); new paragraph "Run- and station-level inference"; Table 5 row corrected;
  B3 run level 3/3 and 2/2 (smallest attainable p 0.125, 0.25); P5 station-disjoint result (27%); K-sensitivity;
  SHA-256 re-check. The real-data tests are described as effect sizes.
- Q5: further baselines paragraph (HB1 7 of 8 groups). GRaSP and DirectLiNGAM are named without a citation
  (no verified bib entry was added); GES, BOSS, DYNOTEARS and VAR-LiNGAM cite existing bib entries
  only where the bib has them (DYNOTEARS, VAR-LiNGAM, PCMCI+); GES/BOSS are named without a new entry.
- Plan count: nine plans, push records for three (b1y has none: the activity feed lagged). New appendix
  "Controls: Plan, Design and Verdicts (b1y)" with the full verdict table; timeline rows added.
- Abstract rebuilt to 247 words (my own count; Springer asks for 150-250: re-count with the journal's tool).

Still open: SVAR spectral radii and generator mixing times (not computed), symbolic faithfulness check,
a full `reproduce.sh all` run end to end, optional Lean lemma for the chain sharpness, Lean statement-map
check by a human, human reading of the load-bearing proofs (including the new Proposition chainsharp and the
Boege reading), OSF registration of any further plan, DOI archive of the repository, AI-statement wording with
model identifiers, the f435a7c committer decision, final bibliographic checks (Granger end page 438,
descriptions of works read only at abstract level).
