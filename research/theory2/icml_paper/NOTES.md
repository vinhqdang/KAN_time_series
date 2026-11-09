# Notes for the authors: ICML 2027 draft

Files: `main.tex` (main text), `appendix.tex` (unlimited appendix, `\input` from main), `refs.bib`, `REFS_LOG.md` (verification of every reference), `Makefile`, `icml2026.sty`/`.bst` plus the helper `.sty` files from the official zip. Build: `make`. It runs pdflatex, bibtex, pdflatex and pdflatex, then fails if the log contains "undefined". The PDF is `main.pdf`.

## Status of this draft (2026-11-04)

- **Style.** No ICML 2027 style was available on icml.cc on 2026-11-04 (`/Conferences/2027/...` returned 404). The draft uses the official **icml2026** style (version of 2025-10-29) from `https://media.icml.cc/Conferences/ICML2026/Styles/icml2026.zip` (sha256 `8b29290f…6bfc6c`). **Swap to icml2027.sty when it is released**: replace the `.sty`/`.bst` files and the two `icml2026` lines in `main.tex`, then re-check the page count.
- **Length (revision 2026-11-04b).** The main text, including Tables 2 and 3 and now three figures, ends at the bottom of page 8 (right column, no slack; unchanged after the X1 revision); the impact and artefact statements and the references start on page 9, and the appendix follows. The PDF is 25 pages in total (main text 8 pages; appendix and references 17 pages). Any further material must go to the appendix. To fit the new ER/loop-free/accuracy material the E1 limitation paragraph (i)-(iv), the E2 paragraph, the Discussion and several intro sentences were shortened; every shortened number is still in Appendix I/J.
- **Revision 2026-11-04b** integrated `../lit2/REPORT.md` (rewording items 1-10, related-work paragraph, 8 new references), `../ext/EXT.md` (X1 ER limits and in-forest identities: main Sec. 3.2 + Fig. 3 + Appendix L; X2 no-self-loop theorem and counterexamples: Sec. 3.1 + Appendix K; X3 accuracy interpretation: end of Sec. 3.2 + Appendix M) and the new Lean files (133 theorems, 3,711 lines). Status labels carried over: ER limits were a sketch at that revision (now proved, see the X1 revision below); X2 limit/periodicity/same-SCC/stationary-without-loops are paper proofs plus enumeration, windowed version only sketched; X3 is elementary.
- **Not done on purpose:** the Lindeberg argument of lit2 section 3 is not used; the joint CLT is stated as an assumption (Prop. 11 and Limitations) with Ginovyan-Taqqu cited for scalar CLTs. The paper text refers to Reisach et al. only as the arXiv entry; no venue is claimed.
- **Build.** Clean: 0 undefined references or citations, 0 overfull boxes. `grep -n 'PH{' *.tex` returns nothing.
- **E1/E2 inserted (2026-11-04)** from `../icml_exp/` (README.md, PLAN4.md, THEORY_ADD.md, summary4.txt, results4.json): main text Section 7 (E1, E2 paragraphs), Table 2 rows, abstract, contribution (6), Discussion and Limitations; Appendix H timeline and deviation (7); new Appendices I (E1) and J (E2). The `\PH{}` macro is still defined in `main.tex` but unused.
- **Claims to keep honest if the text is edited** (all four are stated in the main text):
  1. E1 passes 8/8 only with the linear-dependence graph G_lin (six real causal edges removed by the symmetry lemma, THEORY_ADD B1-B3, which assumes ideal polariser sensors). On the published graph G_D the theorem still beats all eight overall, but on the 442 edges where it predicts strict order the sensor indicator is significantly better, and the theorem's exact match (0.714) is below constant-1/2 (0.815, post hoc). Claim only: the theory predicts linear-dependence relatives.
  2. Most of the separation comes from ties (1,365 of 1,662 predictions are 1/2); strict 0/1 predictions are 297 (citris) and 2,768 (snapshot sets).
  3. One device in a favourable regime; wind-tunnel detection is limited by F1; constant-1/2 has lower absolute error than the theorem at D=1, L=30 (0.311 vs 0.363).
  4. E2: Q7, Q8, Q10 pass, Q9 fails (0.243); P9's failure stands and repeats (0.202 at D=10 on held-out regimes); the 0.267 to 0.338 move from re-drawing subgraphs is variability, and E2 is not presented as rescuing P9.

- **X1 revision (2026-11-04c).** Integrated `../x1proof/X1PROOF.md` and `APPENDIX_L_NOTE.md`: the ER limits L(c), S(c) are now labelled *proved on paper* (main Prop. `prop:er`, Appendix L), not sketch. Exact status: proved from textbook facts (Poisson coupling, Chernoff, Galton-Watson survival and total progeny; van der Hofstad vol. 1, Thms 2.10, 2.19, 3.1, 3.8, 3.16 as numbered in the author's online PDF, opened and matched), with the routine exploration and local-limit couplings written concisely; formulas unchanged (the EXT.md formula is correct). Stated as NOT proved: rates, the critical window c = 1 + lambda p^{-1/3}, almost sure convergence, fixed T as p grows (the limit is T to infinity at fixed p), the layered-DAG S (still a sketch), and the claim that the dichotomy lemma survives <= 3 forced edges (justified in one line). Karp 1990, Luczak 1990, Luczak-Seierstad 2009 and Cooper-Frieze 2004 were not opened and are used in no proof; the first three are named in Appendix L only as results recorded in the introduction of Dovgal et al. (arXiv 2009.12127, opened); Cooper-Frieze is not mentioned in the paper. Lean: ErLemmas.lean adds 5 theorems (W1-W4; W4 has a cardinality corollary), 138 in total (re-counted: `wc -l ../lean/AXIOMS.txt` = 138, all `[propext, Classical.choice, Quot.sound]`), 3,791 lines in 15 files; no probabilistic step is in Lean. Two lines were cut from the abstract and the Lean paragraph to keep the main text at 8 pages (the abstract no longer says "including out-of-sample predictions of detection rates"; it is still in Section 7).

## Revision 2026-11-08 (Tier A of review/00_editorial_decision.md)

Main text now ends on page 7 (right column); Impact, AI-use and Artefact statements and the references follow on page 8; 31 pages in total. Build: 0 undefined references/citations, 0 overfull boxes.

- **A1 (ER proof).** Appendix L now contains the full X1PROOF argument as `app:erproof` (Lemmas dichotomy / large-out-reaches-large-in / local concentration, Cor. structure, Props. N, tree computation, reduction, Thm S, Prop c <= 1). Tightened: (a) Case I-b is a union over all other parents k with forced pairs {(i,j),(k,j)} (no label-dependent parent choice); the local part is the functional phi^beta_K <= indeg(j), handled by Lemma (local)(b) = truncation at M plus uniform integrability (E D^3 bounded); the same device fixes Prop. c <= 1; (b) the forced-edge and conditional-independence bookkeeping is written out once ("admissible explorations": offspring dominate Bin(n', q) and are dominated by 3 + Bin(p, q)); Lemma 1 is stated for any admissible exploration, which covers forced edges and the backward exploration of Lemma 2; (c) Case II-1 (B2): expectation on Omega' = Omega and {max indeg <= C log p} plus Cauchy-Schwarz on the complement. Constants: epsilon < 1/5, beta = c(1 - 4 epsilon) (slack for 3 forced pairs and |R_f| <= 2 epsilon p). The Chernoff lower tail is now derived in one line instead of quoted from memory. **Nothing is left unproved in the statements as given**; the local-coupling Lemma remains written concisely (standard), and no human probabilist has read the appendix. All "sketch"/"heuristic" wording for the ER limits is gone; the iterated-limit order (T to infinity at fixed p, then p to infinity; none needed for S) is stated in Prop. 7, Appendix L and Limitations.
- **A2.** "Pre-registered" replaced by "time-stamped analysis plans" in title, abstract and text; Section 7 "Protocol and its limits" states the agent-driven same-day process, client-set timestamps, no registry, the stage0 exposure (0.51 vs 0.51) and the G_lin motivation; Appendix H has a disclosure paragraph; the CausalRivers F2/F3 accounts are labelled explanatory/post hoc.
- **A3.** AI-use statement added after the Impact Statement, titled "(draft for the authors)". **It is a draft the authors must review, edit and adapt to the ICML 2027 policy once published** (then drop "(draft for the authors)" from the heading). It lists AI-generated artefacts, what is machine-checked, what is checked only by computation, and the paper-only proofs no human has read.
- **A4.** Credits: Brockwell-Davis Ch. 7 (Fejer/Cesaro; Thm agg (a) now explicitly textbook), Bollen 1989 Ch. 7 (three-indicator rule), multivariate Breuer-Major (Arcones 1994, statement from Nourdin-Peccati-Podolskij 2011 Thm 1.1) replaces the assumed joint CLT (proof of Prop. nominal: f_c has Hermite rank 2, condition sum |gamma|^2 < infinity from absolute summability, Cramer-Wold, Slutsky for centring); "inconsistency" renamed "size distortion"; Weiss 1984 and Breitung-Swanson 2002 (aggregation-induced contemporaneous correlation) in related work; Haugh 1976 for E2. See REFS_LOG.md for exactly what was read.
- **A5.** Section 6 says what is and is not certified; breakdown of the declarations; "only criterion" sentence deleted; Reisach's s_rho definition quoted verbatim in Appendix A; 133 leftovers fixed (appendix, lean/README.md).
- **A6.** New Appendix "The real-data estimand s-tilde" (Def. estimand, Prop. sign relation (a)-(e), what is not implied); Lean file `../lean/RelSort/Observed.lean` (10 theorems; (a)-(d) machine-checked; the counterexample (e) by hand). Lean total now **148** declarations, 3,963 lines, 16 files.
- **A7.** Setting and Section 3 say the unrolling is VAR(1)-shaped; new Appendix "Lagged Unrollings" (Prop. lags: lag-weighted distance, same limit, burn-in B >= (p-1) tau_max; paper proof, not machine-checked, not enumerated; lag-0 edges excluded). Discussion gives the burn-in with tau_max.
- **A8.** Abstract about 190 words, no study codes; four contributions; Section 4 (aggregation), F1 and F3 statements, the no-loop section and E1/E2 details moved to the appendix with one-paragraph summaries (Prop. zero, Thm agg, Thm f1, Prop f3 are now stated in Appendices C/D); "a referee" sentence removed; CausalRivers paragraph retitled "a failed test, explained post hoc"; E1 paragraph now leads with its weaknesses, including the fact that on G_lin the prediction coincides with the weak-component-size rule on all 1,662 instances (follows from Cor. statiff: one source per weak component means rel(v) = component; checked by argument, not by re-running code).
- **B1 placeholder.** Subsection "Benchmark impact (B1)" (end of Section 3) with a red \PH{} and a LaTeX comment; no numbers. Replace when `../b1/` reports.

## Reproducibility: what must be done before submission (A9; no code was changed in this revision)

1. **Pinned environment.** One file per language: `requirements.txt` or `environment.yml` with exact versions for every Python script used for paper numbers (numpy, scipy, pandas, networkx, statsmodels, tigramite, lingam as used in ../empirical, ../empirical2, ../realdata, ../icml_exp, ../ext/verify, ../x1proof/verify, ../finite, ../verify); C++ compiler version and flags for `x1check.cpp`, `gw_mc.cpp`, `gw_trunc.cpp`; the Lean toolchain is already pinned (`lean-toolchain`, `lake-manifest.json`, Mathlib v4.35.0).
2. **Configurable data paths.** Replace every hard-coded `/tmp/claude-0/...` and `/home/user/...` path by a single environment variable or command-line argument (e.g. `CAUSALRIVERS_DIR`, `CAUSALCHAMBERS_DIR`, `OUT_DIR`), with a check that fails loudly if missing. Remove `exec`-based code loading in favour of imports.
3. **Data integrity.** Record SHA-256 of the CausalRivers files used (`product.zip`, the regional pickles and time-series CSVs) next to the existing Causal Chambers hashes (PLAN4 section 0), and verify them at load time.
4. **One-command reproduction.** A top-level script (e.g. `make reproduce` or `reproduce.sh`) that (i) builds the Lean project and runs the axiom checks, (ii) re-runs the enumeration and ER checks, (iii) re-runs each real-data study from raw data to its result JSON, and (iv) regenerates every table and figure number, then diffs against the committed result files.
5. **Anonymised supplement** with the above, without names, e-mails, repository URLs or absolute paths.

## Open decisions

1. **Author list and order.** The draft is anonymous (`Anonymous Authors`). Decide the authors before the abstract deadline: for ICML 2026, the author list could not change after the abstract deadline, which was 5 days before the paper deadline. Check the 2027 call. Every author needs an OpenReview account.
2. **AI-use disclosure.** Most of the proofs (THEORY.md, FINITE.md), all of the Lean code, all experiment code and analyses, and this manuscript draft were produced with AI assistance (Claude-based coding agents). The ICML 2026 call for papers says:
   - authors may use generative AI tools;
   - authors take full responsibility for all content;
   - authors are encouraged to explain notable uses in their methodology;
   - LLMs cannot be authors.

   **ICML 2027's policy must be checked by the authors when it is published.** Decide whether and where to disclose: a sentence in the paper (methodology or appendix), the OpenReview form, or both. Given how much of the work was AI-produced, a short explicit statement is advisable. Since 2026-11-08 the draft contains a **draft** AI-use statement (after the Impact Statement, heading marked "draft for the authors"); the authors must review and adapt it.
3. **Human verification of the paper-only proofs.** Lean checks the graph theory (Section 3) completely, and only the deterministic and algebraic core of Sections 4 and 5. **A mathematician (a co-author, not an agent) should read the following proofs line by line before submission:**
   - Prop. zero (zero pattern under per-node filters; MA(∞) support argument; real-analytic genericity).
   - Appendix L (`app:erproof`, full ER proof), Appendix "Lagged Unrollings" (Prop. lags), Appendix "The Real-Data Estimand" part (e).
   - Thm. 8 (a) rate (geometric decay of Γ(h)), (c), (d) (continuity of R² in S; the star inequalities, which are matrix algebra and could be formalised).
   - Lemma D.1 as a probability statement (Gaussian MGF, rotation invariance).
   - Lemma D.2: ‖C^{1/2}MC^{1/2}‖_F² = tr(MCMC) and the operator-norm bounds.
   - Lemma D.3: the union bound and the Chernoff bound for sample means.
   - Thm. 9: assembly of the probability bound and the explicit asymptotic exponent.
   - Prop. 10 and the n_1/2 / n_crit locations. These are asymptotic; since 2026-11-08 the CLT is the multivariate Breuer-Major theorem (Arcones 1994; statement read in Nourdin-Peccati-Podolskij 2011, Thm 1.1); check the Hermite-rank application in Appendix D.
   - Prop. 11: the rate constant c ≈ κ_min²/(18(√B+√D)²).
   - Prop. 12: the finite-sample bracket (Lemma D.3 applied to (X, Z), composed with the Lipschitz bound).
4. **Novelty framing.** The graph results are short consequences of a walk-length lemma and Reisach et al.'s Lemma 1. The introduction says so explicitly. Decide whether to keep this candour, which is recommended. Before submission:
   - re-run a literature search for follow-ups to arXiv 2605.06288 (Semantic Scholar or Google Scholar citations, which were not done);
   - check whether 2605.06288 has appeared at a venue.
5. **Pre-registration evidence.** All three pre-registrations and studies happened within about 5.5 hours on one day, by agents. Reviewers may question how meaningful "pre-registration" is in that setting. Commit timestamps are client-set. Options:
   - provide the hosting service's push records;
   - provide an anonymised mirror of the history;
   - describe the process honestly in the appendix.

   Appendix H already notes that timestamps are client-set. Decide what to supply as supplementary material.
6. **Title.** Current title: "Rel-Sortability of Time-Unrolled Causal Graphs: Exact Values, Finite-Sample Saturation, and Pre-Registered Real-Data Tests". It is long. Alternative: "When Do Time-Unrolled Causal Benchmarks Leak the Causal Order?"
7. **Scope of Section 4.** Aggregation (contribution 3) is the weakest part: it is textbook time-series material and mostly not machine-checked. If space is needed for E1/E2, it can be shortened to a single proposition plus the appendix.

## Which proofs a mathematician should read first (X1)

1. `../x1proof/X1PROOF.md` Section 6, the **reduction of E_up to the local functional Psi_K** (Props. 6-8; Prop. 7 Cases I and II with the witness types alpha1, alpha2, B1, B2). This is the only genuinely new argument; it needs the global Lemma 2 only to exclude large-out-set witnesses, and the first-moment bound c e^{1-c} < 1 (B2) is where c != 1 enters.
2. X1PROOF Lemma 4 (local concentration: mean via Poisson coupling and tightness, variance via node-disjoint edge pairs) and the two-round exploration couplings of Lemmas 1-2. They are standard but compressed; check the conditional-independence bookkeeping (which pairs are revealed) and the claim that Lemma 1 holds with <= 3 forced edges.
3. Then Prop. 5 (N/p to c x^2), the cycle bound for c < 1 and the c = 1 coupling. Textbook theorem numbers (vdH vol. 1) are as numbered in the author's online PDF; check them against the printed book.

## Pre-submission checklist (updated after X1)

- [ ] ICML 2027 style files swapped in; re-check that the main text still fits in 8 pages (on 2026-11-08 it ends at the bottom of page 7, so about one page of slack, needed for B1).
- [ ] ICML 2027 call for papers read: deadlines, AI policy, impact-statement rules, financial conflict-of-interest disclosure, reciprocal reviewing.
- [x] AI-use disclosure **drafted** (2026-11-08, after the Impact Statement). [ ] Authors review and adapt it to the ICML 2027 policy.
- [ ] A mathematician reads the paper-only proofs (decision 3), including the new ones: THEORY_ADD A1-A3 (lagged relatives; A3 constants are sketched) and B1-B3 (linear-zero lemma; B2 is a physical assumption, not checked).
- [ ] E1 assumption B2 (ideal polariser sensors, aligned zero marks): decide whether to test it on other light-tunnel data or to keep it as a stated assumption. Reviewers will ask about G_lin versus G_D.
- [ ] Decide whether to keep the post hoc constant-1/2 comparison in the main text (it is in `../icml_exp/explore4.log`, not pre-registered; the paper labels it post hoc).
- [ ] Every number re-checked against its source file (table below), including the new E1/E2 rows.
- [ ] Pre-registration evidence (decision 5): all four pre-registrations and studies fall within about 5.9 hours on one day, by agents, with client-set commit times. Supply push records or an anonymised mirror. Real times relative to T0 are in Appendix H (plan 4 at T0+5:50, results at T0+5:54).
- [ ] **Re-read the camera-ready of Reisach et al. (arXiv 2605.06288; listed as NeurIPS 2026 accepted on a co-author's page, no proceedings page found) the day it appears**, and re-run the arXiv/Scholar search for citers: their App. A.3 already has the absorption argument, and a v2 might add necessity or a stationary start. Update `reisach2026topological` in refs.bib and the sentence "only sufficiency is shown in the arXiv version" in the introduction accordingly.
- [ ] Literature search refreshed (decision 4); lit2 added 8 references, verified and logged in REFS_LOG.md. Open: the end pages of Bartlett 1935 (536-543) and Clifford 1989 (123-134) are not displayed by any reachable source (only first page and next-article start); check in print. Gong 2017 equation numbers (5), (10) and the classical three-indicator identity source need a human check.
- [x] Joint CLT: now derived from the multivariate Breuer-Major theorem (Arcones 1994; NPP 2011 Thm 1.1 read). [ ] A human checks the Hermite-rank argument.
- [ ] A mathematician reads X2 (Props. in Appendix K, especially the eventual-periodicity limit and the chain argument) and the X1 proofs in the order given above (Appendix L is a summary; the full argument is in `../x1proof/X1PROOF.md`, which must be in the supplement, or its Sections 3-7 moved into Appendix L before submission, since the appendix currently only outlines them). The layered-DAG case stays a sketch.
- [x] Appendix L carries the full X1 proof (2026-11-08), with Reviewer 1's three points tightened. [ ] A human probabilist reads it (B4).
- [ ] Confirm the vdH theorem numbers against the printed book, or drop them from Appendix L. (The Chernoff lower tail is now derived in Appendix L.)
- [ ] Optionally open Karp 1990, Luczak 1990, Luczak-Seierstad 2009, Cooper-Frieze 2004 (not opened); until then they are not cited.
- [ ] Supplementary material assembled and anonymised: Lean project; `../verify`, `../finite`, `../empirical`, `../empirical2`, `../realdata`, `../icml_exp` (code, logs, PLAN4, THEORY_ADD, README; raw data are not committed, SHA-256 values are in PLAN4 section 0); PLAN/PLAN2/PLAN3/PLAN4; axioms report and build log; README. No repository URL, names, e-mails or absolute paths (for example `/tmp/claude-0/...` and `/home/user/...` appear in the experiment folder) in the paper or supplement.
- [ ] Impact statement reviewed (currently the ICML default sentence).
- [ ] Figures and Table 2 checked in print and greyscale (Table 2 is now 19 rows wide and tight; it has no overfull box).
- [ ] Proofread; theorem numbers cited in prose match; the appendix letters cited in the main text (Appendix H, I, J) match the build.
- [ ] OpenReview accounts for all authors; abstract registered before the abstract deadline.
- [ ] Camera-ready only: de-anonymise, acknowledgements, commit hashes for the four pre-registrations (plan 4 is `a0559ec`, results `5f559a1`), a lay summary.

## Where the numbers come from

| paper item | source file |
|---|---|
| Section 3, Table 1, Fig. 1 | `../THEORY.md` §1; Lean `../lean/RelSort/{Limit,Stationary,Examples}.lean` |
| Appendix B enumeration counts | `../THEORY.md` §5, `../verify/t1_enum_results.txt` |
| Section 4 numerical remarks (0.842, 0.546, 0.101; in-tree curve) | `../THEORY.md` §2, `../verify/t2_aggregation_results.txt` |
| Section 5, Appendix E simulation numbers | `../finite/FINITE.md` §§1–4, `../finite/*_results.txt`, `../README_part2.md` |
| Lean counts (148 declarations = 116 + 17 + 5 (ErLemmas) + 10 (Observed); 3,963 lines over 16 files in `RelSort/`; axioms; breakdown 29 headline / 43 graph lemmas / 65 algebraic / 11 examples, counted by name and file with the classification listed in `../lean/README.md`) | `../lean/AXIOMS.txt` (148 lines, all `[propext, Classical.choice, Quot.sound]`), `wc -l ../lean/RelSort/*.lean` (3,963; ErLemmas.lean = 80, Observed.lean = 172), `../lean/BUILD_LOG.txt` (9062 jobs, no sorry/admit/axiom), `../lean/README.md` |
| Sec. 3.1, Appendix K (X2: exact edge value without loops, limit, examples, counts 120 / 3,577 / 196,605 / 8,928 / 4,494; 1,115 graphs; 362 acyclic, 298 in-forests) | `../ext/EXT.md` section 2, `../ext/verify/x2_*_results.txt`; Lean `../lean/RelSort/NoLoop.lean` |
| Sec. 3.2, Fig. 3, Appendix L (proofs and status: `../x1proof/X1PROOF.md` Sections 0-10, `APPENDIX_L_NOTE.md`; Monte Carlo of the tree event 0.38756 +- 0.00077 vs 0.38773 at c = 1.5: X1PROOF Prop. 6; extra exact rows L = 0.69819 +- 0.00024 (c = 1.5, p = 1e5) and 0.62170 +- 0.00012 (c = 2, p = 3.2e5): X1PROOF Section 8; ER formulas; exact table c = 0.5-5; p scaling; giant SCC 0.6353 vs 0.6349; d-effect; in-forest ensembles 0.68739, 0.70859, 0.65783; layered 0.7824 vs 0.7878) | `../ext/EXT.md` section 1 and 1.5, `../ext/verify/x1_er*.txt`, `x1_ens_results.txt`; Lean `../lean/RelSort/ErLemmas.lean` (W1-W4). The curves in `er_curves.tex` were recomputed from the formulas (fixed-point iteration; reproduces the table's formula column and the maxima 0.7542 at c = 1.073 and 0.6650 at c = 1.09) |
| Appendix M (X3: accuracy of the sort rule; enumeration 0.5, 0.5625, 0.592262, 0.605547 vs 0.5, 0.625, 0.660714, 0.669238) | `../ext/EXT.md` section 3, `../ext/verify/x3_bayes.py`; Lean `../lean/RelSort/Predict.lean` |
| Related work, classical credits, new references | `../lit2/REPORT.md` sections 0, 2-5; REFS_LOG.md |
| Table 2, CausalRivers P1–P3 | `../README.md`, `../empirical/results_p.json` |
| Table 2, CausalRivers P4–P6; r̂, r_crit, plug-in predictions | `../README_part2.md`, `../empirical2/results2.json` |
| Table 2, P7–P9, R7–R9; Table 3; S1–S4 | `../realdata/README.md`, `../realdata/results3.json`, `explore3.json` |
| Fig. 2 points | CausalRivers κ 0.71–0.74 and median raw n_eff 113/139, given proxy median |π| 0.50–0.52 and residual n_eff 211/231 (`../README_part2.md`); WT κ̂ 0.03, median n_eff 852; LT κ̂ 0.02, median n_eff 1998 (`../realdata/README.md`). z_c values 3.29 (d = 5) and 4.04 (d = 20) computed from α = 0.01 with Bonferroni |
| Appendix H timeline | `git log --format='%h %at %s' -- research/theory2`, times relative to the first plan commit (T0 = 1793765698). Draft 677c2cb at +20723 s (5:45:23); plan 4 a0559ec at +21050 s (5:50:50); results 5f559a1 at +21266 s (5:54:26) |
| Section 7 E1, Table 2 E1 rows, Appendix I (primary) | `../icml_exp/summary4.txt` blocks `E1_citris_Glin_L0` (E1a 0.000 and 0.969 [0.945, 0.992]; 1,662 instances; exact match 0.992 vs 0.115-0.203; Delta -0.502 [-0.507, -0.496] to -0.533 [-0.536, -0.530]; 10/10 units; Holm p 0.00781; Spearman 0.962 vs 0.105-0.408), `/ detected_edge_subset` (1,338 instances, Delta -0.500), `/ theory_strict_subset` (297 instances; 186 + 1 observed 1 or 1/2, 110 observed 0; Delta -0.539 to -0.991; all-tied baseline +0.443 to 0); `results4.json` `E1_citris_Glin_L0.E1` |
| E1 observed-given-predicted shares (99.2% of 1,365; 99.5% of 187; 99.1% of 110; theorem exact 99.2%, competitors 11-20%) | `summary4.txt` line `obs \| theory value` of `E1_citris_Glin_L0`, README E1 section |
| E1 published graph G_D (Delta -0.323 sensor, -0.424, -0.420, -0.417; exact 0.714; 442 strict instances, exact 0.222 vs sensor 0.319; sensor Delta +0.453 [0.410, 0.492], 0/10 units, reverse Holm 0.002; detection 0.659 [0.643, 0.675]; Spearman 0.341 vs 0.408) | `summary4.txt` blocks `E1_citris_GD_L0` |
| Constant-1/2 comparison (citris G_lin 0.815 and G_D 0.815 exact; 0.092 vs 0.166 mean abs error; snapshot 0.688 vs 0.923; WT D=10 L=0 0.375 exact, abs error 0.313 vs theorem 0.294; WT D=1 L=30 0.378 exact, abs error 0.311 vs theorem 0.363) | `../icml_exp/explore4.log` (post hoc); theorem values from `summary4.txt` |
| Appendix I lagged citris (L=5: detection 0.976 [0.953, 0.992], Delta -0.505 to -0.535, exact 0.998, Spearman 0.996) | `summary4.txt` `E1_citris_Glin_L5` |
| Appendix I snapshot sets (snapshot table: instances 1,712 / 867 / 1,650 / 2,267 / 2,238 / pooled 8,734; theorem exact 0.791 / 1.000 / 0.812 / 1.000 / 1.000 / 0.923; competitor ranges; Delta ranges; pooled Holm 8e-05; 2,768 strict instances, 2,403 observed 1, 349 observed 1/2, 16 observed 0; Spearman 0.913 vs -0.138; detection 0.933 / 1.000 / 0.965 / 1.000 / 1.000) | `summary4.txt` `E1_iid_*`; `results4.json` `E1_iid_*.E1` |
| Appendix I wind-tunnel E1 rows (Delta -0.203 [-0.227, -0.179] 63/64; -0.225 [-0.250, -0.197] 62/64; -0.218 64/64; -0.218; exact 0.539/0.432, 0.443/0.353, 0.411/0.326, 0.533/0.411; detected-edge Delta -0.169 and -0.134; detection 0.592, 0.398) | `summary4.txt` `E2_wt_*`; `results4.json` `E2_wt_*.E1` |
| Section 7 E2, Table 2 E2 rows, Appendix J Table (Q7 0.003 [0.001, 0.005]; Q8 0.553 [0.527, 0.579]; Q9 0.243 [0.193, 0.294]; Q10 +0.069 [0.060, 0.079], non-rel. +0.001 [-0.001, 0.002]; D=10 L=3: 0.011, 0.711 [0.692, 0.729], 0.202 [0.152, 0.253], -0.007 [-0.012, -0.003]; lag 0 D=10: 0.014, 0.742 [0.722, 0.760], 0.202 [0.145, 0.259]; lag 0 D=1: 0.002, 0.552 [0.532, 0.570], 0.120 [0.066, 0.176]; detection 0.328 / 0.398 / 0.592 / 0.585; citris L=5 +0.008 [0, 0.024]) | `summary4.txt` lines `# E2_*_Q`, `# E2_*_detection`; `results4.json` `E2_*` |
| Appendix J per-edge table | `results4.json` `E2_wt_D1_L0`, `E2_wt_D1_L30`, `E2_wt_D10_L0` `.per_edge_detection` |
| Appendix J F checks table | `summary4.txt` lines `# F_*`; `results4.json` `F_*` (0.250 / 0.242; [0.344, 0.426] / 0.332; 0.548 / 0.533; [0.536, 0.554] / 0.524; 0.935 / 0.945; [0.940, 0.988] / 0.945; non-rel. and accuracy columns as printed) |
| P9 variability 0.267 to 0.338 | `../icml_exp/dev4_lags.log` last two lines; PLAN4 development section (study-3 data, subgraph seeds only) |
| z_c 3.90, 4.35, 4.80; factors 1.12 and 1.23 | `../icml_exp/THEORY_ADD.md` section A (computed for 105 pairs, alpha 0.01) |

## Known weaknesses a reviewer will raise

- The positive real-data result is partial: P9 failed, the wind-tunnel graph is bipartite, and the replication uses i.i.d. inputs. E1 addresses the bipartite caveat only for linearly dependent relatives on one light-tunnel series (it fails against the sensor indicator on the published graph's strictly ordered edges); E2 failed Q9, and P9's failure repeated on held-out regimes.
- The CausalRivers F3 agreement is in-sample (r̂ and ℓ̂ are estimated from the tested correlations). The paper says so.
- The non-asymptotic bounds are loose and uninformative at the real sample sizes; only the asymptotic formulas predicted outcomes.
- Self-loops at every node are assumed throughout. Nonlinear mechanisms and contemporaneous effects are not covered.
- [ ] Pre-registration wording: done in the text (2026-11-08); for any new study (B3) register externally (e.g. OSF) before data access.
- [ ] Insert B1 results into the placeholder subsection "Benchmark impact (B1)" and remove the \PH{} (grep -n 'PH{' main.tex).
- [ ] Reproducibility items 1-5 above (A9).
- [ ] Tier B items B2-B4 of review/00_editorial_decision.md (not attempted in this revision).

## Where the numbers of the 2026-11-08 revision come from

| paper item | source |
|---|---|
| stage0 disclosure (0.51 siblings vs 0.51 unrelated vs 0.65 adjacent, 24 h; commits dated 2026-11-03; regions; resolutions 6/12/24 h; shared loaders and sampler) | `../../ts/stage0/README.md` (Results, hidden-driver diagnostic), `../../ts/stage0/gates.md`, `git log -- research/ts/stage0` (3cd4581 08:37 to 814af68 12:54 on 2026-11-03), `../PLAN.md` line 24 and `../empirical/sample.py` line 1 (reuse of the stage0 loaders and sampler) |
| plan-to-results gaps 3 min 41 s (plan 3: cfcab28 09:13:26, results f57d617 09:17:07) and 3 min 36 s (plan 4: a0559ec 10:05:48, results 5f559a1) | `git log --date=iso -- research/theory2` |
| E1 weak-component coincidence (all 1,662 instances) | argument from Cor. statiff applied to G_lin (two weak components with sources red and pol_1); Appendix I "A trivial rule that E1 did not test" |
| Reisach s_rho quote | `scratchpad/dl/r.txt` lines 167-183 (pdftotext of arXiv 2605.06288v1) |
| n_1/2 approx 22 for l = 0.64 (was "approx 25") | 3 + (3.29/atanh 0.64)^2 = 21.8; the old 25 was z_c^2/l^2 |
| Lean counts and breakdown | `../lean/AXIOMS.txt`, `../lean/BUILD_LOG.txt` (incremental rebuild 9064 jobs, exit 0), `../lean/README.md` last section |

## B1 integration (2026-11-09)

- Placeholder `\PH{}` in Sec. 3 replaced by "Does the start artefact change benchmark results?" (Sec. 3.x, Table `tab:b1`); all numbers copied from `../b1/README.md` and `../b1/results/analysis.txt` (commit c2d804e).
- Thm 2 (limit) and Thm 3 (stationary) now state the Delta form (largest finite distance) of the bound and burn-in; Lean still checks the p-1 form, the Delta form is a one-line paper argument checked by enumeration (21,000 cases, 0 violations; B = Delta tight in 445/726). Said so in the text.
- New Appendix "Benchmark Experiment B1" (design, verdicts H1-H3, deviations D1-D3, two artefacts, burn-in cost, Delta check, figures in `figs/`).
- Abstract, contribution (5), Discussion (burn-in >= Delta, standardise) and Limitations (single parametrisation; swaps on baselines; largest swaps from var-sortability) updated.
- Main text still ends on page 8 (33 pages total).

## B3 and B1-NL integration (2026-11-09)

- B1-NL (`../b1nl`, plan 2d19c9b, results c96f018): accuracy gain replicates under NL/SGN/SGNU; ranking reversals do not. Abstract, contribution (5), Sec. 3 B1 paragraph and Limitations toned down accordingly ("robust finding is the sortability shift, not a change of the best method"). New App. `app:b1nl`.
- B3 (`../b3`, plan 827179d, results 7071855, 3.5 min apart; no OSF registration before data access): H1/H2 pass against constant-1/2 and weak-component rules on held-out light-tunnel data; scm_4_blue secondary fails; Malus refinement fails in pol_1_90. Paragraph after E1 in Sec. real data; New App. `app:b3`. `b3/OSF_PREREG.md` is ready but posting it is the user's decision (would only be an archival copy now).
- Main text still ends on page 8 (34 pages total).

## A'/B' integration (2026-11-09, Round-2 roadmap in review/02_rereview_decision.md)

Main text ends on page 8 (left column of p. 9 starts with the Impact Statement); 40 pages in total. Build: 0 undefined references/citations.

- **Title** now "How the Start of a Simulation Makes Time-Unrolled Causal-Discovery Benchmarks Sortable" (A'3). Abstract and contributions lead with (i) strongly connected graphs not protected at short horizons, (ii) burn-in B >= Delta needed and tight (Lean), (iii) fixed start inflates rel- and var-sortability, (iv) finite-sample/real data. Point (iv) of the roadmap ("best method unchanged") had to change after B'1: FLOP is still best on average and in 8/8 cells under the burn-in, but under the fixed start a continuous-optimisation method wins 3/8 cells, and DAGMA/NOTEARS beat FLOP on in-trees under G1 and lose under G2. The text attributes this to var-sortability (it disappears or reverses after standardisation), not to the rel-sortability theory.
- **A'1 (Lean tags).** Thm limit: `sRho_unrolled_bounds_maxFinDist`; Thm stat: `rhoBurn_eq_maxFinDist`, `sBurn_eq_maxFinDist`; Cor statiff: `sBurn_eq_half_iff_maxFinDist`; new Remark `rem:sharp` (B >= Delta cannot be lowered, `edgeLoops_burn_sharp`). The "Lean checks only p-1" caveat is gone. Sec. 6 and App. Lean: 163 declarations (33 headline, 53 graph lemmas, 65 algebraic, 12 examples), 4,159 lines in 16 files, clean rebuild (9,064 jobs). Appendix proofs of Thm limit/stat and Prop. lags now use Delta (lags: d_tau <= d * tau_max <= Delta tau_max; still paper only).
- **A'2.** "track, at about half the size" and H1 named as estimated-vs-exact tracking; B1-NL H3 verdict stated (supported by the plan's rule under NL on one minor swap, discounted; B1's swaps not detected at the replication's power); empty-graph reference; "4 of the 6 cells"; pooled +0.125 (5 non-SC families) vs +0.133 (all 7) labelled; B3 D* coincidence with E1's naive rules and the G_lin failure in pol_1_90 in the main text; B3 timing 3-4 min; appendix overview lists all appendices; "earlier drafts" sentence removed; Impact Statement no longer the default text; "(e) by hand" relabelled.
- **A'3.** Sec. 4 (aggregation) moved to App. C (label `sec:agg` kept there); new Sec. 4 "What the Start Artefact Does to Benchmarks" with Fig. `fig:oracle` (B1 figure: gain vs exact gap and vs detectability oracle; `figs/make_fig_oracle.py` from `../b1x/results/oracle_b3.json`) and Table `tab:b1x`; Discussion recommendation rewritten (stationary start or mixing-checked burn-in; standardise; score without first-slice pairs or as ADMG, with the 0.044/0.273 and 0.041/0.048 numbers); generator-survey paragraph (from `../lit3/GENERATOR_SURVEY.md`). To fit 8 pages, Table `tab:examples`, Fig. `fig:er`, Fig. `fig:sat`, Table `tab:ftheory` and Table `tab:b1` moved to the appendix, and the real-data, E1/E2 and Lean paragraphs were shortened (all numbers remain in the appendix).
- **A'4.** AI statement heading without "(draft for the authors)"; it now names the Delta forms and the sharpness result, says no human has checked Lean-statement fidelity, and lists the new paper-only results (prop:plugin, prop:f1scope, rem:nongauss). **The authors must still check it against the ICML 2027 AI-use policy when published.** App. prereg has a second timeline for the B1, B1-NL, B3 and learner-extension plans. Times are commit times (client-set); the hosting service's push records were not retrieved in this session (no API access to push events), so the text says they can be supplied on request.
- **A'5.** Artefact statement names requirements.txt, DATA.md (SHA-256), reproduce.sh/REPRODUCE.md.
- **A'6.** `../lit3/refs_add.bib` (18 entries) merged into refs.bib, plus `rudelson2013hanson`; cited: FLOP, PC, causal-learn, gadjid, DAGMA, NOTEARS, GOLEM, PCMCI+, tigramite, VAR-LiNGAM, DYNOTEARS, CauseMe, C4C, CausalTime, CDML, Neural-GC, Montagna et al. `flopsearch2026` is in refs.bib but not cited (paper cited instead). tank2021neural uses the early-access form (volume/pages unverified).
- **B'1-B'3** (`../b1x`, plan 5ac1bd2, re-analysis 6045429, results 40d9ee7): main text Sec. 4 and new App. `app:b1x` (learner design and verdicts HX1-HX3, re-analysis with graph-cluster CIs, interactions, empty-graph reference, OA without slice-1 edges, Holm over 21 claims, detectability oracle, generator survey). tab:b1 CIs are now graph-cluster CIs (narrower; point estimates unchanged).
- **B'4** (`../finite2`, 4892ca9): M1-M5 in Sec. 5 (B vs W, F1 scope, plug-in n_eff with the fixed-K caveat: size 0.113 at AR 0.99, K=50; B3 K-sensitivity 0/2,730; non-Gaussian sentence); App. D gets Lemmas acvl2/plugin, Props plugin/f1scope, Remark nongauss and a fixed-K paragraph; ER c=1 convergence sentence after tab:er.

### Still open (user-only)

- Commit f435a7c has committer "Claude <noreply@anthropic.com>" (author is correct). Decide whether to rewrite history (force-push to main) or leave it.
- Post the OSF plan (`../b3/OSF_PREREG.md` is only an archival copy now); any future confirmatory test needs an external registration before data access (B'5).
- A human probabilist should read App. `app:erproof` and the new App. D propositions (B'4).
- Author list, de-anonymised commit hashes, final AI-use statement, ICML 2027 style file and deadline (swap icml2026 -> icml2027 and re-check the 8-page limit).
