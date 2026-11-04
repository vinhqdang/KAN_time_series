# Notes for the authors: ICML 2027 draft

Files: `main.tex` (main text), `appendix.tex` (unlimited appendix, `\input` from main), `refs.bib`, `REFS_LOG.md` (verification of every reference), `Makefile`, `icml2026.sty`/`.bst` plus the helper `.sty` files from the official zip. Build: `make`. It runs pdflatex, bibtex, pdflatex and pdflatex, then fails if the log contains "undefined". The PDF is `main.pdf`.

## Status of this draft (2026-11-04)

- **Style.** No ICML 2027 style was available on icml.cc on 2026-11-04 (`/Conferences/2027/...` returned 404). The draft uses the official **icml2026** style (version of 2025-10-29) from `https://media.icml.cc/Conferences/ICML2026/Styles/icml2026.zip` (sha256 `8b29290f…6bfc6c`). **Swap to icml2027.sty when it is released**: replace the `.sty`/`.bst` files and the two `icml2026` lines in `main.tex`, then re-check the page count.
- **Length.** The main text, including Tables 2 and 3 and both figures, ends 12 lines into page 8 (left column); the impact and artefact statements and the references follow on page 8, and the appendix starts on page 9. The PDF is 19 pages in total. Almost no room is left in the 8-page limit: any reviewer-requested material must go to the appendix, or Section 4 (aggregation, decision 7) must be shortened.
- **Build.** Clean: 0 undefined references or citations, 0 overfull boxes. `grep -n 'PH{' *.tex` returns nothing.
- **E1/E2 inserted (2026-11-04)** from `../icml_exp/` (README.md, PLAN4.md, THEORY_ADD.md, summary4.txt, results4.json): main text Section 7 (E1, E2 paragraphs), Table 2 rows, abstract, contribution (6), Discussion and Limitations; Appendix H timeline and deviation (7); new Appendices I (E1) and J (E2). The `\PH{}` macro is still defined in `main.tex` but unused.
- **Claims to keep honest if the text is edited** (all four are stated in the main text):
  1. E1 passes 8/8 only with the linear-dependence graph G_lin (six real causal edges removed by the symmetry lemma, THEORY_ADD B1-B3, which assumes ideal polariser sensors). On the published graph G_D the theorem still beats all eight overall, but on the 442 edges where it predicts strict order the sensor indicator is significantly better, and the theorem's exact match (0.714) is below constant-1/2 (0.815, post hoc). Claim only: the theory predicts linear-dependence relatives.
  2. Most of the separation comes from ties (1,365 of 1,662 predictions are 1/2); strict 0/1 predictions are 297 (citris) and 2,768 (snapshot sets).
  3. One device in a favourable regime; wind-tunnel detection is limited by F1; constant-1/2 has lower absolute error than the theorem at D=1, L=30 (0.311 vs 0.363).
  4. E2: Q7, Q8, Q10 pass, Q9 fails (0.243); P9's failure stands and repeats (0.202 at D=10 on held-out regimes); the 0.267 to 0.338 move from re-drawing subgraphs is variability, and E2 is not presented as rescuing P9.

## Open decisions

1. **Author list and order.** The draft is anonymous (`Anonymous Authors`). Decide the authors before the abstract deadline: for ICML 2026, the author list could not change after the abstract deadline, which was 5 days before the paper deadline. Check the 2027 call. Every author needs an OpenReview account.
2. **AI-use disclosure.** Most of the proofs (THEORY.md, FINITE.md), all of the Lean code, all experiment code and analyses, and this manuscript draft were produced with AI assistance (Claude-based coding agents). The ICML 2026 call for papers says:
   - authors may use generative AI tools;
   - authors take full responsibility for all content;
   - authors are encouraged to explain notable uses in their methodology;
   - LLMs cannot be authors.

   **ICML 2027's policy must be checked by the authors when it is published.** Decide whether and where to disclose: a sentence in the paper (methodology or appendix), the OpenReview form, or both. Given how much of the work was AI-produced, a short explicit statement is advisable. The draft currently contains **no** disclosure statement.
3. **Human verification of the paper-only proofs.** Lean checks the graph theory (Section 3) completely, and only the deterministic and algebraic core of Sections 4 and 5. **A mathematician (a co-author, not an agent) should read the following proofs line by line before submission:**
   - Prop. 7 (zero pattern under per-node filters; MA(∞) support argument; real-analytic genericity).
   - Thm. 8 (a) rate (geometric decay of Γ(h)), (c), (d) (continuity of R² in S; the star inequalities, which are matrix algebra and could be formalised).
   - Lemma D.1 as a probability statement (Gaussian MGF, rotation invariance).
   - Lemma D.2: ‖C^{1/2}MC^{1/2}‖_F² = tr(MCMC) and the operator-norm bounds.
   - Lemma D.3: the union bound and the Chernoff bound for sample means.
   - Thm. 9: assembly of the probability bound and the explicit asymptotic exponent.
   - Prop. 10 and the n_1/2 / n_crit locations. These are asymptotic and rest on a CLT for quadratic forms that is assumed, not cited. **Add a verified reference** (see REFS_LOG.md).
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

## Pre-submission checklist (updated after E1/E2)

- [ ] ICML 2027 style files swapped in; re-check that the main text still ends on page 8 (it currently ends 12 lines into page 8, with no slack).
- [ ] ICML 2027 call for papers read: deadlines, AI policy, impact-statement rules, financial conflict-of-interest disclosure, reciprocal reviewing.
- [ ] AI-use disclosure written (decision 2).
- [ ] A mathematician reads the paper-only proofs (decision 3), including the new ones: THEORY_ADD A1-A3 (lagged relatives; A3 constants are sketched) and B1-B3 (linear-zero lemma; B2 is a physical assumption, not checked).
- [ ] E1 assumption B2 (ideal polariser sensors, aligned zero marks): decide whether to test it on other light-tunnel data or to keep it as a stated assumption. Reviewers will ask about G_lin versus G_D.
- [ ] Decide whether to keep the post hoc constant-1/2 comparison in the main text (it is in `../icml_exp/explore4.log`, not pre-registered; the paper labels it post hoc).
- [ ] Every number re-checked against its source file (table below), including the new E1/E2 rows.
- [ ] Pre-registration evidence (decision 5): all four pre-registrations and studies fall within about 5.9 hours on one day, by agents, with client-set commit times. Supply push records or an anonymised mirror. Real times relative to T0 are in Appendix H (plan 4 at T0+5:50, results at T0+5:54).
- [ ] Literature search refreshed (decision 4); new references verified and logged in REFS_LOG.md. E1/E2 added no new references.
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
| Lean counts (116 theorems, 3,343 lines, axioms) | `../lean/AXIOMS.txt` (116 lines), `wc -l ../lean/RelSort/*.lean`, `../lean/BUILD_LOG.txt` |
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
