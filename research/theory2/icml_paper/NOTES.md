# Notes for the authors: ICML 2027 draft

Files: `main.tex` (main text), `appendix.tex` (unlimited appendix, `\input` from main), `refs.bib`, `REFS_LOG.md` (verification of every reference), `Makefile`, `icml2026.sty`/`.bst` plus the helper `.sty` files from the official zip. Build: `make`. It runs pdflatex, bibtex, pdflatex and pdflatex, then fails if the log contains "undefined". The PDF is `main.pdf`.

## Status of this draft (2026-11-04)

- **Style.** No ICML 2027 style was available on icml.cc on 2026-11-04 (`/Conferences/2027/...` returned 404). The draft uses the official **icml2026** style (version of 2025-10-29) from `https://media.icml.cc/Conferences/ICML2026/Styles/icml2026.zip` (sha256 `8b29290f…6bfc6c`). **Swap to icml2027.sty when it is released**: replace the `.sty`/`.bst` files and the two `icml2026` lines in `main.tex`, then re-check the page count.
- **Length.** The main text, including Table 2 and both figures, ends near the top of page 7. The 8-page limit leaves about 1.5 pages for E1/E2 results and reviewer-requested material. The impact statement, references and appendix do not count. The PDF is 14 pages in total.
- **Build.** Clean: 0 undefined references or citations, 0 BibTeX warnings, no overfull boxes.
- **Placeholders.** These are the red `\PH{...}` boxes. Search `\PH{` to find them.
  - Main text, Section 7: E1 (discriminating test against naive predictors) and E2 (lagged relatives).
  - Appendix H: timeline row for E1/E2.
  - Appendix I: full E1 and E2 sections.
  - These experiments are being run by another agent in `../icml_exp/`. No numbers were invented. Fill them only from that folder's committed results and README, including any failed pre-registered criteria. If E1 or E2 fails, say so in the main text, as for P2–P9.

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

## Checklist before submission

- [ ] ICML 2027 style files swapped in; page count re-checked (main text ≤ 8 pages).
- [ ] ICML 2027 call for papers read: deadlines (abstract and full paper), AI policy, impact-statement rules, financial conflict-of-interest disclosure (new in 2026), reciprocal-reviewer rules.
- [ ] E1 and E2 results inserted from `../icml_exp/` (main text and Appendices H and I), with pre-registration times; no `\PH{` remaining (`grep -n 'PH{' *.tex`).
- [ ] AI-use disclosure written (decision 2).
- [ ] Probabilistic proofs read by a human mathematician (decision 3); CLT reference added and verified.
- [ ] Every number in the paper re-checked against its source file (table below).
- [ ] Literature search refreshed (decision 4); new references verified and logged in REFS_LOG.md.
- [ ] Supplementary material assembled and anonymised:
  - Lean project (`../lean`; strip author names, licence headers and any paths that contain user names);
  - `../verify`, `../finite`, `../empirical`, `../empirical2`, `../realdata` scripts and logs;
  - PLAN/PLAN2/PLAN3;
  - the axioms report and build log;
  - a README.

  No repository URL may appear in the paper. Check that the files contain no names, e-mails or absolute paths such as `/home/user/...`.
- [ ] Impact statement reviewed (currently the ICML default sentence).
- [ ] Figures checked in print and greyscale. Fig. 2 uses colour plus marker shape, so it is not colour-only.
- [ ] Proofread; check that theorem numbers cited in prose match. The main text uses a single counter (Theorem 1 … Proposition 12).
- [ ] OpenReview accounts for all authors; abstract registered before the abstract deadline.
- [ ] Camera-ready only: de-anonymise, add acknowledgements, commit hashes for the pre-registrations, and a lay summary.

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
| Appendix H timeline | `git log --format='%h %at %s' -- research/theory2`, times relative to the first plan commit |

## Known weaknesses a reviewer will raise

- The positive real-data result is partial: P9 failed, the wind-tunnel graph is bipartite, and the replication uses i.i.d. inputs. E1 is meant to address the bipartite caveat.
- The CausalRivers F3 agreement is in-sample (r̂ and ℓ̂ are estimated from the tested correlations). The paper says so.
- The non-asymptotic bounds are loose and uninformative at the real sample sizes; only the asymptotic formulas predicted outcomes.
- Self-loops at every node are assumed throughout. Nonlinear mechanisms and contemporaneous effects are not covered.
