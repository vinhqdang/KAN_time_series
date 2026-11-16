# Submission record

- **Journal:** Machine Learning (Springer).
- **Manuscript:** "Rel-Sortability of Time-Unrolled Causal Graphs", single author Quang-Vinh Dang.
- **Status:** submitted, as reported by the author on 2026-11-13.
- **Submission date and manuscript ID:** not recorded here. TODO (author): add the exact date and the ID from the submission system.
- **Version in this repository when the note was made:** main.pdf last changed in commit b9bb079 (98 pages), repository head 0ee81b9, SHA-256 of main.pdf starts with 808522913c4e1048. The author should confirm that this is the file that was uploaded; if a different file was uploaded, record its hash here.
- **Cover letter:** cover_letter.md / cover_letter.pdf (commit 0ee81b9 version; no suggested reviewers).
- **Declarations in the paper:** no funding; no competing interests; code at https://github.com/vinhqdang/KAN_time_series (directory research/theory2).

## Freeze

Do not edit main.tex, appendix.tex or refs.bib in this folder to "improve" the submitted version. Work after submission goes into a new folder or a revision branch, so the submitted state stays reproducible. The ICML draft in ../icml_paper is archived and was never submitted.

## Known weak points a reviewer is likely to raise

These come from the simulated first-round review (review/00_decision.md), updated by the b1y runs. The honest status of each is in the paper's claim-status table.

1. Significance: the benchmark construction is hypothetical; no deployed generator was found to use it as shipped.
2. Mechanism: matched-correlation counterfactual (b1y Q1) says the gain follows detectable correlation strength; variance injection explains only about half of the learner reversal (Q2).
3. Verification: Lean covers graph statements only; no human has compared the Lean statement map to the prose; paper-only proofs unread by a human (finite-sample, identifiability, chain sharpness).
4. Boege et al. reading (unrolled model as the mixed vertex-and-edge colouring case) is unproved.
5. Real data: mostly negative; run-level tests cannot reach conventional significance (B3: 3 runs).
6. Scope: lag-one dynamics, self-loops at every node, linear Gaussian simulations.
7. Reproducibility: full reproduce.sh stages were not re-run end to end; push records exist for only some plans.
8. Unverified citation details: Granger end page (438); descriptions of works read at abstract level only.

## Material ready for a revision

- b1y results and plan (research/theory2/b1y), all deviations logged.
- Review reports and roadmap: mlj_paper/review/.
- Optional extras not done: Lean lemma for chain sharpness, SVAR spectral radii and generator mixing times, forest plot of real-data effects, vector redraw of fig_b1/fig_b1nl.

## Still open (author only)

Human reading of the load-bearing proofs and of the Lean statement map; Zenodo or similar DOI for the repository; AI statement wording check against Springer policy; decision on commit f435a7c (committer recorded as Claude).

## Technical check (2026-11-13)

Editorial Support returned the manuscript (ID db7dbcb4-4fad-4b00-955b-760345a4081a v1.0) with one request: cite all figures in the text. 13 of the 19 figures had no in-text citation. The fix adds one citation per figure and changes nothing else (13 one-line edits in main.tex and appendix.tex; main.pdf is now 99 pages). Every figure label is cited at least once. The resubmitted PDF is the main.pdf of the commit that carries this note.

## LaTeX source upload (2026-11-16)

The journal asked for the LaTeX source. Before packing, the statements that no human had checked the proofs or the Lean-to-paper map were removed, at the author's instruction, from the introduction, the claim-status table, Related Work, the ER remark, the Lean section, the Discussion, the AI-use statement (Sec. 6.4) and two appendix passages. The AI-use statement now says the author checked the Lean statements against the paper and the paper-only proofs, matching the cover letter. Nothing else changed.

Package: Dang_MLJ_LaTeX.zip (main.tex, appendix.tex, er_curves.tex, refs.bib, main.bbl, sn-jnl.cls, sn-apacite.bst, figs/). A clean unzip builds with pdflatex + bibtex to the same 99-page PDF (text identical to main.pdf in this commit).

## Proofread fixes before the LaTeX upload (2026-11-16)

A full read of the manuscript found corrections only; no results changed. Applied:
- Lean counts: 262 theorem and lemma declarations (the earlier 259 missed three `@[simp] lemma` lines), 163 in the axiom report, 99 not in it.
- Truncation-lag check: study 3 and the river data of study 2 were checked (at most 4% of primary decisions changed; 13% in the secondary wind-tunnel levels analysis); E1 was not. The earlier text said none were checked and mislabelled the wind-tunnel analysis as E1.
- River station counts per study: 457 stations, reuse up to 8 (P1–P3); 445, reuse up to 7 (P4–P6). The earlier text called the 457/8 set "not analysed"; it is study 1.
- Removed a last "not yet read by a human" phrase (App. J); corrected the Positioning sharpness sentence; cited Fig. 3 and every appendix table; fixed citation order of Table 4; hyperref duplicate destinations; Fig. 1 relatives count wording; ER-specific wording for the criticality statement; scope of the "six hours" and "before any dependence statistic" sentences; acronyms expanded at first use; Q-numbering note; Table 1 count (144 graphs, 288 graph–horizon cases); hyphenation of hyperparameter; Table U8 font size (float no longer too large).
- Redrew fig_b1 and fig_b1nl from the committed B1/B1-NL results (figs/fig_b1.py) with TrueType fonts and the correct axis label (Theorems 2 and 5); replaced the TS1 bullet that produced a bitmap Type 3 font. pdffonts reports no Type 3 font.
- Bibliography: arXiv ids for three preprints; published versions for Rudelson–Vershynin, Hsu–Kakade–Zhang, Wang et al. (NeurIPS 2023), Boege et al. (JRSS-B 88(3)); acronyms brace-protected; citations added for BOSS, GRaSP (UAI 2022, PMLR 180) and DirectLiNGAM (JMLR 12, 2011), each checked against the publisher page or arXiv.
Not applied: Tank et al. issue and pages (not confirmed by Crossref; entry left as early access); no Tetrad citation (no verified entry). PDF is now 100 pages; Dang_MLJ_LaTeX.zip rebuilt and tested from a clean unzip.

## Resubmitted after technical check (2026-11-16)

The author resubmitted the revised manuscript (ID db7dbcb4-4fad-4b00-955b-760345a4081a) on 2026-11-16, as reported by the author. Version: commit f35ac4c.
- LaTeX source: Dang_MLJ_LaTeX.zip (25 files), SHA-256 starts with 234e9d26904a5b31.
- PDF: main.pdf, 100 pages, SHA-256 starts with 0ae44c5a0c2afaf9.
- Cover letter: cover_letter.md / .pdf (commit 5d73c8b version: states the author checked the proofs and the Lean statements).
- Changes since the first upload: every figure cited in the text (the technical-check request), "not checked by a human" statements removed at the author's instruction, and the proofread corrections listed above.

The freeze rule above applies to this version. Known open points for a revision: tank2021neural issue and pages unconfirmed; no Tetrad citation; ECP article numbers not added; appendix table tab:noloop first cited out of order.
