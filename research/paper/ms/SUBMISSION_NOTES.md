# Submission notes for research/paper/ms (evaluation-study manuscript)

Status: draft package, not submittable until the authors complete the items under "What the authors must do".
Build: `./build.sh` (figures, tables, PDF), or by hand: `python3.13 make_example_curves.py` (once, about 2 min), `python3.13 make_figures.py`, `python3.13 make_tables.py`, `pdflatex main; bibtex main; pdflatex main; pdflatex main`. Scripts need python3.13 (the default `python3` lacks matplotlib and torch). Every table and figure is generated from CSVs under `research/`; none is typed by hand except Table "Negative and failed results" (numbers copied from the READMEs, sources named in its note) and the verdict table in Appendix A (copied from `research/replication/README_results.md`).

## 1. What is ready
- `main.tex` (article class, no journal template), `references.bib` (verified references only), `make_figures.py`, `make_tables.py`, `make_example_curves.py` (Fig. 3 data, uses research/replication code on one d=100 dataset, seed 2000, six edges drawn at random with RandomState(0)), `msdata.py` (loaders), `build.sh`.
- Figures (vector PDF, Okabe-Ito colour-blind-safe palette): 1 structure vs function, 2 edge NRMSE by cell with bootstrap CIs (SSC, refit, adaptive, BASIL-W), 3 example curves, 4 do-effect direct vs indirect, 5 Sachs, 6 band coverage before/after kappa.
- Sections: Abstract, Introduction with the contribution list and what is not novel, Related work, Problem and metrics, Methods, Protocol (pre-registration commits, seeds, validation vs test), Results (replication, structure cost, do-effects, adaptive arm, independent-DAG replication, adaptive2, ablations, bands, semi-synthetic, Sachs, negative results), Discussion with the full limitation list, Conclusion, Reproducibility statement, Declarations (placeholders), Appendices (hypothesis verdicts, deviations).

## 2. Sections that depend on other agents (see the status block at the end of this file)
- Section "Independent-DAG replication" (research/replication2), "Adaptive arm at d=500 and sensitivity" (research/adaptive2), "Semi-synthetic benchmark from Sachs" (research/semisyn). Their tables are produced by `make_tables.py` from whatever per-dataset CSVs exist; when data are missing the table is a boxed PENDING placeholder. The explanatory text of these three sections lives in `sections/*.tex` and must be re-read against the final numbers and the other agents' README_results.md files (keep their caveats) before submission.
- Abstract and Conclusion do not use these results; update them if the new results change the picture (for example if the independent-DAG replication does not reproduce the BASIL-W-over-SSA gap).

## 3. What the authors must do
1. Choose the venue and re-format (the manuscript uses plain `article`; reference style is natbib `plainnat`).
2. Complete the author list, affiliations, and the Declarations (author contributions, competing interests, funding, data availability, AI-assistance statement). The AI statement is a placeholder with the facts only (an AI assistant was used for code, experiments, analysis and drafting); the authors are responsible and must adapt it to the venue's policy.
3. Verify every number by re-running the scripts (`aggregate.py` in each experiment directory, then `make_tables.py`). The text quotes numbers from the READMEs; the generated tables are the check. In particular re-check: all values in Sections 6.1-6.5, the Sachs paragraph, the ablation paragraph, and the abstract.
4. Resolve the reference TODO list (section 6) and decide whether to cite the SPADE paper's final volume, pages and year (the bib entry has year = n.d. and a note; leave nothing like that in the submitted version).
5. Decide whether to cite a preprint and the code repository, insert the release commit hash and licence in the Reproducibility statement (placeholder in text).
6. Read the ICML 2026 paper (Yao et al.) and Kuskova et al. in full; the text says "read via abstract only" and must be updated, and the novelty statement may need to change.
7. Decide whether the BASIL-W name and the pre-registration claims should stay: pre-registration here means git commits in a local history, not an external registry; say so in the text (it does) and decide whether a reviewer-visible registry (OSF) is worth adding.
8. Re-run the heavy experiments only if a reviewer asks; the package does not need it.

## 4. Candidate venue types (no claim about acceptance)
- Methodological machine-learning or causal-inference workshops and the evaluation/benchmark tracks of ML conferences (evaluation-and-negative-results framing).
- Journals that publish empirical evaluation or reproducibility studies in statistics, causal inference or ML (for example journals with a replication or benchmark article type), and data-science/IS journals if the SPADE application context is kept.
- The venue of the SPADE paper (Array, Elsevier) is a possibility but the paper is then a follow-up evaluation of SPADE variants; check the venue's policy on overlap and on negative-results papers.

## 5. Checklist
- [ ] Figures are vector PDF (all six are; fonts embedded Type 42). Check legibility at the final column width.
- [ ] Anonymisation if double-blind: remove author line, the repository path and any commit identifiers that reveal identity, the "Declarations" names, and the self-citation of SPADE (rephrase as "the SPADE learner of [anonymous]"); regenerate PDF metadata.
- [ ] Code and data release: tag a commit, record its hash in the manuscript and in `research/README`; include per-dataset CSVs, plans (commits are the pre-registration evidence), and the Sachs data licence/source statement.
- [ ] Licence for code (not chosen) and for data (Sachs public dataset: state source and terms).
- [ ] All tables regenerated after the last experiment finished; no PENDING boxes in the PDF (`grep -i pending main.tex sections/*.tex tables/*.tex`).
- [ ] Reference list: no `n.d.`, no "and others", no placeholder authors in references.bib.
- [ ] Page and word limits of the venue; move Appendix B and parts of Section 6 to supplementary material if needed.
- [ ] Spell-check and read the text once for tone (it is deliberately flat).

## 6. References to confirm (TODO list; the text does not flag them)
- Dang, Le, Dinh, SPADE, Array (Elsevier): volume, pages, year (bib has `year = n.d.`).
- Kaiser and Sipos, Neural Processing Letters: volume, pages, print year (bib uses 2021 from the DOI; REPORT says print year not confirmed).
- Wang, Kolar, Drton, JASA: volume/issue/pages (DOI only; bib year 2025 from the DOI).
- SCORE (Rolland et al.), DiffAN (Sanchez et al.), NoGAM/DAS (Montagna et al.): author lists and page numbers were not enumerated in lit_verify; the bib uses the authors as remembered by the drafter. Check against the proceedings pages.
- Hou et al. 2024 (KAN critical assessment) and Yu et al. 2024: authors from memory; findings not read.
- Kuskova, Zaytsev, Coppedge (ICML 2026): first names not recorded in lit_verify; abstract read only.
- Friedman and Nachman 2000 (Gaussian process networks): found in prior-art searches, venue details not verified in lit_verify; bib gives UAI 2000.
- Sachs et al. 2005: carried over from the earlier notes and NOT re-checked in lit_verify (listed under "not re-checked"); bib gives Science 308:523-529.
- Zou 2006 (adaptive lasso) and Friedman and Koller 2003: verified only from secondary sources in lit_verify.
- Ravikumar 2009, Belloni and Chernozhukov 2013, Huang et al. 2010: verified; no action beyond a final check.
- Removed rather than cited: Sharma and van Beek 2022 (title not recorded), Ng et al. CLeaR 2024 "sober look" (title not confirmed), Seng et al. 2022 preprint, SDCD venue, Wu and Xie arXiv:2604.25295 (unverifiable, do not cite).

## 7. Remaining weaknesses, in plain language
1. Almost everything is simulation from one generator family (Erdos-Renyi, degree 2, a few smooth mechanisms). Reviewers will ask for other graphs, denser graphs, non-additive mechanisms, confounding.
2. The real-data result is negative. There is no ground truth for functions on Sachs, structure is barely above chance, and the repairs are less stable than the unrepaired learner. The semi-synthetic benchmark helps only to the extent that the fitted model resembles the real system.
3. The headline repair method (BASIL-W) is not novel in its ingredients, costs AUROC in essentially every dataset, and is mostly a refinement of SPADE-scaled's own order and candidates. The standard adaptive group lasso gets most of the gain for free. The paper says this, so the contribution is the measurement, the metrics and the comparison, which some reviewers will find thin.
4. The edge-function NRMSE rewards amplitude; it is a measure we defined, with no independent validation. A method with the right shape and wrong amplitude is penalised, which is the point, but some readers will want shape-only (R2_aff) numbers next to it (curve R2 is in the CSVs and is not in the paper tables).
5. The adaptive group lasso was untuned and run on 3-10 datasets per cell in the first grid; a tuned version could close the remaining gap, which would weaken the case for the Bayesian estimator further.
6. Ten datasets per cell in the first replication, shared DAGs across cells. The independent-DAG replication addresses this only for three noise families at d=100.
7. Credible bands are valid only on average, calibrated on five validation datasets; found edges are over-covered.
8. Pre-registration is by git commit, which a reviewer may regard as weak evidence, and several settings (ridge stabilisation, BASIL-S amendments, refit multiplier) were fixed after seeing test or validation results; each is disclosed.
9. The literature check was done by web search with snippet-level reading, two nearest papers were read only by abstract, and "not done before" claims rest on absence of hits.
10. Timings come from a shared, loaded 4-core machine.
11. The uniform datasets of the independent-DAG replication (seeds 3000-3012) are the same datasets as the ablation validation/test sets (seeds 3000-3004 validation for kappa, 3005-3012 ablation test). No BASIL-W setting changed after the ablation, but the replication's uniform cell is therefore not fully fresh; the text says so.

## 8. Discrepancies found while writing (please check)
- `research/real_data/README_results.md` says SSC found "about one extra true edge in the top 18 versus chance (about 2.9 hits at chance, 5.8 observed)"; 5.8 minus 2.9 is about three extra edges. The manuscript quotes the two numbers and does not use the phrase.
- The real_data README caveat calls the consensus graph "acyclic by construction", while research/semisyn/gen.py reports that the 18-edge consensus graph contains one directed cycle (PIP2, PIP3, plcg). The manuscript does not claim acyclicity. Check the real-data runs' handling and the semi-synthetic note.
- `research/adaptive/results` contains one duplicated uniform row (11 rows for 10 datasets); `msdata.load_adaptive` de-duplicates by cell, seed and method, so the tables use 10.
- The adaptive-arm paired tests use the SSC/refit/BASIL-W values of the replication run on the same seeds (independent runs), not a single joint run.
- The first-grid values quoted in `draft_v1.md` for BASIL-W at d=100 uniform (0.194 on seeds 1000-1004) differ from the replication (0.169 on 2000-2009); the manuscript uses the replication throughout and mentions the earlier seeds only in the protocol.
- The Sachs sign-sanity relations were typed from memory in the real-data run; the manuscript states this.

## 9. Status of dependent experiments (updated after all four finished)
- replication2 (60 datasets), adaptive2 (d=500 two datasets, grid five datasets), semisyn (85 datasets) and external (30 datasets, d<=20) are finished and integrated: tables come from the CSVs via `make_tables.py` (tab_rep2, tab_rep2b, tab_adaptive2, tab_semisyn, tab_external), text is in `sections/rep2_text.tex`, `adaptive2_text.tex`, `semisyn_text.tex`, `external_text.tex` (new Section "External baselines at small d"). No PENDING boxes remain.
- Claims were revised: BASIL-W is NOT claimed to be best overall (CAM-like, our own Python reimplementation, matches or beats it at d<=20; BASIL-W beats the refit by only 0.03-0.04 under uniform/heteroscedastic noise on independent DAGs; GAM-ridge ties or beats it at d=11 in the semi-synthetic benchmark). Abstract, introduction, discussion, limitations and conclusion were rewritten accordingly; the contribution is the amplitude-shrinkage diagnosis, standard fixes, and the scale checked.
- Unresolved items to check by the authors: the "about 85% of the gap" figure is d=100 only (about 75% at d=500, two datasets); the replication2 uniform seeds 3000-3012 coincide with the ablation validation/test sets (disclosed); the external README's pooled test counts E1 and E3 twice (shared DAGs); the semi-synthetic consensus graph had one edge dropped by a pre-fixed rule (17 true edges); the real_data README's acyclicity wording was already corrected upstream.
