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
