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

- **Competing interests:** placeholder (red) in the Declarations.
- **Code availability:** placeholder for the public repository URL or archive DOI.
- **Acknowledgements:** placeholder; delete the heading if there are none.
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
