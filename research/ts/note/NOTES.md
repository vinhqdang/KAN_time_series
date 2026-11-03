# Notes for the authors (note/ folder)

Nothing here has been submitted anywhere. `note.pdf` is a draft built from `note.tex`, `references.bib` and `generated/` (made by `make_assets.py` from `../stage0/results.md` and `gates.md`; no new experiments). Build: `/usr/bin/python3 make_assets.py && pdflatex note && bibtex note && pdflatex note && pdflatex note`.

## Decisions you must make
- **Venue and format.** Currently a plain article (about 7 pages). Pick a venue (workshop, arXiv only, benchmark/evaluation track) and switch to its template and page limit.
- **Author list and affiliations.** The author field is a placeholder. Decide who qualifies as an author and the order.
- **Declarations.** Funding, competing interests, ethics/data statement (CausalRivers is public, licence dl-de/by-2-0; state this), code and data availability (which repository state and archive to cite).
- **AI-use statement.** The study, code, and this draft were produced with AI assistance. Write a disclosure that matches the venue's policy and your actual use; verify every number yourself.
- **Title.** The title states the finding in full and is long; shorten if the venue requires.

## Things to check before use
- **Bibliography.** references.bib has only entries marked verified in `../01_literature.md`. Author lists for CausalRivers and DYNOTEARS are truncated ("and others") because the log only verified first authors; complete them from the proceedings or arXiv pages.
- **Not cited (not marked verified there):** NOTEARS, and the sortnregress sources (var-sortnregress, R2-sortnregress). The note describes these without citation; add the correct references after checking them. TECDI, DTCD and ENVAR are deliberately not cited.
- **CausalRivers claims** in the related-work section (resolutions as a tuning knob, magnitude baseline, no separate contemporaneous scoring) come from the stage0 README's reading of arXiv 2503.17452v1; re-check against the final ICLR version.
- **Hindsight reading.** The "skeleton detection, scale leakage, shared forcing" interpretation is post hoc and flagged as such in the text. No paired test of correlation versus PCMCI+/DYNOTEARS AUROC was run; the text only says differences are small.
- The random-score control has accuracy 0.585 [0.542, 0.626] at 12 h (CI excludes 0.5); this is reported but unexplained.
- G1 is below threshold in East Germany (0.624) and 6 h fails G2; both are stated openly. Consider whether a regional gate should have been pre-registered.
