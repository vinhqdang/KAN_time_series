# Reference verification log

All checks were done on 2026-11-04 through the session's HTTPS proxy with TLS verification on. "Metadata verified" means that title, authors, venue and year in `refs.bib` match the page listed. "Content" says whether the claims we attribute to the work were checked against its full text, and where that happened. Every entry in `refs.bib` appears here. Do not add a citation without adding a row.

| key | page opened (2026-11-04) | metadata verified | content behind our claims |
|---|---|---|---|
| reisach2021beware | arxiv.org/abs/2102.13647 (comments: "accepted at ... NeurIPS 2021"); papers.nips.cc 2021 abstract page (title matches) | yes (NeurIPS 34, 2021) | the abstract (read 2026-11-04) states both claims we use: marginal variance increases along the causal order (varsortability), and the performance of some continuous structure-learning algorithms is explained by it and matched by a simple baseline. Full text not re-read |
| reisach2023scale | arxiv.org/abs/2303.18211 (comments: NeurIPS 2023); proceedings.neurips.cc 2023 abstract page (title matches) | yes (NeurIPS 36, 2023) | R²-sortability as a scale-invariant criterion (abstract). Full text not re-read |
| ormaniec2025standardizing | arxiv.org/abs/2406.11601; proceedings.iclr.cc/paper_files/paper/2025 (title and authors listed) | yes (ICLR 2025) | our sentence ("not var-sortable by construction and, empirically, mostly not R²-sortable for common graph families") paraphrases the arXiv abstract, read on 2026-11-04. Full text not read |
| herman2025unitless | arxiv.org/abs/2503.17037; proceedings.mlr.press/v275/herman25a.html (PMLR 275, pp. 1506-1531) | yes | Sec. 5.1, App. D and Sec. 6 read in full in an earlier phase (`../THEORY.md` §4). Our crediting sentence follows that reading |
| reisach2026topological | arxiv.org/abs/2605.06288 (title, 4 authors, 2026-05-07) | yes (arXiv preprint; no venue listed) | full PDF read in an earlier phase (`../PLAN.md`, `../THEORY.md` §0 and §4): definitions, Thm 1, Lemma 1 eq. (10), Sec. 5, Thm 4 (sufficiency only), App. A.3. **Check before submission whether it has since appeared at a venue** |
| lohse2025sortability | arxiv.org/abs/2407.13313 (comments: "Published in Transactions on Machine Learning Research") | yes (TMLR 2025, per arXiv comment and `../THEORY.md` "TMLR 07/2025"). The TMLR OpenReview page was not opened (browser check) | introduction, contributions and Sec. 3 read in an earlier phase (`../THEORY.md` §4) |
| gamella2025chambers | arxiv.org/abs/2404.11341; Crossref record of DOI 10.1038/s42256-024-00964-x (Nat. Mach. Intell. 7(1):107-118, 2025). The doi.org page returned HTTP 429, so the Crossref API was used | yes | data descriptions and the generator script were read when the data were used (`../realdata/PLAN3.md`) |
| stein2025causalrivers | arxiv.org/abs/2503.17452; proceedings.iclr.cc/paper_files/paper/2025 (title and authors listed) | yes (ICLR 2025) | dataset and sampler used in `../empirical`, `../empirical2` |
| runge2019detecting | arxiv.org/abs/1702.07007; Crossref record of DOI 10.1126/sciadv.aau4996 (Sci. Adv. 5(11), 2019). science.org returned 403 | yes; the article number eaau4996 is taken from the DOI | cited only as an example of a time-series discovery method (PCMCI) |
| gong2015subsampled | proceedings.mlr.press/v37/gongb15.html (PMLR 37, pp. 1898-1906) | yes | cited only for the topic (identifiability under subsampling). Full text not read |
| gong2017aggregated | pmc.ncbi.nlm.nih.gov/articles/PMC5995575 (citation_journal_title: UAI proceedings; authors match) | yes (UAI 2017) | cited only for the topic. Full text not read. Earlier phases had seen it only in search snippets |
| zheng2018notears | arxiv.org/abs/1803.01422 (comments: "accepted to NIPS 2018") | yes (NeurIPS 31, 2018) | cited only as a method shown by reisach2021beware to benefit from var-sortability |
| hsu2012tail | arxiv.org/abs/1110.2842; Crossref search: Electron. Commun. Probab. 17, 2012, DOI 10.1214/ecp.v17-2079 | yes | Proposition 1 read in an earlier phase (`../finite/FINITE.md` §6). Our Lemma D.1 is its Gaussian case, proved self-contained |
| laurent2000adaptive | Crossref record of DOI 10.1214/aos/1015957395 (Ann. Statist. 28(5):1302-1338, 2000) | yes | **full text NOT opened** (Project Euclid download was blocked in an earlier phase). Cited only as the source that HKZ derive their Prop. 1 from; our proof does not depend on it |
| demoura2021lean4 | Crossref record of DOI 10.1007/978-3-030-79876-5_37 (CADE 28, LNCS, pp. 625-635, 2021). link.springer.com returned a bot challenge | yes | tool citation |
| mathlib2020 | arxiv.org/abs/1910.09336; Crossref record of DOI 10.1145/3372885.3373824 (CPP 2020, pp. 367-381) | yes | tool citation |

## Not cited (checked or considered, left out)

- Brockwell and Davis, *Time Series: Theory and Methods* (Bartlett's formula, CLT for quadratic forms). The text was not opened and the theorem number is not verified. The paper therefore states the CLT as an assumption ("standard CLT for quadratic forms of stationary Gaussian processes") without a citation. **Before submission, a co-author should add a verified reference for this CLT** (for example a textbook theorem checked in print).
- arXiv 2605.04081 (variable-lag time-series causal discovery): seen only in a search snippet in an earlier phase; not opened, not cited.
- Danks and Plis (subsampling), Hyttinen et al. (subsampling): not opened, not cited.
- Pre-registration methodology references: none cited. Add one if a reviewer asks.

## Unverified items in the draft

- None of the 16 cited entries is unverified at the metadata level.
- Content-level caveats are marked in bold above: the Laurent-Massart full text (not opened) and the venue status of Reisach et al. (2026). For reisach2021beware, reisach2023scale, ormaniec2025standardizing, gong2015subsampled, gong2017aggregated, zheng2018notears and runge2019detecting, only the abstract or landing page was read for this draft. The claims we attribute to them are limited to what their abstracts state.
