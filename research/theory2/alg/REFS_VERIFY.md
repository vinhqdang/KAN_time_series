# Reference verification log (alg/)

Date: 2026-11-10. Every entry of `refs.bib` is listed here with the check that was made. Network access went through the configured proxy with TLS verification on.

How to read the last column. "confirmed" means the venue data came from the API record or the publisher page today. "from memory" means the arXiv record confirms that the paper exists with this title and these authors, but the venue named in `NOVELTY.md` was **not** checked today; `refs.bib` therefore cites those entries as arXiv preprints. Before any of them goes into the paper's `refs.bib`, the venue must be checked.

| bib key | identifier | check | venue status |
|---|---|---|---|
| reisach2026topological (in paper bib) | arXiv:2605.06288v1 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match; PDF downloaded and Sections 5–6 read | preprint, 2026-05-07 |
| friedman1998learning | arXiv:1301.7374 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | UAI 1998: from memory, not checked against the proceedings |
| boege2025colored | arXiv:2404.04024v3 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match; API returns DOI 10.1093/jrsssb/qkaf068; abstract read; HTML v2 read in part (BPEC definition, Thm 5.16, GECS) | JRSS-B confirmed by the DOI; year 2025 inferred from the DOI, volume and pages not checked |
| makam2023symmetries | arXiv:2108.10058v2 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match; Project Euclid page found by web search | EJS 17(2), 2023, DOI 10.1214/23-EJS2192 (from the Project Euclid URL) |
| wu2023partial | arXiv:2308.08959v1 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | journal version not checked |
| shah2012group | arXiv:1111.7061v1 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | journal version (EJS 2012) from memory, not checked |
| segal2005module | arXiv:1212.2517v1 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | UAI 2003 / JMLR 2005 versions from memory, not checked |
| maier2013sound | arXiv:1309.6843v1 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | UAI 2013 from memory |
| yeh2022equivariance | arXiv:2204.03640v1 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | AISTATS 2022 from memory |
| zhou2020metalearning | arXiv:2007.02933v3 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | ICLR 2021 from memory |
| mena2018learning | arXiv:1802.08665v1 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match; journal_ref field says ICLR 2018 | ICLR 2018 confirmed |
| cundy2021bcd | arXiv:2112.02761v1 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | NeurIPS 2021 from memory |
| charpentier2022differentiable | arXiv:2203.08509v1 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | ICLR 2022 from memory |
| zantedeschi2023dag | arXiv:2301.11898v2 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | ICLR 2023 from memory |
| raskutti2018learning | arXiv:1307.0366v4 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | Stat 2018 from memory |
| rolland2022score | arXiv:2203.04413v1 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | ICML 2022 from memory |
| zhao2026learning | arXiv:2606.03227v1 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match; abstract read | preprint, 2026-06-02 |
| peters2014identifiability | arXiv:1205.2536v3 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match; journal_ref and DOI 10.1093/biomet/ast043 in the API record | Biometrika 101(1):219–228, 2014 confirmed |
| chen2019causal | arXiv:1807.03419v2 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match; DOI 10.1093/biomet/asz049 in the API record | Biometrika confirmed by DOI; volume and pages not checked |
| gao2020polynomial | arXiv:2006.11970v2 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | NeurIPS 2020 from memory |
| gao2022optimal | arXiv:2201.10548v2 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | AISTATS 2022 from memory |
| zhou2021efficient | arXiv:2111.01560v1 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | preprint |
| zhao2021learning | arXiv:2111.00740v1 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | preprint |
| stein2024approximating | arXiv:2401.11858v3 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match (v1 had the title 'Approximating a linear dynamical system from non-sequential data', per web search) | preprint |
| guan2026identifying | arXiv:2410.22729v5 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match; journal_ref field: JMLR (2026), 27(219), 1–56 | JMLR confirmed by the arXiv record |
| lavenant2021towards | arXiv:2102.09204v2 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | journal version not checked |
| rimawifine2025simulation | arXiv:2510.16656v2 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | preprint |
| fogel2013convex | arXiv:1306.4805v3 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | NeurIPS 2013 from memory |
| flammarion2019optimal | arXiv:1607.02435v3 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | Bernoulli 2019 from memory |
| berenfeld2024seriation | arXiv:2408.10004v2 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | preprint |
| vogelstein2015fast | arXiv:1112.5507v5 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | PLoS ONE 2015 from memory |
| lyzinski2016graph | arXiv:1405.3133v3 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | IEEE TPAMI 2016 from memory |
| pananjady2016linear | arXiv:1608.02902v1 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | IEEE Trans. Inf. Theory version (different title) not checked |
| unnikrishnan2015unlabeled | arXiv:1512.00115v1 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | journal version not checked |
| hsu2017linear | arXiv:1705.07048v2 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | NeurIPS 2017 from memory |
| abid2017linear | arXiv:1705.01342v2 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | preprint |
| tsakiris2019homomorphic | arXiv:1901.07852v3 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match; journal_ref field: ICML 2019, PMLR 97:6335–6344 | confirmed |
| bauer2016arrow | arXiv:1603.00784v1 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match; PMLR page proceedings.mlr.press/v48/bauer16.html found by web search; abstract read | ICML 2016, PMLR 48:2043–2051 confirmed. Full text not read |
| bang2023wiser | arXiv:2306.01638v1 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | UAI 2023 from memory |
| bang2025constraint | arXiv:2503.21526v3 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | CLeaR 2025 according to a web search snippet, not checked |
| larsen2025score | arXiv:2502.06232v2 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | preprint |
| gunther2023causal | arXiv:2306.12896v1 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | UAI 2023 from memory |
| das2026sc3d | arXiv:2602.02830v3 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | preprint |
| biswas2025cits | arXiv:2508.01920v2 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | preprint |
| sanchez2025causal | arXiv:2510.24639v2 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match; PDF already in ../lit2/pdf/ | TMLR 2025 according to a web search snippet, not checked |
| lowe2020amortized | arXiv:2006.10833v3 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | CLeaR 2022 from memory |
| gobler2023causalassembly | arXiv:2306.10816v2 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | CLeaR 2024 from memory |
| hojsgaard2008graphical | DOI 10.1111/j.1467-9868.2008.00666.x | Crossref API (api.crossref.org/works): title, authors, journal, year, volume, pages, DOI match | JRSS-B 70:1005–1027, 2008 confirmed |
| huang2009learning | DOI 10.1145/1553374.1553430 | Crossref API (api.crossref.org/works): title, authors, journal, year, volume, pages, DOI match | ICML 2009 confirmed |
| peters2009detecting | DOI 10.1145/1553374.1553477 | Crossref API (api.crossref.org/works): title, authors, journal, year, volume, pages, DOI match | ICML 2009, pp. 801–808 confirmed |
| weiss1975time | DOI 10.2307/3212735 (also 10.1017/s0021900200048804) | Crossref API (api.crossref.org/works): title, authors, journal, year, volume, pages, DOI match | J. Appl. Probab. 12:831–836, 1975 confirmed |
| chan2006note | DOI 10.1093/biomet/93.1.221 | Crossref API (api.crossref.org/works): title, authors, journal, year, volume, pages, DOI match | Biometrika 93:221–227, 2006 confirmed |
| trapnell2014dynamics | DOI 10.1038/nbt.2859 | Crossref API (api.crossref.org/works): title, authors, journal, year, volume, pages, DOI match | Nat. Biotechnol. 32:381–386, 2014 confirmed (Crossref lists 10 authors; the bib entry copies that list) |
| saelens2019comparison | DOI 10.1038/s41587-019-0071-9 | Crossref API (api.crossref.org/works): title, authors, journal, year, volume, pages, DOI match | Nat. Biotechnol. 37:547–554, 2019 confirmed |
| magwene2003reconstructing | DOI 10.1093/bioinformatics/btg081 | Crossref API (api.crossref.org/works): title, authors, journal, year, volume, pages, DOI match | Bioinformatics 19:842–850, 2003 confirmed |
| schiebinger2019optimal | DOI 10.1016/j.cell.2019.02.026 | Crossref API (api.crossref.org/works): title, authors, journal, year, volume, pages, DOI match | Cell 176, 2019 confirmed (first page 1517 per Crossref) |
| matsumoto2017scode | DOI 10.1093/bioinformatics/btx194 | Crossref API (api.crossref.org/works): title, authors, journal, year, volume, pages, DOI match | Bioinformatics 33:2314–2321, 2017 confirmed |
| qiu2020inferring | DOI 10.1016/j.cels.2020.02.003 | Crossref API (api.crossref.org/works): title, authors, journal, year, volume, pages, DOI match | Cell Systems 10:265–274.e11, 2020 confirmed |
| pratapa2020benchmarking | DOI 10.1038/s41592-019-0690-6 | Crossref API (api.crossref.org/works): title, authors, journal, year, volume, pages, DOI match | Nat. Methods 17:147–154, 2020 confirmed |
| fonteijn2012event | DOI 10.1016/j.neuroimage.2012.01.062 | Crossref API (api.crossref.org/works): title, authors, journal, year, volume, pages, DOI match | NeuroImage 60:1880–1889, 2012 confirmed |
| bystrova2024macro | DOI 10.1101/2024.10.10.608447 | Crossref API (api.crossref.org/works): title, authors, journal, year, volume, pages, DOI match. A web search snippet showed an earlier title for the same bioRxiv DOI ('Causal discovery from ecological time-series with one timestamp and multiple observations'); the bib uses the Crossref title | bioRxiv preprint, 2024 |
| atkins1998spectral | DOI 10.1137/S0097539795285771 | Crossref API (api.crossref.org/works): title, authors, journal, year, volume, pages, DOI match | SIAM J. Comput. 28:297–310, 1998 confirmed |
| petersen2021data | DOI 10.1093/aje/kwab087 | Crossref API (api.crossref.org/works): title, authors, journal, year, volume, pages, DOI match | Am. J. Epidemiol. 190:1898–1907, 2021 confirmed |
| anderson1982formulation | DOI 10.1016/0304-4076(82)90095-1 | Crossref API (api.crossref.org/works): title, authors, journal, year, volume, pages, DOI match | J. Econometrics 18:47–82, 1982 confirmed |
| hsiao2002maximum | DOI 10.1016/S0304-4076(01)00143-9 | Crossref API (api.crossref.org/works): title, authors, journal, year, volume, pages, DOI match | J. Econometrics 109:107–150, 2002 confirmed |
| binder2005estimation | DOI 10.1017/S0266466605050413 | Crossref API (api.crossref.org/works): title, authors, journal, year, volume, pages, DOI match | Econometric Theory 21, 2005 confirmed; pages not returned by Crossref |
| huang2011learning | proceedings.neurips.cc/paper/2011/hash/6c3cf77d52820cd0fe646d38bc2145ca-Abstract.html | web search; the publisher page was among the results and its listing matched | NeurIPS 24 (2011) confirmed; pages 1548–1556 per the search summary, not in the bib |
| chickering2002optimal | jmlr.org/papers/v3/chickering02b.html | web search; the publisher page was among the results and its listing matched | JMLR 3:507–554, 2002 confirmed |
| niculescu2006bayesian | jmlr.org/papers/v7/niculescu06a.html | web search; the publisher page was among the results and its listing matched | JMLR 7:1357–1383, 2006 confirmed |

## Entries of the paper's bibliography used without a new check

`pamfil2020dynotears` (arXiv:2002.00498, re-confirmed today by the arXiv API), `zheng2018notears` (arXiv:1803.01422, re-confirmed), `bello2022dagma` (arXiv:2209.08037, re-confirmed), `gamella2025chambers` (arXiv:2404.11341, re-confirmed), and `runge2020pcmciplus`, `hyvarinen2010varlingam`, `wienobst2026flop`, which were verified in `../icml_paper/REFS_LOG.md` and not re-checked here.

## Looked for and not verified (not in `refs.bib`, not to be cited)

- Tong and Zhang, "On time-reversibility of multivariate linear processes", Statistica Sinica 2005: recalled from memory; Crossref returned no such record (it returned Chan, Ho, Tong 2006, which is cited instead). Unverified.
- Hashimoto, Gifford, Jaakkola, "Learning population-level diffusions with generative RNNs", ICML 2016: recalled from memory, not searched. Unverified.
- Velorama (Granger causality on pseudotime DAGs), SINCERITIES, SINGE: appeared in search snippets or memory only. Unverified.
- Young et al. 2018 (SuStaIn), Andrews et al. 2021 ("A practical guide to causal discovery with cohort data"): the Crossref query for the latter returned other papers. Unverified.
- Zwiernik and Draisma, "Automorphism groups of Gaussian chain graph models" (arXiv:1501.03013): exists (arXiv API), but a search summary attributed a different title and venue to it; not used.

## Sites that failed

- dblp.org search API: returned a bot-check page ("Making sure you're not a bot") for every query. Not used.
- No other failure. Semantic Scholar and Google Scholar were not queried.

## Searches run (2026-11-10)

Web searches (one line each, abbreviated): unlabelled time-unrolled DAG / recover time labels; VAR from permuted or shuffled variables with Sinkhorn; coloured Gaussian DAG structure learning; arrow of time in multivariate series; causal discovery from cross-sectional snapshots without time order; non-sequential snapshots of a linear system; tiered background knowledge with unknown tiers; Bayesian networks with repeated substructure and unknown parameter sharing; Gaussian graphical model with unknown permutation and block-Toeplitz covariance; follow-ups of Reisach et al. 2026; weight-tied / shift-invariant / periodic DAG learning; gene-network inference on pseudotime; DAG learning by topological layers; Gaussian DAG models with symmetries; shuffled or unlabelled variables in linear dynamical systems. None returned a paper on the unlabelled-column problem of `NOVELTY.md`.
