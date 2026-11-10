# Reference verification log (alg2/)

Date: 2026-11-10. Every entry of `refs.bib` has one row here with the check that was made. Network access went through the configured proxy with TLS verification on.

How to read the last column. "confirmed" means that the venue data came from an API record or a publisher page today. "from memory" means that the arXiv record confirms that the paper exists with this title and these authors, but the venue was **not** checked today; `refs.bib` cites those entries as arXiv preprints. They must be checked before they go into a paper.

**Not verified through an API (3 entries):** `aigner1968estimating`, `donoho2003when`, `gong2017causal`. Details in their rows. Do not cite them without a further check.

Limits that apply to all rows:
- "title, authors, date match" means a match with what I expected before the query. For most arXiv entries the identifier was first written from memory and the API then returned the expected paper; for the others the identifier came from an API title search.
- Abstracts were not read unless the row says so. Statements in `CANDIDATES.md` about what a paper does rest on my prior knowledge of it, on its title, or on a web-search summary.
- Crossref rows: the first hit with the same title and authors was taken. Author first names in `refs.bib` are from memory (Crossref output was reduced to family names).

## 1. Entries

| bib key | identifier | check | venue status |
|---|---|---|---|
| rigobon2003identification | doi:10.1162/003465303772815727 | Crossref API (api.crossref.org/works?query.bibliographic=…): DOI, title, authors, journal, year, volume, pages returned (JSTOR records give the first page only) | Review of Economics and Statistics 2003 confirmed |
| lanne2008identifying | doi:10.1111/j.1538-4616.2008.00151.x | Crossref API (api.crossref.org/works?query.bibliographic=…): DOI, title, authors, journal, year, volume, pages returned (JSTOR records give the first page only) | Journal of Money, Credit and Banking 2008 confirmed |
| lewis2021identifying | doi:10.1093/restud/rdab009 | Crossref API (api.crossref.org/works?query.bibliographic=…): DOI, title, authors, journal, year, volume, pages returned (JSTOR records give the first page only) | The Review of Economic Studies 2021 confirmed |
| sentana2001identification | doi:10.1016/S0304-4076(01)00051-3 | Crossref API (api.crossref.org/works?query.bibliographic=…): DOI, title, authors, journal, year, volume, pages returned (JSTOR records give the first page only) | Journal of Econometrics 2001 confirmed |
| huang2020causal | arXiv:1903.01672 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | JMLR 21 (2020) 1-53 confirmed by the journal_ref field |
| hyvarinen2016unsupervised | arXiv:1605.06336 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | NeurIPS 2016 from memory, not checked |
| wang2023generator | arXiv:2310.19491 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match; abstract read (identification from the law of the solution with a given fixed initial state; necessary and sufficient for additive noise) | NeurIPS 2023 from memory, not checked |
| guan2026identifying | arXiv:2410.22729 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | JMLR 27(219):1-56, 2026 confirmed by the journal_ref field |
| lorch2024causal | arXiv:2310.17405 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | AISTATS 2024 from memory, not checked |
| bellot2022neural | arXiv:2105.02522 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | ICLR 2022 from memory, not checked |
| saggioro2020reconstructing | arXiv:2007.00267 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | Chaos, confirmed by the DOI in the API record; volume and pages not checked |
| varando2020graphical | arXiv:2005.10483 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | UAI 2020 from memory, not checked |
| dettling2023identifiability | arXiv:2209.03835 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | journal version (SIAM J. Matrix Anal. Appl. 2023) from memory, not checked |
| friedman1998learning | arXiv:1301.7374 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | UAI 1998 from memory; arXiv record checked in ../alg/REFS_VERIFY.md, not re-queried today |
| anderson1982formulation | doi:10.1016/0304-4076(82)90095-1 | taken from ../alg/refs.bib, where it was checked through Crossref on 2026-11-10; not re-queried | confirmed there |
| hsiao2002maximum | doi:10.1016/S0304-4076(01)00143-9 | taken from ../alg/refs.bib (Crossref, 2026-11-10); not re-queried | confirmed there |
| binder2005estimation | doi:10.1017/S0266466605050413 | taken from ../alg/refs.bib (Crossref, 2026-11-10); not re-queried | confirmed there |
| komorowski2009bayesian | doi:10.1186/1471-2105-10-343 | Crossref API (api.crossref.org/works?query.bibliographic=…): DOI, title, authors, journal, year, volume, pages returned (JSTOR records give the first page only) | BMC Bioinformatics 2009 confirmed |
| immer2023identifiability | arXiv:2210.09054 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | ICML 2023 from memory, not checked |
| forni2000generalized | doi:10.1162/003465300559037 | Crossref API (api.crossref.org/works?query.bibliographic=…): DOI, title, authors, journal, year, volume, pages returned (JSTOR records give the first page only) | Review of Economics and Statistics 2000 confirmed |
| barigozzi2024fnets | arXiv:2201.06110 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | Journal of Business & Economic Statistics, confirmed by the DOI in the API record; year 2024 from memory |
| basu2019low | arXiv:1812.03568 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | IEEE Trans. Signal Processing, confirmed by the DOI in the API record; year 2019 from memory |
| chandrasekaran2012latent | arXiv:1008.1290 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | Annals of Statistics 40(4):1935-1967, 2012 confirmed by journal_ref and DOI |
| frot2019robust | arXiv:1708.01151 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | JRSS-B 2019 from memory, not checked |
| agrawal2023decamfounder | arXiv:2102.07921 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | JRSS-B 2023 from memory, not checked |
| gerhardus2020high | arXiv:2007.01884 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | NeurIPS 2020 from memory, not checked |
| tran2024estimating | arXiv:2102.06197 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match; abstract read | JRSS-B 2024, DOI 10.1093/jrsssb/qkad165: from a web search result, not from an API record |
| gnecco2021causal | arXiv:1908.05097 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | Annals of Statistics 2021 from memory, not checked |
| engelke2020graphical | arXiv:1812.01734 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | JRSS-B 2020 from memory, not checked |
| gissibl2018max | arXiv:1512.07522 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | Bernoulli 2018 from memory, not checked |
| bodik2023causality | arXiv:2112.10858 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | Extremes (2023) confirmed by journal_ref and DOI |
| li2026causal | arXiv:2604.21620 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | preprint |
| routh2026rate | arXiv:2610.15051 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | preprint |
| rayas2022learning | arXiv:2206.07083 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match; abstract summary read | NeurIPS 2022 confirmed (proceedings.neurips.cc page found by web search) |
| mada2025structure | arXiv:2504.03189 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | preprint |
| jayadev2019learning | arXiv:1905.08716 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | preprint; journal version not checked |
| rajeswaran2015network | arXiv:1506.00438 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | preprint; journal version not checked |
| deka2015structure | arXiv:1501.04131 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | journal version (IEEE Trans. Control of Network Systems) from memory, not checked |
| feigin1994limit | doi:10.1016/0304-4149(94)90022-1 | Crossref API (api.crossref.org/works?query.bibliographic=…): DOI, title, authors, journal, year, volume, pages returned (JSTOR records give the first page only) | Stochastic Processes and their Applications 1994 confirmed |
| davis1989estimation | doi:10.1016/0304-4149(89)90090-2 | Crossref API (api.crossref.org/works?query.bibliographic=…): DOI, title, authors, journal, year, volume, pages returned (JSTOR records give the first page only) | Stochastic Processes and their Applications 1989 confirmed |
| aigner1968estimating | – | NOT VERIFIED through an API: Crossref returned no record. A web search (econbiz.de record) gives the same authors, title, journal, volume and pages | unverified by API; bibliographic data from a library catalogue page |
| aigner1977formulation | doi:10.1016/0304-4076(77)90052-5 | Crossref API (api.crossref.org/works?query.bibliographic=…): DOI, title, authors, journal, year, volume, pages returned (JSTOR records give the first page only) | Journal of Econometrics 1977 confirmed |
| koenker1978regression | doi:10.2307/1913643 | Crossref API (api.crossref.org/works?query.bibliographic=…): DOI, title, authors, journal, year, volume, pages returned (JSTOR records give the first page only) | Econometrica 1978 confirmed |
| donoho2003when | – | NOT VERIFIED: Crossref has no record; the NeurIPS proceedings URL tried returned no title. Title, authors and venue are from memory | unverified |
| arora2012computing | arXiv:1111.0952 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | STOC 2012 from memory, not checked |
| huang2014nonnegative | doi:10.1109/TSP.2013.2285514 | Crossref API (api.crossref.org/works?query.bibliographic=…): DOI, title, authors, journal, year, volume, pages returned (JSTOR records give the first page only) | IEEE Transactions on Signal Processing 2014 confirmed |
| fu2019nonnegative | arXiv:1803.01257 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | IEEE Signal Processing Magazine, confirmed by the DOI in the API record; year 2019 from memory |
| plumbley2003algorithms | doi:10.1109/TNN.2003.810616 | Crossref API (api.crossref.org/works?query.bibliographic=…): DOI, title, authors, journal, year, volume, pages returned (JSTOR records give the first page only) | IEEE Transactions on Neural Networks 2003 confirmed |
| wang2023causal | arXiv:2007.11131 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | JMLR 24, 2023: jmlr.org/papers/v24/21-1329.html found by web search, page not opened |
| mesters2026causal | arXiv:2605.04381 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match; abstract read (mean-independence restrictions replace independence; LiMIAM) | preprint |
| misiakos2023learning | arXiv:2305.15936 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match; abstract read (few root causes, L0 objective) | NeurIPS 2023 confirmed by the journal_ref field |
| tagasovska2020distinguishing | arXiv:1801.10579 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | ICML 2020 from memory, not checked |
| stein2025causalrivers | arXiv:2503.17452 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | ICLR 2025 confirmed (proceedings.iclr.cc PDF found by web search) |
| das2026sc3d | arXiv:2602.02830 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | preprint |
| ruiz2026autocause | arXiv:2608.00198 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match; found by the arXiv query all:CausalRivers; title only, not cited in the text | preprint |
| gamella2025causal | doi:10.1038/s42256-024-00964-x | Crossref API (api.crossref.org/works?query.bibliographic=…): DOI, title, authors, journal, year, volume, pages returned (JSTOR records give the first page only) | Nature Machine Intelligence 2025 confirmed |
| efron2004large | doi:10.1198/016214504000000089 | Crossref API (api.crossref.org/works?query.bibliographic=…): DOI, title, authors, journal, year, volume, pages returned (JSTOR records give the first page only) | Journal of the American Statistical Association 2004 confirmed |
| carvalho2008centroid | doi:10.1073/pnas.0712329105 | Crossref API (api.crossref.org/works?query.bibliographic=…): DOI, title, authors, journal, year, volume, pages returned (JSTOR records give the first page only) | Proceedings of the National Academy of Sciences 2008 confirmed |
| li2009controlling | – | publisher page jmlr.org/papers/v10/li09a.html: title and first author in the citation meta tags; second author and pages from memory | JMLR vol. 10 confirmed by the URL |
| strobl2019estimating | arXiv:1607.03975 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | journal version (ACM TIST 2019) from memory, not checked |
| runge2020discovering | arXiv:2003.03685 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | UAI 2020 from memory, not checked |
| gong2015discovering | – | PDF downloaded from proceedings.mlr.press/v37/gongb15.pdf; abstract, Section 1 and Section 3.2 read (Gaussian statement, Proposition 1, Theorem 2) | ICML 2015 confirmed by the PDF header |
| gong2017causal | – | NOT VERIFIED through an API (Crossref and the arXiv title search returned nothing). A web search result lists the title and summarises the abstract; authors and venue are from memory | unverified |
| tank2019identifiability | arXiv:1704.02519 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | Biometrika 2019 from memory, not checked |
| hyttinen2016causal | arXiv:1602.07970 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | PGM 2016 from memory, not checked |
| fan2024recoverability | arXiv:2406.02191 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | ICML 2024 from memory, not checked |
| palm1984missing | doi:10.2307/1913513 | Crossref API (api.crossref.org/works?query.bibliographic=…): DOI, title, authors, journal, year, volume, pages returned (JSTOR records give the first page only) | Econometrica 1984 confirmed |
| phillips1973problem | doi:10.1016/0304-4076(73)90021-3 | Crossref API (api.crossref.org/works?query.bibliographic=…): DOI, title, authors, journal, year, volume, pages returned (JSTOR records give the first page only) | Journal of Econometrics 1973 confirmed |
| hansen1983dimensionality | doi:10.2307/1911996 | Crossref API (api.crossref.org/works?query.bibliographic=…): DOI, title, authors, journal, year, volume, pages returned (JSTOR records give the first page only) | Econometrica 1983 confirmed |
| mccrorie2003problem | doi:10.1023/A:1025858121378 | Crossref API (api.crossref.org/works?query.bibliographic=…): DOI, title, authors, journal, year, volume, pages returned (JSTOR records give the first page only) | Acta Applicandae Mathematicae 2003 confirmed |
| marcellino1999some | doi:10.1080/07350015.1999.10524802 | Crossref API (api.crossref.org/works?query.bibliographic=…): DOI, title, authors, journal, year, volume, pages returned (JSTOR records give the first page only) | Journal of Business & Economic Statistics 1999 confirmed |
| wang2023neural | arXiv:2311.03309 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | ICLR 2024 from memory, not checked |
| jorda2005estimation | doi:10.1257/0002828053828518 | Crossref API (api.crossref.org/works?query.bibliographic=…): DOI, title, authors, journal, year, volume, pages returned (JSTOR records give the first page only) | American Economic Review 2005 confirmed |
| stock2018identification | doi:10.1111/ecoj.12593 | Crossref API (api.crossref.org/works?query.bibliographic=…): DOI, title, authors, journal, year, volume, pages returned (JSTOR records give the first page only) | The Economic Journal 2018 confirmed |
| mertens2013dynamic | doi:10.1257/aer.103.4.1212 | Crossref API (api.crossref.org/works?query.bibliographic=…): DOI, title, authors, journal, year, volume, pages returned (JSTOR records give the first page only) | American Economic Review 2013 confirmed |
| rothenhausler2015backshift | arXiv:1506.02494 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | NeurIPS 28 (2015) 1513-1521 confirmed by the journal_ref field |
| peters2016causal | arXiv:1501.01332 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | JRSS-B 78(5):947-1012, 2016 confirmed by the journal_ref field |
| pfister2019invariant | arXiv:1706.08058 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | JASA 2019 from memory, not checked |
| mooij2020joint | arXiv:1611.10351 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | JMLR 21(99):1-108, 2020 confirmed by the journal_ref field |
| wagenmaker2020active | arXiv:2002.00495 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | COLT 2020 from memory, not checked |
| lorch2022amortized | arXiv:2205.12934 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | NeurIPS 2022 from memory, not checked |
| ke2022learning | arXiv:2204.04875 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match; the API output was cut after eight authors; the last two names are from memory | ICLR 2023 from memory, not checked |
| stein2024embracing | arXiv:2402.09305 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | AAAI 2024 workshop (as recorded in ../../ts/01_literature.md), not re-checked |
| montagna2025demystifying | arXiv:2405.16924 | arXiv API (export.arxiv.org/api/query?id_list=…): title, authors, date match | TMLR 2025 confirmed by the journal_ref field |
| shimizu2006linear | – | publisher page jmlr.org/papers/v7/shimizu06a.html: title and first author in the citation meta tags; co-authors from memory | JMLR vol. 7 confirmed by the URL; not cited in the text |

## 2. Works looked for and not entered

- SCOTCH was first looked up under arXiv:2311.18477 (written from memory). That identifier is a different paper (functional GARCH for FX volatility). The correct record, arXiv:2311.03309, was found by a title search and is the one in `refs.bib`.
- The "naive reverse-physical baseline" of the CausalRivers paper is taken from `../../ts/01_literature.md`; the benchmark code was not opened today.

## 3. Novelty searches made on 2026-11-10

arXiv API (`search_query`), with the number of records returned:

| query | records | relevant |
|---|---|---|
| all:"causal discovery" AND all:nonnegative AND all:"conic hull" | 0 | – |
| all:"structure learning" AND all:"conservation" AND all:flow AND all:directed | 1 | Mada and Anguluri (2025) |
| all:"causal" AND all:"river network" AND all:topology AND all:discharge | 1 | none (hydrological modelling) |
| all:LiNGAM AND all:nonnegative | 0 | – |
| all:"max-linear" AND all:"causal" AND all:river | 3 | Tran et al.; Li and Castro-Camilo; Routh |
| ti:"few root causes" | 1 | Misiakos et al. |
| all:"causal discovery" AND all:"one-sided" AND all:noise | 1 | Mesters et al. |
| all:"mass conservation" AND all:"causal discovery" | 0 | – |
| all:"nonnegative" AND all:"structural equation" AND all:identifiab AND all:"support" | 0 | – |
| all:"causal discovery" AND all:"river" AND all:"flow" AND all:"physics" | 0 | – |
| all:CausalRivers | 2 | Stein et al.; AutoCause |
| all:"stochastic frontier" AND all:"causal discovery" | 0 | – |
| all:"extreme quantile" AND all:"causal discovery" AND all:confound | 0 | – |
| all:"subsampled" AND all:"vector autoregressive" AND all:identifiability AND all:Gaussian AND all:"second-order" | 0 | – |
| ti:"heteroskedasticity" AND all:"causal discovery" AND all:"structural vector autoregress" | 0 | – |
| all:"transient" AND all:"causal discovery" AND all:"initial condition" AND all:identifiability | 0 | – |

The arXiv API searches titles and abstracts only. An empty result is weak evidence.

Web searches (11), by topic: nonnegative sources and conic-hull identifiability in linear SEMs; river-network topology from discharge by mass balance and linear programming; structure learning under conservation laws; QTree; CausalRivers follow-up methods; nonnegative coefficients and noise with extreme rays or separability; Gong et al. (2015) and the Gaussian case; transient response and covariance trajectory for instantaneous effects; conserved flow networks with a lower-boundary or minimum-ratio estimator; Aigner and Chu (1968); identification from a support boundary under latent confounding.

What they found that was not known to me before: Mada and Anguluri (2025), Jayadev et al. (2019), Rajeswaran and Narasimhan (2015), Routh (2026), Li and Castro-Camilo (2026), Mesters et al. (2026). None of them uses the lower support boundary of a nonnegative additive model for directed structure learning with dependent inputs.

Not searched: Semantic Scholar or Google Scholar citation lists; Water Resources Research, Journal of Hydrology and similar journals; the transportation literature on origin-destination estimation.
