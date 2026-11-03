# Literature verification and prior-art check

Date: 2026-11-03. Method: web search and page fetches only (WebSearch/WebFetch). Status labels: VERIFIED = existence, title, authors, venue/year confirmed on a primary or proceedings/arXiv page seen in this pass; VERIFIED (partial) = existence and venue confirmed, but a field (full authors, pages) not enumerated; CORRECTED = prior note was wrong or incomplete; UNVERIFIABLE = could not be found this pass; NOT RE-CHECKED = carried over from 01_literature_review.md, not independently re-confirmed here (do not cite without a check).
Limit: search snippets are model-summarised; where a result was a mirror/aggregator it is noted. Nothing here was read as full text except where stated.

## 1. Verification table

### 1a. Items flagged as unverified in 01_literature_review.md section 6

| Reference | Status | URL | Notes |
|---|---|---|---|
| Zhang et al., Truncated matrix power iteration, NeurIPS 2022 | CORRECTED (authors completed) | https://proceedings.neurips.cc/paper_files/paper/2022/hash/74fc5575632191d96881d8015f79dde3-Abstract.html ; arXiv 2208.14571 | Authors: Zhen Zhang, Ignavier Ng, Dong Gong, Yuhang Liu, Ehsan Abbasnejad, Mingming Gong, Kun Zhang, Javen Qinfeng Shi (8). |
| Siems et al., NeurIPS 2023 | CORRECTED (title/authors) | https://arxiv.org/abs/2305.11475 | "Curve Your Enthusiasm: Concurvity Regularization in Differentiable Generalized Additive Models". Siems, Ditschuneit, Ripken, Lindborg, Schambach, Otterbach, Genzel (7). NeurIPS 36. |
| Seng et al., "Learning large DAGs is harder than you think" | CORRECTED (full title) | https://iclr.cc/virtual/2024/poster/18125 | Full title: "Learning Large DAGs is Harder than you Think: Many Losses are Minimal for the Wrong DAG". Seng, Zecevic, Dhami, Kersting. ICLR 2024. |
| Seng et al., preprint 2022 (scale sensitivity) | VERIFIED (identification inferred) | https://www.alphaxiv.org/abs/2206.07195 | Most likely "Tearing Apart NOTEARS: Controlling the Graph Prediction via Variance Manipulation", arXiv 2206.07195 (same group). Author list on that page not read; check before citing. |
| Kaiser & Sipos | CORRECTED (title, venue partly) | https://arxiv.org/abs/2104.05441 | arXiv title "Unsuitability of NOTEARS for Causal Graph Discovery"; journal title "...when Dealing with Dimensional Quantities", Neural Processing Letters, DOI 10.1007/s11063-021-10694-5. Volume, pages and print year NOT confirmed (DOI suggests 2021, not 2022). |
| Pamfil et al., DYNOTEARS, AISTATS 2020 | VERIFIED | https://proceedings.mlr.press/v108/pamfil20a.html | Pamfil, Sriwattanaworachai, Desai, Pilgerstorfer, Georgatzis, Beaumont, Aragam. PMLR 108:1595-1605. Author order of Beaumont/Georgatzis differs between arXiv (Beaumont first) and PMLR (Georgatzis first); use the PMLR order. |
| Ravikumar et al., SpAM, JRSSB | VERIFIED | https://ideas.repec.org/a/bla/jorssb/v71y2009i5p1009-1030.html | Ravikumar, Lafferty, Liu, Wasserman. JRSS-B 71(5):1009-1030, 2009. |
| Montagna et al., NeurIPS 2023 | VERIFIED | https://arxiv.org/abs/2310.13387 | "Assumption violations in causal discovery and the robustness of score matching". Montagna, Mastakouri, Eulig, Noceti, Rosasco, Janzing, Aragam, Locatello (8). A web-search check that listed "Zhang" as a co-author is wrong. |
| Geffner et al., DECI | VERIFIED (partial) | https://arxiv.org/abs/2202.02195 | 12 authors on arXiv: Geffner, Antoran, Foster, Gong, Ma, Kiciman, Sharma, Lamb, Kukla, Pawlowski, Allamanis, Zhang. TMLR 2024 (June) per search; TMLR page not opened. |
| SDCD, Nazaret et al. | VERIFIED (partial); venue UNCONFIRMED | https://arxiv.org/abs/2311.10263 | Paper exists. Review says ICML 2024; one source said ICML 2023, which is inconsistent with a Nov 2023 arXiv date. Confirm venue on proceedings.mlr.press. |
| Wu & Xie, arXiv:2604.25295 | UNVERIFIABLE | none | Not found. Do not cite. |

### 1b. References in the Related Work paragraph of draft_v0.md

| Reference | Status | URL | Notes |
|---|---|---|---|
| Hoyer et al., NIPS 2008 | VERIFIED | https://papers.nips.cc/paper/2008/hash/f7664060cc52bc6f3d620bcedc94a4b6-Abstract.html | Hoyer, Janzing, Mooij, Peters, Schoelkopf. NIPS 21, pp. 689-696 (proceedings 2009). |
| Peters et al., JMLR 2014 | VERIFIED | https://jmlr.org/papers/v15/peters14a.html | Peters, Mooij, Janzing, Schoelkopf. JMLR 15:2009-2053. |
| NOTEARS, Zheng et al. NeurIPS 2018 | VERIFIED | https://arxiv.org/abs/1803.01422 | Zheng, Aragam, Ravikumar, Xing. |
| NOTEARS-MLP, Zheng et al. AISTATS 2020 | VERIFIED | https://proceedings.mlr.press/v108/zheng20a.html | "Learning Sparse Nonparametric DAGs". Zheng, Dan, Aragam, Ravikumar, Xing. PMLR 108:3414-3425. |
| DAGMA, Bello et al. NeurIPS 2022 | VERIFIED | https://arxiv.org/abs/2209.08037 | Bello, Aragam, Ravikumar. |
| GOLEM, Ng et al. NeurIPS 2020 | VERIFIED | https://arxiv.org/abs/2006.10201 | Ng, Ghassami, Zhang. |
| GraN-DAG, Lachapelle et al. ICLR 2020 | VERIFIED | https://arxiv.org/abs/1906.02226 | Lachapelle, Brouillard, Deleu, Lacoste-Julien (order differs on ICLR page). |
| CAM, Buhlmann et al. | VERIFIED | https://stat.ethz.ch/Manuscripts/buhlmann/CAM-AOS.pdf | Buhlmann, Peters, Ernest. Ann. Stat. 42(6):2526-2556, 2014. |
| SCORE, Rolland et al. ICML 2022 | VERIFIED (partial) | https://proceedings.mlr.press/v162/rolland22a.html | Author list not enumerated in this pass. |
| NoGAM, Montagna et al. CLeaR 2023 | VERIFIED (partial) | https://proceedings.mlr.press/v213/montagna23a.html ; arXiv 2304.03265 | Title "Causal Discovery with Score Matching on Additive Models with Arbitrary Noise". Confirms the NoGAM/DAS id fix in 01 section 1. |
| DAS, Montagna et al. CLeaR 2023 | VERIFIED (partial) | https://proceedings.mlr.press/v213/montagna23b.html | "Scalable Causal Discovery with Score Matching", PMLR 213:752-771. |
| NAM, Agarwal et al. NeurIPS 2021 | VERIFIED | https://arxiv.org/abs/2004.13912 | Agarwal, Melnick, Frosst, Zhang, Lengerich, Caruana, Hinton. |
| KAN, Liu et al. ICLR 2025 | VERIFIED | https://arxiv.org/abs/2404.19756 | Liu, Wang, Vaidya, Ruehle, Halverson, Soljacic, Hou, Tegmark. |
| KAN critiques: Yu et al. 2024 | VERIFIED | https://arxiv.org/abs/2407.16674 | "KAN or MLP: A Fairer Comparison" (arXiv only). |
| KAN critiques: Hou et al. 2024 | VERIFIED (partial) | https://arxiv.org/abs/2407.11075 | "Kolmogorov-Arnold Networks: A Critical Assessment of Claims, Performance, and Practical Viability". Findings not read. |
| Adaptive lasso, Zou JASA 2006 | VERIFIED (secondary source) | reference lists in search results | JASA 101:1418-1429. Primary page not opened. |
| SCAD, Fan & Li JASA 2001 | VERIFIED (secondary source) | reference lists in search results | JASA 96:1348-1360. |
| MCP, Zhang 2010 | UNVERIFIABLE this pass | none | Journal/pages not confirmed. Check before citing. |
| Xu et al. 2022 (NOTEARS-AL) | VERIFIED | https://arxiv.org/abs/2209.02946 | "On the Sparse DAG Structure Learning Based on Adaptive Lasso". Danru Xu, Erdun Gao, Wei Huang, Andy Song, Mingming Gong. Preprint (revised to Feb 2023); no venue found. |
| ICML 2026 NOTIME-CAM paper | VERIFIED (abstract only) | https://icml.cc/virtual/2026/poster/64429 | See section 2. |
| DiBS, Lorch et al. NeurIPS 2021 | VERIFIED | https://papers.nips.cc/paper/2021/hash/ca6ab34959489659f8c3776aaf1f8efd-Abstract.html | Lorch, Rothfuss, Schoelkopf, Krause. pp. 24111-24123. |
| BayesDAG, Annadani et al. NeurIPS 2023 | VERIFIED | https://arxiv.org/abs/2307.13917 | Annadani, Pawlowski, Jennings, Bauer, Zhang, Gong. |
| GP-based order method (arXiv 2402.14781) | CORRECTED (venue) | https://proceedings.mlr.press/v258/toth25a.html | Toth, Knoll, Pernkopf, Peharz, "Effective Bayesian Causal Inference via Structural Marginalisation and Autoregressive Orders", AISTATS 2025, PMLR 258:4240-4248. |
| Order MCMC, Friedman & Koller | VERIFIED (secondary source) | https://www.datalearner.com/academic/journal-papers/0885-6125/volumes-and-issues/50/paper-detail/10992 | Machine Learning 50, 2003. |
| Viinikka et al. NeurIPS 2020 | VERIFIED | https://arxiv.org/abs/2010.00684 | Viinikka, Hyttinen, Pensar, Koivisto. Notes that order samplers are biased towards graphs compatible with many orders (relevant to our stated limitation). |
| Wang, Kolar, Drton, confidence sets for orderings | VERIFIED | https://arxiv.org/abs/2305.14506 | JASA, DOI 10.1080/01621459.2025.2542552 (vol 121, issue 553 per MCML listing). |
| Varsortability, Reisach et al. NeurIPS 2021 | VERIFIED | https://arxiv.org/abs/2102.13647 | Reisach, Seiler, Weichwald. |
| R2-sortability, Reisach et al. NeurIPS 2023 | VERIFIED | https://arxiv.org/abs/2303.18211 | "A Scale-Invariant Sorting Criterion to Find a Causal Order in Additive Noise Models". Reisach, Tami, Seiler, Chambaz, Weichwald. |
| iSCM, Ormaniec et al. ICLR 2025 | VERIFIED | https://arxiv.org/abs/2406.11601 | "Standardizing Structural Causal Models". Ormaniec, Sussex, Lorch, Schoelkopf, Krause. |

### 1c. Other references of 01_literature_review.md checked in this pass

| Reference | Status | URL | Notes |
|---|---|---|---|
| Huang, Horowitz, Wei, Ann. Stat. 2010 | VERIFIED (new, relevant) | https://projecteuclid.org/journals/annals-of-statistics/volume-38/issue-4/Variable-selection-in-nonparametric-additive-models/10.1214/09-AOS781.full | 38(4):2282-2313. Adaptive group lasso for spline additive models. Not in our review; should be cited. |
| Tank et al., Neural Granger Causality, TPAMI 2022 | VERIFIED | https://pmc.ncbi.nlm.nih.gov/articles/PMC9739174 | TPAMI 44(8):4267-4279; arXiv 1711.08160. |
| Dagma-DCE, Waxman et al. | VERIFIED | https://arxiv.org/abs/2401.02930 | Waxman, Butler, Djuric. IEEE OJSP vol 5, pp 393-401, 2024. |
| NAVAR, Bussmann et al. | VERIFIED | https://arxiv.org/abs/2010.09429 | Discovery Science 2021, LNCS 12986, pp 446-460. |
| Kuskova et al. 2026 | CORRECTED (no longer preprint-only) | https://arxiv.org/abs/2605.26408 ; https://proceedings.mlr.press/v306/kuskova26a.html | Kuskova, Zaytsev, Coppedge, "Function-Valued Causal Influence in Nonlinear Time Series", ICML 2026 (also arXiv). |
| Gentzel et al. NeurIPS 2019 | VERIFIED | https://mlanthology.org/neurips/2019/gentzel2019neurips-case | Gentzel, Garant, Jensen. |
| DiffAN, AVICI | VERIFIED (partial) | | DiffAN (ICLR 2023, "Diffusion Models for Causal Discovery via Topological Ordering") confirmed by search summary; AVICI arXiv 2205.12934, NeurIPS 2022 venue not confirmed in snippets. |
| Belloni & Chernozhukov, Bernoulli 2013 | VERIFIED | https://arxiv.org/abs/1001.0188 | Bernoulli 19(2):521-547. Post-lasso OLS refit. |
| Sharma & van Beek, PGM 2022 | VERIFIED (new) | https://proceedings.mlr.press/v186/sharma22a.html | MARS-spline scores for BN structure learning. |

NOT RE-CHECKED this pass (carried from 01, verified there only by the AI workers): Zhang & Hyvarinen 2009; Peters & Buhlmann 2014; DAG-GNN; DCDI; Yu et al. no-curl; Fang et al. low-rank; DCD-FG; Wei et al. (No Fears); Deng et al. (topological swaps); Ng et al. CLeaR 2024; Rolland/NoGAM author lists; SciNO; CaPS; Hiremath et al.; OSLow; CORL; Zhu et al.; SEA; SADA; Shah et al.; DCILP; GENIE3; Strobl & Visweswaran; GA2M/Lou; Nori; NODE-GAM; Eilers & Marx; Gupta et al.; Zhang et al. 2025 concurvity; Sovrano; Almodovar; Salmeron; Lin et al., Liu et al. (KAN Granger); Stability selection; Friedman et al. 1999; Louizos; Zhou et al. ICML 2024; Biza; Machlanski; Faller; Constantinou; Kitson; Brouillard CLeaR 2025; Sachs; Mooij 2016; Smith 2011; Marbach; Schaffter; CausalBench; Gamella; CausalTime; CausalRivers; Peters & Buhlmann SID; VAR-LiNGAM; PCMCI/PCMCI+; NTS-NOTEARS; Rhino; CUTS; Marcinkevicius & Vogt; Huang et al. 2020; Hooker; Meinshausen relaxed lasso (search returned nothing).

## 2. ICML 2026: "Identifiable Nonlinear Differentiable Causal Discovery via Independence and Adaptive Group Sparsity"

Source limits: the OpenReview forum (https://openreview.net/forum?id=Nix6kuhB6P) and PDF sit behind a browser check and could not be fetched; no arXiv version was found in three searches. What follows rests on the ICML poster abstract (https://icml.cc/virtual/2026/poster/64429; authors Ruicong Yao, Tim Verdonck, Mihaela van der Schaar, Jakob Raymaekers) and on an aggregator page (Lacuna, https://lacuna.tiptreesystems.com/work/identifiable-nonlinear-differentiable-causal-discovery-via-independence-and/wrk_ace34afce398e1e88e16c45e6647ca15) that names the method NOTIME-CAM. The aggregator is a secondary, possibly machine-written summary; treat its details as unconfirmed.

What it does (abstract, reliable): extends NOTIME (Berrevoets, Raymaekers, van der Schaar, Verdonck, Yao; AISTATS 2025, PMLR 258:3115-3123, https://proceedings.mlr.press/v258/berrevoets25a.html) to nonlinear additive noise models with arbitrary noise. NOTIME minimises an independence criterion among residuals. Claimed: the global minimiser recovers the true DAG up to extra "constant edges"; an adaptive group lasso penalty on whole columns of the first-layer MLP weights is added to remove them.
Aggregator-only details (unconfirmed): criterion is dHSIC; CAM-based ordering initialisation; synthetic ER graphs d=10-70 with Gaussian, lognormal, Student-t noise and reversed-variance schemes; baselines NOTEARS, DAGMA, CAM; Sachs SHD 14 vs NOTEARS 22 vs CAM 30; limitation that dHSIC loses power as d grows.
Evaluation: structure only (SHD) as far as can be seen. Nothing indicates edge-function amplitude, shape, or interventional-effect evaluation, and nothing indicates a refit or debiasing step.

Relation to our work:
(a) Adaptive group sparsity for debiasing: here the adaptive group lasso is used for selection (pruning constant edges, oracle-style selection), not for amplitude recovery. It is the same family as Xu et al. 2022 and Huang-Horowitz-Wei 2010, and it shares the idea that adaptive weights reduce shrinkage of strong groups. It is the natural competitor to our refit baseline and we should add an adaptive-group-lasso arm (not yet run). Its existence means "adaptive group lasso for DAG learners" is not new; "does not fix amplitude" would need to be shown by us, not assumed.
(b) Independence term: this is their primary loss; it targets identifiability under non-Gaussian, heteroscedastic noise, not amplitude. Our dropped independence-term idea is therefore not novel and, if revived, must cite NOTIME and this paper. It supports our decision to frame the contribution around measurement and repair, not identifiability.
A full read is needed (OpenReview PDF) before the paper characterises it beyond "structure-focused".

## 3. Prior-art searches

Queries ran in standard mode; all returned only snippet-level evidence unless a page was fetched.

| # | Query (short) | Closest hits | Overlap with our claim | Verdict |
|---|---|---|---|---|
| a1 | Bayesian structure learning additive/spline/GAM + order MCMC + closed-form ML | Ni, Stingo, Baladandayuthapani, Biometrics 2015 (spline DAG, MCMC, edge and functional selection; https://ideas.repec.org/a/bla/biomet/v71y2015i3p585-595.html); Friedman & Nachman 2000 GP networks (marginal-likelihood scoring); Functional Bayesian Networks (arXiv 2210.12832, B-spline coefficients, MCMC); Sharma & van Beek 2022 (MARS scores, not Bayesian ML) | Bayesian spline DAG with edge and function posteriors exists (Ni et al.), at small scale and MCMC over structures. | Estimator novelty NOT claimable. Spline basis + g-prior closed form + order MCMC + warm start from a learner may be unpublished, but the closest hits cover the ingredients. |
| a2 | basis expansion + g-prior + closed-form + order MCMC, nonlinear ANM | Toth et al. AISTATS 2025 (nonlinear ANM, GP, autoregressive orders); BayesDAG; Chang et al. order-based (arXiv 2202.05150, equal variance); Viinikka et al. | Same problem class, different machinery. | Combination not found in 3 searches; absence of evidence only. |
| a3 | Bayesian CAM, spline posterior | Morville & Park 2025 CAM with smooth backfitting (not Bayesian); Chaudhuri, Bhattacharya, Ni (arXiv 2508.00993): Bayesian DAG selection, linear SEM, non-Gaussian errors | Bayesian DAG selection theory but linear. | Not a threat; cite as context only if wanted. |
| b | benchmark/metric scoring recovered edge functions | DOS metric (arXiv 2409.19377, graph + interventional, six dimensions); Kuskova et al. ICML 2026 (function-valued edges via ICE, qualitative shapes in synthetic data, no accuracy metric visible in abstract) | Kuskova et al. treat edge functions as the object of interest, in time series. No ground-truth edge-function NRMSE found. | The metric looks novel for cross-sectional DAG learners; "no benchmark scores edge functions" is supported by searches but must be hedged ("to our knowledge"). Kuskova et al. must be cited as the nearest. Full text not read. |
| c | do-effect error to evaluate learned nonlinear SEMs | Toth et al. 2025 reports MAE between predicted and true average causal effects on nonlinear models (snippet); Gentzel et al. 2019 and Garant & Jensen 2016 argue for interventional evaluation; DOS metric; Active Bayesian Causal Inference (NeurIPS 2022) | Interventional-effect error as an evaluation is established. | Do-effect NRMSE as a metric is NOT novel in spirit; novelty at most lies in using it to expose amplitude shrinkage in sparse additive learners. |
| d1 | refit after group lasso in sparse additive models | Huang, Horowitz, Wei 2010 (adaptive group lasso, B-splines); Belloni & Chernozhukov 2013 (post-lasso OLS, smaller bias); block refitting in l12 regularisation (arXiv 1910.11186); debiased group lasso for varying-coefficient models (Honda 2021, snippet only) | The "refit unpenalised on the selected support" remedy is classical (post-lasso). | Plain refit is NOT novel. Its use inside acyclicity-constrained differentiable DAG learners and its evaluation on edge curves is an application. |
| d2 | threshold-then-refit in NOTEARS-style methods | Xu et al. 2022 NOTEARS-AL; Ng et al. CLeaR 2024 "sober look" (thresholding least-squares coefficients, https://arxiv.org/abs/2304.02146) | Debiasing and refit are discussed for linear NOTEARS. | Supports the claim that no dedicated nonparametric edge-curve study exists, but a "sober look" style refit exists for linear weights. |
| e | amplitude shrinkage of group lasso in neural Granger / DAG methods | Tank et al. (group lasso, exact zeros); search found no study of amplitude underestimation or refit for neural Granger models | None directly. | Claim (shrinkage distorts edge curves in KAN/spline DAG learners) survives, as an empirical finding; the general phenomenon (lasso bias) is textbook. |
| f | DiBS/BayesDAG/GP-order: function recovery reporting | BayesDAG: E-SHD, edge F1, NLL-type metrics; Toth et al.: ACE MAE; a 2025 comparison of DiBS, DCDI, BayesDAG, SDCD uses ESHD/F1 | None report per-edge function NRMSE in the snippets. | Survives, with hedge. Toth et al. is the strongest Bayesian competitor and should be a baseline or at least discussed. |
| g | KAN + acyclicity cross-sectional DAG learning | none found | | Consistent with 01 (only SPADE). |
| h | ICML 2026 independence + adaptive group sparsity | see section 2 | | Structure-focused. |
| i | Kuskova et al. function-valued influence | arXiv 2605.26408 | Edge-function shape as a first-class object, time series, NAVAR. | Partially overlaps the "interpretation of edge functions" motivation. |

## 4. Honest assessment of the contributions

1. Observation that sparse additive DAG learners find edges but shrink amplitudes (our run: slope 0.44 at d=100): partially novel. Lasso shrinkage is classical; documenting it for spline/KAN DAG learners with a ground-truth measurement appears new, but only as an empirical result from one generator family and few datasets. Adaptive group lasso (Huang et al. 2010; the ICML 2026 paper) is a known mitigation we have not tested.
2. Edge-function NRMSE metric (shape/amplitude vs truth): likely novel as a benchmark metric in cross-sectional nonlinear causal discovery, but the idea of scoring recovered functions is elementary, and Kuskova et al. (time series, qualitative) are adjacent. Claim only "to our knowledge".
3. Do-effect NRMSE: not novel as an idea (interventional evaluation: Gentzel et al., Garant & Jensen, DOS, Toth et al. ACE error). Novel only in its use here.
4. Plain refit: already done in spirit (post-lasso, Belloni & Chernozhukov; Huang et al.). Present it as a strong baseline, not a contribution. Note that the project's own F2 criterion already treats it that way.
5. BASIL-W (g-prior marginal likelihood, order MCMC, warm start): partially novel at best. Bayesian spline DAG learners exist (Ni et al. 2015; Friedman & Nachman; Functional Bayesian Networks), nonlinear Bayesian order methods exist (Toth et al.), and order MCMC is Friedman & Koller. The specific combination and the warm start from a continuous learner were not found, but three-plus searches are weak evidence. The draft's statement "novelty of the estimator is NOT claimed" is the right posture.
6. Joint claim (scale + shrinkage repair + shape-faithful metrics for additive DAG learners): the framing as a measurement-and-repair study seems unoccupied, but this rests on absence of evidence. Risks: (i) reviewers pointing to adaptive group lasso as the standard fix; (ii) Toth et al. and Ni et al. as Bayesian precedents; (iii) the ICML 2026 paper if its full text evaluates something beyond SHD.

Recommended next checks: read the ICML 2026 PDF and Kuskova et al. in full; read Toth et al. experiment section; run an adaptive-group-lasso arm; search Google Scholar for "Bayesian additive DAG order MCMC spline" once more; confirm SDCD venue and the Seng 2022 preprint author list.

## 5. Sources actually opened or returned in this pass (selection)
ICML poster 64429; Lacuna summary; PMLR NOTIME page (via search); arXiv 2605.26408 (abstract page); arXiv 2508.00993 (abstract page); proceedings/arXiv/NeurIPS pages for the references in section 1 (via search results). OpenReview Nix6kuhB6P: blocked.
