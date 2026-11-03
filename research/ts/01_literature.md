# Scoping review: time-series causal discovery with lagged and instantaneous effects, forecasting, and interpretable per-edge functions

Date of review: 2026-11-03. Scope: scoping review, not systematic. Searches were done with WebSearch/WebFetch; arXiv abstract pages were fetched directly for most references (see Section 5 for the status of each reference). This is a fresh start and does not reuse any earlier project storyline.

Conventions. "V-arXiv" means I fetched the arXiv abstract page and read title, authors, date and any venue note. "V-search" means title, authors and venue were confirmed from search results (publisher or proceedings pages) but I did not open the page myself. "Partial" means some fields are unconfirmed. Numbers quoted in Section 3 come from three sources: (a) Rhino tables, which I read directly in the PDF text; (b) CausalRivers, CausalTime, KANGCI and SC3D tables, which were extracted by the WebFetch summariser from the HTML versions and have not been re-read by me, so spot-check them before putting them in a paper; (c) the sandbox downloads in Section 2, which I ran myself.

---

## 0. Bottom line (honest assessment)

1. Method-wise the area is crowded and, on the easy benchmarks, saturated. Lagged-only neural Granger methods reach AUROC of about 1.0 on Lorenz-96 and VAR data (KANGCI paper's own table: nearly every method is between 0.98 and 1.00 at p=10). Another KAN-flavoured cMLP is unlikely to be publishable on its own.
2. On the real or realistic benchmarks, current methods are weak. DREAM3 AUROCs sit between 0.50 and 0.76 for all published methods; CausalRivers AUROCs are 0.5 to 0.85 and a physics-based naive baseline or plain linear VAR is competitive with or better than most causal discovery methods; CausalTime AUROCs range from below 0.5 to 0.89 depending on dataset and method. A paper that claims a new SOTA by 0.01 AUROC will be a hard sell.
3. The real gap is in evaluation and in the instantaneous setting. I found no benchmark with real data whose ground truth has instantaneous (contemporaneous) edges. Every real or realistic benchmark I checked has lagged or physical-flow ground truth. Methods that model instantaneous effects (PCMCI+, VAR-LiNGAM, DYNOTEARS, Rhino, NTS-NOTEARS, SC3D) are therefore mostly validated on synthetic data, with the one exception of Rhino's argument on netsim (Table 3 below).
4. The "per-edge interpretable function" niche is not empty any more. In 2026 a group published function-valued causal influence (ICE curves on NAVAR) and forecast-necessity edge ablation. Both were lagged-only as far as I could tell from abstracts. Any KAN or spline paper has to position against these, plus GCKAN/KANGCI (lagged only).
5. Ranked recommendation: Gap A (temporal-aggregation CausalRivers benchmark with physics-known orientation) is the most feasible and most novel. Gap B (forecast loss as an unsupervised model-selection signal for causal discovery) is the next. The KAN/spline instantaneous model (Gap C) is feasible but incremental and carries the highest reviewer risk. Details and kill criteria are in Section 4.

---

## 1. Taxonomy of methods

Columns: Lag = models lagged effects; Inst = models instantaneous (same-step) effects; Lin/NL = linear or nonlinear; Interp = per-edge or per-variable effect readable from the fitted model; Scal = reported to handle roughly 100 or more variables; Fcst = the fitted model is a predictive model that can forecast. The Interp, Scal and Fcst columns are my assessment from the abstracts and the tables I read, not claims made by the papers, unless stated. "?" means I could not determine it.

| Method (ref) | Lag | Inst | Lin/NL | Interp | Scal | Fcst | Notes |
|---|---|---|---|---|---|---|---|
| PCMCI (Runge et al. 2019) | yes | no | both (CI test dependent) | graph and test statistics, no functions | medium | no (not a predictive model) | Constraint-based; CI test power drops with small T (Rhino paper says this explains DREAM3 failure) |
| PCMCI+ (Runge 2020) | yes | yes | both | graph and test statistics | medium | no | Adds contemporaneous link discovery and orientation; assumes causal sufficiency |
| LPCMCI (Gerhardus and Runge 2020) | yes | yes | both | graph | low-medium | no | Allows latent confounders; stationary data; reviewers noted no real-data evaluation |
| TiMINo (Peters et al. 2013) | yes | yes | nonlinear, additive noise | graph | low | yes (regression model) | Identifiability for nonlinear lagged and instantaneous effects, independence-test based |
| VAR-LiNGAM (Hyvarinen et al. 2010) | yes | yes | linear, non-Gaussian | coefficient matrices | medium | yes (VAR part) | Identified only with non-Gaussian noise; fails when d > T by default implementation (per Rhino paper) |
| DYNOTEARS (Pamfil et al. 2020) | yes | yes | linear | coefficient matrices | medium | yes (linear) | Continuous acyclicity penalty on instantaneous block |
| NTS-NOTEARS (Sun et al. 2023) | yes | yes | nonlinear (1D CNN per variable) | weights only | medium | yes | Nonparametric DBN with prior knowledge |
| Rhino (Gong et al. 2023) | yes | yes | nonlinear, history-dependent noise | graph with edge probabilities | low-medium (d up to 100 on DREAM3) | yes (generative, autoregressive flow) | Identifiability result for instantaneous parents; ICLR 2023 oral |
| SC3D (Das et al. 2026) | yes | yes | nonlinear (MLP) | lag-specific adjacency + instantaneous DAG | up to d=50 in synthetic | node-wise predictors exist but forecasting not evaluated | Two-stage, spectral-radius acyclicity; evaluated on CausalRivers d=10,20 |
| TECDI (as named in SC3D paper) | yes | yes | nonlinear | ? | ? | ? | I could not find the TECDI paper; characterisation is the SC3D authors' only. Unverified |
| DOTS (Sanchez et al. 2025, TMLR) | yes | yes (via causal ordering) | ? | graph | ? | ? | "Causal Ordering for Structure Learning from Time Series"; only abstract metadata checked |
| TCDF (Nauta et al. 2019) | yes (learns delays) | partial (authors claim instantaneous effects can be learned) | nonlinear (attention CNN) | attention scores, kernel delays | medium | yes (predictive CNN) | Poor on CausalTime AQI (AUROC 0.41) and on CausalRivers d=10,20 per SC3D table |
| NAVAR (Bussmann et al. 2021) | yes | no | nonlinear, additive per input variable | yes: one function per (cause, effect) pair | medium | yes | Closest ancestor of per-edge-function models |
| cMLP / cLSTM (Tank et al. 2022) | yes | no | nonlinear | group-lasso weights, no shapes | medium | yes | Neural Granger baseline everyone compares to |
| SRU / eSRU (Oliva 2017; Khanna and Tan 2020) | yes | no | nonlinear | weights | medium | yes | eSRU is strong on Lorenz-96 and DREAM3 baselines |
| JRNGC (Zhou et al. 2024) | yes | no | nonlinear | Jacobian-based weighted causal matrix | high (one model for all targets) | yes | ICML 2024 |
| ACD (Lowe et al. 2022) | yes | no | nonlinear (amortised, NRI-style) | graph | medium | yes (decoder) | Amortised over samples sharing dynamics, different graphs |
| CUTS / CUTS+ (Cheng et al. 2023, 2024) | yes | no | nonlinear | graph | CUTS+ targets high-dim | imputation plus prediction | Handles irregular sampling; CUTS+ is among best on CausalTime |
| GC-KAN (Lin et al. 2024) | yes | no | nonlinear (KAN) | KAN splines, weights | low (tested on VAR, Lorenz-96) | yes | cMLP with KANs; arXiv only as far as I found |
| KANGCI (Liu et al. 2025) | yes | no | nonlinear (KAN) | base weights, spline functions available but not used for shape reporting | p=100 on DREAM | yes (trained with prediction loss, forecasting not evaluated) | Time-reversal check; best on Lorenz-96 p=40 and 3 of 5 DREAM-3 nets in its own tables |
| GRNGC (Liu et al. 2025) | yes | no | nonlinear (KAN or other) | gradient L1 | ? | yes | Industrial processes; two arXiv-version titles |
| NAVAR plus ICE (Kuskova et al. 2026) | yes (as far as abstract says) | not stated | nonlinear additive | yes: function-valued influence curves | ? | yes (forecast-necessity test in sister paper) | Directly overlaps with the "per-edge function" claim |
| Causal Pretraining (Stein et al. 2024) | yes | not stated | learned | none | depends | no | Supervised mapping series to graph; reported CausalRivers AUROC 0.65 (Transformer, 5-node random) |
| Zero-shot TS foundation models (Wang et al. 2026) | yes | ? | learned | none | depends | yes (it is a forecaster) | Chronos-2 as noisy zero-shot discoverer, per search summary only |

Forecasting-focused KAN work (no causal discovery): T-KAN/MT-KAN (Xu et al. 2024), KAN4TSF/RMoK (Han et al. 2024), iTFKAN (Liang et al. 2025), AR-KAN (Zeng et al. 2025), HaKAN (Hasan et al. 2026). None of the abstracts I read mention causal structure or instantaneous effects.

Surveys: Gong et al. (ACM CSUR) is the broad temporal-data survey. CausalCompass (2026) and TCD-Arena (ICLR 2026) are robustness benchmarks.

Observation: in my search I found no method that simultaneously has (i) per-edge learned univariate functions, (ii) an acyclic instantaneous block, (iii) forecasting evaluation, and (iv) evaluation on a real benchmark with known ground truth. Absence from search results is not proof of absence, so state this carefully if used.

---

## 2. Benchmarks with ground truth

Download tests were run from this sandbox with curl through the pre-configured proxy and the default CA bundle. TLS verification was never disabled. Files were put in /tmp/claude-0/dl (not in the repo).

| Benchmark | Size | Ground truth type | Instantaneous effects in GT? | Licence / availability | Download in sandbox? |
|---|---|---|---|---|---|
| CausalRivers (Stein et al., ICLR 2025) | RiversEastGermany 666 stations, 651 edges; RiversBavaria 494 stations, 490 edges; RiversElbeFlood 42 nodes (Sep-Oct 2024 flood, 15-min); main span 2019-2023 at 15 min, about 175k steps; missingness about 8% (East Germany), about 1% (Bavaria) | Real series, graph from upstream-to-downstream river topology, manually checked | Not explicitly. Authors discuss lag misalignment with sampling rate and test resolutions from 15 min to 24 h; at coarse resolution upstream-downstream travel times shrink below the step, so effects become effectively contemporaneous (my inference, not a stated property). Autoregressive links excluded from scoring | Data under Datenlizenz Deutschland dl-de/by-2-0 (README and paper); some federal states' data not freely available (paper). Code: github.com/CausalRivers/benchmark | YES. `product.zip` from the GitHub release downloaded: 133,228,548 bytes. Contents: rivers_ts_east_germany.csv (618.7 MB uncompressed), rivers_ts_bavaria.csv (466.0 MB), rivers_ts_flood.csv (0.78 MB), graph pickles and metadata CSVs. The github.com HTML page returned 403 from the proxy but release assets and raw.githubusercontent.com work |
| CausalTime (Cheng et al., ICLR 2024) | AQI: 36 variables, length 8760 (hourly PM2.5); Traffic: 20 variables, length 52116; Medical (MIMIC-IV based): 20 variables, 2-hour steps, 1000 patients | Realistic generated series from fitted normalising-flow models; ground truth is the graph of the generating model (not an observed real-world graph) | No, lagged | Website causaltime.cc; code github.com/jarrycyx/unn. No licence stated in the paper text I read | PARTIAL. causaltime.cc returned 200 and the repo README on raw.githubusercontent.com returned 200. I did not locate or download the dataset files. MIMIC-derived data likely needs PhysioNet credentialing (general knowledge, not verified) |
| CauseMe (Runge et al., Nat. Commun. 2019) | Many experiments (synthetic mimicking real challenges plus some real data with high-confidence structure); sizes vary per experiment | Mixed: synthetic ground truth and some real data | Some synthetic experiments may include contemporaneous links; I could not check because the site was unreachable | Platform causeme.net, which redirects to causeme.uv.es | NO. causeme.net returned 302 to https://causeme.uv.es/, where curl fails with "unable to get local issuer certificate" (server chain issue). I did not bypass verification. No `causeme` package exists on PyPI (JSON endpoint 404) |
| netsim fMRI (Smith et al., NeuroImage 2011) | 28 simulations, 50 subjects each. Nodes: 5 (most), 10, 15 (sim3), 50 (sim4). T per subject: 50, 100, 200, 1200, 2400, 5000. Read from the .mat files I downloaded | Simulated BOLD with known directed network (networks differ slightly between subjects) | Not explicit; BOLD smoothing makes lagged-only models awkward, and Rhino's ablation argues modelling instantaneous effects helps (Table 3 below) | FMRIB page states no licence or terms | YES. `sims.tar.gz` downloaded: 49,291,293 bytes, 28 files sim1.mat to sim28.mat |
| DREAM3 in-silico (Prill et al. 2010; Marbach et al. 2009 generator) | Time-series use: 5 networks (2 E. coli, 3 yeast), d=100, 46 perturbation trajectories of 21 time points | Simulated gene regulation (GeneNetWeaver) with known network | No (Rhino reports its fitted instantaneous parameters gave no obvious help) | Hosted on Synapse (registration/terms likely); GNW is open source | NOT DOWNLOADED. Synapse front page, GNW page and the PLOS paper returned 200, but I did not obtain the data files |
| DREAM4 in-silico | KANGCI paper uses 5 networks, p=100, T=210 | Simulated | No | as DREAM3 | NOT TESTED |
| Lorenz-96 (Lorenz, ECMWF seminar 1995/96) | Any d and T, generated | Analytic: variable i depends on i-2, i-1, i+1 | No (continuous-time ODE; discretisation can make sub-step effects look instantaneous) | Public equation, nothing to download | N/A. Needs only a simulator (numpy/torch present in sandbox) |
| TimeGraph (Ferdous et al., KDD 2025 D&B) | Synthetic benchmark datasets, sizes not read | Synthetic | Per search summary: focus on trend, seasonality and noise, not instantaneous | not checked | NOT TESTED |
| DTCD "Composable Causality" (NeurIPS 2026 listing) | Synthetic generator | Synthetic contemporaneous plus lagged adjacency | Yes (per listing snippet only) | not checked | NOT TESTED; reference unverified |
| TCD-Arena (ICLR 2026) | Toolkit; about 30 million CD attempts over 33 assumption violations | Synthetic with controlled violations | not stated in abstract | not checked | NOT TESTED |
| CausalCompass (2026) | Eight misspecification scenarios | Synthetic and (per abstract) released datasets | not stated | not checked | NOT TESTED |

Takeaways. (1) The only real-data benchmark with a ground-truth graph and a sensible size is CausalRivers. (2) None of the real or realistic ones announce instantaneous ground truth. (3) CausalRivers and netsim are the two I can use in this sandbox without registration; CPU-only torch is available, but the large CSVs are 0.6 GB and 0.47 GB, so subgraph sampling is required, as the benchmark intends.

---

## 3. What published results say on the real benchmarks

All numbers are AUROC unless stated. Source names are the papers listed in Section 5.

### CausalRivers (Stein et al. 2025; fetch-summariser extraction, not independently re-read)
- Experiment 1, random 5-node subgraphs: VAR 0.80, CDMI 0.80, VAR-LiNGAM 0.75, naive reverse-physical baseline RP 0.75, PCMCI 0.65, Causal Pretraining (Transformer) 0.65, DYNOTEARS 0.61, null 0.50.
- Experiment 2, 42-node Elbe flood graph, 5-node subgraphs, full series: VAR 0.85, CDMI 0.83, DYNOTEARS 0.79, CP 0.65, PCMCI 0.60.
- Authors' conclusion (summarised): many established methods underperform on real data, simple baselines sometimes match them, and Granger-type methods are most robust. Fine-tuning Causal Pretraining on Bavaria improved it.
- SC3D follow-up (Das et al. 2026), East Germany connected subgraphs, lagged structure only: d=10 AUROC SC3D 0.610, PCMCI+ 0.590, NTS-NOTEARS 0.552, TECDI 0.557, DYNOTEARS 0.509, NeuralGC 0.500, TCDF 0.489. d=20: NTS-NOTEARS 0.636, SC3D 0.623, PCMCI+ 0.611, TECDI 0.567, DYNOTEARS 0.504, NeuralGC 0.500, TCDF 0.506. Differences between the top methods are inside the reported standard deviations (for example SC3D 0.610 +/- 0.110 at d=10). Note these are not directly comparable to the original paper's numbers (different subgraph sampling and different sets of baselines).

### CausalTime (Cheng et al. 2024; fetch extraction)
AUROC, AQI / Traffic / Medical: CUTS+ 0.893 / 0.618 / 0.820; LCCM 0.857 / 0.555 / 0.801; eSRU 0.823 / 0.599 / 0.756; NGC 0.717 / 0.603 / 0.574; PCMCI 0.527 / 0.542 / 0.699; TCDF 0.415 / 0.503 / 0.633; CUTS 0.601 / 0.624 / 0.374. Traffic is near chance for every method (best 0.624). PCMCI, VAR-LiNGAM, DYNOTEARS, NAVAR and Rhino were not in that table except PCMCI.

### DREAM3 (Rhino paper Table 2, read directly in the PDF)
Summary-graph AUROC, E.coli1 / E.coli2 / Yeast1 / Yeast2 / Yeast3: cMLP 0.644 / 0.568 / 0.585 / 0.506 / 0.528; eSRU 0.66 / 0.629 / 0.627 / 0.557 / 0.55; TCDF 0.614 / 0.647 / 0.581 / 0.556 / 0.557; DYNOTEARS 0.590 / 0.547 / 0.527 / 0.526 / 0.510; PCMCI+ 0.530 / 0.519 / 0.530 / 0.510 / 0.512; Rhino 0.685 / 0.680 / 0.664 / 0.585 / 0.567. Baseline numbers were copied from Khanna and Tan by the Rhino authors (single run). KANGCI's own table (fetch extraction) reports 0.758 / 0.680 / 0.667 / 0.552 / 0.562 and CUTS+ 0.703 / 0.675 / 0.661 / 0.612 / 0.554. Best published values are therefore 0.55 to 0.76, with Yeast2 and Yeast3 near chance.

### netsim (Rhino paper Table 3, read directly; sim3, subjects 2 to 6, d=15, T=200)
cMLP 0.93, TCDF 0.91, DYNOTEARS 0.90, eSRU 0.88, VAR-LiNGAM 0.84, PCMCI+ 0.83, cLSTM 0.83, SRU 0.80, Rhino without instantaneous effects 0.93, Rhino 0.99. The authors read the gap between Rhino and Rhino-NoInst (0.93 to 0.99) as evidence that instantaneous modelling matters on this data. This is a single simulation with five subjects, so it is thin evidence. KANGCI reports AUROC over all 28 netsim sims and claims best in 22 of 28 (fetch extraction; the summariser flagged one text/table inconsistency for Sim22, so verify before citing).

### Lorenz-96 and VAR (saturated)
KANGCI table (fetch extraction): p=10, T=1000, AUROC 1.0 for eSRU, GVAR, CUTS+, JRNGC and KANGCI; p=40, T=500: eSRU 0.953, CUTS+ 0.961, JRNGC 0.956, KANGCI 0.972, cMLP 0.843, TCDF 0.565. VAR: nearly everything at or above 0.99.

### Robustness benchmarks
CausalCompass (abstract): no single method wins in all eight misspecified scenarios, but the best across scenarios are nearly always deep-learning-based. TCD-Arena (abstract): robustness profiles vary widely; ensembles may improve robustness. These are synthetic.

### Reading across
- Gap between synthetic and real: methods at 0.95 to 1.0 on Lorenz-96 drop to 0.5 to 0.76 on DREAM3 and to 0.5 to 0.64 on CausalRivers subgraphs.
- Method ordering is not stable across benchmarks (TCDF is mid-pack on DREAM3, worst on CausalRivers and CausalTime AQI; PCMCI is weak on DREAM3 and CausalRivers but best-ish on CausalTime Medical among classical methods).
- Evidence for the value of instantaneous modelling on real data is a single Rhino ablation plus a statement of motivation. No benchmark can currently test it properly.

---

## 4. Research gaps, ranked

Ranking is by feasibility in this environment (CPU-only, public data I can already download) combined with novelty. All "novelty" statements mean "I did not find prior work in this scoping search", not "none exists".

### Gap A (rank 1): a real-data benchmark for instantaneous effects by temporal aggregation of CausalRivers
Idea. Aggregate the 15-min discharge series to 6 h, 12 h, 24 h (and 2-day). Upstream-to-downstream travel times that are shorter than the step become contemporaneous edges whose direction is known from physics (upstream causes downstream), while longer ones stay lagged. Using station metadata (distance and the physical-lag proxies the authors mention: distance, elevation, volume) one can estimate per edge whether it is expected to fall in the lag-0 or lag-1 bucket at each resolution. This gives real data with known orientation of contemporaneous edges, which the benchmarks above lack.
Testable hypothesis. At resolutions where the median edge travel time is below the step, instantaneous-capable methods (PCMCI+, VAR-LiNGAM, DYNOTEARS, SC3D-style, Rhino) recover the orientation of contemporaneous edges above chance and better than lag-only Granger methods, which cannot orient them at all.
What counts as a negative result. (i) Orientation accuracy of every instantaneous-capable method on contemporaneous edges is within two standard errors of 0.5 across at least 30 subgraphs at 24 h; or (ii) lag-only VAR/Granger on the undirected skeleton scores equal or higher AUROC than all instantaneous-capable methods. Either would show that the instantaneous machinery does not help on real data, which is itself a publishable benchmark-paper finding. Another failure mode to check first: rain and other unobserved common drivers create hidden confounding at coarse resolution and may make "ground truth" orientation unrecoverable even in principle. If the physics-based oracle (orient by known upstream direction) is not itself consistent with the data, the benchmark is invalid and this gap dies.
Feasibility: high (data already downloaded and verified; no training required for the first stage). Novelty: moderate to high. The original paper already studies resolutions 15 min to 24 h but, as far as the summary shows, scores only lagged structure with autoregressive links excluded.

### Gap B (rank 2): does forecasting loss select good causal graphs on real data? (model selection without ground truth)
Idea. Most neural causal discovery methods need regularisation strengths and lag orders tuned against ground truth, which does not exist in practice. Models that are also forecasters give held-out predictive loss for free. Test whether held-out forecast loss (and an edge-ablation "forecast necessity" criterion, in the spirit of the 2026 Kuskova et al. preprint) tracks graph accuracy across hyperparameter grids and seeds on CausalRivers, CausalTime and netsim.
Testable hypothesis. Across a hyperparameter grid per method and dataset, validation forecast error and AUROC have a Spearman correlation of magnitude at least 0.5 in the expected direction on real benchmarks, so forecast-based selection recovers most of the oracle-tuned AUROC.
What counts as a negative result. Median Spearman correlation below 0.3 across dataset-method pairs, or selecting by forecast loss yields AUROC no better than a random draw from the grid (permutation test, p > 0.05). Sparse models that forecast equally well with fewer edges (redundancy, persistence) would produce exactly this negative outcome, and the Kuskova abstract already hints at that redundancy problem.
Feasibility: high-medium (only small models on subgraphs; CPU is enough for 5 to 20 node graphs). Novelty: medium. Adjacent work exists (AutoCause, Causal-Audit, both 2026 from the same group; I read only their metadata) that automates expert decisions or audits assumption violations, so a clear differentiation (selection by predictive criteria validated against real ground truth) is needed.

### Gap C (rank 3): additive spline (KAN-style) structural VAR with an acyclic instantaneous block, per-edge functions and forecasting, validated on real benchmarks
Idea. Each edge (cause, lag) gets a univariate spline function, as in NAVAR and KAN, and instantaneous edges get splines under an acyclicity constraint. The model is simultaneously a forecaster and a causal graph with readable per-edge shapes.
Why it is open and why it is risky. GCKAN/KANGCI are lagged-only and report weights not shapes; NAVAR+ICE (2026) gives shapes but, per its abstract, not instantaneous effects; SC3D has the instantaneous block but MLP and no per-edge functions; NTS-NOTEARS uses CNNs. A combined model is therefore a natural but incremental step, and it will be judged against the saturation and weak-real-benchmark facts in Sections 0 and 3.
Testable hypothesis. (i) On synthetic SVAR with known nonlinear per-edge shapes, the model recovers edge shapes with normalised error below a pre-set threshold and orients instantaneous edges better than SC3D-style MLP baselines. (ii) On CausalRivers (at the Gap A resolutions), netsim and CausalTime it is not worse than VAR and PCMCI+ in AUROC. (iii) Forecast RMSE is within 5% of an unconstrained MLP of the same size.
What counts as a negative result. Any of: shape-recovery error not better than fitting the same spline model without the instantaneous block on data that has none; AUROC on CausalRivers within noise of linear VAR (the likely outcome given Section 3); forecast RMSE more than 10% worse than MLP; instantaneous-edge orientation at chance on data with true instantaneous effects (identifiability fails without non-Gaussianity or additive-noise assumptions being met).
Feasibility: high. Novelty: low to medium. I would only pursue it as the model half of a paper whose main contribution is Gap A or B.

### Gap D (rank 4): amortised or pretrained discovery with instantaneous effects, tested on aggregated real data
Idea. Causal Pretraining reached 0.65 AUROC on CausalRivers 5-node subgraphs and improved with fine-tuning; whether pretraining on synthetic SVAR data with instantaneous edges transfers to real aggregated data is untested in what I found. I did not verify whether Causal Pretraining's training data include instantaneous effects.
Testable hypothesis. A small transformer pretrained on synthetic SVAR with random lagged and instantaneous graphs and then evaluated zero-shot on the Gap A aggregated subgraphs beats VAR-LiNGAM and PCMCI+ on contemporaneous-edge orientation.
What counts as a negative result. Zero-shot AUROC and orientation accuracy no better than PCMCI+ at equal compute, and fine-tuning on Bavaria does not close the gap. Montagna et al. (TMLR 2025) already argue that amortised transformers remain bound by identifiability theory, so a negative result is plausible and should be reported honestly.
Feasibility: low to medium on CPU only (training a transformer on millions of synthetic series is expensive). Novelty: medium.

### Gap E (rank 5, probably not worth a paper alone): netsim and DREAM sanity check of the "instantaneous helps" claim
The only real-ish evidence that instantaneous modelling helps is Rhino's single netsim ablation. A replication across all 28 netsim simulations with and without instantaneous terms, same architecture, same tuning, would settle whether the effect is general. Negative result: the AUROC gain from enabling instantaneous terms is under 0.02 in at least 20 of 28 sims. Feasibility: high (data downloaded). Novelty: low. It is useful as a section inside Gap A or Gap C work.

---

## 5. Verification log

Status key as defined at the top. "Unverified" entries are mentioned in the text only as flagged and should not be cited without checking.

| # | Reference as used | Status | What was checked |
|---|---|---|---|
| 1 | Runge, Nowack, Kretschmer, Flaxman, Sejdinovic. Detecting causal associations in large nonlinear time series datasets (PCMCI). Science Advances 5(11), eaau4996, 2019. arXiv 1702.07007 | V-arXiv | Title, first authors, date, journal ref from arXiv page; co-author list truncated after three names in my extraction |
| 2 | Runge. Discovering contemporaneous and lagged causal relations in autocorrelated nonlinear time series datasets (PCMCI+). UAI 2020, PMLR 124:1388-1397. arXiv 2003.03685 | V-arXiv, V-search | arXiv page and PMLR citation |
| 3 | Gerhardus, Runge. High-recall causal discovery for autocorrelated time series with latent confounders (LPCMCI). NeurIPS 2020. arXiv 2007.01884 | V-arXiv, V-search | arXiv page; NeurIPS venue from search |
| 4 | Hyvarinen, Zhang, Shimizu, Hoyer. Estimation of a structural vector autoregression model using non-Gaussianity. JMLR 11:1709-1731, 2010 | V-search | jmlr.org page appeared in results |
| 5 | Peters, Janzing, Scholkopf. Causal inference on time series using restricted structural equation models (TiMINo). NeurIPS 2013, pp. 154-162 (earlier arXiv 1207.5136 under a different title) | V-search | papers.nips.cc abstract page |
| 6 | Pamfil et al. DYNOTEARS: structure learning from time-series data. AISTATS 2020. arXiv 2002.00498 | V-arXiv | Comment field: accepted to AISTATS 2020 |
| 7 | Sun, Schulte, Liu, Poupart. NTS-NOTEARS: learning nonparametric DBNs with prior knowledge. AISTATS 2023 (PMLR 206). arXiv 2109.04286 | V-arXiv, V-search | arXiv comment AISTATS 2023 |
| 8 | Nauta, Bucur, Seifert. Causal discovery with attention-based convolutional neural networks (TCDF). Machine Learning and Knowledge Extraction 1(1):19, 2019 | V-search | MDPI and Crossref pages appeared in results. I did not fetch an arXiv ID; my first guess of 1809.07812 was wrong (it resolved to an unrelated physics paper), so no arXiv ID is cited |
| 9 | Bussmann, Nys, Latre. Neural additive vector autoregression models for causal discovery in time series (NAVAR). Discovery Science 2021, LNCS 12986. arXiv 2010.09429 | V-arXiv | Journal ref on arXiv page |
| 10 | Tank, Covert, Foti, Shojaie, Fox. Neural Granger causality. IEEE TPAMI (accepted version per arXiv comment). arXiv 1802.05842 | Partial | Title and first authors from arXiv; volume, year and pages not checked; only first three authors extracted |
| 11 | Oliva, Poczos, Schneider. The statistical recurrent unit. ICML 2017, pp. 2671-2680. arXiv 1703.00381 | V-search | Search results with CMU record |
| 12 | Khanna, Tan. Economy statistical recurrent units for inferring nonlinear Granger causality. ICLR 2020. arXiv 1911.09879 | V-arXiv, V-search | ICLR poster page in results |
| 13 | Gong, Jennings, Zhang, Pawlowski. Rhino: deep causal temporal relationship learning with history-dependent noise. ICLR 2023 (oral). arXiv 2210.14706 | V-arXiv, V-search; PDF text read | Tables 1 to 3 read directly. One aggregator listed ICLR 2022, which conflicts with the ICLR site listing 2023 |
| 14 | Lowe, Madras, Zemel, Welling. Amortized causal discovery. CLeaR 2022. arXiv 2006.10833 | V-arXiv | Comment: accepted at CLeaR 2022. Only first three authors extracted (Lowe, Madras, Zemel et al.); fourth author Welling is from my memory, unverified |
| 15 | Cheng, Yang, Xiao et al. CUTS: neural causal discovery from irregular time-series data. ICLR 2023. arXiv 2302.07458 | V-arXiv | Journal ref field |
| 16 | Cheng, Li, Xiao, Li, Suo, He, Dai. CUTS+: high-dimensional causal discovery from irregular time-series. AAAI 2024, 38(10):11525-11533. arXiv 2305.05890 | V-arXiv, V-search | AAAI OJS page in results |
| 17 | Zhou, Bai, Yu, Zhao, Chen. Jacobian regularizer-based neural Granger causality (JRNGC). ICML 2024, PMLR 235:61763-61782. arXiv 2405.08779 | V-arXiv, V-search | |
| 18 | Stein, Shadaydeh, Denzler. Embracing the black box: heading towards foundation models for causal discovery from time series data (Causal Pretraining). AAAI 2024 Workshop AI4TS. arXiv 2402.09305 | V-arXiv | Comment field |
| 19 | Montagna, Cairney-Leeming, Sridhar et al. Demystifying amortized causal discovery with transformers. TMLR 2025. arXiv 2405.16924 | V-arXiv | Journal ref field |
| 20 | Mahajan, Gladrow, Hilmkil et al. Amortized inference of causal models via conditional fixed-point iterations (Cond-FiP). TMLR 2025. arXiv 2410.06128 | V-arXiv | Non-time-series; cited only as context |
| 21 | Wang, Wang, Huang. Transformer is inherently a causal learner. arXiv 2601.05647, 2026 | V-arXiv (metadata only) | The Chronos-2 zero-shot claim comes from a search summary, not read |
| 22 | Sanchez, Machlanski, McDonagh et al. Causal ordering for structure learning from time series (called DOTS in a search summary). TMLR 2025. arXiv 2510.24639 | V-arXiv (metadata only) | Method details not read; the name DOTS comes from a search snippet |
| 23 | Xu, Mameche, Vreeken. Information-theoretic causal discovery in topological order (TOPIC). AISTATS 2025 | V-search, partial | Only appeared in search results; not used in the tables |
| 24 | Das, Chakraborty, Maulik. SC3D: dynamic and differentiable causal discovery for temporal and instantaneous graphs. arXiv 2602.02830, 2026 (preprint) | V-arXiv; tables from fetch extraction | Last 14,782 characters of the page not read |
| 25 | TECDI (named only in the SC3D comparison) | UNVERIFIED | I could not locate the paper (a first guess, arXiv 2206.06169, turned out to be Lippe et al., CITRIS, a different paper). Do not cite |
| 26 | Liu, Wang, Vaidya, Ruehle, Halverson, Soljacic, Hou, Tegmark. KAN: Kolmogorov-Arnold networks. ICLR 2025. arXiv 2404.19756 | V-search | arXiv listing and ICLR page in results |
| 27 | Lin, Ren, Barucca et al. Granger causality detection with Kolmogorov-Arnold networks (GC-KAN). arXiv 2412.15373, Dec 2024 | V-arXiv | No venue found |
| 28 | Liu, Xu, Li et al. Kolmogorov-Arnold networks for time series Granger causality inference (KANGCI / GCKAN). arXiv 2501.08958, Jan 2025 | V-arXiv; tables from fetch extraction | No venue found. Name in the paper is KANGCI; search results call it GCKAN |
| 29 | Liu, Dong, Yang et al. A gradient-based causal discovery framework with applications to complex industrial processes (GRNGC). arXiv 2507.11178, 2025 | V-arXiv | A search result lists the ar5iv title as "Gradient Regularization-based Neural Granger Causality", so the title may differ between versions |
| 30 | Xu, Chen, Wang. KANs for time series: bridging predictive power and interpretability (T-KAN, MT-KAN). arXiv 2406.02496, 2024 | V-arXiv | |
| 31 | Han, Zhang, Wu et al. Are KANs effective for multivariate time series forecasting? (KAN4TSF / RMoK). arXiv 2408.11306, 2024 | V-arXiv | |
| 32 | Liang, An, Fan et al. iTFKAN: interpretable time series forecasting with KAN. arXiv 2504.16432, 2025 (under review at IEEE TKDE per comment) | V-arXiv | |
| 33 | Zeng, Xu, Wang. AR-KAN. arXiv 2509.02967, 2025 | V-arXiv (metadata only) | |
| 34 | Hasan, Hamza, Bouguila. Time series forecasting with Hahn Kolmogorov-Arnold networks (HaKAN). arXiv 2601.18837, 2026 | V-arXiv | |
| 35 | Kuskova, Zaytsev, Coppedge. Function-valued causal influence in nonlinear time series. PMLR 306, 2026. arXiv 2605.26408 | V-arXiv; abstract read via fetch | Page range in my extraction was truncated; full text not read, so whether instantaneous effects are handled is unknown |
| 36 | Kuskova, Zaytsev, Coppedge. Beyond coefficients: forecast-necessity testing for interpretable causal discovery in nonlinear time-series models. arXiv 2604.18751, 2026 | V-arXiv; abstract read | Preprint, no venue noted |
| 37 | Zaytsev, Kuskova, Coppedge. From causal discovery to dynamic causal inference in neural time series (DCNAR). arXiv 2603.20980, 2026 | V-arXiv; abstract read | Preprint |
| 38 | Stein, Shadaydeh, Blunk et al. CausalRivers: scaling up benchmarking of causal discovery for real-world time-series. ICLR 2025. arXiv 2503.17452 | V-arXiv, V-search; data downloaded | Numbers from fetch extraction |
| 39 | Cheng, Wang, Xiao et al. CausalTime: realistically generated time-series for benchmarking of causal discovery. ICLR 2024. arXiv 2310.01753 | V-arXiv, V-search | Numbers from fetch extraction; dataset files not downloaded |
| 40 | Stein, Penzel, Piater, Denzler. TCD-Arena: assessing robustness of time series causal discovery methods against assumption violations. ICLR 2026. arXiv 2605.03045 | V-arXiv | Abstract only |
| 41 | Yi, Shen, Wu et al. CausalCompass: evaluating the robustness of time-series causal discovery in misspecified scenarios. arXiv 2602.07915, 2026 | V-arXiv; abstract read | A NeurIPS 2026 poster listing appeared in search but I did not confirm acceptance |
| 42 | Ruiz, Arana-Catania, Ardila et al. Causal-Audit: a framework for risk assessment of assumption violations in time-series causal discovery. arXiv 2604.02488, 2026 | V-arXiv (metadata only) | Not read |
| 43 | Ruiz, Arana-Catania, Ardila et al. AutoCause: a Python framework that automates expert decisions in environmental time-series causal discovery. arXiv 2608.00198, 2026 | V-arXiv (metadata only) | Not read. Uses 30 five-station Bavarian subgraphs per search snippet |
| 44 | Ferdous, Hossain, Gani. TimeGraph: synthetic benchmark datasets for robust time-series causal discovery. KDD 2025 Datasets and Benchmarks. arXiv 2506.01361 | V-arXiv | |
| 45 | "Composable Causality" / DTCD toolkit, NeurIPS 2026 listing | UNVERIFIED | Seen only as a search snippet; no authors or arXiv ID checked. Do not cite |
| 46 | Gong, Yao, Zhang, Li, Bi. Causal discovery from temporal data: an overview and new perspectives. ACM Computing Surveys, DOI 10.1145/3705297. arXiv 2303.10112 | V-arXiv, V-search | Journal author list could not be matched exactly |
| 47 | Runge, Bathiany, Bollt et al. Inferring causation from time series in Earth system sciences (introduces CauseMe). Nature Communications 10:2553, 2019, DOI 10.1038/s41467-019-10105-3 | V-search | Author list beyond Runge not extracted (the search said Runge plus 19 co-authors) |
| 48 | Smith, Miller, Salimi-Khorshidi, Webster, Beckmann, Nichols, Ramsey, Woolrich. Network modelling methods for FMRI. NeuroImage 54(2):875-891, 2011, DOI 10.1016/j.neuroimage.2010.08.063 | V-search; data downloaded | FMRIB netsim page fetched; NeuroImage 2010 vs 2011 issue date: the page says 2010 (online), journal volume is 2011 |
| 49 | Prill et al. Towards a rigorous assessment of systems biology models: the DREAM3 challenges. PLoS ONE 5(2):e9202, 2010 | Partial | Title and venue from Rhino reference list; PLOS landing page returned 200; author list not checked |
| 50 | Marbach, Schaffter, Mattiussi, Floreano. Generating realistic in silico gene networks for performance assessment of reverse engineering methods. J. Comput. Biol. 16(2):229-239, 2009, DOI 10.1089/cmb.2008.09TT | V-search | |
| 51 | Lorenz. Predictability: a problem partly solved. ECMWF Seminar on Predictability, Vol. I, pp. 1-18 (listed as 1995 by ECMWF, 1996 in most citations) | V-search | Year inconsistency noted; cite as "Lorenz 1996" only with this caveat |
| 52 | ENVAR (linear SVAR(1) under equal noise variance), arXiv 2605.21846 | UNVERIFIED | Snippet only; mentioned nowhere in the tables |

Not cited because not verified: any claim that a particular KAN-based causal discovery method has been published in a peer-reviewed venue (I found only arXiv versions for GC-KAN, KANGCI and GRNGC); the exact author lists marked "et al." above; the number of methods tested in TCD-Arena and CausalCompass.

### Environment notes
- Downloads: netsim sims.tar.gz (49.3 MB) and CausalRivers product.zip (133.2 MB) both succeeded over HTTPS via the proxy with default verification. causeme.uv.es failed TLS verification (incomplete server chain); not retried with verification off. github.com HTML pages and the GitHub API returned 403 from the proxy; raw.githubusercontent.com and release assets worked. zenodo.org returned 503 on both tests.
- arXiv API (export.arxiv.org) timed out; arXiv abs pages worked and were used for metadata.
- Python in the sandbox has numpy 1.26.4, scipy, torch (CPU), tigramite 5.2.10.1 installed (system python3 without -I; `python3 -I` does not see site packages).
