# B'1 analysis plan: does the start artefact move competitive continuous-optimisation learners?

Roadmap item B'1 of `../icml_paper/review/02_rereview_decision.md` (findings C4, R-N5). It extends B1 (`../b1/PLAN.md`, `../b1/README.md`) to DAGMA, NOTEARS and GOLEM, run on raw and on standardised data, and adds one time-series method that uses the time labels, as a reference.

This plan is committed and pushed **before any simulation that produces a B'1 score**. Computations run before this commit:
- Method fits on one validation graph per (family, p) (seed 9600), to measure run time only (`timing_val.sh`, `run_b1x.py --timing`). The timing mode fits every method and records only the seconds; no metric is computed.
- A finite-difference check of the GOLEM gradients (max abs error 3e-9).

The B'2 and B'3 parts of this folder (`reanalysis_*.py`, `oracle_b3.py`) re-analyse committed B1 / B1-NL data post hoc. They are not part of this plan.

## 0. Question and threat model

The threat model is B1's (§0). The pT window variables are handed to an i.i.d. learner as anonymous columns, with the time labels hidden. In B1 the fixed start (G1) raised the order accuracy of the sort baselines. It reversed some baseline-vs-baseline comparisons but never changed the best method (FLOP).

The review asks two questions about competitive continuous-optimisation methods:
1. Do they move?
2. Does their standing relative to FLOP change?

DAGMA, NOTEARS and GOLEM-EV are known to exploit var-sortability on raw data, and to lose much of their accuracy after standardisation (Reisach, Seiler, Weichwald, NeurIPS 2021). In B1, G1 created var-sortability (+0.20 in Var-SnR order accuracy on raw data), so these methods are expected to gain from G1 on raw data. The open questions are:
- How large is the gain, relative to FLOP?
- Does a gain remain on standardised data, where only the scale-free artefacts (rel- and R²-sortability) are left?

## 1. Design (fixed now)

### 1.1 Graphs, SEM, generators

The graph, SEM and generator code is B1's (`../b1/common.py`), imported unchanged:
- **Families.** `ring` (strongly connected), `chain`, `intree`, `ER2`.
  - `chain` and `intree` carry B1's two substantive reversals.
  - `ring` has the largest exact gap.
  - `ER2` is the densest non-SC family.
- **Sizes.** p ∈ {5, 10}, T ∈ {3, 5}, so the window size d ∈ {15, 25, 30, 50}.
- **Dropped cells.** T = 10 and the families symER, outtree and ER1 are dropped to fit the one-core budget. At d = 100, NOTEARS takes about 130 s and GOLEM about 75 s per fit.
- **SEM.** B1's linear-Gaussian VAR(1) with positive weights Unif(0.5, 1), spectral cap 0.8 and R-het noise (sd_v ~ Unif(0.5, 1)). n = 1000 independent trajectories.
- **Generators.**
  - G1: fixed start x_1 = e_1.
  - G2: latent burn-in B = 100, keep the next T slices.
  - Ground truth for both: B1's window DAG.
- **Regimes.**
  - R-het: raw data.
  - R-std: the same data with every column standardised.
  - Each data set is analysed in both regimes, so raw and std are paired.

### 1.2 Seeds and size

- **Test graphs.** Seeds 3000–3011, i.e. 12 graphs per (family, p) and 96 summary graphs.
  - Graph and parameter RNG: `default_rng([301, family_index, p, seed])`.
  - Data RNG: `default_rng([302, family_index, p, seed, T, gen_index, 13])`.
  - Both streams are new and distinct from B1 (no prefix), B1-NL (101–203) and validation.
- **Data sets.** 4 families × 2 p × 12 seeds × 2 T × 2 generators = **384**.
- **Projected time.** The completed validation timing (one graph per (family, p), 32 data sets, machine shared with other jobs) took 1,304 s per seed. That projects **4.35 h** for 12 seeds. NOTEARS dominates on chains under G2: 151 s for one d = 50 fit.
- **Pre-stated size rule.** If the completed validation timing projects more than 4.5 h, the seeds are cut to 3000–3009 before the test run, and this is logged as a deviation. Otherwise the size above is run. The projection is 4.35 h, below the 4.5 h limit, so **12 seeds are run**.
- The test seeds are run once, in 8 resumable shards (one per (family, p); one row is appended per finished data set).

### 1.3 Methods (time labels hidden; columns randomly permuted per data set)

**Continuous optimisation (CO), each on raw and on standardised data, with the authors' defaults and threshold 0.3:**
- **DAGMA linear** (Bello, Aragam, Ravikumar, NeurIPS 2022).
  - Implementation: the authors' PyPI package `dagma` 1.1.1, numpy class `DagmaLinear(loss_type="l2")`.
  - Settings: `fit` defaults, i.e. lambda1 = 0.03, w_threshold = 0.3, T = 5, s = [1, .9, .8, .7, .6], warm_iter 3e4, max_iter 6e4, lr 3e-4.
- **NOTEARS linear** (Zheng, Aragam, Ravikumar, Xing, NeurIPS 2018).
  - Implementation: a line-by-line copy of the authors' reference `notears/linear.py`, in `methods_x.py`.
  - Settings: the README example, i.e. lambda1 = 0.1, l2 loss, max_iter 100, h_tol 1e-8, rho_max 1e16, w_threshold = 0.3.
- **GOLEM-EV and GOLEM-NV** (Ng, Ghassami, Zhang, NeurIPS 2020).
  - Implementation: a numpy port of the authors' TensorFlow code (`methods_x.py`).
  - Kept from the original: the score, Adam (lr 1e-3, 1e5 iterations, B_0 = 0), the defaults (EV: lambda1 = 2e-2, lambda2 = 5; NV: lambda1 = 2e-3, lambda2 = 5, initialised at the EV solution), and the post-processing (threshold 0.3, then remove the weakest edges until acyclic).
  - What differs: the data enter through the sample covariance (the score depends on X only through it), and the gradients are analytic. A finite-difference check gives max error 3e-9.
  - No GPU framework is used. torch 2.15 (CPU) is installed but not needed.
  - **GOLEM is run only on the p = 5 cells** (d = 15, 25), because of run time: 1e5 iterations × 2 models × 2 regimes cost about 30–40 s per data set at d ≤ 25 and about 75 s at d = 50.

**Comparators (B1 implementations, `../b1/common.py`):**
- M1 rel-SortnRegress, M2 R²-SortnRegress, M3 Var-SortnRegress on raw data, std_M3 Var-SortnRegress on standardised data (a random order, since all variances are 1).
- M5 FLOP (flopsearch 0.3.0, BIC penalty 2, 10 restarts, standardised input).
- The **empty graph**.

M1, M2 and FLOP are scale invariant, so they are identical in R-het and R-std.

**Reference that uses the time labels (not part of any hypothesis):**
- **PCMCI+** (Runge, UAI 2020) via tigramite 5.2.10.1.
- Settings: ParCorr (analytic), tau_min = 0, tau_max = 1, pc_alpha = 0.01, on the time-indexed raw data. The n trajectories enter as n datasets (analysis_mode = "multiple").
- Mapping to the window: a lag-1 link i → j becomes (t, i) → (t+1, j) for every t, and contemporaneous links become within-slice edges in every slice.
- PCMCI+ pools over time and assumes stationarity, so G1 violates its assumptions while G2 (window from a stationary chain) satisfies them.

### 1.4 Metrics (per data set and method)

- **SHD/|E|.** B1's SHD (each unordered pair counts 1 if its state differs), divided by the number of true window edges. The empty graph has SHD/|E| = 1 exactly. SHD_x1 (first-slice pairs excluded) is also recorded.
- **SIDn.** Normalised SID (gadjid 0.1.0), computed on the directed part of the estimate. It is missing if that part is cyclic, and the share of missing values is reported.
- **AOA (ancestral order accuracy).** The mean over true edges (i, j) of 1 if i is an ancestor of j in the estimate (and not vice versa), 0 if the reverse, and 1/2 otherwise. AOA turns any output graph into an order-accuracy analogue. For the sort methods the order accuracy OA is also recorded. FLOP's undirected edges count as incomparable.
- **Orientation score and number of edges.** As in B1.

Graph-level value: for a summary graph (family, p, seed), the mean over T ∈ {3, 5} of the paired data-set quantity.

### 1.5 Statistics

- **Units.** The unit of every test is the summary graph: 96 graphs for DAGMA/NOTEARS, 48 for GOLEM.
- **Tests.** Two-sided Wilcoxon signed-rank (scipy, `zero_method="wilcox"`) on graph-level values.
- **CIs.** Graph-cluster percentile bootstrap, 10,000 resamples, seed 0.
- **Holm** is applied within each hypothesis, as stated below.

## 2. Hypotheses and decision rules (fixed now)

Notation: CO variants V = {DAGMA_raw, DAGMA_std, NOTEARS_raw, NOTEARS_std, GOLEMEV_raw, GOLEMEV_std, GOLEMNV_raw, GOLEMNV_std}, 8 variants. "Favourable" means lower SHD/|E| or SIDn, or higher AOA.

**HX1 (does the fixed start move the CO methods?).**
- **Statistic.** For each variant v and metric k ∈ {SHD/|E|, SIDn, AOA}, the graph-level G1 − G2 difference d_{v,k}.
- **Test.** 24 tests, Holm at 0.05.
- **Per-cell verdict.** "G1 moves v on k" if Holm-significant, with direction.
- **Overall verdict.** HX1 is supported if at least one raw CO variant among DAGMA_raw, NOTEARS_raw and GOLEMEV_raw has a Holm-significant favourable G1 − G2 difference on AOA or SHD/|E|. Otherwise HX1 is not supported, i.e. no start effect was detected on CO methods.
- **Prediction.** Favourable on raw data for all of them (var-sortability), with larger gains in AOA than in SHD.

**HX2 (does any CO method's rank relative to FLOP change?).**
- **Statistic.** For each variant v and family f, the graph-level paired difference Δ_{v,f,g} = metric_v − metric_FLOP under generator g. There are 24 graphs per family, or 12 for GOLEM (p = 5 only).
- **Test.** Two-sided Wilcoxon, Holm over all (variant × family × generator) tests of the metric: 8 variants × 4 families × 2 generators = 64 per metric.
- **Rank change.** Opposite-sign means of Δ under G1 and G2, both Holm-significant.
- **Verdict.** The primary metric is SHD/|E|; HX2 is supported if at least one rank change exists there. SIDn and AOA are evaluated by the same rule and reported, but are not used for the verdict.
- **Interaction effects (reported in full).** I_{v,f} = Δ_{v,f,G1} − Δ_{v,f,G2}, per family and pooled over families, with graph-cluster 95% CIs, for SHD/|E|, SIDn and AOA. Per-cell interactions (12 graphs) are reported descriptively. The same interactions are reported for v against Var-SnR (raw) and against rel-SnR.
- **Prediction.** No rank change on SHD/|E|: FLOP stays better under both generators. Raw CO variants may close part of the gap under G1, giving interactions in favour of CO.

**HX3 (does standardisation remove the CO methods' gain?).**
- **Statistic.** For each CO method m ∈ {DAGMA, NOTEARS, GOLEM-EV, GOLEM-NV} and metric k ∈ {AOA, SHD/|E|}, the graph-level difference of gains (d_{m_raw,k} − d_{m_std,k}).
- **Test.** 8 tests, Holm at 0.05.
- **Verdict for each (m, k):**
  - **"removed":** the raw gain is favourable and Holm-significant in HX1, the difference of gains is Holm-significant in HX3 in the direction "raw gain more favourable than std gain", and the 95% CI of the std gain includes 0 or |mean std gain| < 0.02.
  - **"reduced, not removed":** the difference of gains is significant but the std gain is outside both conditions.
  - **"not reduced":** the difference of gains is not significant.
  - **"no gain to remove":** the raw gain is not significant.
- **Prediction.** The gain is "reduced, not removed" for DAGMA and NOTEARS on AOA. G1 also raises the scale-free R²- and rel-sortability, so some gain survives standardisation.

**Descriptive (no verdict).**
- Levels per method (SHD/|E|, SIDn, AOA, G1 / G2) with the empty graph.
- The best method without labels, per family and generator.
- PCMCI+ SHD/|E| and SHD_x1/|E| under G1 and G2.
- Rate of cyclic CO outputs.
- The rel-SnR OA gain (as a replication of B1 H1 on the new seeds), with the interaction of rel-SnR vs R²-SnR on chains (B1's mechanism reversal).

**Sanity rules (a failure is reported and limits interpretation).**
- XS1: pooled rel-SnR OA gain G1 − G2 > 0 with CI > 0 (B1 replicated at this size).
- XS2: CO fits return finite weights in every data set. A failed fit is recorded as missing and reported.

## 3. Reporting

`README.md` gives:
- the verdicts, numbers and CIs;
- deviations, with timing;
- a main-text-ready paragraph and a table;
- an honest reading, including null results.

`results/b1x_test.csv.gz` (merged shards) is committed. Shard row files are ignored (`.gitignore`). Any change after this commit is logged as a deviation.
