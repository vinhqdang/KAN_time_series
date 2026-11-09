# B1x: statistical hygiene (B'2), mechanism oracle (B'3) and competitive learners (B'1) for the B1 start artefact

This folder answers roadmap items B'1, B'2 and B'3 of `../icml_paper/review/02_rereview_decision.md` (findings C4, R-N3, R-N4, R-N5, R-N6, R-N12).
- It imports `../b1/common.py` and `../b1nl/common_nl.py` unchanged. No committed `b1/` or `b1nl/` file was modified.
- One CPU core was used, with `OMP_NUM_THREADS=1`.

| part | status | files |
|---|---|---|
| **B'2** re-analysis of committed B1 / B1-NL data | **post hoc, descriptive, no new data, not pre-registered** | `reanalysis_orders.py`, `reanalysis_b2.py`, `multiplicity.py` → `results/b2_reanalysis.{txt,json}`, `results/multiplicity.{txt,csv}`, `results/orders_x1.csv.gz`, `results/empty_sid.csv.gz`, `results/b1_interactions_shd.csv` |
| **B'3** detectability oracle | **post hoc**; population computation on the B1 test graphs and parameters (same seeds); no new data | `oracle_b3.py` → `results/oracle_b3.{txt,json,csv.gz}`, `results/fig_oracle.{png,pdf}` |
| **B'1** DAGMA / NOTEARS / GOLEM + PCMCI+ | **new confirmatory experiment**; `PLAN.md` committed and pushed in **5ac1bd2** before any B'1 score | `methods_x.py`, `run_b1x.py`, `run_shards.sh`, `merge_shards.py`, `analyse_b1x.py` → `results/b1x_test.csv.gz`, `results/analysis.{txt,json}` |

---

## Part B'2: statistical hygiene on the committed B1 / B1-NL data (post hoc)

All CIs below are percentile bootstraps with 10,000 resamples. The **unit is the summary graph**: the 3 horizons of a graph are resampled together.

### 1. tab:b1 with graph-cluster CIs (B1, R-het)

Each family has 60 graphs (30 seeds × 2 p) and 180 data-set pairs.
- The Δ column also shows the naive data-set CI that was printed in the paper.
- The cluster CIs are **narrower** than the naive ones. T ∈ {3, 5, 10} is a fixed design factor with a large systematic effect. The naive bootstrap re-randomises the mix of horizons, while the cluster bootstrap keeps each graph's three horizons together, so the between-T variation no longer enters the uncertainty. The point estimates are unchanged. In either case the paper's CIs were not anti-conservative.

| family | exact s_ρ G1 / G2 [cluster CI] | rel OA G1 / G2 | ΔOA rel, cluster CI {naive CI} | var OA G1 / G2 | ΔOA var |
|---|---|---|---|---|---|
| symER (SC) | .782 [.771,.793] / .500 | .627 [.619,.634] / .501 [.492,.510] | +.126 [.115,.137] {.110,.142} | .640 / .498 | +.142 [.133,.152] |
| ring (SC) | .954 [.941,.966] / .500 | .687 [.677,.697] / .507 [.498,.516] | +.180 [.167,.193] {.163,.197} | .728 / .503 | +.224 [.213,.236] |
| chain | .887 [.879,.896] / .500 | .749 [.742,.756] / .532 [.524,.539] | +.217 [.209,.226] {.200,.235} | .930 / .699 | +.231 [.215,.247] |
| out-tree | .844 [.836,.852] / .500 | .694 [.684,.705] / .550 [.536,.565] | +.144 [.127,.161] {.125,.163} | .944 / .707 | +.237 [.222,.253] |
| in-tree | .792 [.786,.798] / .664 [.655,.673] | .743 [.736,.749] / .673 [.661,.685] | +.069 [.057,.081] {.056,.082} | .945 / .722 | +.223 [.214,.232] |
| ER(c=1) | .765 [.749,.779] / .560 [.543,.577] | .670 [.652,.687] / .585 [.566,.604] | +.085 [.068,.103] {.069,.103} | .855 / .646 | +.209 [.199,.219] |
| ER(c=2) | .830 [.820,.840] / .520 [.510,.532] | .698 [.682,.713] / .588 [.567,.610] | +.109 [.092,.127] {.092,.128} | .737 / .587 | +.150 [.138,.161] |

- Pooled over the non-SC families: +0.125 [0.116, 0.134] (300 graphs; unchanged).
- The var-OA CIs for each generator are in `results/b2_reanalysis.txt`, and are all ≤ ±0.025.
- B1-NL Δ with cluster CIs:
  - NL: +0.108 [0.100, 0.116];
  - SGN: +0.137 [0.128, 0.145];
  - SGNU: +0.137 [0.122, 0.151];
  - per family: `results/b2_reanalysis.txt` §1.

### 2. Tracking at the unit level and with graph-cluster inference

| experiment | cell-level Spearman (42 cells) [graph-cluster CI] | unit-level Spearman (graph × T) [CI] | OLS slope [CI] | graph-cluster permutation p |
|---|---|---|---|---|
| B1 | 0.798 [0.758, 0.823] | 0.616 [0.589, 0.643] (1,260 pairs) | 0.530 [0.503, 0.560] | ≤ 1e-4 |
| B1-NL NL | 0.677 [0.595, 0.709] | 0.443 [0.398, 0.487] (1,050) | 0.341 [0.305, 0.378] | ≤ 1e-4 |
| B1-NL SGN | 0.817 [0.757, 0.824] | 0.637 [0.610, 0.667] (1,050) | 0.561 [0.526, 0.598] | ≤ 1e-4 |
| B1-NL SGNU | 0.775 [0.709, 0.811] | 0.621 [0.575, 0.666] (420) | 0.564 [0.507, 0.626] | ≤ 1e-4 |

The permutation moves whole graphs (their T-vectors of gains) across graphs, keeping the horizons aligned. A second version permutes graphs separately within each T, which removes the common T trend from the null. Both reach the floor of 1/10,001 in all four data sets. The paper's p = 1.2e-10 treated the 42 cells as independent; the defensible statement is "p ≤ 1e-4 by graph-cluster permutation; unit-level Spearman 0.62".

### 3. H3 as interaction effects instead of swap counting

For each method pair, cell and metric, I = [m1 − m2](G1) − [m1 − m2](G2), with unit = graph.
- I < 0 on SHD or SID, and I > 0 on orientation, means m1 gains relative to m2 from the fixed start.
- Per cell: bootstrap CI and Wilcoxon on the per-graph I, with Holm over all pair × cell tests of the metric.
- Per family: pooled over p and T, with a graph-cluster CI.

**How common are interactions?** Count of Holm-significant pair × cell interactions:

| | SHD | SID | orientation |
|---|---|---|---|
| B1 (6 methods, 595 / 575 / 595 tests) | 294 | 241 | 313 |
| B1-NL NL (5 methods, 420 tests) | 113 | 137 | 172 |
| B1-NL SGN | 232 | 228 | 259 |
| B1-NL SGNU (10 graphs, min attainable p 0.002) | 0 (CI excludes 0 in 301) | 0 (302) | 0 (330) |

So the start artefact changes the *size* of method differences almost everywhere. Swap counting (opposite sign and both significant) picked out only the few cases where an interaction was large enough to flip the sign.

**Focus contrasts** (family-pooled I [graph-cluster 95% CI]; G1 and G2 differences in parentheses):

| contrast | metric | B1 | NL | SGN | SGNU |
|---|---|---|---|---|---|
| chain: rel-SnR − R²-SnR | SHD | **−27.8** [−36.6, −19.4] (+9.8 → +37.6) | +1.2 [−2.8, +5.1] (−19.3 → −20.5) | +3.7 [−1.0, +8.6] | −0.6 [−8.6, +7.5] |
| | SID (norm.) | **−0.110** [−0.123, −0.097] | **−0.076** [−0.088, −0.064] | **−0.065** [−0.080, −0.049] | **−0.077** [−0.100, −0.055] |
| | orientation | **+0.135** [0.123, 0.147] | **+0.069** [0.056, 0.083] | **+0.086** [0.071, 0.101] | **+0.087** [0.060, 0.114] |
| chain p = 10, T = 3 (B1 swap cell): rel − R² | SHD | **−52.1** [−58.9, −45.3] | **−16.4** [−22.0, −10.6] | **−31.8** [−37.8, −26.2] | −35.1 [−45.0, −25.4] |
| chain p = 10, T = 5 (B1 swap cell) | SHD | **−51.3** [−60.8, −41.8] | −8.4 [−16.0, −0.5] | −9.3 [−18.8, +0.1] | −11.2 [−29.9, +5.4] |
| in-tree: Var-SnR − FLOP | SHD | **−68.4** [−77.0, −60.4] (−4.3 → +64.1) | **−49.9** [−56.0, −44.2] (+16.3 → +66.2) | **−75.5** [−87.8, −64.2] (+23.1 → +98.6) | **−75.5** [−95.8, −56.4] |
| | SID | **−0.121** [−0.134, −0.110] | **−0.123** [−0.138, −0.108] | **−0.123** [−0.138, −0.110] | **−0.124** [−0.140, −0.109] |
| | orientation | **+0.200** [0.185, 0.216] | **+0.170** [0.152, 0.188] | **+0.213** [0.198, 0.228] | **+0.215** [0.197, 0.233] |
| in-tree: Var-SnR − PC (B1 only) | SHD | **−41.4** [−44.4, −38.6] (−13.1 → +28.3) | – | – | – |

How to read these contrasts:
- **R-N3 resolved in substance.** The in-tree Var-SnR vs FLOP interaction is the same in all four data sets on all three metrics, with magnitudes within each other's CIs. B1-NL "did not replicate the reversal" on SHD only because FLOP is already better under G1 there (+16 to +27 SHD), so the shift of about 50–75 SHD does not cross zero. Under SGN, Var-SnR is ahead under G1 by SID (−0.056) and orientation (+0.052), and behind under G2.
- **rel vs R² on chains.**
  - The SID and orientation interactions replicate in direction in every configuration, at about 60–80% of B1's size.
  - On SHD, the family-pooled interaction does not replicate: about 0 in NL, SGN and SGNU. In the two B1 swap cells it replicates in sign but is smaller. At p = 10, T = 3: −16 (NL), −32 (SGN), −35 (SGNU), against B1's −52; at T = 5 the CIs touch 0.
  - The B1 SHD swap therefore combines a real, replicating start effect on rel-SnR vs R²-SnR with B1-specific levels that put the two methods close enough for the sign to flip.
- **Wording.** "Reversals not detected" in the paper should become: "the G1 − G2 interaction of the in-tree Var-SnR vs FLOP contrast replicates on SHD, SID and orientation in all three parametrisations; the rel- vs R²-SortnRegress chain interaction replicates on SID and orientation and in the short-horizon cells on SHD; sign reversals on SHD did not recur because the B1-NL level differences are larger".

### 4. Empty-graph reference and SHD/|E| (R-N5)

SHD of the empty graph = |E|, so SHD/|E| = 1 is the empty-graph reference. Mean SHD/|E|, G1 / G2, B1 (R-het), over all 7 families:

| | rel-SnR | R²-SnR | Var-SnR | Random-SnR | FLOP | PC | Time-SnR (labels) |
|---|---|---|---|---|---|---|---|
| SHD/\|E\| | 1.05 / 1.70 | 1.33 / 1.96 | 0.61 / 1.39 | 1.72 / 2.04 | **0.27 / 0.46** | 0.40 / 0.64 | 0.04 / 0.27 |
| share of data sets with SHD/\|E\| > 1 | 0.50 / 0.88 | 0.78 / 1.00 | 0.25 / 0.86 | 0.90 / 0.94 | 0.00 / 0.00 | 0.00 / 0.01 | 0 / 0 |
| normalised SID | 0.32 / 0.45 | 0.36 / 0.43 | 0.24 / 0.38 | 0.51 / 0.52 | **0.18 / 0.23** | 0.28 / 0.37 | 0.00 / 0.00 |

The empty graph has normalised SID 0.50 (B1 average; 0.35 for in-trees, 0.64 for ER2).
- Under G2, every SortnRegress variant is worse than the empty graph on SHD (1.39–2.04). On SID it is better than the empty graph only marginally (0.38–0.45 vs 0.50).
- Under G1, Var-SnR (0.61) beats the empty graph, and rel-SnR is at it (1.05).
- B1-NL is similar: SGN rel-SnR 1.39 / 2.09 and FLOP 0.18 / 0.37; NL rel-SnR 0.98 / 1.26 and FLOP 0.35 / 0.53.
- **R-N5 is correct.** The SHD reversals are among methods that are no better than the empty graph on SHD. FLOP and PC are never worse than the empty graph. SID is the more informative metric for the sort baselines.

### 5. Order accuracy excluding the edges that leave the first window slice

True edges always join slice t to slice t+1, so no true edge lies *inside* slice 1. The relevant restriction drops the edges whose tail is in slice 1, i.e. the latently confounded slice under G2.
- **Reproduction.** The orders of rel-, R²- and Var-SnR were recomputed with the committed RNG streams. The recomputed OA reproduces the committed OA exactly (max difference 0) in B1 and in all B1-NL configurations.

G1 − G2, non-SC families (graph-cluster CI):

| | rel-SnR all edges | rel-SnR excl. slice-1 edges | R²-SnR all / excl. | Var-SnR all / excl. |
|---|---|---|---|---|
| B1 | +0.125 [0.116, 0.134] | +0.089 [0.079, 0.100] | +0.075 / +0.096 | +0.210 / +0.195 |
| NL | +0.108 | +0.067 [0.057, 0.077] | +0.079 / +0.085 | +0.218 / +0.194 |
| SGN | +0.137 | +0.087 [0.076, 0.097] | +0.099 / +0.098 | +0.246 / +0.216 |
| SGNU | +0.137 | +0.095 [0.077, 0.113] | +0.096 / +0.097 | +0.242 / +0.213 |

- **The gain survives and stays large** without the slice-1 edges, in every family and configuration (per family: `results/b2_reanalysis.txt` §5). So it is not produced by the change of target at the confounded first slice.
- **Part of the gain sits on slice-1 edges, and the theory predicts this.** Thm 1.5's value-1 edge copies are concentrated near the start. The B'3 population oracle predicts that the non-SC gain drops from +0.130 to +0.096 when these edges are excluded; the observed drop is +0.125 → +0.089.

### 6. One multiplicity statement (B1, B1-NL, B3 and B'1)

`multiplicity.py` gives each pre-registered claim one claim-level p:
- for "≥ k of m families" rules, the k-th smallest Holm-adjusted p;
- for "≥ 1 swap" rules, the best max(Holm G1, Holm G2);
- for tracking, the graph-cluster permutation p;
- for H4, the max of the parts (a) and (c) p-values;
- for B3, the plan's Holm p.

Holm is then applied across all claims, with family-wise α = 0.05. With the three B'1 claims there are 21 (`results/multiplicity.txt`):

- **Survive family-wise control:**
  - B1 H1, tracking, H2b, H2c and H3 (claim p 2.0e-3, Holm 7.9e-3);
  - B1-NL H1 ×3, tracking ×3 and H4 ×3;
  - B1-NL H3-NL (claim p 0.011, Holm 0.033: the single, substantively minor swap, R²- vs Random-SnR);
  - B3 H1 (1e-5, the Monte Carlo floor) and B3 H2 (2e-5);
  - B'1 HX1 (5e-16), HX2 (claim p 0.012, Holm 0.033) and HX3 (2e-16).
- **Does not survive:** B1-NL H3-SGN (p 0.108; no swap).
- **Not claims:**
  - B1 H2a failed, as predicted, and is not a claim.
  - B1-NL H3-SGNU was untestable by design.
  - The B3 secondary analyses (S1, S3, S4) are not confirmatory.
- **Caveat.** Several p-values are at their resolution floors (1e-4 permutation, 1e-5 sign flip). Holm on floors is still valid but conservative. B3 is internally time-stamped only (no external registration).

---

## Part B'3: detectability oracle for the mechanism (post hoc; R-N12)

For every B1 test graph and parameter set (420 graphs; same seeds and RNG streams; R-het) and T ∈ {3, 5, 10}, we computed the exact population covariance of the window:
- **G1:** x_1 = e_1, Σ_{t+1} = Aᵀ Σ_t A + D, Cov(x_s, x_t) = Σ_s A^{t−s}.
- **G2:** the same recursion, started after the 100 burn-in steps that G2 simulates.

**Check.** Population relatives (|ρ| > 1e-20) reproduce the exact graph s_ρ of Thm 1.5 (G1) and Thm 1.9 (G2) in all 2,520 cases (max |diff| 1e-16). At a threshold of 1e-12, one ER2 G2 case differs, because a far-past common ancestor gives |ρ| = 4e-13.

We then computed two predictions:
- **Oracle-threshold sort.** rel-SnR's Alg. 1 counts with every sample correlation replaced by its population value, using the same threshold ε = 0.0620 (t-test, α = 0.05, n = 1000). Prediction = edge-sortability of those counts.
- **Detection-probability oracle.** P(|r_ij| > ε) under Fisher z at n = 1000. The count difference of an edge's endpoints is treated as a sum of independent Bernoulli differences (the shared pair cancels). Prediction = P(c_i < c_j) + ½ P(tie): the expected rel-SnR order accuracy, up to dependence between the indicators. It uses no simulated data.

**Which explains the observed G1 − G2 rel-SnR gain better?**

| predictor | cell level (42): Spearman / Pearson R² / OLS slope / MAE | unit level (1,260): Spearman / R² / slope / MAE |
|---|---|---|
| exact graph gap s_ρ(G_T) − s_ρ^stat | 0.80 / 0.61 / 0.55 / 0.168 | 0.62 / 0.34 / 0.53 / 0.177 |
| oracle-threshold sort | 0.97 / 0.90 / 0.70 / 0.029 | 0.76 / 0.54 / 0.69 / 0.072 |
| **detection-probability oracle** | **0.99 / 0.97 / 1.03 / 0.012** | **0.79 / 0.61 / 1.03 / 0.058** |

- The detection oracle explains **97%** of the cell-level variance with slope 1.03 and intercept −0.010, and 61% at the unit level. The graph gap explains 61% and 34%.
- Unit-level R²(oracle) − R²(gap) = +0.27 [0.23, 0.30], graph-cluster bootstrap.
- In a joint regression the graph gap gets coefficient −0.06 (cell) and −0.04 (unit), i.e. it adds nothing once detectability is known.
- The oracle also reproduces the levels, as well as the differences. Rel-SnR OA G1 / G2 predicted vs observed:

| family | predicted | observed |
|---|---|---|
| chain | 0.756 / 0.528 | 0.749 / 0.532 |
| ring | 0.702 / 0.503 | 0.687 / 0.507 |
| in-tree | 0.746 / 0.677 | 0.743 / 0.673 |
| ER2 | 0.698 / 0.584 | 0.698 / 0.588 |

**Ring p = 10, T = 10.**
- Exact gap 0.500 (s_ρ(G_T) = 1, since T ≤ p); observed gain only +0.079.
- Oracle-threshold predicts +0.080 and the detection oracle +0.094.
- Reason: only 39–41% of the graph-relative pairs have a population correlation detectable at n = 1000 (expected recall G1 / G2 0.39 / 0.41). Distant relatives along the 10-cycle with spectral radius 0.8 have |ρ| below ε.
- Similarly symER p = 5, T = 10: gap +0.111, observed +0.019, oracle +0.025 (the threshold oracle even predicts −0.018).

**Reading (answers R-N12).**
- The graph quantity of Thm 1.5/1.9 is the correct *population limit with perfect detection*, and it determines the sign: every observed cell gain is positive where the gap is.
- The *size* of the benchmark effect at n = 1000 is set by which relatives are detectable, i.e. by the correlation decay along paths. The "slope 0.53" is not an unexplained attenuation: a population computation with the same threshold accounts for it almost exactly (slope 1.03).
- This is still the paper's mechanism, i.e. relatives counts and the start, with a finite-sample detection layer. It is not a generic transient. The oracle uses the start-dependent covariance only through the relatives counts, and predicts the gain with slope 1.
- The variance and R² ramps (the generic-transient alternative) are not used by the oracle. They cannot be what drives rel-SnR, which sees only thresholded correlations.
- **Claim for the paper.** "The exact gap predicts the sign and ranks the cells (Spearman 0.80); a detectability oracle computed from the exact window covariances predicts the magnitude (cell-level R² 0.97, slope 1.03)."

---

## Part B'1: DAGMA, NOTEARS, GOLEM and PCMCI+ (pre-registered)

**Timeline.**
- Plan `PLAN.md`, code and validation timing were committed and pushed in **5ac1bd2** at 08:20 UTC, before any B'1 score.
- The test run went from 08:28 to 12:41 UTC: 384 data sets, one core, 4.21 h. It was run once and analysed once with `analyse_b1x.py` as committed.
- No threshold, hypothesis, seed or method was changed.

**Design.** B1's generators and SEM, run on a new grid of 384 data sets:
- Families: ring, chain, in-tree, ER2. p ∈ {5, 10}, T ∈ {3, 5}.
- 12 new graphs per (family, p), i.e. 96 summary graphs; G1 and G2 for each.
- Every data set was analysed on raw (R-het) and on standardised (R-std) data, with the time labels hidden.
- Methods and default settings:

| method | implementation | settings |
|---|---|---|
| DAGMA linear | PyPI `dagma` 1.1.1 | λ = 0.03, threshold 0.3 |
| NOTEARS linear | the authors' reference code | λ = 0.1, threshold 0.3 |
| GOLEM-EV / -NV | numpy port of the authors' code | λ = 2e-2 / 2e-3, λ₂ = 5, 1e5 Adam steps, threshold 0.3; **p = 5 cells only**, for run time, as planned |

- Comparators: FLOP, rel-, R²- and Var-SnR, and the empty graph.
- PCMCI+ (tigramite, ParCorr, τ_max = 1) on the time-indexed data, as a labels-using reference.
- Metrics: SHD/|E| (empty graph = 1), normalised SID, and ancestral order accuracy (AOA). Unit of inference: summary graph.

### Verdicts

| hypothesis | rule | result | verdict |
|---|---|---|---|
| **HX1** the fixed start moves CO methods | some raw CO variant Holm-significant favourable on AOA or SHD/\|E\| (24 tests) | All four raw variants improve under G1 on all three metrics: SHD/\|E\| −0.41 to −0.55, SID −0.13 to −0.19, AOA +0.16 to +0.23 (all Holm ≤ 4e-8; numbers below) | **supported** |
| **HX2** some CO method's rank relative to FLOP changes | opposite-sign, both Holm-significant (64 tests on SHD/\|E\|) | **2 rank changes, both on in-trees.** DAGMA-raw − FLOP: −0.218 under G1 (Holm 0.012) vs +0.306 under G2 (1.4e-5). NOTEARS-raw − FLOP: −0.176 (0.0011) vs +0.193 (0.020). The same 2 changes appear on SID and on AOA | **supported** |
| **HX3** standardisation removes the gain | removed / reduced / not reduced (8 tests) | DAGMA SHD: **removed** (std gain −0.005 [−0.027, +0.016]). All other (method, metric) pairs: **reduced, not removed**. On AOA the std "gain" of DAGMA, GOLEM-EV and GOLEM-NV is significantly *negative* (−0.06 to −0.10): after standardisation G1 makes them orient *worse*. NOTEARS-std keeps a small favourable gain (AOA +0.036, SHD −0.077) | **supported** (gain removed or reversed for 3 of 4 methods on AOA) |
| XS1 rel-SnR gain replicates | CI > 0 | +0.187 [0.169, 0.205] | pass |
| XS2 failed fits | reported | 0 failed fits. Cyclic outputs: NOTEARS-std 15.6% (SID missing in 60 of 384), NOTEARS-raw 2.9%, DAGMA-std 0.3%, others 0 | – |

**HX1 detail.** Graph-level G1 − G2 differences [graph-cluster 95% CI]:

| variant | SHD/\|E\| | SID (norm.) | AOA |
|---|---|---|---|
| DAGMA raw | −0.548 [−0.663, −0.438] | −0.147 [−0.163, −0.132] | +0.197 [0.177, 0.216] |
| NOTEARS raw | −0.419 [−0.521, −0.321] | −0.127 [−0.142, −0.114] | +0.156 [0.136, 0.175] |
| GOLEM-EV raw (p = 5) | −0.491 [−0.602, −0.385] | −0.182 [−0.204, −0.160] | +0.225 [0.204, 0.248] |
| GOLEM-NV raw (p = 5) | −0.409 [−0.491, −0.330] | −0.193 [−0.217, −0.170] | +0.232 [0.205, 0.258] |
| DAGMA std | −0.005 [−0.027, +0.016] | +0.004 [−0.006, +0.014] | **−0.064** [−0.079, −0.048] |
| NOTEARS std | −0.077 [−0.103, −0.050] | −0.073 [−0.087, −0.060] | +0.036 [0.022, 0.049] |
| GOLEM-EV std | −0.326 [−0.440, −0.217] | **+0.022** [0.005, 0.038] | **−0.093** [−0.115, −0.073] |
| GOLEM-NV std | −0.039 [−0.073, −0.006] (n.s. after Holm) | **+0.029** [0.014, 0.045] | **−0.097** [−0.118, −0.077] |

Bold marks an unfavourable, Holm-significant effect.

**Comparison: B1's sort baselines on the same data.**
- Var-SnR SHD/|E| moves from 0.50 (G1) to 1.29 (G2), and FLOP from 0.19 to 0.52.
- The raw CO gains are as large as Var-SnR's, which is consistent with exploiting the G1 var-sortability.

**HX2 detail and interactions vs FLOP.** I = (v − FLOP)(G1) − (v − FLOP)(G2) on SHD/|E|, graph-cluster CI. I < 0 means v gains on FLOP under G1.

| variant | ring | chain | in-tree | ER2 | all |
|---|---|---|---|---|---|
| DAGMA raw | +0.17 [0.12, 0.22] | **−0.66** [−0.91, −0.44] | **−0.52** [−0.60, −0.44] | +0.11 [−0.02, 0.23] | −0.23 [−0.33, −0.13] |
| NOTEARS raw | +0.32 [0.27, 0.36] | **−0.51** [−0.69, −0.34] | **−0.37** [−0.46, −0.27] | +0.18 [0.06, 0.27] | −0.10 [−0.19, −0.01] |
| GOLEM-EV raw | +0.09 [−0.02, 0.20] | −0.45 [−0.66, −0.26] | −0.38 [−0.55, −0.23] | +0.01 | −0.18 [−0.29, −0.08] |
| GOLEM-NV raw | +0.07 | −0.20 [−0.36, −0.04] | −0.32 [−0.43, −0.21] | +0.04 | −0.10 [−0.19, −0.02] |
| DAGMA / NOTEARS / GOLEM std | +0.11 to +0.41 (FLOP gains) | +0.25 to +0.37 (GOLEM-EV std −0.41) | +0.11 to +0.19 (GOLEM-EV std −0.17) | +0.14 to +0.30 | −0.02 to +0.32 |

Reading the interactions:
- On the tree-shaped families (chain, in-tree), the fixed start helps every raw CO method relative to FLOP, by 0.2–0.66 SHD/|E|.
- On ring and ER2 the interaction is null or reversed: the CO methods lose ground to FLOP under G1.
- After standardisation, FLOP gains relative to the CO methods under G1 in almost every family.
- Per cell, 72 of 96 (variant × cell) interaction CIs exclude 0, without multiplicity correction.

**Best method without labels** (lowest mean SHD/|E| per (family, p)): FLOP in all 8 cells under G2. Under G1, a CO method wins in 3 of 8:

| cell | G1 winner | FLOP under G1 |
|---|---|---|
| chain p = 5 | DAGMA-raw 0.09 | 0.10 |
| in-tree p = 5 | NOTEARS-raw 0.09 | 0.27 |
| in-tree p = 10 | DAGMA-raw 0.06 | 0.33 |

Pooled over the in-tree data sets, SHD/|E| and SID under G1 and G2:

| method | SHD/\|E\| G1 / G2 | SID G1 / G2 |
|---|---|---|
| DAGMA-raw | 0.08 / 0.87 | 0.03 / 0.16 |
| NOTEARS-raw | 0.13 / 0.76 | 0.05 / 0.18 |
| FLOP | 0.30 / 0.56 | 0.10 / 0.12 |

- **The in-tree rank change is not a slice-1 confounding artefact.** Excluding first-slice pairs (SHD_x1/|E|), DAGMA-raw is at 0.08 / 0.73 against FLOP's 0.30 / 0.37.

**Levels** (all cells, G1 / G2; GOLEM p = 5 only):

| method | SHD/\|E\| | SID (norm.) | AOA |
|---|---|---|---|
| FLOP | **0.19 / 0.52** | **0.10 / 0.20** | **0.89 / 0.77** |
| DAGMA raw | 0.35 / 0.90 | 0.23 / 0.37 | 0.80 / 0.60 |
| NOTEARS raw | 0.47 / 0.89 | 0.26 / 0.39 | 0.74 / 0.59 |
| GOLEM-EV raw (p = 5) | 0.44 / 0.93 | 0.29 / 0.47 | 0.79 / 0.56 |
| GOLEM-NV raw (p = 5) | 0.39 / 0.80 | 0.28 / 0.47 | 0.79 / 0.56 |
| DAGMA std | 0.87 / 0.87 | 0.53 / 0.53 | 0.35 / 0.41 |
| NOTEARS std | 0.76 / 0.84 | 0.43 / 0.51 | 0.49 / 0.46 |
| GOLEM-EV std (p = 5) | 0.97 / 1.30 | 0.64 / 0.62 | 0.33 / 0.42 |
| GOLEM-NV std (p = 5) | 0.95 / 0.99 | 0.65 / 0.62 | 0.31 / 0.41 |
| Var-SnR raw | 0.50 / 1.29 | 0.18 / 0.33 | 0.86 / 0.63 |
| rel-SnR | 0.78 / 1.60 | 0.22 / 0.38 | 0.75 / 0.57 |
| empty graph | 1 / 1 | 0.42 / 0.42 | 0.5 / 0.5 |
| PCMCI+ (labels; reference) | 0.024 / 0.022 | 0.00 / 0.00 | 1.00 / 1.00 |

Points from the levels table:
- **Raw CO methods beat the empty graph on SHD on average under both generators**, but under G2 only narrowly (0.80–0.93). On chains under G2 they are worse than the empty graph (1.14–1.25). So R-N5's point ("worse than an empty graph") applies less to the competitive methods than to the sort baselines, but it does not vanish.
- **Standardised CO methods are anti-sortable on tree families under G1.** AOA is 0.26–0.27 for DAGMA-std and GOLEM-std on chains and in-trees, i.e. they orient most edges backwards, and are near the empty graph on SHD.
- **PCMCI+ with labels is almost perfect under both generators.** SHD/|E| is 0.02 and orientation ≥ 0.997. It pools a lag-1 model over time, and the fixed start did not hurt it at n = 1000 trajectories.
- **Repeat of B1's chain swap.** rel-SnR vs R²-SnR on chains swaps sign again on the new seeds: SHD/|E| −0.60 (G1) vs +0.30 (G2), I = −0.90 [−1.08, −0.73]; OA I = +0.216 [0.184, 0.247].

### Main-text-ready paragraph

> **Competitive learners.** In a separately pre-registered extension we ran DAGMA, NOTEARS and GOLEM-EV/NV (authors' code or a faithful port; default hyperparameters; threshold 0.3) on raw and on standardised data, on 384 new data sets: ring, chain, in-tree and ER(c = 2); p ∈ {5, 10}; T ∈ {3, 5}; 12 graphs per (family, p); GOLEM at p = 5 only.
> - **On raw data, all four improve sharply under the fixed start.** Over 96 graphs, SHD/|E| falls by 0.41–0.55, normalised SID by 0.13–0.19, and ancestral order accuracy rises by 0.16–0.23 (graph-level Wilcoxon, Holm ≤ 4·10⁻⁸).
> - **This changes the ranking against the strongest score-based learner.** On in-trees, DAGMA and NOTEARS beat FLOP under G1 (SHD/|E| 0.08 and 0.13 vs 0.30), and lose to it under G2 (0.87 and 0.76 vs 0.56). Both reversals are Holm-significant under both generators, and recur on SID and order accuracy.
> - **Standardisation removes the effect or turns it around.** DAGMA's SHD gain vanishes (−0.005 [−0.027, 0.016]). For DAGMA and GOLEM the start effect on order accuracy reverses (−0.06 to −0.10).
> - **So the fixed start mainly hands var-sortability to the continuous-optimisation methods.** The scale-free start artefact that remains after standardisation helps only rel-SortnRegress, which gains +0.19 here as in B1.
> - **Labels make the artefact irrelevant.** A time-series method that uses the labels (PCMCI+) recovers the window almost perfectly under both generators (SHD/|E| 0.02).

### Main-text-ready table

| method (raw data unless noted) | SHD/\|E\| G1 / G2 | SID G1 / G2 | order acc. G1 / G2 | Δ order acc. [95% CI] |
|---|---|---|---|---|
| FLOP | 0.19 / 0.52 | 0.10 / 0.20 | 0.89 / 0.77 | – |
| DAGMA | 0.35 / 0.90 | 0.23 / 0.37 | 0.80 / 0.60 | +0.197 [0.177, 0.216] |
| NOTEARS | 0.47 / 0.89 | 0.26 / 0.39 | 0.74 / 0.59 | +0.156 [0.136, 0.175] |
| GOLEM-NV (p = 5) | 0.39 / 0.80 | 0.28 / 0.47 | 0.79 / 0.56 | +0.232 [0.205, 0.258] |
| DAGMA, standardised | 0.87 / 0.87 | 0.53 / 0.53 | 0.35 / 0.41 | −0.064 [−0.079, −0.048] |
| NOTEARS, standardised | 0.76 / 0.84 | 0.43 / 0.51 | 0.49 / 0.46 | +0.036 [0.022, 0.049] |
| in-trees only: DAGMA / NOTEARS / FLOP | 0.08 / 0.13 / 0.30 (G1); 0.87 / 0.76 / 0.56 (G2) | | | |
| PCMCI+ (time labels used) | 0.02 / 0.02 | 0.00 / 0.00 | 1.00 / 1.00 | – |

Δ is the graph-level G1 − G2 difference with a graph-cluster CI. Order accuracy for graph outputs is ancestral (1 / 0 / ½ for correct / reversed / incomparable).

### Deviations and notes (with timing)

- **No deviation from the plan.**
  - The size rule was applied before the test run: the validation projection was 4.35 h, below 4.5 h, so 12 seeds were kept. The actual run took 4.21 h.
  - Before the plan commit, the HX3 decision rule was clarified in both PLAN and code: the difference of gains must be significant *in the direction* raw more favourable than std. This was before any score.
- **NOTEARS (reference code) emits overflow RuntimeWarnings** in `expm` during some L-BFGS-B line searches, as the original does. No fit failed. Cyclic outputs after thresholding are reported (XS2). For cyclic outputs SID is undefined and missing, and AOA scores 2-cycles as incomparable (½).
- **GOLEM is a numpy port, not the TensorFlow original.** It has the same score, optimiser, defaults and post-processing, uses analytic gradients (finite-difference error 3e-9), and has the data enter through the sample covariance. It was run only at p = 5, as planned.
- **CPU.** The B'3 oracle was rerun once, niced (about 1.5 min), during the first minutes of the test run to add the slice-1 comparison. Otherwise the test run had the core to itself among my processes.
- **Not done.**
  - T = 10 and the families symER, out-tree and ER1 were excluded by design (budget).
  - No survey of deployed generators' initialisation (roadmap B'1's second half): not in this brief.

### Honest reading

1. **Contribution.** B1's limitation, "reversals concern baselines only", no longer holds. The fixed start reverses DAGMA's and NOTEARS's standing against FLOP on in-trees. It also makes a continuous-optimisation method the best label-free method in 3 of 8 (family, p) cells, against 0 of 8 under the burn-in.
2. **Which artefact.** Both the reversal and the gains come from the **var-sortability** created by the fixed start, which is outside the paper's theory:
   - they disappear or reverse after standardisation;
   - they are largest on chains and in-trees, where G1 makes the window almost perfectly var-sortable (Var-SnR order accuracy 0.95–0.96 under G1, against 0.77 on rings and ER2);
   - they are absent or reversed on rings and ER2.
3. **What the paper should say.** "The fixed start inflates var- and rel-sortability; the first changes the ranking of competitive continuous-optimisation methods (on raw data), the second, predicted by Thm 1.5/1.9, changes the scale-free sort baselines."
4. **What it should not say.** It should not claim that the paper's rel-sortability mechanism moves the competitive methods: after standardisation their start effect is near zero or adverse.
5. **Scope.** One linear-Gaussian parametrisation; n = 1000; d ≤ 50; GOLEM only at d ≤ 25; defaults, not tuned hyperparameters.
