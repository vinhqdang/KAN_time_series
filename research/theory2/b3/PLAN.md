# B3 analysis plan: a discriminating test of the source-SCC rule (Thm 1.9 / Cor 1.10) against tie rules, on held-out data with several source SCCs per weak component

Roadmap item B3 (`../icml_paper/review/00_editorial_decision.md`, finding C5, Devil's Advocate CRITICAL 3). Builds on `../THEORY.md` (Thm 1.9, Cor 1.10, Cor 2.9, Section 6 rule), `../icml_exp/PLAN4.md` (E1) and `../icml_exp/THEORY_ADD.md` (Lemma B1-B3).

**Registration status.** This plan is committed and pushed to the project repository before any value of any held-out dataset named below is loaded. It is **only internally time-stamped** (git commit and push to `origin/main`). **No external registration (OSF or other registry) was made before data access**, unless the user posts `OSF_PREREG.md` to OSF before the analysis commit, which the agent cannot do. A reader must therefore trust the repository history. Commit timestamps are client-set; the push to the remote is the stronger time-stamp.

## 0. The problem this test fixes

The old E1 passed mostly through ties. On its primary graph G_lin every weak component has a single source, so the theorem's prediction coincides on every instance with the trivial rule "½ iff same weak component, else compare observed component sizes". The competitors could not output ½, and no constant-½ rule was pre-registered.

Cor 1.10 says more than "ties within a weak component". When a weak component has several source SCCs, an edge (i, j) is strictly sorted (value 1) iff Src(j) ≠ Src(i). On an observed node set S, the Section-6 rule gives value 1 iff S contains a relative of j that is not a relative of i. So the **same edge** changes from ½ to 1 depending on whether a second, independent source of its head is observed. No tie rule and none of the eight old naive rules predicts this switch:
- the tie rules (constant ½, weak-component rule) predict ½ on every within-component edge;
- the eight old naive rules (sensor indicator, ancestor counts, in-degrees, depths, every-edge-1) predict 1 on every source → sensor edge.

The theorem agrees with the tie rules where no second source is observed and with the naive sort rules where one is. **It is the only predictor that is right on both kinds of instance, if the data behave as it says.**

## 1. Data (held out) and what was done before this commit

### 1.1 Exposure check
- Studies 3, 4 (E1/E2) and the CausalRivers studies used: WT `wt_walks_v1` (actuators_random_walk_*, regime_jumps_*), LT `lt_walks_v1/actuators_white`, `lt_crl_benchmark_v1/{citris_1, buchholz_1_obs}`, `lt_camera_v1/scm_{1,2,4,5}_reference` (`../realdata/README.md`, `../icml_exp/README.md`, `../icml_exp/PLAN4.md` §0).
- The datasets below were never downloaded by this project. A repository grep finds no reference to `ar_1_uniform`, `camera_walks`, `scm_5_red`, `scm_5_blue`, `scm_5_green`, `scm_4_green`, `scm_4_blue` or `scm_4_red` outside this folder (one hit is the text of an unrelated paper in `lit2/`). PLAN4 explicitly says the interventional `scm_*_<target>` sets were "not used" and the camera image sets were "not used".
- **Prior exposure to the device is substantial and is disclosed.** The same physical light tunnel produced all LT data analysed before, and E1 found the sensors respond strongly and almost instantly to the colour actuators. The development check in §1.4 used study-3 data from the same device.

### 1.2 Datasets (Causal Chambers, Gamella, Peters, Bühlmann 2025; CC BY 4.0)

| role | archive | member | rows (from generator) | zip CRC-32 (central directory) |
|---|---|---|---|---|
| **H1 primary (time series)** | `lt_camera_walks_v1.zip` | `ar_1_uniform_ref.csv` | 30,000 | b153930d |
| H1 | same | `ar_1_uniform_pol_1_90.csv` | 30,000 | 894684f7 |
| H1 | same | `ar_1_uniform_pol_12_30.csv` | 30,000 | dcab420f |
| **H2 co-primary (actuator collider R → G ← B, snapshot)** | `lt_camera_v1.zip` | `scm_5_red.csv` | 10,000 | a3d5a406 |
| H2 | same | `scm_5_blue.csv` | 10,000 | 08eb6c50 |
| S4 secondary (multi-source after intervention) | same | `scm_5_green.csv`, `scm_4_green.csv`, `scm_4_blue.csv` | 10,000 each | 158bdeb3, de1c4f3d, efc3eb05 |
| S4 control (single source) | same | `scm_4_red.csv` | 10,000 | 9fdbe1ed |

- The CRC-32 values come from the remote zip's central directory, read before this commit without reading any data bytes. **SHA-256 values are recorded on first download (`fetch_b3.py` → `results/fetch_report.json`) before any statistic is computed.** The download is an HTTP range fetch of the single csv member (`../icml_exp/remote_zip.py`, TLS verification on). `zipfile` checks the CRC-32 on extraction.
- Raw data go to the session scratchpad (`.../scratchpad/b3data/`) and are not committed.

### 1.3 Design facts (generators, read; no data)
- **ar_1_uniform_\*** (`lt_camera_walks_v1/generators/actuators_ar.py`; camera configuration).
  - R, G, B, θ1 and θ2 are five independent stationary AR(1) series. Each has uniform innovations, its own seed (100·k + i + 1728296067) and φ = 0.5, 0.6, 0.7, 0.8, 0.9. Each series is min-max scaled, then mapped to μ ± Δ: (128, 120) for the colours; (0, 30) for θ in `ref`; θ2 at μ = 90 in `pol_1_90`; θ1, θ2 at μ = 30 in `pol_12_30`.
  - The dataset README says θ1 is shifted in `pol_1_90` and Δ = 20; the generator says θ2 and Δ = 30. The design check in `fetch_b3.py` regenerates the protocol and compares it with the recorded inputs; neither choice changes the graph.
  - Every other manipulable variable is constant. There is **no edge among the actuators**, so every actuator is its own source SCC.
- **scm_5_\* and scm_4_\*** (`lt_camera_v1/generators/scm_{4,5}.py`).
  - Linear SCMs among R, G, B with truncated-normal noise; the polarisers are fixed at 0.
  - scm_5 has R → G ← B, scm_4 has R → G → B. Experiment `scm_k_<target>` removes the edges into the target.
  - Graphs:
    - scm_5_red and scm_5_blue: R → G ← B (collider, sources R and B);
    - scm_5_green: no actuator edge (three sources);
    - scm_4_green: G → B, R isolated (sources R, G);
    - scm_4_blue: R → G, B isolated (sources R, B);
    - scm_4_red: chain R → G → B (one source; the control).
- **Sensor edges** (causalchamber 0.2.8 `ground_truth`, `_edges_lt_standard`; the camera configuration adds only the image node, which is not a column): R, G, B → ir_1, vis_1, ir_2, vis_2, ir_3, vis_3, current (21 edges); θ1, θ2 → ir_3, vis_3 (Malus, 4 edges); θk → angle_k. Constant parameters are dropped.

### 1.4 Done before this commit (complete list)
1. Repository exposure check (§1.1).
2. Zip central-directory listings with sizes and CRC-32 (§1.2), dataset READMEs, generators, `variables.csv`.
3. **Graph-only predictions** (`graphs_b3.py`, `predict_b3.py` → `results/predict_b3_results.txt`).
4. **Development check on study-3 data only** (`dev_b3.py` → `results/dev_b3.log`). It computed per-edge correlations and detection of the 21 colour edges and the 3 colour-colour pairs on `lt_walks_v1/actuators_white.csv`, which study 3 had already analysed, with this plan's estimator on 3,000-row segments.
   - Result: all 21 colour edges were detected in 6/6 segments (median |r| from 0.12, red–vis_3, to 0.86); the colour–colour pairs had |r| ≤ 0.014 and were never detected.
   - **This means the outcome of H1 and H2 on D\* was expected before the test.** The test is discriminating with respect to the competitors, but it is not a high-risk test of the theorem on this device (see §7).
5. A smoke test of `test_b3.py` on **synthetic** csv files written by a throwaway script in the scratchpad (not Causal Chambers data), to check that the code runs.

No byte of any csv member in §1.2 has been read.

## 2. Graphs, predictors, instances

### 2.1 Relatives graph and scored edges
- Scored edges: the design graph **G_D** (all edges in §1.3, induced in S).
- **H1 (ar_1).** The relatives graph is **G_lin = G_D minus the 4 Malus edges**, as in E1.
  - Reason: THEORY_ADD Lemma B1-B3. Under a design symmetric under θ → −θ about the polariser zero (or about μ for both polarisers, or with θ2 − θ1 symmetric about 90), Malus' cos²(θ1 − θ2) is even, so it has zero covariance with θk. The AR designs are symmetric only up to the min-max scaling and the rounding, so the lemma is approximate here.
  - **The primary analysis does not depend on this choice.** Graph-only (`predict_b3_results.txt`): on every D\* instance the theorem gives the same value on G_lin and on G_D (0 disagreements of `theory_GD` within D\*), and both tie rules give ½ on either graph. G_lin matters only in secondary analyses S1, S2 and S5.
  - G_lin has three weak components: the colour block {R, G, B, 7 sensors} with **three source SCCs**, {θ1, angle_1} and {θ2, angle_2}.
- **H2, S4 (scm).** The relatives graph is G_D. It has one weak component, with 2 (scm_5_red, scm_5_blue, scm_4_green, scm_4_blue), 3 (scm_5_green) or 1 (scm_4_red) source SCCs.

### 2.2 Predictors (each gives 1, ½ or 0 for an induced scored edge (u, v) on node set S)
- **theory**: THEORY.md Section-6 rule on the relatives graph. ρ(v) = #{w ∈ S : v, w have a common ancestor in the full graph}, value 1/½/0 for ρ(u) <, =, > ρ(v). This is the rule used in studies 3 and 4.
- The **eight old naive predictors**, unchanged from E1 (`../icml_exp/graphs4.py`) and computed on the same relatives graph as the theory, as in E1: `sensor_indicator`, `n_anc_full`, `n_anc_S`, `indeg_full`, `indeg_S`, `depth_full`, `depth_S`, `all_one`.
- **const_half**: every edge ½.
- **wc_rule** (weak-component-size rule): ½ iff u and v lie in the same weak component of the full relatives graph; otherwise compare |C(u) ∩ S| with |C(v) ∩ S|.
- **theory_GD**: the theory rule applied to the published design graph G_D (identical to `theory` for H2 and S4).

Predictors that make identical predictions on the rows of an analysis are merged into one test, so Holm counts distinct predictors.

### 2.3 Units, node sets, estimator
- **H1.** Each ar_1 experiment is cut into 10 consecutive segments of 3,000 rows, giving 30 units (unit j = 10·e + s, with experiments e in the order ref, pol_1_90, pol_12_30).
  - Levels, standardised per unit. The AR(1) inputs are stationary, with φ ≤ 0.9.
  - Nodes (14): red, green, blue, pol_1, pol_2, ir_1, vis_1, ir_2, vis_2, ir_3, vis_3, current, angle_1, angle_2.
- **H2 and S4.** Each scm experiment is cut into 10 segments of 1,000 rows. Nodes (10): red, green, blue and the 7 colour sensors. The angle sensors have no random ancestor and are not nodes, as in E1.
- **Estimator: E1's, unchanged.** The PLAN3 effective-n Fisher-z test with the Bartlett factor, K = 50 (H1) or K = 20 (H2, S4), lag L = 0, α = 0.01, and Bonferroni over the pairs of each node set. The observed count is ρ̂(v) = 1 + #declared partners in S, and o_e ∈ {0, ½, 1}.
- **Node sets per unit.** The full set, plus 10 subgraphs of size 5 and 10 of size 8. Each subgraph contains at least one scored edge. They are drawn by `graphs4.sample_subgraphs` with seed 10·(base + j) + d, where base is 9900 (ar_1), 9940 / 9950 / 9960 / 9970 / 9980 / 9990 (scm_5_red / scm_5_blue / scm_5_green / scm_4_green / scm_4_blue / scm_4_red), and j is the unit index within the experiment (0..29 for ar_1, 0..9 otherwise).
  - A node constant in a unit is removed, and subgraphs are then drawn among the remaining nodes (reported). Rows with missing values are dropped (reported).

### 2.4 Graph-only power (`results/predict_b3_results.txt`, assuming no node is removed)

| analysis | instances | D\* = {theory ≠ wc_rule} | share | units with D\* |
|---|---|---|---|---|
| H1 ar_1 (30 units) | 4,238 | 2,361 | 0.557 | 30/30 |
| H2 scm_5_red + scm_5_blue (20 units) | 4,422 | 2,078 | 0.470 | 20/20 |
| S4 scm_5_green, scm_4_green, scm_4_blue (30 units) | 6,328 | 4,890 | 0.773 | 30/30 |
| S4 control scm_4_red | 2,194 | 0 | 0 | — |

- On D\* the theory predicts 1 on every instance, and wc_rule and const_half predict ½.
- **On H1's D\*, all eight old predictors and theory_GD agree with the theory**, so they are untestable there. On H2's D\*, `sensor_indicator` disagrees on 264 instances: the actuator-collider edges R → G and B → G with the other source in S, where it predicts ½.
- On the instances where the theory predicts ½ (1,263 in H1, 2,344 in H2), the naive rules predict 1: all eight on every one in H1; in H2 `n_anc_*`, `indeg_*`, `depth_*` and `all_one` on all 2,344, and `sensor_indicator` on 2,221.
- With one-sided sign flips over 30 or 20 units, the smallest attainable p is about 10⁻⁵ (Monte Carlo, 100,000 flips), far below the Holm-adjusted levels.

## 3. Metric and test (as E1)
- Loss: |o_e − p_e|.
- For a competitor C and a set of instances, take the instances where theory ≠ C. Then Δ_{C,u} = mean loss(theory) − mean loss(C) within unit u.
- **Test.** One-sided sign-flip test of mean_u Δ_{C,u} < 0: exact if there are ≤ 16 units, otherwise Monte Carlo with 100,000 flips, seed 1. Holm over the distinct testable competitors of that analysis.
- **Family-wise α = 0.025 per hypothesis** (Bonferroni over the two hypotheses H1 and H2).
- Pooled Δ_C with a unit-cluster bootstrap 95% CI (2,000 resamples, seed 0); the number of units in which the theory is better; and the reverse test (competitor better), also Holm-corrected.
- On D\* (theory 1, tie rules ½), Δ against a tie rule equals 0.5 − P(o = 1). So **Δ ≤ −0.10 means the estimated counts are strictly sorted in the predicted direction on at least 60% of D\* instances.**
- Both estimator failure modes push o towards ½, which favours the tie rules on D\*:
  - no power (nothing declared): all counts equal;
  - saturation (everything declared): all counts equal |S|.
  - So, unlike E1, a pass on D\* cannot come from power-induced ties.

## 4. Hypotheses and pre-registered criteria

**H1 (primary; the B3 claim; real time series).** On H1's D\*, the theory beats every testable competitor: pooled Δ_C ≤ −0.10 **and** Holm-adjusted one-sided p ≤ 0.025. The testable competitors are those with at least one disagreeing instance in D\*; by §2.4 these are const_half and wc_rule, which are merged because they are identical on D\*.

**H2 (co-primary; foregrounded scm_5 collider; real snapshot data).** The same criterion on D\* of scm_5_red + scm_5_blue (20 units). By §2.4 the testable competitors are const_half (= wc_rule) and sensor_indicator.
- Reported in addition, without a separate verdict: on the collider edges R → G and B → G, the share of o = 1 when the other source is in S (theory 1) and the share of o = ½ when it is not (theory ½).

**Prerequisites (reported; they gate only the interpretation of a failure).** On the full node sets, pooled over units:
- the share of relatives-graph non-relative pairs declared dependent is < 0.5;
- the share of relative pairs detected is ≥ 0.8 (E1's thresholds).

**Secondary analyses (pre-specified; criteria fixed; they qualify the claim, see kill rules):**
- **S1 (the theory's ties against the naive sort rules).** Take the instances where the theory predicts ½ **and** the scored edge's own endpoint pair is declared dependent (E1c analogue: a tie there is not a non-detection). Test the theory against every competitor that disagrees: pass iff Δ < 0 and Holm p ≤ 0.025 for all.
  - The unrestricted version (all theory-½ instances) is reported without a verdict.
- **S2 (all instances, E1-style).** All 11 competitors, Δ ≤ −0.10 and Holm p ≤ 0.025. Reported, no verdict.
- **S3 (within-edge switch, H1 only).** For the 21 colour-source → colour-sensor edges, take per unit the mean o when a second colour source is in S minus the mean o when none is. Under G_lin this is exactly the theorem's 1-versus-½ switch, predicted +0.5. Pass iff the mean is ≥ 0.25 and the one-sided sign-flip p ≤ 0.025. Reported for the other sets as well, where it is not an exact image of the theorem.
- **S4.** H1's criteria applied to the pooled multi-source interventional sets (scm_5_green, scm_4_green, scm_4_blue; 30 units), plus per-set results. scm_4_red (one source) is a control, where theory = tie rules on every instance; only S1 and S2 are reported for it.
- **S5 (refinement check, H1).** theory (G_lin) against theory_GD on the instances where they disagree. These are the Malus-dependent instances, and they lie outside D\*. Reported; no verdict.
- **S6 (mechanism check on D\*).** Among the D\* instances observed as 1, the share where the head v is declared dependent with an observed source in Src(v) \ Src(u) and the tail u with none of them. This says whether the predicted mechanism produced the sorting.
- Also reported: the per-edge observed values split by second source present or absent; the per-experiment H1 results; median n_eff; removed nodes; the regenerated-protocol design check.

## 5. Kill rules and verdict logic (fixed)
- **K1.** If H1 fails and its prerequisites fail, H1 is reported as uninformative (estimator failure), neither support nor refutation. If H1 fails while the prerequisites pass, it is a **refutation of the source-SCC prediction on these data**, and the paper must say so.
- **K2.** If any competitor is significantly better than the theory on D\* (reverse Holm p ≤ 0.025), the paper must state that this trivial rule out-predicts the theorem on these data.
- **K3.** If H1 passes but S1 fails, the paper may claim only that "the source-SCC rule beats tie rules where it predicts strict order", not that it beats every competitor.
- **K4.** If S6 < 0.5 for H1 (or H2), the D\* pass of that hypothesis is not attributed to the source-SCC mechanism.
- **Claims:**
  - "Thm 1.9 / Cor 1.10 separated from the trivial tie rules on held-out real time series" iff H1 passes and K2 and K4 do not trigger.
  - "Separated from tie rules and naive sort rules" (full separation) iff, in addition, S1 passes for H1.
  - The collider claim is the same, with H2 in place of H1.
  - Nothing else counts: not S2, S4 or S5, and not the per-experiment results.

## 6. Honest predictions (written before the data)
- **H1 and H2: pass, with P(o = 1 | D\*) ≥ 0.9 expected.** This follows from the development check (§1.4): all colour edges are strongly detected on this device and independent colours are never declared dependent.
  - The main risks are a weak colour → sensor edge at n_eff ≈ 1,000 to 3,000 (B → vis_3? R → vis_3, r ≈ 0.12 in development), and a slow common drift of the light source making R, G and B look dependent.
- **S1 (H1): uncertain.**
  - Ties on theory-½ instances need ρ̂(colour) = ρ̂(sensor) when no second colour is in S. That requires every sensor–sensor pair in S to be detected (sensor pairs share latent colour sources, so they should be) and no detected Malus residual between ir_3/vis_3 and a polariser or angle node in S.
  - Under the approximate symmetry, a Malus residual is possible at n_eff in the thousands, and it would break ties at ir_3 and vis_3. The citris E1 observations (Malus pairs never detected) suggest it will not.
- **S1 (H2):** pass expected (in scm_5_reference all 2,238 E1 instances were exact).
- **S3:** pass. **S5:** theory (G_lin) expected to do better than theory_GD, as on citris; uncertain because of the approximate symmetry. **S6:** ≥ 0.9.

## 7. What this test can and cannot show (stated in advance)
- It removes the specific objection that **the theorem's success is indistinguishable from "½ iff same weak component"**. Here the two disagree on 56% (H1) and 47% (H2) of instances. Any estimator failure favours the tie rule, and the old naive predictors are kept as well, so the theorem has to be right in both directions.
- It does **not** make the theorem a risky prediction on this device. The theorem is a deterministic function of the graph. Given the published graph and faithful, well-powered dependence detection, its predictions follow. What the test checks is that the estimated relatives counts on real held-out data follow the source-SCC structure rather than the weak-component structure. Development data from the same device already indicated they would.
- The H2 and S4 data are snapshots (i.i.d.). They test the source-SCC mechanism (FINITE Cor 1.6 analogue), not a time-series effect. H1 is a real stationary multivariate time series.

## 8. Code and outputs
- `fetch_b3.py` → `results/fetch_report.json` (SHA-256, rows, columns, design check).
- `test_b3.py` → `results/results_b3.json`, `results/test_b3.log`. The per-dataset caches are not committed; they are regenerable.
- Reproduce: `OMP_NUM_THREADS=1 python3 -I fetch_b3.py <data_dir> results/fetch_report.json`, then `OMP_NUM_THREADS=1 python3 -I test_b3.py <data_dir> results`.
- No threshold, unit, seed, K, node set, graph or competitor changes after results are seen. Any deviation, including code fixes, is listed in `README.md`.
