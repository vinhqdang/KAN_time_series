# PLAN4 (pre-registration): a discriminating real-data test of Thm 1.9 (E1) and a lagged relatives estimator (E2)

This plan builds on `../THEORY.md` (Thm 1.9, Cor 1.10, Prop 2.8, Cor 2.9), `../finite/FINITE.md` (F1 to F3 and the erratum, §8), `../realdata/PLAN3.md` and `../realdata/README.md`. The new theory needed here (lagged relatives, and a linear-zero lemma for one experiment) is in `THEORY_ADD.md`.

**This file is committed and pushed before any correlation, autocorrelation or other dependence statistic is computed on any dataset used for a verdict below.**

What was done before this commit:
- Downloads, and SHA-256 of every file used (listed below).
- Listing of the remote zip archives.
- Column names, row counts, missing-value counts and sampling intervals. Distinct-value counts were taken for experimenter-set inputs only.
- Regeneration of the citris_1 input protocol from its published generator. It matches the recorded inputs exactly: `check_citris_design.py` → `check_citris_design_results.txt`. This step reads input columns only.
- Graph-only predictions (`graphs4.py`, `predict4.py` → `predict4_results.txt`).
- **Development runs** of the analysis code on the PLAN3 data only (WT actuators_random_walk_2..16; `dev4.*`, `dev4_lags.*`). These data were already analysed and are not used for any verdict here. The development section below discloses everything computed in those runs.

No sensor column of any held-out dataset has been summarised in any way.

## 0. Data and provenance (Causal Chambers, Gamella, Peters, Bühlmann 2025; CC BY 4.0; downloaded 2026-11-04 through the session proxy, TLS verification on)

| file | source | rows | SHA-256 (first 16) | used for |
|---|---|---|---|---|
| `citris_1.csv` | `lt_crl_benchmark_v1.zip` (2.76 GB, images included). This member alone was fetched by HTTP range requests (`remote_zip.py`), and the zip CRC-32 was checked on extraction | 100,000 | 2291eb10b18b1740 | E1 primary; E2 secondary |
| `buchholz_1_obs.csv` | same zip | 10,000 | 4844a04cef6a8097 | E1 secondary (i.i.d.) |
| `scm_{1,2,4,5}_reference.csv` | `lt_camera_v1.zip` (4.6 GB), range-fetched the same way | 5,000 / 10,000 / 10,000 / 10,000 | 16d7747601cae1b0 / d6c652559a3770e9 / df72c6dbb1c5e15a / 827d045e91dc58a1 | E1 secondary (i.i.d.) |
| `regime_jumps_{single,multi}.csv` | `wt_walks_v1.zip` (md5 19bb4e92…, matches `directory.yaml`). These are in the same archive as PLAN3's data, but PLAN3 did not analyse them | 320,000 each | ae00d09adddde4db / 6e5e28d969067263 | E2 primary; E1 tertiary |

Raw data are not committed. They are kept in `/tmp/claude-0/dl4/{crl,cam}` and `/tmp/claude-0/dl3/cc/wt_walks_v1/`.

## 1. Stage A for E1: which Causal Chambers experiments have multi-layer ground truth?

The published chamber graphs (causalchamber 0.2.8 `ground_truth`) are bipartite actuators → sensors in every configuration except `pressure-control`. In that configuration the controller adds pressure_downwind → load_in and pressure_downwind → load_out, giving a cycle.

Multi-layer structure among *measured* variables therefore arises in two ways:
- in experiments whose **generator makes some actuators functions of other actuators** (a structural causal model, SCM, among the inputs);
- in the controller configuration.

| candidate | ground truth beyond bipartite? | data | decision |
|---|---|---|---|
| **`lt_crl_benchmark_v1/citris_1`** | yes. The time-series SCM among the inputs is: red (reflected random walk) → green → blue, each step moving towards its parent; pol_1 (random walk) → pol_2, whose direction depends on sign(blue − red). The sensors form a third layer (R, G, B → ir, vis, current; pol → ir_3, vis_3, angle). The chain red → green → blue → ir_1 has length 3. green and blue are both effects and causes | 100,000 steps, about 1.8 s apart, no missing values. The inputs equal the generator output exactly | **E1 primary** (the only real time series with multi-layer ground truth found) |
| `lt_crl_benchmark_v1/buchholz_1_obs` | yes. A linear Gaussian SCM among the actuators (sempler convention, read in lganm.py): blue → green → red, pol_1 → red, green → pol_2, blue → pol_2; sensors as a third layer | 10,000 i.i.d. | E1 secondary (snapshot) |
| `lt_camera_v1/scm_1_reference`, `scm_2_reference` | yes. Generator A matrix: red → green → blue, red → pol_1, pol_2 → green, pol_2 → blue | 5,000 / 10,000 i.i.d. | E1 secondary (snapshot) |
| `lt_camera_v1/scm_4_reference`, `scm_5_reference` | chain red → green → blue (scm_4) and collider red → green ← blue (scm_5), with the polarisers fixed | 10,000 i.i.d. each | E1 secondary (snapshot) |
| interventional `scm_*_<target>`, `buchholz_1_<target>` | the graph is mutilated by a do-intervention | | not used (one environment per graph is enough; interventions change the graph) |
| `*_synth_det` | | synthetic | rejected (not real) |
| `wt_pressure_control_v1` | yes (cycle) | the only experiment, `hatch_0`, keeps every input except the controlled loads constant, so there is no independent random source and Cor 1.10(i) predicts ½ everywhere trivially | rejected |
| `lt_spice_v1` (Light Tunnel Mk2, `linked_leds`) | possibly (linked LEDs). Its graph is not in causalchamber 0.2.8; only a configuration PDF exists | 40 MB, released 2026-04 | not examined beyond the README. Its ground truth would have had to be transcribed from a PDF; documented as a limitation |
| `wt_regime_jumps_*` | bipartite (the hatch is constant within a regime) | 64 regimes × 10,000 | E2 primary. E1 tertiary: disagreements there arise only from latent-node ties |
| `loads_hatch_mix_*`, `color_mix` | bipartite; inputs are deterministic periodic waves, so time itself is a common driver of the inputs | | rejected |
| `smooth_polarizers`, `wt_changepoints_v1`, `wt_intake_impulse_v1`, `*_validate_v1`, `*_test_v1`, `lt_interventions_standard_v1`, `lt_malus_v1`, `wt_bernoulli_v1`, camera image sets | bipartite, a single varying input, or block-wise non-stationary designs | | not used |

**Amount of disagreement** (`predict4_results.txt`, graph-only): theory and competitors disagree on 1,067 to 1,205 of the 1,342 subgraph edge instances of citris_1 (G_lin; 25 to 28 of the 32 full-set edges). On the i.i.d. sets they disagree on 447 to 2,037 instances per experiment, and on the WT regime subgraphs on 2,691 of 5,198. So the data allow the theory and the naive rules to disagree. The question is whether the estimated counts side with the theory.

## 2. Development runs on PLAN3 data (disclosed; not evidence)

`dev4.log`, `dev4.json` (from `test4.py dev`) and `dev4_lags.py` → `dev4_lags.log` used WT actuators_random_walk_2..16 with the code in `test4.py`.

- **Code check.** With L = 0, the code reproduces PLAN3 exactly: non-relatives declared 0.039, relatives detected 0.396, and P9 = 0.2665 with PLAN3's subgraph seeds (10·(9000+run)+d). With seeds 10·(9000+j)+d (j = 0..14 instead of the run number), the same data give Spearman 0.338. **The P9 statistic therefore moves by about 0.07 with the subgraph draw alone.** This is reported as a fact about P9's variability. It does not change P9's verdict, which stays "failed".
- **E1 on WT dev (bipartite; disagreement only from latent-node ties).**
  - The theory's loss is 0.206 against 0.455 for every competitor (Δ = −0.249, 15/15 runs).
  - On the detected-edge subset, Δ = −0.235.
  - Edges predicted ½ were observed exactly ½ in 59% of instances.
- **E2 on WT dev (relatives detected / non-relatives declared).**

  | setting | relatives detected | non-relatives declared |
  |---|---|---|
  | D = 10, L = 0 | 0.396 | 0.039 |
  | D = 10, L = 1 | 0.393 | 0.037 |
  | D = 10, L = 3 | 0.389 | 0.036 |
  | D = 3, L = 0 | 0.346 | 0.038 |
  | D = 3, L = 10 | 0.382 | 0.039 |
  | D = 1, L = 0 | 0.179 | 0.025 |
  | D = 1, L = 10 | 0.246 | 0.028 |
  | D = 1, L = 30 | 0.241 | 0.028 |

  - At D = 1, L = 30: accuracy on the edges predicted 1 is 0.632 and Spearman is 0.238. At D = 1, L = 0: 0.709 and 0.138.
  - At D = 10, L = 3: accuracy 0.651 and Spearman 0.337.
  - So lags help at fine resolution (D = 1) and do nothing once block means over about 3 s are taken. No setting exceeds about 0.40 detection on dev. Development therefore indicates that most of the 16 undetectable WT edges are not lag problems.
- **What the development runs changed in this plan:**
  - the choice of the primary E2 configuration: D = 1, where lags matter in principle;
  - the honest predictions in §6;
  - the decision to add the power safeguards to E1 (§3.5), after seeing that ties can come from non-detection.

  No threshold was tuned on dev data. All thresholds are copied from PLAN3 or fixed a priori below.

## 3. E1: discriminating test

### 3.1 Theory prediction (fixed)
- The THEORY.md Section 6 rule: for node set S, ρ(v) = #{w ∈ S : v and w related in the full graph of the experiment's random variables}.
- An edge (u, v) has value 1, ½ or 0 according as ρ(u) <, = or > ρ(v).
- Scored edges are the induced true causal edges (design graph G_D).
- **Primary graph for relatives: citris_1 G_lin** (THEORY_ADD Cor B3: the six design edges with provably zero linear cross-covariance at all lags are removed; the scored edges stay G_D). Secondary: G_D itself.
- For the i.i.d. and WT sets, G_lin = G_D (no such lemma applies).

### 3.2 Competitors (fixed)
Each competitor is a node score c, with edge value 1, ½ or 0 for c(u) <, =, > c(v). Every competitor uses the same graph as the theory in that analysis.
1. `sensor_indicator`: 1 for sensors, 0 for experimenter-set inputs. This is the naive explanation named in PLAN3's caveat.
2. `n_anc_full`: number of ancestors in the full graph.
3. `n_anc_S`: number of ancestors within S.
4. `indeg_full`: in-degree in the full graph.
5. `indeg_S`: in-degree within the induced subgraph.
6. `depth_full`: longest directed path ending at the node, in the full graph.
7. `depth_S`: the same within the induced subgraph.
8. `all_one`: every edge 1. This is the finite unrolled-DAG limit of THEORY.md Thm 1.5 for crossing edges, i.e. Reisach et al.'s G_T.

### 3.3 Data, units, estimator (E1 primary: citris_1)
- **Nodes (14).** red, green, blue, pol_1, pol_2, ir_1, vis_1, ir_2, vis_2, ir_3, vis_3, current, angle_1, angle_2. The flag column is excluded (THEORY_ADD Rem B4). All other columns are constants.
- **Units.** 10 consecutive segments of 10,000 rows. The data are levels, standardised per unit. The inputs are stationary by design: the resets mix at rate 0.13 per step.
- **Estimator.** The PLAN3 effective-n Fisher z with K = 50 (levels are persistent), L = 0, α = 0.01 and Bonferroni over the pairs of the node set.
- **Node sets per unit.** The full set, plus 10 subgraphs of size 5 and 10 of size 8 (`graphs4.sample_subgraphs`, seed 10·(9300 + j) + d, j = unit index). This gives 210 node sets and 1,342 + 10 × 32 edge instances.
- A node constant in a unit is removed from that unit, and subgraphs are then drawn among the remaining nodes (reported). Rows with missing values are dropped (none exist).

### 3.4 Metric and test (fixed)
- For each edge instance e: the observed value o_e ∈ {0, ½, 1} from the estimated counts ρ̂ = 1 + number of declared partners in S, and the predicted values p_e^T (theory) and p_e^C (competitor C).
- Loss: |o_e − p_e|.
- Orientation accuracy alone (the mean of o_e) does not depend on the predictor, so the comparison is on agreement between predicted and observed edge values.
- For each competitor C, take the disagreement instances (p^T ≠ p^C). For each unit u: Δ_{C,u} = mean loss(theory) − mean loss(C) over that unit's disagreement instances.
- **Test.** One-sided exact sign-flip permutation test of mean_u Δ_{C,u} < 0 over the 10 units (1,024 flips). Holm correction over the 8 competitors at family-wise α = 0.05.
- Reported for each competitor: the pooled Δ_C over all disagreement instances, with a cluster-bootstrap 95% CI over units (2,000 resamples, seed 0); the number of units in which the theory is better; and the reverse test (competitor better), also Holm-corrected.

### 3.5 Thresholds, safeguards and kill rules (E1 primary)
- **E1a (prerequisites; full node sets, pooled over units).**
  - The share of G_lin non-relative pairs (40 per unit) declared dependent is < 0.5 (no saturation, as P7).
  - The share of G_lin relative pairs (51 per unit) detected is ≥ 0.8. This is the power condition: ties must not come from non-detection.
- **E1b (main).** For every one of the 8 competitors: pooled Δ_C ≤ −0.10 **and** Holm-adjusted one-sided p ≤ 0.05.
- **E1c (robustness against power-induced ties).** The same test restricted to disagreement instances whose own endpoint pair is declared dependent. For every competitor: Δ_C < 0 with Holm-adjusted p ≤ 0.05. A competitor with no disagreement instance in this subset is reported as untestable, not as failed.
- Also reported, without a verdict:
  - the "all estimated counts equal" baseline Δ (the value of Δ if the estimator declared nothing);
  - Δ on the theory-strict subset (theory value 0 or 1);
  - per-edge observed values;
  - exact-match rates;
  - P9-style Spearman(predicted mean edge value, observed accuracy) over subgraphs for every predictor whose prediction varies.
- **"Thm 1.9 separated from the naive explanations on real data" is claimed iff E1a, E1b and E1c all pass on citris_1.** Nothing else counts: not the secondary sets, not G_D, not the WT tertiary analysis.
- **Kill rules.**
  - K1: if E1a fails, E1 is uninterpretable on citris_1 (saturated or underpowered), and no separation is claimed.
  - K2: if E1b fails for `sensor_indicator`, or if any competitor is significantly better than the theory (reverse Holm p ≤ 0.05), the paper must not claim that Thm 1.9 explains real-data relatives-sortability better than that competitor. This is reported as a negative result.
  - K3: if E1b passes but E1c fails, the advantage is attributed to power-induced ties, and no separation is claimed.

### 3.6 Secondary E1 analyses (reported; no effect on the verdict)
- **S-E1-1.** citris_1 with G_D relatives, the theory as stated without Lemma B3. Predicted to do worse than G_lin on the six zero-covariance edges.
- **S-E1-2.** citris_1 with the lagged estimator, L = 5.
- **S-E1-3.** The i.i.d. SCM sets: buchholz_1_obs, scm_1_reference, scm_2_reference, scm_4_reference and scm_5_reference.
  - Units: consecutive segments of 1,000 rows, 45 units.
  - Estimator: K = 20, L = 0.
  - Subgraph seeds: 10·(base + j) + d with base 9400, 9500, 9600, 9700 and 9800 respectively.
  - Same criteria as E1a to E1c, both pooled (a Monte Carlo sign-flip test with 100,000 flips, seed 1, over the 45 units) and per experiment.
  - This is the snapshot analogue (FINITE Cor 1.6). It tests the source-SCC mechanism, not a time-series effect.
- **S-E1-4 (tertiary).** WT regime_jumps with the E2 preprocessings.
  - The graph is bipartite, so every disagreement is a latent-node tie (theory ½ against all competitors 1).
  - It is flagged in advance as power-confounded, since relatives detection is about 0.4 on dev.

## 4. E2: lagged relatives estimator

### 4.1 Definition and theory link
- The estimator is the one defined in THEORY_ADD §A: max over |h| ≤ L of |atanh r̂_uv(h)|·√(n̂_eff,uv(h) − 3), with n̂_eff,uv(h) = (n − |h|)/max(1, 1 + 2Σ_{k≤K} r̂_u(k)r̂_v(k)), compared with z_c(L) = Φ^{-1}(1 − α/(2m(2L+1))).
- **Why it targets the same population relatives sets.** THEORY_ADD Prop A1:
  - under (Z), non-relatives have γ_uv(h) = 0 at every lag, so lags add no false target;
  - under lag-0 faithfulness the sets coincide for every L;
  - under the weaker lag-L faithfulness the lagged sets equal rel_S while the lag-0 sets may not.
  - So Thm 1.9's predicted s̃ is unchanged.
- **What changes in F1** (THEORY_ADD Prop A3):
  - ρ_min becomes ρ*_min,L = min over relatives of max_{|h|≤L}|ρ(h)| (a gain);
  - the failure-probability union bound and the Bonferroni count gain a factor 2L+1: z_c goes from 3.90 to 4.35 (L = 3) or 4.80 (L = 30) for 105 pairs (a cost);
  - the null Bartlett factor is the same at every lag (Lemma A2).
- **Framing.** E2 is not a re-analysis of P9. P9 failed on the PLAN3 data, and that verdict is final. E2 asks whether the lagged estimator improves detection on held-out data, and whether the PLAN3-style criteria hold there.

### 4.2 Held-out data and preprocessing (E2 primary)
- **Data.** WT `regime_jumps_single` and `regime_jumps_multi`: 2 × 32 regimes = 64 units.
  - Within a regime, load_in, load_out, pot_1 and pot_2 follow independent mirrored random walks (generator `regime_jumps.py`, read).
  - The hatch and the sensor parameters are constant within a regime (checked: one distinct value per regime). The hatch is 45 in 2 + 6 regimes and 0 otherwise.
- **Nodes (15).** load_in, load_out, pot_1, pot_2 and the 11 sensors (pressure_ambient included, as in PLAN3; it has no random ancestor).
- **Graph.** PLAN3's WT edges without the hatch: 20 edges. 19 are predicted 1; pot_1 → signal_1 is predicted ½. There are 54 non-relative and 51 relative pairs.
- **Preprocessing.**
  - Drop the first 100 rows of each regime (about 30 s of transient after a regime change).
  - **Primary: D = 1** (first differences of the raw rows, n = 9,899 per unit).
  - Secondary: PLAN3's D = 10 block means then differences (n = 989).
  - Standardise per unit, with K = 20.
- **Configurations.** Primary: L = 30 (about 9 s at 0.3 s per row) against L = 0 at D = 1. Secondary: L = 3 (about 9 s) against L = 0 at D = 10.
- **Subgraphs.** 10 of size 5 and 10 of size 8 per unit, seed 10·(9200 + j) + d (j = 0..31 for single, 32..63 for multi). This gives 1,280 subgraphs.
- Constant nodes in a unit are removed and reported (e.g. a tachometer at res = 0 could read constant).

### 4.3 Pre-registered tests (E2 primary: D = 1, lagged L = 30)
- **Q7.** The pooled share of non-relative pairs (54 × 64) declared dependent is **< 0.5** (cluster-bootstrap CI reported).
- **Q8.** The accuracy on the full-set edges predicted 1 (19 × 64 = 1,216 instances) is **> 0.5, with cluster-bootstrap 95% CI lower bound > 0.5**. pot_1 → signal_1 is reported separately (predicted ½).
- **Q9.** **Spearman(s̃, estimated accuracy) ≥ 0.3** over the 1,280 subgraphs. The p-value and a cluster-bootstrap CI over units are reported.
- **Q10 (detection).** The share of the 51 relative pairs detected, lagged minus lag-0 on the same units, is **> 0, with paired cluster-bootstrap 95% CI lower bound > 0**. The same difference for non-relatives (the cost) is reported.
- **E2 success** iff Q7, Q8, Q9 and Q10 all pass at the primary configuration.
- **Kill rules.**
  - If Q7 fails, the lagged estimator saturates and Q8, Q9 are not interpreted as support.
  - If Q10 fails, lags do not address F1 detection on these data, and the paper does not recommend the lagged estimator.
- **Reported, no verdict:**
  - the same Q7 to Q10 at the secondary configuration (D = 10, L = 3);
  - the lag-0 estimator's Q7 to Q9 at D = 1 and D = 10 on these held-out regimes. This is a held-out replication of the PLAN3 estimator. It is reported as such and does not modify P9;
  - per-edge detection rates for lag 0 against lagged;
  - citris_1 with L = 5 against L = 0 (Q7 with G_lin non-relatives, and Q10).

## 5. Out-of-sample F-theory checks (fixed)
For every E2 configuration (D = 1: L = 0 and L = 30; D = 10: L = 0 and L = 3) and for citris_1 (L = 0 and L = 5):
- Split each unit into halves. From the first half, compute the per-lag detection probabilities for the second half (THEORY_ADD §A, last paragraph), and from them:
  - the predicted relatives detection, in the bracket [mean p_max, mean p_ind] (F1);
  - the predicted non-relatives declared (F2);
  - the predicted orientation accuracy on the edges predicted 1, by Monte Carlo of independent Bernoulli(p_ind) decisions (2,000 draws, seed 1).
- These are compared with the second-half test.
- **Check criterion (fixed).** The observed second-half relatives detection lies in [mean p_max − 0.03, mean p_ind + 0.03]. Reported as pass or fail per configuration. It is not part of the E1 or E2 verdicts.

## 6. Predictions written before the data (honest expectations)
- **E1 citris.**
  - Strong colour-block correlations should make detection nearly complete, giving exact ties inside the colour block.
  - I expect E1a and E1b to pass.
  - The main risks:
    1. a weak pol_1–pol_2 correlation (it does not break ties; see THEORY_ADD §B);
    2. a polariser zero-mark offset making ir_3 and vis_3 linearly related to the polariser block, which would break the ties of the 6 + 4 edges at ir_3 and vis_3;
    3. a slow common drift of the light source, which would create cross-block dependence (E1a would detect it).
  - The two anti-sorted predictions (red → pol_2 and blue → pol_2 = 0) are the riskiest.
- **E1 i.i.d.** I expect high power at n = 1,000 and a pass.
- **E2.** From development:
  - Q7 and Q8 should pass;
  - Q9 at D = 1 will probably fail (dev 0.24);
  - Q10 at D = 1 should pass (dev +0.06);
  - at D = 10, Q10 should fail (dev −0.007).
  - Lags are not expected to bring WT detection much above 0.4. The undetectable WT edges are mainly weak or nonlinear effects, not lagged ones.

## 7. Code and outputs
- `test4.py main` → `results4.json`, `test4.log`.
- Reproduce: `OMP_NUM_THREADS=1 python3 -I test4.py <this dir> main results4.json` (at most 2 cores).
- No threshold, unit, seed, K, L, D, node set or graph changes after results are seen. Any deviation will be listed in `README.md`.
