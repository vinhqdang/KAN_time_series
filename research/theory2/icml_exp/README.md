# ICML experiments E1 and E2: a discriminating real-data test of Thm 1.9, and a lagged relatives estimator

Pre-registration: `PLAN4.md`, commit **a0559ec**, pushed before any dependence statistic of any PLAN4 dataset was computed. Theory addendum: `THEORY_ADD.md`, same commit. It contains the lagged-relatives propositions A1 to A3 and the citris linear-zero lemma B1 to B3.

## Files

Code:
- `test4.py`: all pre-registered analyses.
- `graphs4.py`: graphs, the theory rule and the competitor predictors.
- `predict4.py`: graph-only predictions.
- `summarise4.py`: tables from `results4.json`.
- `explore4.py`: post hoc.
- `check_citris_design.py`: design check.
- `remote_zip.py`: fetches single CSV members of large zips by HTTP range requests, with TLS verification.
- `dev4_lags.py`: development runs.

Outputs:
- `results4.json`, `test4.log`, `summary4.txt`;
- `predict4_results.txt`, `check_citris_design_results.txt`;
- `dev4.json`, `dev4.log`, `dev4_lags.log` (development, PLAN3 data);
- `explore4.log` (post hoc).

Raw data are not committed. SHA-256 values are listed in PLAN4 §0 (prefixes) and in full in `../DATA.md`; `test4.py` finds the data through `../datapaths.py` (`THEORY2_DATA_ROOT` or `THEORY2_CC_*`).

Reproduce with `OMP_NUM_THREADS=1 python3 -I test4.py <this dir> main results4.json`. It takes about 2 minutes on one core.

## Datasets chosen and why

- **E1 primary: `lt_crl_benchmark_v1/citris_1`** (light tunnel, 100,000 steps, about 1.8 s apart).
  - This is the only real time series in the Causal Chambers repository whose ground truth has several layers among measured variables. The generator makes green follow red and blue follow green, so red → green → blue → sensors is a chain of length 3. pol_2 depends on pol_1 and on sign(blue − red).
  - The recorded inputs equal the regenerated generator output exactly.
  - Every published chamber graph (causalchamber 0.2.8) is bipartite except `pressure-control`. Its only experiment varies no independent input, so it was rejected.
- **E1 secondary: i.i.d. structural causal models among the actuators** (snapshots): `buchholz_1_obs`, `lt_camera_v1/scm_{1,2,4,5}_reference`. These include a 5-node DAG, a chain R → G → B and a collider R → G ← B.
- **E1 tertiary and E2 primary: WT `regime_jumps_single` and `regime_jumps_multi`** (64 regimes of 10,000 steps). They are in the same archive as PLAN3's data, but PLAN3 did not analyse them.
- **Development only** (PLAN4 §2): PLAN3's WT random-walk runs.

## E1: does Thm 1.9 beat the naive predictors?

Metric:
- For every scored edge instance (full node set plus 20 subgraphs per unit), take the observed edge value o ∈ {0, ½, 1} from the estimated relatives counts.
- Loss is |o − predicted|. Take the instances where the theory and a competitor disagree.
- Δ = loss(theory) − loss(competitor), compared with an exact sign-flip test over units and Holm over the 8 competitors.

### Primary: citris_1, linear-dependence graph G_lin, lag-0 PLAN3 estimator (K = 50), 10 units

| criterion | threshold | result | verdict |
|---|---|---|---|
| E1a non-relative pairs declared (40 per unit) | < 0.5 | **0.000** [0, 0] | pass |
| E1a relative pairs detected (51 per unit) | ≥ 0.8 | **0.969** [0.945, 0.992] | pass |
| E1b, all 8 competitors | Δ ≤ −0.10 and Holm p ≤ 0.05 | Δ from **−0.502** (sensor indicator, CI [−0.507, −0.496]) to **−0.533** (all-one, i.e. the finite-G_T limit); 10/10 units every time; Holm p = 0.0078 for each (8 × 1/1024, the smallest attainable) | **pass (8/8)** |
| E1c, detected-edge subset | Δ < 0 and Holm p ≤ 0.05 | Δ = −0.500 for all 8 (1,338 instances, all observed exactly ½), Holm p = 0.0078 | **pass (8/8)** |

- **Verdict: E1 passes on the primary dataset. Kill rules K1 to K3 are not triggered.**
- The theory's predicted edge value equals the observed one in **99.2%** of the 1,662 instances. The competitors match in 11% to 20%.
- The predictions split as follows:
  - **½** (1,365 instances): observed exactly ½ in 99.2%;
  - **1** (187 instances, polariser → ir_3 and vis_3): observed 1 in 99.5%;
  - **0**, the anti-sorted predictions red → pol_2 and blue → pol_2 (110 instances): observed 0 in 99.1%.
- This last group is the riskiest prediction: a true causal edge whose head has *fewer* relatives than its tail.
- On the theory-strict subset (theory 0 or 1), the theory also wins against every competitor (Δ −0.54 to −0.99). There, an estimator with no power would favour the competitors (the "all ties" baseline Δ is +0.44 to 0), so this part of the advantage cannot come from power-induced ties.
- P9-style Spearman over 200 subgraphs: **0.962** for the theory. The competitors reach 0.10 to 0.41.

### Secondary (no verdict)

| analysis | E1a | E1b | E1c | theory exact match |
|---|---|---|---|---|
| citris, design graph G_D (theorem applied without Lemma B3) | 0.000 / 0.659 | pass 8/8 (Δ −0.32 to −0.42) | pass | 0.714 |
| citris, G_lin, lagged estimator L = 5 | 0.000 / 0.976 | pass 8/8 (Δ −0.505 to −0.535) | pass | – |
| i.i.d. SCM sets, pooled (45 units, 100,000 sign flips) | 0.000 / 0.93 to 1.0 | pass 8/8 (Δ −0.487 to −0.489, 45/45 units) | pass | 0.923 |
| WT regimes, D = 10, L = 0 (bipartite; latent ties only) | 0.014 / 0.592 | pass (Δ −0.20, 63/64 units) | pass (Δ −0.17) | 0.539 |
| WT regimes, D = 1, L = 30 | 0.003 / 0.398 | pass (Δ −0.23) | pass (Δ −0.13) | 0.443 |

Notes on the secondary rows:
- **citris with G_D.** On the theory-strict edges (G_D predicts 1 at ir_3 and vis_3 and on the polariser edges), the sensor indicator is significantly better (Δ = +0.45, reverse Holm p = 0.002).
- **i.i.d. sets.** Per experiment, everything passes except scm_1 (5 units; the minimum attainable p of 1/32 cannot survive Holm). The scm_4 chain and the scm_5 collider behave exactly as Cor 1.10 predicts: the chain gives all ½ (2,267 of 2,267 instances observed ½), and the collider gives 1 on edges into the collider and its sensors and ½ elsewhere (2,238 of 2,238 exact).
- **WT rows.** WT detection is 0.40 to 0.59, so part of the advantage there is power-induced (PLAN4 flagged this in advance). The E1c subset still favours the theory.

Post hoc (`explore4.log`, not pre-registered): a constant "every edge ½" predictor matches 81.5% of the citris instances, against 99.2% for the theory (G_lin). On the i.i.d. sets it matches 68.8%, against 92.3%. With G_D, however, the theory (71.4%) is *worse* than the constant predictor on citris.

## E2: lagged relatives on held-out WT regimes (64 units)

The primary configuration is D = 1 (raw increments), comparing lagged L = 30 (about 9 s) with lag 0.

| test | threshold | result | verdict |
|---|---|---|---|
| Q7 non-relatives declared (54 per unit) | < 0.5 | **0.003** [0.001, 0.005] | pass |
| Q8 accuracy on edges predicted 1 (1,216 instances) | > 0.5, CI excludes 0.5 | **0.553** [0.527, 0.579] | pass |
| Q9 Spearman(s̃, accuracy), 1,280 subgraphs | ≥ 0.3 | **0.243** [0.193, 0.294] | **fail** |
| Q10 relatives detected, lagged − lag 0 | > 0, CI lower bound > 0 | 0.398 − 0.328 = **+0.069** [0.060, 0.079] (non-relatives: +0.001 [−0.001, 0.002]) | pass |

- **Verdict:** Q9 failed, so E2 does not succeed. No kill rule is triggered, because Q7 and Q10 pass.
- Lags raise detection at fine resolution without adding false positives, as Prop A1 and A3 predict.
- **Secondary configuration (D = 10, L = 3 vs 0):**
  - Q7: 0.011; Q8: 0.711 [0.692, 0.729]; Q9: 0.202 (fail).
  - Q10: **−0.007** [−0.012, −0.003] (fail). Lags *reduce* detection here, because the Bonferroni cost exceeds the gain, as development predicted.
- **Held-out replication of the PLAN3 lag-0 estimator:**
  - D = 10: Q7 0.014, Q8 0.742 [0.722, 0.760], Q9 **0.202** (fail).
  - D = 1: Q8 0.552, Q9 0.120.
  - So the P9-type failure replicates on held-out wind-tunnel data. **P9's original verdict (0.267, failed) stands.** E2 does not rescue it, and it is not presented as doing so.
- **Per edge (D = 1, lag 0 → L = 30):**
  - recovered: load_in → pressure_upwind 0.56 → 0.98; load_out → pressure_upwind 0.02 → 0.81; the downwind edges 0.73/0.75 → 0.95/1.00; load_in → rpm_out 0 → 0.17;
  - lost to the larger threshold: pot_1 → signal_1 0.36 → 0.16 and pot_1 → signal_2 0.27 → 0.11;
  - still never detected: every mic edge, pot_2 → signal_2 and the cross-current edges.
  - Even at its best (D = 10, lag 0: 0.59), WT detection stays far below the light tunnel's. Lags are not the main cause of the F1 shortfall.
- **citris (secondary), L = 5 vs 0:** detection 0.976 vs 0.969, difference +0.008 [0, 0.024], so Q10 fails. Detection there is already nearly saturated at lag 0.

## Out-of-sample F-theory checks (first half predicts second half; criterion fixed in PLAN4 §5)

| configuration | predicted relatives detection [p_max, p_ind] | observed | check | predicted / observed non-relatives declared | predicted / observed accuracy (predicted-1 edges) |
|---|---|---|---|---|---|
| WT D = 1, L = 0 | 0.250 | 0.242 | pass | 0.007 / 0.003 | 0.573 / 0.590 |
| WT D = 1, L = 30 | [0.344, 0.426] | 0.332 | pass (within 0.03) | [0.021, 0.074] / 0.003 | 0.504 / 0.506 |
| WT D = 10, L = 0 | 0.548 | 0.533 | pass | 0.014 / 0.008 | 0.649 / 0.632 |
| WT D = 10, L = 3 | [0.536, 0.554] | 0.524 | pass | [0.016, 0.027] / 0.008 | 0.628 / 0.618 |
| citris L = 0 | 0.935 | 0.945 | pass | 0.007 / 0.000 | – (G_lin has no full-set edge predicted 1 inside a block) |
| citris L = 5 | [0.940, 0.988] | 0.945 | pass | [0.010, 0.045] / 0.000 | – |

- The F1/F2 asymptotics again predict out-of-sample detection to within about 0.02, and the orientation accuracy to within 0.02.
- With lags, the observed detection sits at or slightly below p_max. This is consistent with strongly correlated lag statistics, for which the max over lags adds little beyond the best single lag.
- The independence formula p_ind overstates both detection and false positives, as THEORY_ADD §A said it would.

## Deviations and clarifications

1. None of thresholds, units, seeds, K, L, D, graphs or node sets changed after the results were seen.
2. In the commit after PLAN4, a stray `__pycache__` file that had been committed with PLAN4 was removed. No content changed.
3. `F1_check_pass` uses the PLAN4 §5 tolerance (±0.03). For citris, "predicted accuracy" is undefined (NaN in the JSON), because no full-set G_lin edge inside a block has predicted value 1. The predicted-1 edges at ir_3 and vis_3 are not edges of G_lin; they are scored only as design edges in E1.
4. scm_4 has no non-relative pair, so its E1a non-relative share is undefined (NaN). The pooled i.i.d. E1a uses the other sets.
5. The constant-½ comparison (`explore4.py`) is post hoc.

## Honest assessment: does E1 now separate Thm 1.9 from the naive explanations?

- **Yes, on one real multi-layer time series, under a pre-registered and stated refinement.**
  - On citris_1 the estimated relatives counts are tied along the whole chain red → green → blue → sensors, and between pol_1, pol_2 and the angle sensors.
  - They are strictly ordered, and in the anti-causal direction, on red → pol_2 and blue → pol_2.
  - Every one of the eight competitors predicts something else: sensor indicator, ancestors, in-degree, depth (each in two versions) and the finite-G_T "all edges sorted" limit.
  - Only the source-SCC rule of Thm 1.9 / Cor 1.10 predicts these counts. The effect is as large as it can be (Δ ≈ −0.5, 10/10 units). It survives the power safeguard, and on the theory-strict edges it is opposite in sign to what a power artefact would produce.
  - The snapshot sets replicate it cleanly, including the textbook chain-vs-collider contrast (scm_4 all ½, scm_5 as predicted).
- **Caveats that a paper must state.**
  1. The primary prediction uses the linear-dependence graph G_lin. G_lin removes six real causal edges that, by a symmetry lemma (THEORY_ADD B1 to B3; physical assumption B2 on the sensors), carry no linear covariance at any lag. Applying Thm 1.9 to the published design graph G_D still beats all eight competitors overall. But it loses to the sensor indicator on the edges where G_D predicts strict order, and it is worse than a constant-½ predictor (71% vs 82%). The correct statement is therefore: Thm 1.9 predicts the relatives sort of a correlation-based estimator when "relatives" means linearly dependent relatives. Faithfulness of the graph to linear dependence is a real assumption, and it fails here by design.
  2. The theory's advantage is concentrated in ties. In single-source components it predicts "unsortable". The naive rules cannot express that, so a large part of the separation is "the theory correctly predicts ½ where others predict 1". The strict (0/1) predictions are fewer: 297 instances on citris, 2,768 on the i.i.d. sets.
  3. All of this comes from one light-tunnel device. The actuator structural causal models are experimenter-designed, and the sensors respond almost instantaneously and strongly (relatives detection 0.97). This is close to the favourable regime of FINITE F1.
  4. On the wind tunnel (bipartite, detection 0.4 to 0.6), the theory still beats the naive rules through latent-node ties, also in the detected-edge subset. But the graded P9-type criterion fails again on held-out data (0.20 to 0.24), and a constant-½ predictor has a lower absolute error there than the theory. On the wind tunnel, the limiting factor remains F1 (weak, nonlinear or undetectable effects), not the theorem, and lags do not fix it.
