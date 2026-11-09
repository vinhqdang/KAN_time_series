# Theory 2, part 3: real data with designed independent sources (Causal Chambers)

Pre-registration: `PLAN3.md`, commit cfcab28. It was pushed before any correlation, autocorrelation or other dependence statistic of the chosen data was computed. Code: `test3.py` (pre-registered analyses), `explore3.py` (post hoc), `cc_graph.py` (graphs), `predict_graph.py` (graph-only predictions), `inspect_design.py` and `inspect_walks.py` (Stage A metadata). Outputs: `results3.json`, `test3.log`, `explore3.json`, `explore3.log`, `predict_graph_results.txt`, `inspect_design_results.txt`. Raw data are not in the repository. They are the CC BY 4.0 Causal Chambers files `wt_walks_v1.zip` and `lt_walks_v1.zip` (md5 values in PLAN3), kept outside the repository; sources, SHA-256 values and the data-root variables are in `../DATA.md` (one-command run: `../reproduce.sh`).

Reproduce: `OMP_NUM_THREADS=1 python3 -I test3.py <this dir> <wt_walks_v1 csv dir> <lt_walks_v1 csv dir>` takes about 40 s on one core.

## Stage A outcome (details in PLAN3.md)

- **Causal Chambers (Gamella, Peters, Bühlmann 2025): downloaded and chosen.** These are real physical systems with a published ground-truth graph. In the wind-tunnel `actuators_random_walk` runs, the five actuators are independent, experimenter-generated random walks. So the sources are separate source SCCs by construction, and Cor 1.10's non-trivial case holds by design. Restricted to the 16 random variables, 25 of the 26 edges are predicted to have value 1. The light-tunnel `actuators_white` run (11 i.i.d. inputs, 20 variables, 37 of 39 edges predicted 1) is the replication.
- **CausalRivers + DWD precipitation: data reachable, not chosen.** The station list and archives download from opendata.dwd.de; 321 Bavarian and 284 East German stations were active in 2019; the data are open. Rain is a measured driver, but it reaches each gauge through its own catchment filter. By FINITE Prop 3.6, a contemporaneous rainfall proxy then leaves a residual even at reliability 1. This candidate would need a new per-gauge rainfall-response model, so it would not test the theory as it stands.
- **causalAssembly: rejected.** The released data are semi-synthetic and cross-sectional.
- **fMRI with physiological regressors: rejected.** There is no ground-truth graph.

## Pre-registered results

Wind tunnel (WT, primary): 15 runs. Each series is block-meaned (D = 10), first-differenced and standardised, giving n = 999 per run. The test is the effective-n Fisher z (Bartlett, K = 20), α = 0.01, Bonferroni-corrected. CIs come from a cluster bootstrap over runs. Light tunnel (LT, replication): 10 segments of 2,000 steps, levels, same test.

| | WT (primary) | LT (replication) |
|---|---|---|
| median n_eff per unit | 852 | 1998 |
| **P7/R7** non-relative pairs declared dependent (< 0.5) | **0.039 [0.031, 0.047]** (63 pairs × 15) | **0.000 [0, 0]** (126 × 10) |
| **P8/R8** accuracy on edges predicted 1 (> 0.5, CI excludes 0.5) | **0.656 [0.637, 0.675]** (375 edge instances) | **1.000 [1, 1]** (370) |
| edges predicted ½ (expected ½) | pot_1 → signal_1: 0.40 | pol → angle: 0.50 |
| **P9/R9** Spearman(predicted s̃, estimated accuracy) (≥ 0.3) | **0.267** (p = 3e-6), CI [0.163, 0.368], 300 subgraphs | **0.779** (p = 4e-42), CI [0.696, 0.864], 200 subgraphs |
| share of true relative pairs detected | 0.396 | 0.688 |

Verdicts:
- **P7 passes.** Unlike CausalRivers (0.98), only about 4% of graph non-relative pairs test dependent. The data are not saturated.
- **P8 passes.** Edges into confluences are oriented correctly by the relatives count in 65.6% of cases, and the CI excludes 0.5.
- **P9 fails** by the pre-registered threshold: 0.267 < 0.3, although the CI contains 0.3 and the correlation is clearly positive.
- **Kill rule: not triggered**, because P7 passes and P8 passes.
- **Positive real-data demonstration: not achieved as pre-registered**, because P9 failed. That definition required P7, P8 and P9 to pass on WT.
- **Replication (LT): R7, R8 and R9 all pass by wide margins.** As PLAN3 stated beforehand, this does not count as the demonstration. The inputs are i.i.d., so the data are close to snapshots and the time-series aspect is weak.

## Secondary analyses (pre-registered, no verdicts)

| WT | non-rel. dependent | acc. (pred. 1) | Spearman | median n_eff |
|---|---|---|---|---|
| S2 nominal n | 0.048 | 0.668 | 0.242 | 999 |
| S1 levels, K = 100 | 0.054 [0.042, 0.066] | 0.719 | 0.380 | 68 |
| S3 D = 1 | 0.025 | 0.709 | 0.100 | 9552 |
| S3 D = 30 | 0.023 | 0.619 | 0.336 | 276 |
| S4 given pressure_ambient, levels | 0.006 [0, 0.014] | 0.667 | – | 74 |
| S4 given pressure_ambient, primary | 0.001 [0, 0.004] | 0.648 | – | 888 |

- **S1: my prediction was wrong.** Levels of the random walks did not produce many spurious actuator dependences (0.054 against 0.039). Spearman is higher (0.380). With K = 100 the Bartlett n_eff is about 68, so the effective-n test is conservative enough on these runs.
- **S2: the nominal test gives slightly more false positives, as Cor 1.8 predicts.** The difference is small because the differenced series have little autocorrelation.
- **S3: D = 1 loses most relatives** (recovery 0.18; Spearman 0.10). This is consistent with FINITE Rem 1.10: responses that are slower than one step look weak at lag 0.
- **S4 (a measured driver).** The ground-truth graph has no edge between the in-tunnel barometers and pressure_ambient. They nevertheless share atmospheric pressure. Their level correlation with pressure_ambient is 0.65 to 0.93 per run, and these are the only non-relatives declared in at least 20% of runs (`explore3.json`).
  - Conditioning on the measured driver removes them: the non-relative fraction goes from 0.054 to 0.006 in levels and from 0.039 to 0.001 in the primary analysis.
  - Orientation accuracy barely changes.
  - The Phase E triad reliability estimator (FINITE Prop 3.8) does not apply here: it assumes a driver that loads on every node, but this one reaches only the four barometers, and those barometers are mutual relatives through their common actuator ancestors. The computed values (r̂ = 0.02 and 0.0006) are meaningless and carry no interpretation.
- **LT Malus caveat: the pre-registered prediction is confirmed.** The pol_k–ir_3/vis_3 pairs have |r| ≤ 0.04 and are never detected, although they are relatives. Their orientation is unaffected (R8 = 1.000).

## Do the F-theory quantities predict the outcome?

The F-theory check is out of sample: plug-in values from the first half of each unit predict the test on the second half, using the Fisher-z asymptotics of FINITE Prop 1.7 and 2.3.

| | WT predicted | WT observed | LT predicted | LT observed |
|---|---|---|---|---|
| non-relatives declared (F2) | 0.031 | 0.028 | 0.006 | 0.000 |
| relatives detected (F1) | 0.360 | 0.343 | 0.679 | 0.681 |
| orientation accuracy, edges predicted 1 (Monte Carlo of independent decisions) | 0.630 | 0.620 | 0.999 | 1.000 |

- **F2 (saturation).**
  - The median non-relative |r̂| is κ̂ = 0.03 (WT) and 0.02 (LT). The implied onset sample sizes are n_1/2 = 3 + (z_c/atanh κ̂)² ≈ 15,600 and 33,800. These are far above n_eff ≈ 430 to 1,000 (half series). So F2 predicts no saturation, which is what we observe.
  - CausalRivers was the opposite case: κ ≈ 0.64 gave n_1/2 ≈ 25, against n of 700 to 3,600.
  - Post hoc, the barometer–ambient pairs have κ = 0.19 to 0.32, giving n_1/2 = 145 to 426, which is below n_eff. F2 therefore predicts that these pairs are detected, and they are, in 67% to 93% of runs, in the order of κ.
- **F1 (recovery) is the binding constraint on WT.** The first-half ρ̂_min is about 0.002, and only 40% of relatives are detected.
  - Post hoc (`explore3.log`), every hatch edge (|r| ≤ 0.02), every mic edge, pot_2 → signal_2, and the cross edges load_in → current_out, load_in → rpm_out, load_in → pressure_intake, load_out → rpm_in, load_out → current_in and load_out → pressure_intake are essentially undetectable at this n. Together these are 16 of the 26 graph edges.
  - With these near-faithfulness violations, the estimated relative counts are compressed (for example, ρ̂(mic) = 1 against a true 14). This explains why the graded agreement P9 is weak while P8 holds.
  - The F1 plug-in prediction reproduces the observed orientation accuracy to within 0.01.
- **The nonasymptotic F1–F3 bounds are not informative** at these n, as FINITE.md §4 anticipated; ρ_min ≈ 0.002 makes them vacuous. The asymptotic formulas carry all the predictive content.

## Deviations and clarifications

1. `test3.py` was written after the PLAN3 commit, and it implements the plan as written. No threshold, unit, seed, K, D or node set was changed.
2. The S1 expectation in PLAN3 (more spurious dependence in levels) did not materialise. I report it as a wrong secondary prediction.
3. The F-check item 4 triad estimator is not applicable (see S4). Its output is reported and not interpreted.
4. For S4, the plan said "non-relative fraction among the barometer pairs". Within the 15-node conditioned set, the in-tunnel barometers are all mutual relatives, so that fraction is empty. I report instead the barometer–ambient correlations (in `results3.json`) and the overall fraction.
5. The P9 failure is close to the threshold, and its CI contains 0.3. As pre-registered, the point estimate decides.
6. Post hoc analyses (`explore3.py`: per-edge detection rates, barometer κ against n_1/2) carry no verdict.

## Honest assessment

- **This is the first real-data setting in this project where the theory's two preconditions hold:** no saturating unmeasured driver, and several independent source SCCs.
- **What holds.** Saturation is absent, exactly as F2 predicts from the measured κ. The relatives count orients edges into confluences better than chance on real physical time series (0.656, CI [0.637, 0.675]). The F1/F2 asymptotics predict the out-of-sample dependence rates and the orientation accuracy to within about 0.01 to 0.02.
- **What does not hold.** The graded per-subgraph agreement P9 missed its threshold (0.267), because many ground-truth edges of the wind tunnel are too weak to be detected by a lag-0 correlation at n ≈ 1,000. This is an F1 / faithfulness limitation, not a driver problem. So, strictly by the pre-registration, there is no positive real-data demonstration on the primary dataset.
- **The light-tunnel replication passes all three criteria cleanly** (0.000, 1.000, 0.779). Its inputs are i.i.d., so it shows the relatives-sort mechanism on a real system rather than a time-series-specific effect.
- **What the paper can claim.**
  - A partial positive result: on real systems with independent sources, the estimated relatives-sort orients confluence edges above chance, with no saturation.
  - A clean replication on a second real system.
  - A quantitative account of when the method saturates (CausalRivers: n ≫ n_1/2) and when it does not (Causal Chambers: n ≪ n_1/2).
  - The paper should not claim that the graded theorem-predicted s_ρ is recovered on real time series. That is shown only on the near-snapshot light-tunnel data.
- **Caveats.**
  - The WT graph is bipartite (sources → sensors), so the theorem's prediction coincides with "sensors have more relatives than actuators". The test does not separate Thm 1.9 from that simpler statement for this graph.
  - All units of each chamber come from one physical device.
