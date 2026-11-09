# B1-NL: does the start artefact survive nonlinear, signed-weight and non-Gaussian parametrisations?

This is a pre-registered replication of B1 (`../b1/`). It answers the criticism that B1 used "only one linear-Gaussian parametrisation with positive weights".
- **Plan.** `PLAN.md` was committed and pushed together with the simulation code (commit 2d19c9b) before any simulation that produced an algorithm score.
- **Notes before the test run.** Notes N1 and N2 were added to `PLAN.md` §5 (commit 2f6882a) after the validation run and before the test run.
- **Test run.** Run once on one core, in 42 resumable shards. It took 7,815 s (2.2 h) for 5,040 data sets.
- **Changes after results.** No threshold, hypothesis or seed was changed after results were seen.

## Main-text paragraph (draft)

> **Robustness to the parametrisation.** We repeated B1 with new graphs and seeds under three other parametrisations: a nonlinear additive VAR (NL), a linear VAR with signed weights ±Unif(0.5, 1) (SGN), and SGN with uniform noise (SGNU).
> - In NL, each cross edge applies tanh, sin or a bounded rational-quadratic to its parent, with a linear self-loop.
> - The NL, SGN and SGNU data sets numbered 2,100, 2,100 and 840, with families, horizons, generators and methods as in B1, and decision rules fixed in advance.
>
> The fixed start raised rel-SortnRegress order accuracy in every family and every one of the 42 (family, p, T) cells, in all three configurations:
> - **NL:** mean gain +0.108 over the non-strongly-connected families (95% CI [0.100, 0.116]). All five families were Holm-significant, with p ≤ 1.4e-9.
> - **SGN:** +0.137 [0.128, 0.146].
> - **SGNU:** +0.137 [0.123, 0.152].
>
> The per-cell gain again tracked the exact population gap s_ρ(G_T) − s_ρ^stat: Spearman 0.68 (NL), 0.82 (SGN) and 0.78 (SGNU), against 0.80 in B1.
> - With signed weights the slope of gain on gap (0.56) matched B1's (0.53).
> - Under nonlinearity it was attenuated (0.34), as we predicted before the run: a correlation-count relatives estimator recovers only about 2/3 of the graph relatives there.
>
> The separate var-sortability artefact also replicated. Var-SortnRegress gained +0.19 to +0.24 on raw data and nothing after standardisation.
>
> Significant SHD rank swaps were rarer than in B1:
> - NL: one swap, R²- vs Random-SortnRegress on chains with p = 10, T = 10.
> - SGN: none.
> - On secondary metrics, the B1 in-tree reversal of Var-SortnRegress versus FLOP reappeared under SGN (SID: 1 swap; orientation: 2 swaps).
>
> The start artefact in sortability and in sort-based order accuracy therefore does not depend on linearity, positive weights or Gaussian noise. Its effect on significant method rankings is fragile and confined to sort-based baselines.

**Table.** Mean over 2 p × 3 T × graphs per family (NL and SGN: 25 graphs per (family, p), i.e. 150 data-set pairs; SGNU: 10, i.e. 60 pairs).
- **Columns.** "OA" is the rel-SnR order accuracy under G1 / G2. Δ is the G1 − G2 difference with a bootstrap 95% CI over data-set pairs. "Recall" is the share of graph-relative pairs that the correlation-count estimator detects in NL.
- **Exact s_ρ.** The values are those of the NL graphs. The SGN graphs, a different draw, are within 0.02 (`results/exact_predictions.txt`).
- **B1 comparison.** B1's Δ (linear Gaussian, positive weights) is given for reference.

| family | exact s_ρ G1 / G2 | NL OA | NL Δ [95% CI] | SGN OA | SGN Δ [95% CI] | SGNU OA | SGNU Δ [95% CI] | B1 Δ | NL recall G1 / G2 |
|---|---|---|---|---|---|---|---|---|---|
| symER (SC) | 0.780 / 0.500 | 0.577 / 0.501 | +0.077 [+0.064, +0.090] | 0.605 / 0.483 | +0.122 [+0.104, +0.142] | 0.613 / 0.488 | +0.125 [+0.096, +0.157] | +0.126 | 0.49 / 0.51 |
| ring (SC) | 0.954 / 0.500 | 0.621 / 0.496 | +0.125 [+0.107, +0.143] | 0.706 / 0.501 | +0.205 [+0.184, +0.228] | 0.695 / 0.524 | +0.171 [+0.142, +0.201] | +0.180 | 0.50 / 0.38 |
| chain | 0.887 / 0.500 | 0.698 / 0.511 | +0.187 [+0.171, +0.203] | 0.712 / 0.505 | +0.206 [+0.186, +0.226] | 0.730 / 0.530 | +0.200 [+0.162, +0.240] | +0.217 | 0.76 / 0.62 |
| out-tree | 0.842 / 0.500 | 0.622 / 0.514 | +0.107 [+0.088, +0.127] | 0.613 / 0.459 | +0.154 [+0.134, +0.174] | 0.615 / 0.436 | +0.180 [+0.141, +0.220] | +0.144 | 0.80 / 0.82 |
| in-tree | 0.797 / 0.659 | 0.725 / 0.652 | +0.073 [+0.059, +0.088] | 0.743 / 0.658 | +0.085 [+0.071, +0.100] | 0.735 / 0.662 | +0.073 [+0.051, +0.097] | +0.069 | 0.82 / 0.75 |
| ER(c=1) | 0.773 / 0.558 | 0.647 / 0.566 | +0.081 [+0.066, +0.096] | 0.657 / 0.549 | +0.108 [+0.090, +0.127] | 0.664 / 0.560 | +0.104 [+0.072, +0.137] | +0.085 | 0.74 / 0.72 |
| ER(c=2) | 0.829 / 0.523 | 0.643 / 0.551 | +0.092 [+0.078, +0.106] | 0.654 / 0.523 | +0.132 [+0.111, +0.153] | 0.665 / 0.535 | +0.129 [+0.098, +0.163] | +0.109 | 0.57 / 0.55 |

**Figure.** `results/fig_b1nl.png` (pdf alongside) has one panel per configuration. Each plots the per-cell gain against the exact gap, as in B1's Fig. (a). All 126 cell means are above 0, and all but one are below the diagonal.

## Verdicts (test seeds, run once; rules from PLAN §2)

| hypothesis | rule | NL | SGN | SGNU |
|---|---|---|---|---|
| **H1** fixed start helps rel-SnR on non-SC families | ≥ 4/5 Holm-significant and pooled mean ≥ 0.02 | 5/5 (Holm p ≤ 1.4e-9); pooled **+0.108** [0.100, 0.116] (250 graphs) → **supported** | 5/5 (Holm p ≤ 7.6e-10); pooled **+0.137** [0.128, 0.146] → **supported** | 5/5 (Holm p ≤ 4.1e-5); pooled **+0.137** [0.123, 0.152] (100 graphs) → **supported** |
| H1 kill / reversal | < 3/5 or pooled < 0.01; significant opposite sign | not triggered; no reversal | not triggered; no reversal | not triggered; no reversal |
| **H1 tracking** | Spearman(cell d, exact gap) ≥ 0.5, one-sided p < 0.05 | **0.677** (p = 4e-7) → **supported**. Unit level 0.44; OLS slope 0.34 | **0.817** (p = 2e-11) → **supported**. Unit level 0.64; slope 0.56 | **0.775** (p = 8e-10) → **supported**. Unit level 0.62; slope 0.56 |
| **H3** ≥ 1 Holm-significant SHD rank swap (840 tests) | as B1 | **1** swap → **supported (weak)** | **0** swaps → **not supported** | not testable (10 seeds; pre-stated) |
| **H4** var/rel split: (a) Var-SnR raw gain ≥ 0.05, CI > 0; (b) \|std gain\| < 0.02; (c) rel gain CI > 0 | all three | +0.189 / +0.000 / +0.106 → **holds** | +0.243 / +0.006 / +0.145 → **holds** | +0.239 / −0.008 / +0.140 → **holds** |
| **Replication summary** | H1 and tracking in both NL and SGN | — | — | — |

**Replication summary: REPLICATES** (H1 and tracking supported in both NL and SGN, and in SGNU as well).

### H3 detail (ranking changes)

- **NL, SHD.**
  - The single significant swap is **R²-SnR vs Random-SnR on chains, p = 10, T = 10**. R²-SnR is better under G1 by 41 SHD (Holm 0.010) and worse under G2 by 38 (Holm 0.011).
  - It survives SHD_x1 (Holm 0.025 / 0.019).
  - This is a swap between two weak baselines, comparable to B1's small ring swap. It is not the rel-SnR vs R²-SnR mechanism swap of B1.
  - Pairs that swap by mean: 50 of 420.
  - The best method changes in 2/42 cells; median Kendall τ between the G1 and G2 rankings is 0.80.
  - No significant swaps on SID, orientation or OA.
- **SGN, SHD.**
  - 0 significant swaps; 59 swaps by mean; the best method changes in 2/42 cells.
  - Secondary metrics show **Var-SnR vs FLOP on in-trees**:
    - SID, p = 5, T = 5: Var-SnR is better under G1 and FLOP better under G2 (Holm 0.020 / 0.030).
    - Orientation, p = 5 and p = 10, T = 3: same direction (2 swaps).
  - This is B1's variance-artefact reversal. It replicates in direction, but on SHD it no longer passes Holm with 25 seeds and 5 methods.
- **SGNU (descriptive only).** 0 significant swaps; 58 by mean.
- **What does not change.** With labels hidden, FLOP is the best method in every configuration under both generators:
  - normalised SHD G1 / G2: 0.35 / 0.53 (NL), 0.19 / 0.37 (SGN), 0.18 / 0.37 (SGNU);
  - the ordering among the other methods is also largely stable (Var-SnR < rel-SnR < R²-/Random-SnR).

### Relatives under nonlinearity and signed weights (pre-stated diagnostics)

- **NL detection is partial, as pre-stated.**
  - The correlation-count estimator detects 67% (G1) and 62% (G2) of the graph-relative pairs. Its false-positive rate is about 5%, i.e. at the test level α.
  - On SC families recall is only 0.38–0.51.
  - The estimated rel-sortability still rises under G1 in every family (S2 passes: +0.07 to +0.19). The gain is attenuated: NL's OLS slope of d on the exact gap is 0.34, against 0.56 (SGN) and 0.53 (B1).
  - The attenuation is strongest where recall is lowest (symER: +0.077 vs B1 +0.126).
  - This matches the direction predicted in PLAN §1.2.
- **SGN sample relatives deviate from graph relatives.**
  - Recall is 0.75 / 0.73, below what positive weights give, because of near-cancellations.
  - One visible consequence: under G2, out-trees are slightly *anti*-rel-sortable in the sample (estimated 0.455, rel-SnR OA 0.459), whereas the graph value is 0.5. The exact s_ρ is a graph quantity and does not see this.
  - The G1 − G2 gain is nonetheless at least as large as in B1.
- **SGNU vs SGN.** Same graphs and weights; only the noise distribution differs. The gains agree within CIs in every family. As expected, the correlation-based estimator is insensitive to non-Gaussian noise.

### Strongly connected families (descriptive)

- **Positive in every cell.** The gain is positive in all 36 SC cells (3 configs × 2 families × 2 p × 3 T). Only one cell's CI touches 0: SGNU ring, p = 5, T = 10, +0.027 [−0.001, +0.051].
- **Short horizons.** At T < p the gain is large in every configuration (symER and ring, p = 10, T = 3: +0.10 to +0.31).
- **Declines with T.** The gain declines with T, as the exact gap does.

### Sanity rules

| rule | NL | SGN | SGNU |
|---|---|---|---|
| S1 \|OA_M1(G2) − OA_M1(G2L)\| < 0.02 per family | max 0.007 → pass | max 0.013 → pass | max 0.015 → pass |
| S2 est. rel-sortability G1 > G2, non-SC families | +0.07 to +0.19 → pass | +0.08 to +0.20 → pass | +0.08 to +0.20 → pass |
| S3 coupling error < 1e-3 | 5.6e-5 → pass | 6.8e-6 → pass | 8.7e-6 → pass |

## Deviations and notes (with timing)

None of these changed a threshold, hypothesis, seed or method.
- **Before the plan commit (parameter-only, no scores).** S3 coupling was checked on every test parameter set. The nonlinearity strength was checked on validation parameters (PLAN preamble).
- **N1 (after the validation run, before the test run).** A faster transitive closure for the recall diagnostic. Output is identical, verified on 252 cases.
- **N2 (after the validation run, before the test run).** S1 has limited power because it compares independent data sets. The rule was kept; S1 passed in all configurations anyway.
- **Size rule.** The validation timing projected 2.3 h (< 3.0 h), so the full pre-registered size was run (25 / 25 / 10 seeds). The actual run took 2.2 h.
- **Reductions relative to B1, fixed in the plan for the one-core budget:**
  - 25 (SGNU: 10) graphs per (family, p) instead of 30;
  - no PC;
  - no R-eq regime;
  - G2L (burn-in 500) instead of the exact stationary start G2s.

  Without PC, 5 of B1's 11 SHD swaps (Var-SnR vs PC on in-trees) cannot occur here by construction.
- **Validation runs (seeds 9100–9102)** were used only for debugging and timing. They are not evidence.

## Honest reading

1. **The core claim replicates cleanly.** The fixed start makes the anonymous unrolled window more rel-sortable. Sort-based order accuracy rises by roughly the amount the exact theory predicts:
   - The effect is present in every family and every cell, under nonlinear dynamics, with signed weights and with uniform noise.
   - It is not an artefact of positive weights. With signed weights the gains are slightly *larger* than in B1, partly because G2 sample rel-sortability falls below the graph value.
   - The criticism "only one linear-Gaussian parametrisation with positive weights" is answered for the sortability and order-accuracy claims.
2. **Nonlinearity attenuates the effect, as predicted.**
   - What attenuates is the estimator, not the graph quantity: a correlation-count estimator sees only part of the nonlinear dependence.
   - Tracking survives (0.68), but the slope falls to about 0.34.
   - A dependence-based relatives estimator (e.g. HSIC counts) would likely see more of the gap. This was not tested.
3. **The ranking claim is weaker than B1 suggested.**
   - In B1, H3 rested on 11 SHD swaps. Most of them involved Var-SnR against FLOP/PC on in-trees, and the most interesting one was rel-SnR vs R²-SnR on chains.
   - Here (fewer seeds, no PC) there is a single, substantively minor SHD swap under NL, and none under SGN.
   - The Var-SnR vs FLOP in-tree reversal reappears under SGN only on SID and orientation.
   - The rel-SnR vs R²-SnR chain swap does not replicate on SHD. On chains with p = 10, T ≤ 5, rel-SnR keeps a lower SHD than R²-SnR under G2 too (NL: 86 vs 114; SGN: 137 vs 167).
   - Pooled over cells, the order-accuracy reversal by mean does replicate in all three configurations, as in B1. rel-SnR is ahead of R²-SnR under G1 and behind it under G2: NL 0.647 vs 0.615 → 0.541 vs 0.545; SGN 0.670 vs 0.640 → 0.525 vs 0.540; SGNU 0.674 vs 0.639 → 0.533 vs 0.542.
   - This reversal is small under G2 and not cell-wise significant.
   - H3-NL is formally "supported", but it should not be used as evidence that the start artefact reverses meaningful comparisons.
   - The defensible statement is: "the start artefact shifts absolute scores of sort-based methods robustly, and it can occasionally flip baseline-vs-baseline comparisons. Under every parametrisation tried, the ranking of the strongest method (FLOP) is unaffected."
4. **The variance artefact (H4) is the most robust secondary finding.**
   - Var-SnR gains +0.19 to +0.24 from the fixed start in every configuration, and exactly nothing after standardisation.
   - Reisach et al. propose these generators partly to avoid var-sortability, so this is worth stating in the paper.

## Files and reproduction

- `common_nl.py`: the NL/SGN/SGNU parameters and simulator, the coupling check and the window graph relatives. Everything else is imported unchanged from `../b1/common.py`.
- `run_b1nl.py`, `run_shards.sh`: the resumable one-core test run, in shards per (config, family, p) under `results/shards/` (ignored by git).
- `analyse_b1nl.py results/b1nl_test.csv.gz results`: writes `analysis.txt`, `analysis.json` and `fig_b1nl.{png,pdf}`.
- `results/b1nl_test.csv.gz`: 5,040 data sets, 0.6 MB, merged shards.
- `results/exact_predictions.txt`: graph-only predictions, computed before the plan.
- Environment as in B1:
  - Python 3.13, numpy, scipy, scikit-learn, pandas;
  - flopsearch 0.3.0, gadjid 0.1.0;
  - `OMP_NUM_THREADS=1`.
