# Stage 0 validity gates (computed before any method comparison)

Plan: research/ts/02_plan_stage0.md (commit 3cd4581). Subgraphs: all 30 seeds (5000..5029) per size and region, 120 subgraphs, 797 edge occurrences (edges counted once per subgraph in which they occur).
Gates use only the sampled graphs, labels and the three scale-only controls. No method was run.

## G1  physics oracle consistent with data (threshold: >= 70% of sampled edges)

Criterion per edge: peak lag (over 0..14 d, upstream leading, first-differenced 15-min discharge) is > 0 and the correlation at the peak exceeds the correlation at the mirrored negative lag.

| group | edges | frac G1-consistent | frac tau_e=0 | median tau_e (15-min steps) |
|---|---|---|---|---|
| ALL | 797 | 0.797 | 0.202 | 15 |
| bavaria d=5 | 123 | 0.984 | 0.016 | 19 |
| bavaria d=10 | 275 | 0.964 | 0.036 | 17 |
| east_germany d=5 | 124 | 0.694 | 0.298 | 15 |
| east_germany d=10 | 275 | 0.593 | 0.407 | 8 |
| bavaria (both sizes) | 398 | 0.970 | 0.030 | 17 |
| east_germany (both sizes) | 399 | 0.624 | 0.373 | 11 |

Unique edges: 400; G1 fraction over unique edges: 0.765

**G1 = 0.797 -> PASS**

## G2  enough contemporaneous edges (24 h: >= 30% of edges and >= 100 contemporaneous edges across the d=10 subgraphs of each region)

Label: contemporaneous iff tau_e < Delta/2 (Delta/2 = 3 h, 6 h, 12 h = 12, 24, 48 steps of 15 min).

| resolution | all edges: frac contemp | ES d=10: n contemp / n | BA d=10: n contemp / n | ES frac | BA frac | verdict (all conditions) |
|---|---|---|---|---|---|---|
| 6h | 0.404 | 147/275 | 84/275 | 0.511 | 0.296 | FAIL |
| 12h | 0.617 | 183/275 | 170/275 | 0.624 | 0.611 | PASS |
| 24h | 0.795 | 227/275 | 225/275 | 0.777 | 0.814 | PASS |

Sensitivity of the label rule (fraction contemporaneous, all edges):

| resolution | tau<Delta/4 | tau<Delta/2 (primary) | tau<Delta |
|---|---|---|---|
| 6h | 0.302 | 0.404 | 0.617 |
| 12h | 0.404 | 0.617 | 0.795 |
| 24h | 0.617 | 0.795 | 0.862 |

**G2 at 24 h: PASS**; resolutions passing: ['12h', '24h'].

## G3  scale leakage: orientation accuracy of scale-only controls on contemporaneous edges

Orientation of a true edge u->v by the control's causal order (increasing score = earlier). Ties count 0.5. var-sortnregress: score = marginal variance of the series used (standardised: all variances are 1, so every pair is a tie by construction). R2-sortnregress: score = R^2 of a node regressed on all other nodes at the same time step (OLS, series standardised or raw). Extra row (not in the plan): RP = order by series mean (the CausalRivers 'reverse physical' baseline).

Threshold: control orientation accuracy >= 0.8 on standardised data means orientation is leaked by something other than causal mechanism.

| control | data | 6h contemp (n) | 12h contemp (n) | 24h contemp (n) | 6h lagged | 12h lagged | 24h lagged |
|---|---|---|---|---|---|---|---|
| var | standardised | 0.500 (322) | 0.500 (492) | 0.500 (634) | 0.500 | 0.500 | 0.500 |
| var | raw | 0.922 (322) | 0.943 (492) | 0.937 (634) | 0.933 | 0.908 | 0.902 |
| R2 | standardised | 0.742 (322) | 0.744 (492) | 0.729 (634) | 0.707 | 0.705 | 0.736 |
| R2 | raw | 0.742 (322) | 0.744 (492) | 0.729 (634) | 0.707 | 0.705 | 0.736 |
| mean(RP) | standardised | 0.500 (322) | 0.500 (492) | 0.500 (634) | 0.500 | 0.500 | 0.500 |
| mean(RP) | raw | 0.941 (322) | 0.939 (492) | 0.931 (634) | 0.922 | 0.915 | 0.926 |

**G3: no leakage flag: neither var- nor R2-sortnregress reaches 0.8 on standardised contemporaneous edges at any resolution.**
Raw-scale rows show how much of the orientation is available from magnitude alone.

## Diagnostic: residual correlation among sibling stations

VAR(1) (OLS, standardised) residual correlations, averaged over node pairs in the sampled subgraphs. Siblings: two stations that are both upstream parents of the same downstream station. Adjacent: true edge u->v. A high sibling value at coarse resolution points to shared forcing (rain) creating contemporaneous dependence without a direct edge.

| resolution | sibling mean r (n) | adjacent mean r (n) | other mean r (n) |
|---|---|---|---|
| 6h | 0.320 (518) | 0.413 (791) | 0.303 (1991) |
| 12h | 0.402 (518) | 0.538 (791) | 0.398 (1991) |
| 24h | 0.508 (518) | 0.648 (791) | 0.513 (1991) |

## Operationalisations and deviations fixed at this commit (before any method run)

D1. Subgraph pool. The plan drops a subgraph if any station keeps a gap after interpolating gaps of <= 2 steps. Applied literally, only 92 of 666 East Germany stations are gap-free at all three resolutions, and their induced river graph has no connected component with 10 nodes (largest 6). d=10 East Germany subgraphs would not exist. Deviation: stations must pass the 20% missing rule at every resolution; a sampled subgraph is accepted only if a contiguous window of at least 365 days exists in which every node is complete (after the <= 2-step interpolation) at 6 h, 12 h and 24 h; the longest such window is used by every method and by the tau_e estimation for that subgraph. Candidate subgraphs without such a window are rejected and redrawn from the same seeded stream. Window length is therefore subgraph-specific (377 to 1826 days). Consequence: selection toward well-measured stations. This was decided before any label, gate or method result was seen.
D2. Aggregation: mean of the 15-min values in each window, with the step set missing if fewer than 50% of its 15-min values are present (the plan does not state this rule).
D3. Seeds: the subgraph for seed s in a cell is drawn with numpy default_rng(s); if its node set equals one already drawn in the same cell the draw continues from the same stream.
D4. Edge counts: an edge that occurs in several subgraphs counts once per occurrence. d=5 and d=10 induced subgraphs of the river graph occasionally contain more than d-1 edges (river splits).
D5. G1/G2/G3 use all 30 seeds per cell (they involve no tuned method). G3 uses the two plan controls; the RP row is extra.

Decision rules, operationalised now (they do not change any threshold):
- Hypothesis metrics use the test seeds 5005..5029 only (25 per size and region = 100 subgraphs), pooled over regions and sizes. Orientation accuracy is a pooled-edge ratio with a cluster bootstrap over subgraphs (2000 resamples, percentile 95% CI, SE = bootstrap SD); subgraphs without a contemporaneous edge contribute nothing to it.
- Contemporaneous-capable methods for the verdicts: PCMCI+, VAR-LiNGAM, DYNOTEARS. The two-stage NOTEARS-SVAR is the plan's "reference": reported, excluded from verdicts; the verdict is also shown with it included.
- H1 at resolution R: some capable method with accuracy >= 0.60, 95% CI lower bound > 0.5, mean paired directed-AUROC gain over lag-only VAR >= 0.03 and Holm-adjusted two-sided paired Wilcoxon p < 0.05. Holm family: every (capable method, VAR) pair at each of 6 h, 12 h, 24 h (primary analysis, p=1, standardised) = 9 tests (12 with NOTEARS-SVAR). H1 needs R=12 h and R=24 h both satisfied.
- Negative (i): at 24 h, every capable method has |accuracy - 0.5| < 2 SE, with >= 30 subgraphs containing a contemporaneous edge. Negative (ii): at 24 h, mean directed AUROC of VAR >= that of every capable method (12 h and 6 h reported too).
- Lag-only VAR and correlation score 0.5 on orientation of contemporaneous edges by definition; their coefficient-based orientation is reported as a secondary column.
- Hyperparameters per method and resolution are chosen by mean directed AUROC on the validation subgraphs (seeds 5000..5004 of every cell = 20 subgraphs).
