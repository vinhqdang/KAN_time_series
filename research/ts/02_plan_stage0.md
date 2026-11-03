# Stage 0 plan: is temporal aggregation of CausalRivers a valid real-data test for contemporaneous edges?

Pre-registered by commit BEFORE any code is run on the data. Nothing here touches the published SPADE manuscript or the earlier shrinkage/BASIL line of work.

## Question

If CausalRivers discharge series are aggregated to 6 h, 12 h and 24 h, upstream-to-downstream edges with short travel time become contemporaneous. Direction is known from physics (upstream causes downstream). Does this give a benchmark on which (a) the physics orientation is consistent with the data, and (b) methods that model contemporaneous effects do better than lag-only methods?

This stage decides whether a paper exists. It does not build any new method.

## Data

- `product.zip` (CausalRivers release, already downloaded in the earlier session; re-download from the GitHub release asset if missing; TLS verification stays on). Use RiversEastGermany and RiversBavaria. Licence dl-de/by-2-0, cite Stein et al., ICLR 2025.
- Aggregate the 15-min series by mean to 6 h, 12 h, 24 h. Stations with more than 20% missing steps at the aggregated resolution are dropped; gaps of at most 2 aggregated steps are linearly interpolated; the rest are dropped with the subgraph that contains them. Standardise each node (zero mean, unit variance) before any method (primary analysis). A secondary analysis uses raw scale to measure scale-based leakage.
- Subgraphs: connected subgraphs of the river graph, sizes d=5 and d=10, sampled with fixed seeds (5000..5029 for each size and each region). 30 per size and region at each resolution. The same subgraph sets are reused across resolutions and methods.

## Edge labels (fixed now)

- For every edge u->v in a sampled subgraph, estimate its empirical travel time tau_e at the 15-min resolution as the lag (in 15-min steps, 0 to 14 days) that maximises the cross-correlation of first-differenced discharge between u and v, upstream leading. This uses only the fine-resolution series, independent of the coarse analysis.
- At resolution Delta, the edge is labelled **contemporaneous** if tau_e < Delta/2, **lagged** otherwise. Labels are therefore fixed before any method runs.
- Ground-truth orientation is the river direction (upstream -> downstream). Autoregressive links are not scored.

## Validity gates (checked first, before any method comparison)

- **G1, physics oracle consistent with data:** at 15 min, for at least 70% of sampled edges, the cross-correlation peak lies at a positive lag with upstream leading (peak lag > 0 and the correlation at that lag exceeds the correlation at the mirrored negative lag). Fail means the physics orientation is not recoverable even in principle; the whole direction stops.
- **G2, enough contemporaneous edges:** at 24 h, at least 30% of edges are labelled contemporaneous, and at least 100 contemporaneous edges exist across the d=10 subgraphs of each region. Fail means there is no usable test set at that resolution; report which resolutions pass.
- **G3, scale leakage:** report how well var-sortnregress and R2-sortnregress orient contemporaneous edges on standardised and on raw data. Discharge grows downstream, so raw-scale orientation can be trivial. If a scale-only control reaches orientation accuracy of at least 0.8 on standardised data, orientation is leaked by something other than causal mechanism and the result must say so.

## Methods

Lag-only: VAR with Granger-style scoring (undirected skeleton, directed by lag; for contemporaneous edges it cannot orient, scored at 0.5 orientation by definition), and a plain correlation-magnitude score. Contemporaneous-capable: PCMCI+ (tigramite), VAR-LiNGAM (lingam), DYNOTEARS (own implementation if no maintained package installs; say so), and a linear NOTEARS-style SVAR as a reference. Controls: var-sortnregress, R2-sortnregress, random scores. Lag order p=1 at 6 h and 12 h and 24 h (also p=2 as a sensitivity run). Software versions and hyperparameters are written in `research/ts/stage0/README.md`; they are fixed on seeds 5000..5004 of each cell (validation) and the reported numbers use seeds 5005..5029 (test) only.

## Metrics

1. Orientation accuracy on true contemporaneous edges, given the true skeleton (isolates direction from detection). Chance 0.5.
2. Directed-edge AUROC and AUPRC over all ordered pairs, excluding autoregressive links.
3. Per method, bootstrap 95% CI over subgraphs; paired comparisons use a paired Wilcoxon test with Holm correction over the method pairs reported.

## Hypotheses and decision rules (fixed now)

- **H1 (positive):** at 24 h and at 12 h, at least one contemporaneous-capable method has orientation accuracy >= 0.60 on contemporaneous edges with a bootstrap CI excluding 0.5, and its directed AUROC exceeds lag-only VAR by >= 0.03 (paired, Holm p < 0.05).
- **Negative, either is enough to kill the benchmark claim:** (i) every contemporaneous-capable method has orientation accuracy within 2 standard errors of 0.5 across at least 30 subgraphs at 24 h; or (ii) lag-only VAR has directed AUROC at least equal to every contemporaneous-capable method.
- **Ambiguous:** anything between. Report as such.

## What happens next

- G1 fails: stop. Report why. No new method is built on this benchmark.
- G1 passes and negative: the finding is "contemporaneous machinery does not help on real aggregated data"; propose a short benchmark/evaluation paper for a modest venue.
- G1 passes and H1 holds: proceed to Stage 1 (spline additive SVAR with per-edge lag and contemporaneous functions plus forecasting), with its own pre-registered plan.

## Stated risks

- Hidden common drivers (rain) can create spurious contemporaneous dependence at coarse resolution; G1 does not exclude this. Report the correlation of residuals among sibling stations (stations sharing a downstream neighbour) as a diagnostic.
- The label rule tau_e < Delta/2 is a choice; sensitivity at Delta/4 and Delta is reported.
- The existing CausalRivers paper already studies resolutions down to 24 h; novelty here is only the contemporaneous-edge scoring, which must be checked against that paper's text (state what was read).
