# Real-data evaluation on Sachs (pre-registration)

Status: written and committed BEFORE any code in this directory is run. Question: do SSC (SPADE-scaled), SSR_k1.5 (top 1.5 d edges of SSC, refit with lambda=0), BASIL-W and full BASIL behave sensibly on a real benchmark with a (contested) graph and no ground truth for edge functions?

## Data and protocol
- data/sachs/cyto_full_data.csv (7,466 cells, d=11), consensus graph data/sachs/cyto_full_target.csv (18 directed edges). Preprocessing identical to scripts/sachs_benchmark.py: natural log, then z-score (on the full matrix first by log only; z-score again within each subsample).
- 10 subsamples of n=2000 rows without replacement: subsample r (r=0..9) = `np.random.RandomState(r).choice(7466, 2000, replace=False)`; every method gets the same z-scored subsample and seed r.
- Methods: SSC, SSR_k1.5, BASIL-W (as in research/spire/bench.py: m=20, iters=200 d, 2 chains, select=True, SPADE-scaled candidates and warm start), BASIL (full, fit_basil with g=n, pi=2/(d-1), kmax=3, default MCMC settings), GAM-ridge, NOTEARS-linear, DAGMA-linear (METHODS of scripts/instantaneous_dag_benchmark.py), var-sortnregress. No hyperparameter is tuned on Sachs.
- Structure scores: SSC, SSR: edge importance; BASIL(-W): posterior edge probability P; GAM-ridge: its score; NOTEARS/DAGMA: |W|; var-sortnregress: |lasso coef|.

## Metrics (per subsample, then mean/sd over 10)
1. Directed AUROC and AUPRC on the 110 ordered pairs vs the 18 consensus edges (chance AUPRC 0.164). Ties are handled by sklearn (no jitter).
2. Orientation accuracy on the 18 true edges: 1 if score[c,e] > score[e,c], 0.5 if equal.
3. Skeleton AUROC: max(S, S.T) on the 55 unordered pairs vs the symmetrised truth.
4. Function stability: for each method with curves (SSC, SSR, BASIL-W, BASIL, GAM-ridge), take the edges ranked top-18 by that method in each subsample. For each edge that is in the top-18 in at least 3 subsamples, evaluate its curve (as function of the z-scored cause) on a common grid of 50 points in [-1.8, 1.8] (np.interp from the subsample's own data points, curve centred), and compute the mean pairwise R2_aff (squared Pearson correlation, i.e. R2 after affine alignment) and the mean pairwise signed correlation across the subsamples where the edge is in the top 18. Report the average over those edges and, as a stricter variant, the same over edges that are consensus edges. A method that gives a curve in every subsample (even when the edge is not in its top 18) is not penalised for that; the selection step is part of what we measure only through the edge set used.
5. Amplitude: for edges in the top 18 of SSC, in each subsample, std of the centred curve of SSC, SSR, BASIL-W on the data rows; report mean ratio SSR/SSC and BASIL-W/SSC (shrinkage of SSC is visible without ground truth; the previous synthetic result predicts SSC amplitude about 0.44 of truth, so ratios of 1.5-3 are predicted if the real signal is like the simulations). Paired over subsamples.
6. Sign sanity (only a few relations that are textbook MAPK/PI3K signalling and also in the consensus graph): praf -> pmek (activating), pmek -> p44/42 (activating), PIP3 -> pakts473 (activating). Statistic: sign of the correlation between the edge curve and the (z-scored) cause value; counted as correct if positive. I did NOT verify these against the text of Sachs et al. 2005 in this session (no network check); they are standard and are labelled as such. Inhibitory relations (e.g. PKA on Raf) are skipped because the measured phospho-site makes the expected sign ambiguous.

## Paired comparisons
Across the 10 subsamples: Wilcoxon signed-rank (two-sided) and bootstrap 95% CI (10,000 resamples) of the mean paired difference, for AUROC, AUPRC, orientation (each vs SSC and vs the best baseline NOTEARS-linear), and for function stability (SSR vs SSC, BASIL-W vs SSC). With n=10 the Wilcoxon minimum p is 0.002; we report effect sizes first.

## Expectations (stated honestly, no promise of a win)
- Earlier runs on these exact subsamples (experimental_results/sachs_raw.csv, disclosed: that is where SPADE gets mean AUROC 0.60, AUPRC 0.29 and NOTEARS-linear 0.57/0.24) say structure on Sachs is weak for every method. I expect all methods in 0.50-0.70 AUROC, differences between SSC, SSR, BASIL-W of about 0.05 or less, and var-sortnregress near chance or below (log+z-score does not remove var-sortability of raw data but we z-score, so near 0.5). I expect BASIL-W and SSR to be slightly lower than SSC in AUROC (as in synthetic data) and possibly not better than the linear baselines.
- Stability: I expect refit and BASIL-W curves to be at least as stable as SSC for top edges (more signal), but this could be reversed (lambda=0 refit can overfit and so be unstable). Not known.
- Amplitude: SSR/SSC > 1 and BASIL-W/SSC > 1, ratios probably 1.5-3.

## Verdicts (decided now)
- Structure: POSITIVE if SSC (or SSR/BASIL-W) has mean AUROC >= 0.60 and AUPRC > 0.164 + 0.05 with a bootstrap CI for AUROC excluding 0.5. NEGATIVE if the best of our methods has mean AUROC CI including 0.5 or is below NOTEARS-linear by more than 0.05. Otherwise AMBIGUOUS.
- Refit/BASIL-W vs SSC structure: "no loss" if the mean AUROC difference > -0.03 (the synthetic kill line).
- Stability: SSR/BASIL-W "more stable" if mean paired R2_aff exceeds SSC's with bootstrap CI of the difference excluding 0; "less stable" if the CI is entirely below 0 (this is a negative result for the claim that the corrected curves are usable on real data).
- Amplitude: shrinkage "visible" if the mean ratio CI excludes 1.
- Sign sanity: reported as counts out of 3 per method over subsamples; a method whose curves have the wrong sign for the majority of subsamples on any of the three is flagged.
- Overall NEGATIVE for the paper's real-data claim if structure is NEGATIVE and the corrected curves are less stable than SSC.

## Caveats known in advance
Sachs is small (d=11, 18 edges, the AUPRC differences of 0.05 are about one edge); the consensus graph is contested and omits known relations (and contains edges that are cyclic-free by construction while biology has feedback); the 7,466 cells pool different interventional/perturbation conditions, so the data are a mixture and violate iid and the single-SEM assumption; interventions are not used; curve stability across subsamples measures reproducibility, not correctness (all subsamples share the same 7,466 cells, so they overlap by about 27% pairwise and the subsamples are not independent: stability is optimistic and p-values are anti-conservative); only 10 subsamples.
