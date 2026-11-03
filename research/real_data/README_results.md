# Sachs real-data results (protocol: PLAN.md, committed before running)

Code: run_sachs.py (resume-safe, 1 thread, ~10 min total), analyze_sachs.py. Raw: results/structure.csv, results/scores/, results/curves/; full output results/analysis.txt; figure fig_curves.png. 10 subsamples of 2000 cells (seeds 0-9), log + z-score. All numbers are mean over the 10 subsamples; CIs are bootstrap over subsamples. The subsamples overlap (all drawn from 7,466 cells), so they are not independent and CIs are optimistic.

## Structure (18 consensus edges; chance AUPRC 0.164)
| method | AUROC (95% CI) | AUPRC | skeleton AUROC | orientation acc. | hits in top 18 |
|---|---|---|---|---|---|
| SSC | **0.613** [0.591,0.639] | **0.291** | 0.690 | 0.51 | 5.8 |
| SSR_k1.5 | 0.595 [0.569,0.620] | 0.266 | 0.688 | 0.53 | 6.1 |
| BASIL-W | 0.563 [0.529,0.596] | 0.248 | 0.703 | 0.58 | 5.8 |
| BASIL (full) | 0.544 [0.513,0.574] | 0.227 | 0.683 | 0.51 | 5.3 |
| GAM-ridge | 0.584 [0.570,0.598] | 0.287 | 0.735 | 0.40 | 6.1 |
| NOTEARS-linear | 0.566 [0.557,0.575] | 0.239 | 0.634 | 0.53 | 5.0 |
| DAGMA-linear | 0.435 [0.416,0.450] | 0.168 | 0.688 | 0.31 | 3.0 |
| var-sortnregress (z-scored, order = ties/noise) | 0.503 [0.460,0.544] | 0.207 | 0.664 | 0.44 | 3.3 |
| var-sortnregress-logscale (extra, not in PLAN) | 0.555 [0.530,0.578] | 0.255 | 0.659 | 0.49 | 4.1 |

Paired vs SSC (AUROC diff, Wilcoxon): SSR -0.017 [-0.034,+0.002] p=0.049; BASIL-W -0.050 [-0.097,-0.008] p=0.105; BASIL -0.069 [-0.106,-0.033] p=0.014; NOTEARS-linear -0.047 [-0.079,-0.019] p=0.012 (so SSC > NOTEARS-linear by 0.047; AUPRC +0.053 [0.013,0.099], p=0.084). SSR and BASIL-W vs NOTEARS-linear: +0.029 [0.000,0.060] and -0.003 [-0.038,+0.028], ties. Orientation differences are all within noise (CIs span 0) except DAGMA-linear and GAM-ridge, which orient worse than chance-level 0.5.

Verdict (PLAN): structure POSITIVE by the pre-registered rule only for SSC (AUROC 0.613 >= 0.60, AUPRC 0.291 > 0.214, CI excludes 0.5), but weakly: about one extra true edge in the top 18 versus chance expectation (about 2.9 hits at chance, 5.8 observed), AUROC 0.61. SSR passes the "no loss" rule (-0.017 > -0.03) and is itself just below the 0.60 positive bar (0.595). BASIL-W and full BASIL lose more than 0.03 AUROC vs SSC (-0.050, -0.069): the "no loss" criterion FAILS for them, consistent with the direction on synthetic data but larger. Full BASIL is no better than BASIL-W. DAGMA-linear is below chance (its error is systematic, not noise). I expected DAGMA/NOTEARS to be weak and they are; I did not expect full BASIL to be the weakest of our methods.

## Function stability (mean pairwise R2_aff of an edge's curve across subsamples)
- PLAN definition (edges in the method's own top 18 in >=3 subsamples; curves only from subsamples where the edge is top-18): SSC 0.986 (24 edges), BASIL-W 0.950 (26), BASIL 0.954 (21), GAM-ridge 0.974 (18), SSR 0.832 (24). Consensus edges only: SSC 0.988, BASIL-W 0.957, SSR 0.835.
- Common edge set (24 edges that are SSC top-18 in >=3 subsamples), counting an unselected edge (zero curve) as R2=0, paired over subsamples: SSC 0.978, GAM-ridge 0.945, BASIL 0.618, BASIL-W 0.538, SSR 0.491; SSR - SSC = -0.487 [-0.504,-0.472], BASIL-W - SSC = -0.440 [-0.482,-0.405], both Wilcoxon p=0.002. This is dominated by selection instability: SSR / BASIL-W / BASIL return a zero curve for 35% / 34% / 28% of (edge, subsample) cells, SSC never does (its lasso penalty leaves small non-zero curves).
- Conditional on the edge being selected in both subsamples, R2_aff is high for all: SSC 0.978, SSR 0.968, BASIL-W 0.961, BASIL 0.953.
- Verdict: by the pre-registered rule, SSR and BASIL-W are "LESS STABLE" than SSC (CI entirely below 0). The honest reading: the corrected curves are stable in shape when the edge is kept, but the edge set itself flips between subsamples, much more than for SSC. That is a negative result for using the corrected methods as-is on real data. A caveat in the other direction: SSC's 0.98 reflects, in part, that it is a smooth, always-on fit that changes little with the sample; high stability is not correctness. Stability here is optimistic (overlapping subsamples).
- Stability is not evidence of causal meaning: the top SSC edges by mean score are p44/42 -> pakts473 (not in consensus), pmek -> praf (reverse of consensus), PKC -> P38 (consensus), PKA -> p44/42 (consensus), plcg -> PKA (not in consensus), pjnk -> PKC (reverse). Only 2 of the top 6 are consensus edges in the consensus direction.

## Amplitude (std of centred curve, on the SSC top-18 edges of each subsample)
SSR / SSC = 1.60 [1.57,1.64], BASIL-W / SSC = 1.41 [1.32,1.50], BASIL / SSC = 1.22 [1.15,1.30], GAM-ridge / SSC = 1.63. Mean curve std: SSC 0.243, SSR 0.385, BASIL-W 0.316. Shrinkage of SSC is visible (CI excludes 1), the direction predicted by the simulations. The size (1.4-1.6x) is below the ~2.3x (1/0.44) the synthetic slope would imply, and there is no ground truth to say which amplitude is right. GAM-ridge, an unpenalised-ish reference, agrees with SSR in amplitude, which is mild support for SSR.

## Sign sanity (weak, see caveat)
Marginal correlations in the log data are praf-pmek +0.785, pmek-p44/42 -0.071, PIP3-pakts473 -0.135, so for two of the three textbook-activating relations the data themselves do not show a positive dependence, and the test is mostly uninformative.
- praf -> pmek: curve sign positive in 10/10 (SSC) and 10/10 (GAM-ridge); SSR / BASIL-W select the edge in only 3 / 1 of 10 subsamples (always positive when selected).
- pmek -> p44/42: SSC positive 2/10, negative 8/10; BASIL-W and GAM-ridge negative 10/10 (BASIL-W puts probability 1.0 on this edge); SSR never selects it. Wrong or non-monotone relative to the textbook sign.
- PIP3 -> pakts473: SSC positive 1/10, GAM-ridge 2/10; the other methods never select it.
The three relations are standard MAPK/PI3K signalling that I did not verify against the text of Sachs et al. 2005 in this session. Sign flags per PLAN: SSC, BASIL-W, GAM-ridge are flagged (wrong sign for the majority of subsamples on pmek -> p44/42). Sign errors may reflect the pooled interventional conditions (e.g. U0126 inhibits MEK activity, not phosphorylation, so pmek level need not track pERK) rather than method error; this cannot be separated here.

## Overall verdict against PLAN.md
Structure: weakly positive for SSC, ambiguous for SSR, negative-leaning for BASIL-W and BASIL (they lose AUROC relative to SSC and tie NOTEARS-linear). Stability: corrected curves less stable than SSC through selection flips (negative on the pre-registered criterion), shape-stable otherwise. Amplitude: shrinkage visible, plausible. Sign sanity: not supportive, but uninformative. The pre-registered overall-NEGATIVE condition (structure negative AND corrected curves less stable) is NOT met because SSC structure passes; but for the paper's specific claim that refit/BASIL-W improve edge functions, the real-data evidence is neutral to unfavourable: no way to check accuracy, and the structure/selection stability cost is visible.

## Caveats
- d=11, 18 edges: a difference of 0.05 AUPRC is about one edge. The consensus graph is contested and, as research/semisyn/gen.py found, contains one directed cycle (so it is not a DAG as used here); reverse-of-consensus edges (pmek -> praf, pjnk -> PKC) rank high and are counted as errors although feedback is known biology.
- The pooled data mix interventional conditions (not iid, not one SEM); interventions are not used.
- Subsamples overlap (~27% pairwise) and share cells; bootstrap/Wilcoxon p-values are anti-conservative; Wilcoxon with n=10 has minimum p 0.002.
- No ground truth for functions: stability and amplitude measure reproducibility and shrinkage, not accuracy.
- Same subsamples/seed as the earlier experimental_results/sachs_raw.csv runs (disclosed in PLAN; SPADE numbers there are similar, e.g. 0.60 AUROC). The plan's SSC is the SPADE-scaled variant, whose Sachs numbers had not been seen before.
- var-sortnregress on z-scored data has all variances equal, so its order is tie-breaking noise (as in the synthetic iSCM benchmark); an extra variant on the pre-z-score log variances is reported separately and was not in the plan.
- An initial run hit multi-threading (BLAS) and was restarted pinned to one core; resume reused only the first subsample's completed rows (identical seeds, deterministic code).
- Sign-sanity relations are from memory of the standard cascade, not verified in this session.
