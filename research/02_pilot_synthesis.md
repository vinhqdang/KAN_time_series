# Pilot synthesis (five parallel pilots, G1-G5)

Numbers below are the pilot agents' own reports, spot-checked for consistency across pilots but NOT independently re-run. Each pilot was small (10-30 graphs, one generator family, about 20-45 minutes on one CPU thread), so treat them as feasibility signals, not results.

| Pilot | Direction | Verdict | What held up | What did not |
|---|---|---|---|---|
| G1 | Debiased/stable edge selection | marginal | Group-lasso shrinks curve amplitude: calibration slope 0.67 (d=10), 0.48 (d=20). Adaptive group-lasso restores it (0.95-0.96), curve R2 +0.11/+0.29, better on 20/20 graphs, ranking mostly unchanged | Refit + stability selection (the full pipeline) lowered AUPRC by 0.03-0.04; P-spline smoothness did not help |
| G2 | Shape-fidelity metric/benchmark | marginal | Metric suite is cheap; "edge found" and "shape faithful" differ by 0.2-0.3 recall; concurvity hurts discovery more than curves | Rankings mostly the same as AUROC; a plain GAM-ridge baseline (no DAG, no sparsity) matched SPADE; class-agreement metric unreliable |
| G3 | Interpretable additive discovery at scale | not worth pursuing as built | Screened spline fit with the TRUE order: AUROC 0.93-1.00 up to d=200 | Estimated order agrees with only 49-64% of true edges (50% = chance); never reached the earlier acyclicity-based screened SPADE; non-linear baselines unavailable at this size |
| G4 | Benchmark validity audit | methods content, weak alone | Project generator is strongly varsortable (0.91-0.95); a variance-sorting baseline beats NOTEARS/DAGMA/GOLEM and trails SPADE by only 0.04-0.07 (SPADE loses at d=50); on a standardized generator SPADE's margin over NOTEARS-linear survives (+0.06 to +0.32); run-to-run noise floor about 0.04-0.06 AUROC (threads, float32 vs float64) | Mostly established ideas; sortability audits exist (Reisach et al.) |
| G5 | Non-stationary temporal data | marginal | SPADE on levels fails under I(1) (AUPRC 0.35-0.52 below differences) | Error-correction recipe added only +0.022 AUPRC (t about 2.6, no multiplicity correction); easy regime where linear baselines also reach AUROC ~1 |

## Cross-pilot consistency
- G1 and G4 both used a standardized (iSCM-style) generator: varsortability 0.55 and 0.5 respectively. G3's "std" generator was NOT sortability-free (0.76-0.88), so its sortability claims are void.
- G1 (amplitude bias) and G2 (shape fidelity) point at the same object, the learned curve; G2's GAM-ridge result is the main threat to any claim that SPADE's curves are better than a simple additive fit.
- G4's noise floor (about 0.04-0.06 per run) sits above single-run gains reported in the earlier ad-hoc extension (+0.014); only paired designs over at least 30 datasets can support such gains.

## Recommendation (for discussion)
Merge G1 + G2 (+ G4 as the evaluation protocol) into one study: measure and repair amplitude/shape bias of sparse spline edge functions in additive-noise discovery, evaluated on standardized generators against real additive baselines (CAM or GAM backfitting, NAM, DAGMA-DCE, KAN), at least 30 datasets per cell, a variance floor, and a sortnregress control. G5 is a limits appendix. G3 needs a better order estimator before it is worth anything.
