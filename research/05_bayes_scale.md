# 05 — Scaling BASIL (candidate-parent screening + local order MCMC)

Committed BEFORE the scalable code runs. Motivation (user): without good scaling the method is not worth pursuing. Full BASIL costs ~11 min/dataset at d=50 (subset enumeration over all d-1 candidates).

## Method change (BASIL-S)
1. **Candidate screening.** For node i, rank all j by marginal spline-R² of x_i on f(x_j) (closed form: whitened features, R²_ij = ||Z_jᵀ y_i||² / (n ||y_i||²)); keep the top m = 20 as candidate parents. Cost O(n d² K).
2. **Per-node score tables only over subsets of the candidates**, |pa| ≤ kmax = 3, same g-prior marginal likelihood as BASIL (g = n, π = 2/(d-1)). Sets per node: 1+20+190+1140 = 1351, independent of d.
3. **Order MCMC with local moves** (adjacent swaps and swaps within distance ≤ 5; only affected nodes are rescored), initialised by a greedy order (repeatedly add the node whose best score given already-placed nodes improves most over its empty-parent score).
4. Same outputs: edge posterior probabilities; model-averaged curves for true-edge slope diagnostics.
Known risk: a true parent weak in marginal R² (e.g. effects that only show conditionally) is dropped at step 1 and can never be recovered. We therefore report **screening recall** (share of true edges inside the candidate sets) in every cell.

## Hypotheses and kill criteria
Generator/bench as in 03 §7 (iSCM, ER mean degree 2, uniform noise). Validation seeds 0–4 are used only to confirm m and the move set at d=50; no tuning on test seeds 1000+.
- **S1 (approximation is faithful).** At d=50, n=500, BASIL-S AUROC within 0.02 of full BASIL on the same datasets (6 datasets already available in results/cells) and ≥ 20× faster. Kill: gap > 0.03 or speed-up < 10×.
- **S2 (scales).** d ∈ {100, 200}, n=1000, 5 datasets each (d=500 attempted if time permits): AUROC ≥ 0.85 and wall-clock ≤ 10 min per dataset at d=200. Kill: AUROC < 0.80 or time > 15 min or screening recall < 0.85.
- **S3 (competitive with the scalable incumbent).** BASIL-S AUROC ≥ SPADE-scaled − 0.02 and AUPRC ≥ SPADE-scaled − 0.02 (SPADE-scaled = project's λ=0.2/d + ExtraTrees K=15, frozen settings, run on the same iSCM datasets). Kill: AUROC gap beyond −0.05.
- **S4 (what BASIL-S adds).** Orientation accuracy ≥ SPADE-scaled − 0.03; curve slope on found true edges within [0.85,1.10]; ECE ≤ 0.10. Kill: orientation < SPADE-scaled − 0.08.
- **S5 (control).** var-sortnregress AUROC at least 0.15 below BASIL-S.

## Limitations stated in advance
Order MCMC mixing at d ≥ 100 with local moves is untested; only 5 datasets per cell (CIs will be wide, no significance claims); SPADE-scaled takes ~9 min/dataset at d=200 so baselines limit the number of datasets; novelty vs DiBS/BayesDAG unresolved.
