# G4 pilot results: benchmark validity / sortability / variance floor

Protocol: docstring of `g4_audit.py` (written before runs). Raw outputs: `results_A_raw.csv`, `results_B_std.csv`, `variance_floor_*.csv`, `analysis_output.txt` (`analyze.py`). Seeds 800-829 only. n=1500. Seeds per cell: d=6,10: 8; d=20: 6; d=50: 4 (so SEs are wide). A_raw = project generator; B_std = same graph/mechanisms/noise, each variable z-scored right after generation.

Self-caught bug: the first random-order control used RandomState(seed), the same stream the generator uses for its permutation, so it silently used the TRUE order (AUROC 0.63-0.92 instead of ~0.6-0.7). Fixed (seed+1e6) and recomputed; old values kept in column `leaked_auroc_old`. Same class of leakage the checklist warns about.

## Sortability (mean over seeds)
| d | A varsort raw | A R2sort | B varsort raw* | B R2sort |
|---|---|---|---|---|
| 6 | 0.950 | 0.525 | 0.51 | 0.564 |
| 10 | 0.921 | 0.602 | 0.35 | 0.493 |
| 20 | 0.909 | 0.446 | 0.56 | 0.394 |
| 50 | 0.912 | 0.573 | 0.52 | 0.488 |
After z-scoring varsortability is 0.5 by construction (my numbers deviate, 0.46-0.68, only through float-rounding ties; ignore). *B's value is rounding noise around 0.5. R2-sortability is scale-invariant (same raw/standardized). The project generator is strongly varsortable (0.91-0.95; Reisach et al. flag >0.9 as the exploitable regime) but near-chance R2-sortable.

## AUROC mean (sd across seeds)
| gen | d | var-sortnreg | R2-sortnreg | random-order sortnreg | SPADE | NOTEARS-lin | DAGMA-lin | GOLEM | NOTEARS-MLP |
|---|---|---|---|---|---|---|---|---|---|
| A | 6 | .898 | .618 | .589 | .941 | .687 | .540 | .605 | .631 |
| A | 10 | .879 | .722 | .691 | .947 | .871 | .606 | .808 | .593 |
| A | 20 | .853 | .651 | .634 | .915 | .849 | .595 | .751 | .649 |
| A | 50 | .863 | .725 | .682 | .823 | .797 | .589 | .779 | .498 |
| B | 6 | .595 | .666 | .584 | .977 | .661 | .509 | .563 | .650 |
| B | 10 | .592 | .698 | .692 | .996 | .834 | .534 | .755 | .648 |
| B | 20 | .649 | .644 | .636 | .996 | .824 | .573 | .743 | .683 |
| B | 50 | .682 | .695 | .675 | .870 | .814 | .553 | .730 | .501 |
(sd typically 0.02-0.16; see `analysis_output.txt`.)

Findings:
1. On the raw project generator a trivial variance-sorting + lasso baseline reaches AUROC 0.85-0.90, which beats every fast baseline at every d and beats SPADE at d=50 (paired SPADE minus var-sortnregress: +0.04, +0.07, +0.06, -0.04 for d=6,10,20,50). Standardizing the generator drops var-sortnregress to 0.59-0.68 and SPADE rises (d=10: 0.947 to 0.996; d=20: 0.915 to 0.996). So the raw generator is NOT a fair test of causal structure learning for any method that can see scale.
2. The five methods z-score internally, so they never see raw variances, and per-dataset Spearman correlation of their AUROC with varsortability is weak (SPADE -0.05, NOTEARS-linear -0.08, GOLEM -0.19, DAGMA-linear -0.25, n=26 datasets, not significant). Hence their AUROC is mostly NOT explained by varsortability; the benchmark's flaw is that a scale-aware baseline is a free strong competitor, not that SPADE exploits it. R2-sortnregress tracks its own R2-sortability (rho 0.66-0.70) and is a weak baseline here (0.62-0.73).
3. Standardizing the generator widens SPADE's margin over linear baselines (d=20: +0.07 to +0.17 over NOTEARS-linear) and the claim survives, in the narrow sense of this generator family. NOTEARS-MLP is near chance at d=50 (0.50) under both generators (probably under-trained at 600 epochs; not tuned). SPADE degrades at d=50 (0.82/0.87, 4 seeds).
4. Caveat: this generator has bounded smooth mechanisms and uniform noise, so internal standardization removes much of the variance signal anyway; the result may not carry to other generator families. 4-8 seeds per cell.

## Variance floor (SPADE, d=10, A_raw)
- Between datasets (30 seeds, 800-829): SPADE 0.927 sd 0.060; NOTEARS-linear 0.832 sd 0.071. Paired SPADE minus NOTEARS-linear: +0.095, sd 0.078, SE 0.014. With 5 seeds the SE would be ~0.035.
- Same data, method seeds 0-4: within-dataset sd 0.022 on average (per-dataset range max-min 0.012 to 0.109; dataset 804: 0.935 +/- 0.044).
- Re-run, same data, same seed, same threads: bit-identical (dAUROC 0).
- Threads 1 vs 2 (same seed): dAUROC RMS 0.036 (values 0, .032, .032, -.022, .062) although max normalised score change is only 0.001-0.007. AUROC is a rank metric over ~90 pairs with near-ties among tiny scores, so tiny float perturbations flip ranks.
- float32 vs float64 (same seed): dAUROC RMS 0.054 (-.015, -.085, +.052, -.014, +.065); two datasets changed scores by 0.83 and 0.99 (different optimisation basin). This matches the project's observed 0.926 vs 0.892 d=10 gap between library environments (0.034), i.e. that gap is inside the implementation-noise floor.
- Practical floor for d=10: a single-run AUROC from one method seed and one dataset has SD about 0.04-0.06 from implementation noise alone (0.022 seeds, 0.036 threads, 0.054 dtype, partly overlapping); a claimed gain below ~0.03-0.05 AUROC must be shown paired over >=30 datasets with method-seed repeats and an SE reported.

## Recommended evaluation protocol (checklist)
1. Report varsortability and R2-sortability of every generator (raw and after standardization) per d.
2. Always include var-sortnregress, R2-sortnregress and a random-order control; if var-sortnregress is within one SE of the proposed method on raw data, the raw benchmark is uninformative.
3. Evaluate on internally standardized (or equal-noise-variance) SCMs as the primary setting; raw-scale data only as a labelled secondary setting.
4. Control order leakage: never derive baseline orders, seeds or hyperparameters from the generator's RNG stream (we hit this ourselves); use disjoint seed ranges for tuning and reporting.
5. >=30 datasets per cell at d<=20; report paired differences with SE, not unpaired means.
6. Report method-seed variance on fixed data (>=5 seeds) separately from between-dataset variance.
7. Pin and log torch/numpy/BLAS versions, dtype, thread count, device; report AUROC under at least float32 and float64 and 1 vs N threads; treat gains below the measured implementation floor as null.
8. Use identical preprocessing (z-scoring) for all methods and state it; give binary-output methods (SCORE, NoGAM) a separate metric or an edge-score variant.
9. Release raw per-seed CSVs and the exact generator code.
10. Do not tune on reporting seeds; log everything tried.

## Limits
Single n, single generator family, small seed counts at d=20,50; SCORE, NoGAM, GraN-DAG, DAGMA-nonlinear not run; NOTEARS-MLP and others untuned; thread test used only 1 vs 2 threads; library-version effect (the project's two environments) not reproduced directly, only dtype/thread proxies.
