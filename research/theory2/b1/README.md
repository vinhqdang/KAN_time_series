# B1: does the fixed start of time-unrolled benchmark generators change how causal-discovery methods score and rank?

Plan: `PLAN.md`, committed and pushed (commit 6e51eb1) before any simulation that produced an algorithm score. Deviations D1 and D2 were recorded in `PLAN.md` §5 (commit 3d6b113) after the validation run and before the test run. D3 is below. No threshold or hypothesis was changed.

## Main-text paragraph (draft)

> **Does the start artefact matter for benchmarks?** We unrolled 420 summary graphs over T ∈ {3, 5, 10} slices: symmetrised ER and directed rings (strongly connected), chains, out-trees, in-trees and directed ER with c = 1, 2, at p ∈ {5, 10}. We simulated n = 1000 independent linear-Gaussian trajectories in two ways. G1 uses Reisach et al.'s fixed start, where slice 1 is pure noise. G2 discards a latent burn-in of 100 steps. The methods saw the pT variables with time labels hidden. The fixed start raised the order accuracy of rel-SortnRegress in every family, by +0.125 on average over the non-strongly-connected families (95% CI [0.116, 0.134]; Wilcoxon p < 1e-10 in each of 5 families, Holm). Across the 42 (family, p, T) cells, the gain tracked the exact population gap s_ρ(G_T) − s_ρ^stat of Theorems 1.5 and 1.9 (Spearman 0.80). It also appeared on strongly connected graphs at short horizons: +0.22 to +0.31 at T = 3, where s_ρ(G_T) is 0.93–1.0 instead of 1/2. The start also inflated var-sortability: Var-SortnRegress order accuracy rose by +0.20 on raw data and by 0.00 after standardisation. These shifts reversed method rankings. On SHD, 11 method pairs swapped significantly between G1 and G2 (Holm over 1,190 tests). On chains, rel-SortnRegress beat R²-SortnRegress under G1 and lost to it under G2. On in-trees, Var-SortnRegress beat FLOP and PC under G1 and lost to both under G2. FLOP was the best method on average under both generators. With time labels visible, sorting by time orients every edge (order accuracy 1). The artefact therefore matters only when the generator is used as Reisach et al. propose, i.e. as anonymous i.i.d. variables.

**Table (main text).** R-het regime (heterogeneous noise, raw data). Each entry is the mean over 30 graphs × 2 p × 3 T = 180 data sets per family. "OA" is the fraction of true edges that the method's order orients correctly. SID is normalised (lower is better). The Δ CI is a bootstrap over the 180 data sets.

| family | exact s_ρ G1 / G2 | rel-SnR OA G1 / G2 | Δ rel-SnR OA [95% CI] | Var-SnR OA G1 / G2 | R²-SnR OA G1 / G2 | rel-SnR SID G1 / G2 | Var-SnR SID G1 / G2 | FLOP SID G1 / G2 |
|---|---|---|---|---|---|---|---|---|
| symER (SC) | 0.782 / 0.500 | 0.627 / 0.501 | +0.126 [+0.110, +0.142] | 0.640 / 0.498 | 0.584 / 0.503 | 0.51 / 0.62 | 0.54 / 0.64 | 0.33 / 0.43 |
| ring (SC) | 0.954 / 0.500 | 0.687 / 0.507 | +0.180 [+0.162, +0.198] | 0.728 / 0.503 | 0.609 / 0.509 | 0.40 / 0.60 | 0.41 / 0.59 | 0.28 / 0.35 |
| chain | 0.887 / 0.500 | 0.749 / 0.532 | +0.217 [+0.199, +0.235] | 0.930 / 0.699 | 0.753 / 0.671 | 0.22 / 0.43 | 0.05 / 0.23 | 0.09 / 0.12 |
| out-tree | 0.844 / 0.500 | 0.694 / 0.550 | +0.144 [+0.125, +0.162] | 0.944 / 0.707 | 0.681 / 0.622 | 0.30 / 0.43 | 0.05 / 0.21 | 0.09 / 0.12 |
| in-tree | 0.792 / 0.664 | 0.743 / 0.673 | +0.069 [+0.057, +0.082] | 0.945 / 0.722 | 0.792 / 0.702 | 0.16 / 0.19 | 0.03 / 0.16 | 0.13 / 0.14 |
| ER(c=1) | 0.765 / 0.560 | 0.670 / 0.585 | +0.085 [+0.068, +0.102] | 0.855 / 0.646 | 0.686 / 0.604 | 0.21 / 0.27 | 0.12 / 0.23 | 0.12 / 0.14 |
| ER(c=2) | 0.830 / 0.520 | 0.698 / 0.588 | +0.109 [+0.093, +0.127] | 0.737 / 0.587 | 0.634 / 0.569 | 0.48 / 0.60 | 0.47 / 0.62 | 0.22 / 0.29 |

**Figure (main text).** `results/fig_b1.png` (pdf alongside). Panel (a) plots the per-cell G1−G2 gain in rel-SnR order accuracy against the exact population gap; all 42 cell means lie below the diagonal (attenuation from estimation) and above 0 (smallest +0.005, ER(c=1) p = 5 T = 10). Panel (b) shows normalised SID per method under G1 (o) and G2 (x). B2 figure: `results/fig_srho_vs_T.png` shows the exact s_ρ(G_T) against T = 2..20 next to the stationary value.

## Verdicts (test seeds, run once)

| hypothesis | rule (PLAN §2) | result | verdict |
|---|---|---|---|
| **H1** fixed start helps rel-SnR on non-SC families | ≥ 4/5 families Holm-significant and pooled mean ≥ 0.02 | 5/5 families. Holm p ≤ 1.1e-10. Family means +0.069 (in-tree) to +0.217 (chain). Pooled +0.125 [0.116, 0.134] over 300 graphs | **supported** |
| H1 tracking | Spearman(cell d, exact gap) ≥ 0.5, one-sided p < 0.05 | 0.798 over 42 cells (p = 1.2e-10). Unit level 0.616. OLS d = 0.53·gap − 0.027 | **supported** |
| H1 kill / contradiction | < 3/5, pooled < 0.01, or significant opposite sign | none | not triggered |
| **H2a** "small at T ≥ p" on SC families (review's form) | \|mean d\| < 0.03 in every SC cell with T ≥ p | fails: symER p=5 T=5 +0.116, ring p=5 T=5 +0.217, ring p=10 T=10 +0.079. Only symER p=5 T=10 (+0.019) is small | **fails, as predicted from the exact gap before the run** |
| **H2b** non-negligible at T < p | some SC cell with T < p has d ≥ 0.03, CI > 0 | all 6 such cells: +0.13 to +0.31 | **holds** |
| **H2c** symER gain decreases with T | d(T=3) > d(T=10) | +0.195, p = 1.2e-11 | **holds** |
| **H3** ranking changes (SHD) | ≥ 1 pair with opposite-sign, Holm-significant differences under G1 and G2 | 11 significant swaps (1,190 tests) | **supported** |

The 11 SHD swaps (R-het):
- **chain, p = 10, T = 3 and T = 5: rel-SnR vs R²-SnR.** rel-SnR is better under G1 (SHD −37 and −26) and worse under G2 (+15 and +25). This swap is driven by rel-sortability, i.e. the paper's mechanism.
- **in-tree, p = 5 and p = 10, T = 3, 5, 10: Var-SnR vs FLOP (3 cells) and Var-SnR vs PC (5 cells).** Var-SnR is better under G1 and worse under G2. In the G2 cells the differences are large, e.g. p = 5, T = 10: Var-SnR has 54 more errors than PC.
- **ring, p = 5, T = 10: R²-SnR vs Random-SnR.** This is a small effect.

Secondary metrics:
- SID: 9 significant swaps, including rel-SnR vs R²-SnR on chains, Var-SnR vs FLOP on chains, out-trees and in-trees, and rel-SnR vs PC.
- Orientation score: 9 significant swaps. Order accuracy: 2 significant swaps (rel-SnR vs R²-SnR, chain T = 3).
- SHD_x1, which excludes pairs inside the first window slice and so removes the sufficiency cost of the latent burn-in: 9 significant swaps remain. The swaps are therefore not an artefact of slice-1 confounding.
- The best method (lowest mean SHD) changes in 10 of 42 cells; the median Kendall τ between the G1 and G2 rankings is 0.73.
- Pooled over all cells: rel-SnR has higher order accuracy than R²-SnR under G1 (0.695 vs 0.677), and the order reverses under G2 (0.562 vs 0.597).
- What does not change: FLOP is the best method without time labels under both generators (normalised SHD 0.27 vs 0.46; SID 0.18 vs 0.23).

### Var- versus rel-sortability (PLAN §2, descriptive)

G1 − G2 differences, pooled over all families (per family: `results/analysis.txt`):
- **rel-SnR order accuracy:** +0.133 in R-het, +0.134 in R-eq. It is unchanged in R-std, since the method is invariant to scale.
- **Var-SnR order accuracy:** +0.202 in R-het and +0.244 in R-eq, but **+0.002 in R-std**.
- **Estimated var-sortability:** +0.202 in R-het, 0 in R-std.
- **R²-SnR order accuracy:** +0.080.

So the fixed start creates two separate artefacts. In the stationary window the variance of each node is constant over time. Under the fixed start, variance accumulates from slice 1, and the resulting var-sortability benefits Var-SnR. It disappears under standardisation. The relatives artefact predicted by Thm 1.5 benefits rel-SnR and survives standardisation. The largest ranking reversals (Var-SnR vs FLOP/PC on in-trees) come from the variance artefact. Only the rel-SnR vs R²-SnR reversals come from the mechanism the paper's theory describes.

### Sanity rules

| rule | result |
|---|---|
| S1: G2 vs exact stationary start G2s, rel-SnR OA, \|mean\| < 0.02 per family | between −0.013 and +0.009 → pass. G2s reproduces G2 |
| S2: estimated rel-sortability G1 > G2 in every non-SC family | +0.06 to +0.21 → pass |
| S3: mixing error < 1e-3 | 1.4e-10 (after D1) → pass |

### Cost of the latent burn-in (causal sufficiency)

Time-SnR uses the time labels and is a reference method only:
- normalised SHD is 0.044 under G1 and 0.273 under G2;
- excluding first-slice pairs, it is 0.041 vs 0.048. Almost all of its G2 errors are spurious edges among the confounded first-slice nodes.
- G2s gives the same numbers (0.273 / 0.048).

Any stationary-start benchmark should therefore either score the window without first-slice pairs or treat the window as an ADMG.

## B2: exact s_ρ(G_T) vs T and the Δ-sharpened bound

`results/exact_curves.txt` and `fig_srho_vs_T.png` give the mean over the 30 test graphs of each (family, p):
- **Short horizons are almost fully rel-sortable.** At T = 3, s_ρ(G_T) is between 0.81 (in-tree p = 5) and 1.00 (ring).
- **The approach to the limit is slow.** At T = 20, symER is still at 0.553 / 0.569 (p = 5 / 10) against the limit 1/2, and the ring at p = 10 is at 0.737. For the ring, s_ρ(G_T) = 1 exactly for every T ≤ p, because Δ = p − 1 and every copy of every edge lies within distance Δ of the start.
- **Non-SC families converge to L > 1/2.** Chains and trees tend to 0.72–0.74, while the stationary value is 0.50 for chains and out-trees and 0.66 for in-trees.
- **"T ≥ p" does not make the artefact small.** The excess decays like min(Δ, T−1)/(2(T−1)).

**Δ-sharpened bound.** 0 ≤ s_ρ(G_T) − L ≤ min(Δ, T−1)·|E_∘| / (2(T−1)|E_S|), where Δ is the largest finite shortest-path distance.
- Proof: in the proof of Thm 1.5, a value-1 copy of an E_∘ edge needs t ≤ d(u,i) < ∞, so t ≤ Δ.
- The burn-in condition of Thm 1.9 likewise only needs B ≥ Δ.
- Brute-force check by enumeration of relatives (`results/delta_check.json`), on 3,000 random summary graphs × 7 horizons = 21,000 cases:
  - **0 violations** of the upper bound, 0 of the lower bound;
  - the bound is attained in 3,164 cases;
  - it is strictly sharper than the p − 1 bound in 10,186 cases.
- Burn-in check, on 726 graphs:
  - B = Δ gave the stationary value in **726/726**;
  - B = Δ − 1 gave a different value in 445/726. So Δ cannot be lowered in general.
- No counterexample was found.

## Deviations

- **D1 (before the test run).** The burn-in was raised from 50 to 100. The validation run failed S3 (mixing error 2.8e-3; 9.0e-3 on a test parameter set with a non-normal chain); see PLAN §5.
- **D2 (before the test run).** SID is missing when PC returns a directed part with a cycle. PC is then dropped from the SID swap analysis in those cells (930 SID tests instead of 1,190).
- **D3 (test-run execution).** The first test-run attempt was killed by a container restart after about 900 of 1,800 tasks of the (symER, ring) shard.
  - It had written no results file. Only progress counters were seen, and no partial test scores were inspected.
  - The test run was then redone from scratch, with the same seeds, in resumable per-family shards (`run_shards.sh`). The shard files were concatenated into `results/b1_test.csv.gz`.
  - All seeds are deterministic, so the rerun reproduces exactly what the killed run would have produced.
- Not run:
  - DAGMA/NOTEARS (DAGMA needs torch, which is not installed).
  - PC at p = 10, T = 10 (35 s per data set; as planned).
- FLOP was run (PyPI `flopsearch` 0.3.0, the authors' implementation).

## Honest reading for the paper

1. **What B1 shows.** The benchmark impact is real and is predicted by the theory. Hiding the time labels and changing only the start of the trajectories moves the order accuracy of sort-based baselines by 0.07–0.22. The size of the shift ranks with the exact s_ρ gap (Spearman 0.80). In some cells it reverses significant pairwise comparisons.
2. **The R-std result is a second, separate artefact.** It shows that the fixed start also creates var-sortability. That has practical weight in its own right, since Reisach et al. propose these generators partly to escape sortability artefacts.
3. **Limits of the result.**
   - The reversals concern baselines (rel-SnR vs R²-SnR; Var-SnR vs FLOP/PC on tree-shaped graphs). FLOP, the strongest method, stays on top on average under both generators.
   - The most dramatic reversals come from the variance artefact, not from rel-sortability.
   - The setting is a single linear-Gaussian parametrisation with positive weights and n = 1000.
   - The SC families show that "strongly connected" (Reisach et al.'s Thm 4 condition) does not protect a short-horizon benchmark. For T ≤ 10 the exact s_ρ is still 0.6–1.0.
4. **What the paper can claim.** "The fixed start inflates rel- and var-sortability by amounts that Thm 1.5/1.9 predict exactly, and benchmark conclusions about sort-based baselines change accordingly." It should not claim that the state-of-the-art ranking flips.
5. **Remedy.** Use a burn-in of at least Δ (not p − 1). Standardise. Score windows without first-slice pairs, or as ADMGs, because the burn-in breaks causal sufficiency in slice 1.

## Files and reproduction

- `common.py`: graphs, SEM, generators, exact s_ρ (Thm 1.3/1.5/1.9) and an enumeration check, methods, metrics.
- `exact_curves.py {predict|curves|delta} results`: graph-only computations (predictions, B2 curves, Δ check).
- `run_b1.py`: one row per data set. `run_shards.sh`: resumable test run (about 2 h on 2 cores).
- `analyse_b1.py results/b1_test.csv.gz results`: produces `analysis.txt`, `analysis.json` and `fig_b1.*`.
- Environment:
  - Python 3.13, numpy 2.5.4, scipy 1.16.3, scikit-learn 1.9.1, pandas 3.0.5;
  - causal-learn 0.1.4.8, flopsearch 0.3.0, gadjid 0.1.0.
  - Always set `OMP_NUM_THREADS=1`.
- `results/b1_test.csv.gz` (6,300 data sets, 670 KB) is committed. Shard files and simulated data are not.
