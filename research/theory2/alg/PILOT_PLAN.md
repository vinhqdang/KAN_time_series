# DRAFT — pilot plan for TiDAL (not to be run before the author approves it)

Status: **DRAFT, 2026-11-10. Nothing in this plan has been run.** No code exists yet. The plan is to be frozen (commit and push) before any pilot score is computed; until then it may be edited.
Method and notation: `DESIGN.md`. Reason for the pilot: `NOVELTY.md`, Section 5.

## 0. Question

Does a learner that is given no time labels, but assumes that the columns are an unrolled lag-1 process with tied parameters (TiDAL, `free` start law, standardised input), recover the window DAG of a burn-in sample clearly better than FLOP, the best label-free method of B1 and B'1?

The pilot decides only whether the method is worth a confirmatory experiment. Its numbers are not evidence for the paper and are not to be reported there as results.

## 1. Data (validation seeds only)

Generator, SEM and metrics are those of B1 (`../b1/common.py`), imported unchanged.
- Families: `ring`, `chain`, `intree`, `ER2` (the B'1 families).
- p ∈ {5, 10}, T ∈ {3, 5}, so d ∈ {15, 25, 30, 50}; n = 1000 trajectories.
- Generators: G2 (burn-in 100; primary) and G1 (fixed start; secondary).
- Weights Unif(0.5, 1), spectral cap 0.8, heterogeneous noise (R-het), as in B'1.
- **Seeds 9700–9703**: 4 summary graphs per (family, p), 32 graphs, **128 data sets**.
  - Graph and parameter RNG: `default_rng([401, family_index, p, seed])`.
  - Data RNG: `default_rng([402, family_index, p, seed, T, gen_index, 17])`.
  - Column shuffle RNG: `default_rng([403, family_index, p, seed, T, gen_index])`.
  - These streams are disjoint from the B1 test seeds (1000–1029, no prefix), B1-NL (prefixes 101–203), B'1 (3000–3011, prefixes 301/302) and the earlier validation seeds (9000–9004, 9100–9102, 9600).
- Reserve seeds 9710–9713 (same prefixes) for the single allowed revision (Section 6).
- Every method receives the same shuffled, column-standardised matrix. PCMCI+ receives the labelled raw data.

## 2. Methods

All TiDAL settings are those of `DESIGN.md`, Section 4, fixed before the run: 30 restarts (1 spectral, 29 random, perturbation seed = data seed), FAQ tolerance 1e-6, at most 300 iterations, ridge 1e-6, exact best-subset BIC for the rows of A, T chosen by BIC among the divisors of d with T ≥ 3 and p ≥ 2.

| id | method | role |
|---|---|---|
| M1 | **TiDAL**, `free` law, T selected | primary |
| M2 | TiDAL, true (p, T) given | shows the cost of selecting T |
| M3 | TiDAL with the oracle alignment (only the support of A is learned) | ceiling of the model |
| M4 | TiDAL-untied (shape tying only), true (p, T) given | ablation of parameter tying |
| M5 | FLOP, B'1 settings (flopsearch 0.3.0, BIC penalty 2, 10 restarts) | the method to beat |
| M6 | two-stage baseline: spectral seriation into slices, then PCMCI+ on the recovered labels (`DESIGN.md`, Section 8) | is the likelihood needed? |
| M7 | PCMCI+ with the true labels (B'1 settings) | ceiling with labels |
| M8 | empty graph | reference |

DAGMA and NOTEARS are **not** rerun (a NOTEARS fit at d = 50 took up to 151 s in B'1). Their B'1 test levels (SHD/|E| 0.90 and 0.89 under G2, raw) are context only.
TiDAL-s and the Gumbel–Sinkhorn variant are not part of the pilot.

## 3. Outcomes

Per data set and method, as in B'1: SHD/|E| (B1's SHD; the empty graph has 1), SHD_x1/|E| (pairs inside the first slice excluded), ancestral order accuracy, number of edges, run time.
For M1, M2 and M4 also:
- slice accuracy: share of columns placed in the correct slice (well defined by T1);
- whether the selected T is the true one (M1);
- BIC gap: BIC(found) − BIC(M3). A positive gap means the optimiser missed a better alignment (risk R3).

**Primary outcome**: SHD_x1/|E| under G2, M1 against M5. SHD_x1 is primary because the truth graph has no edges inside slice 1 while the window distribution has dependence there; FLOP is penalised for edges it is right to find, and TiDAL does not output them by design. SHD/|E| is reported next to it.

## 4. Kill criterion (fixed before the run)

Let Δ = [metric of FLOP] − [metric of TiDAL (M1)], per data set, under G2, standardised input.
The method goes forward to a confirmatory experiment **only if K1, K2 and K3 all hold**:

- **K1 (size).** Mean Δ over the 64 G2 data sets is at least **0.10** on SHD_x1/|E| **and** at least **0.20** on SHD/|E|.
- **K2 (breadth).** TiDAL has the lower mean SHD_x1/|E| in at least **6 of the 8** (family, p) cells under G2 (8 data sets per cell).
- **K3 (level).** The mean SHD/|E| of TiDAL under G2 is at most **0.25**. (FLOP was at 0.52 and PCMCI+ with labels at 0.02 in B'1; 0.25 is slightly better than halfway.)

If any of K1–K3 fails, the method is dropped from the paper. The identifiability statements T1–T4 of `DESIGN.md` do not depend on the pilot and may still be used.

Conditions that do not kill but restrict what may be claimed:
- **C1 (fixed start).** If under G1 the mean SHD/|E| of M1 exceeds FLOP's by more than 0.05, the method is claimed for burn-in windows only, and the fixed start needs TiDAL-s.
- **C2 (two-stage baseline).** If M6 is within 0.05 of M1 on the primary outcome, the paper must say that a simple pipeline recovers the labels as well, and the likelihood method is presented as one of two constructions.
- **C3 (parameter tying).** If M4 is within 0.03 of M2, the paper must say that the gain comes from the shape of the unrolling and not from tied weights.
- **C4 (selection of T).** If the selected T is wrong in more than 10% of the data sets, only M2 (T given) may be claimed, with that caveat.

Diagnostic, not a criterion: if the BIC gap is positive in more than 20% of the data sets, the failure is attributed to the optimiser (R3), and the single revision of Section 6 may be used.

## 5. Stage 0: population checks (no data, no scores)

Run before Stage 1; they use exact covariances and graphs only.
- **S0a (layerings).** For every summary graph with loops on p ≤ 4 units (up to isomorphism) and T ∈ {3, 4}: enumerate the layerings of U (`DESIGN.md`, C5). Output: the list of graphs with more than one layering.
- **S0b (brute force).** For (p, T) ∈ {(2, 3), (2, 4), (3, 3)} and 20 random parameter draws per weakly connected summary graph (B1 weight law): evaluate the exact population Kullback–Leibler value of every alignment (720, 40,320 and 362,880 alignments), with dense A. Output: whether all minimisers have the true column DAG.
- **S0c (reversal).** For the 32 pilot graphs: the largest off-diagonal entry of D̃ (`DESIGN.md`, T4), scaled.
- **S0d (faithfulness).** For the p = 5 pilot graphs: the smallest absolute partial correlation among the adjacent pairs of H and the largest among pairs that H separates, on exact window covariances, for conditioning sets of size ≤ 2.

Gate: if S0b finds a draw whose minimisers include a wrong column DAG outside the cases of T4, stop; revise C5 and this plan before any data are simulated. S0a, S0c and S0d are reported and decide how C5 is worded.

## 6. Rules

- No hyperparameter is tuned on the pilot seeds. If any setting of Section 2 is changed after a pilot score has been seen, the change is logged as a revision and the pilot is rerun **once** on the reserve seeds 9710–9713; only that rerun counts for K1–K3. A second revision is not allowed; the method is then dropped.
- The pilot is run once per seed set, in resumable shards, one row appended per finished data set.
- All data sets enter the means. A method failure (exception, non-finite output) counts as the empty graph (SHD/|E| = 1) and is listed.
- Statistics are descriptive: means per cell and pooled, with a graph-cluster percentile bootstrap interval (10,000 resamples, seed 0) for the K1 differences. No significance claim is made from 32 graphs.

## 7. Budget

Limit: **3 CPU-hours on one core** (`OMP_NUM_THREADS=1`) for Stages 0 and 1 together.
- Stage 0 estimate: under 20 minutes (S0b at (3, 3) dominates).
- Stage 1 estimate, not measured: TiDAL M1 10–60 s per data set, M2–M4 less, FLOP and PCMCI+ a few seconds; about 1.5–2 h for 128 data sets.
- Timing step first: seed 9700 only, recording seconds and no metric. If it projects more than 2.5 h for Stage 1, restarts are cut from 30 to 10 for all TiDAL variants before any score is computed, and this is logged. If the projection is still above 2.5 h, the candidate values of T are limited to those with p ≥ 5. No other change is allowed.

## 8. If the pilot passes

A confirmatory experiment needs its own plan, committed before its run, with fresh test seeds (for example 5000–5011, new RNG prefixes), DAGMA and NOTEARS rerun, the families and horizons of B'1, at least one B1-NL configuration (nonlinear or non-Gaussian), and hypothesis tests with the summary graph as the unit. The thresholds of that plan should be set from the pilot, and the pilot data sets must not be reused.

## 9. Files to be created (none exists yet)

`tidal.py` (method), `stage0.py`, `run_pilot.py`, `analyse_pilot.py`, `pilot/` (results). All under `research/theory2/alg/`.
