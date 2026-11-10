# Design: TiDAL, tied DAG learning with latent alignment

Date: 2026-11-10. Status: design only. Nothing here has been run, simulated or checked numerically.
Statements are marked **[proved here]** (short paper proof given), **[standard]**, or **[conjecture]**.
Background and prior work: `NOVELTY.md`. Pilot (DRAFT): `PILOT_PLAN.md`.

## 1. Problem

Data: n i.i.d. rows Z ∈ R^d, d = pT. The columns are anonymous. The generator is the one of the paper (`../b1/common.py`):
- a summary graph G_S on p units with a self-loop at every unit;
- X_{t+1} = A X_t + ε_t for t = 1..T−1, with A_ji ≠ 0 iff i → j in G_S, and ε_t ~ (0, D), D diagonal, the same A and D for every t;
- a start law for X_1;
- Z = Π X for an unknown bijection Π from the d cells (unit j, slice t) to the d columns.

Target: the **column DAG** G*, with an edge Π(i,t) → Π(j,t+1) for every summary edge i → j and every t < T. This is the ground truth used in B1 and B'1 (`true_dag`).
The alignment Π itself is a means, not the target. It is not unique (T1 below), and it need not be.

Not given to the method: Π, G_S, A, D, the start law. p and T are also not given; only d is (Section 4.4).

## 2. Model

Parameters θ = (Π, A, D, Σ_1), with A_jj ≠ 0 for all j and D diagonal positive.

Three start laws:

| name | Σ_1 | covers |
|---|---|---|
| `free` (default) | any positive definite matrix | fixed start, burn-in window, exact stationary start, any other start |
| `fix` | diagonal D_0 (free) | the fixed start of Reisach et al. (G1) |
| `stat` | Γ(A, D), the solution of Γ = AΓAᵀ + D | a stationary window (G2 in the limit) |

`free` is the default because it does not use the start artefact: the first slice is treated as a nuisance, in the same way as the prior network of a dynamic Bayesian network.

Covariance and precision of X (cells ordered by slice). The precision Θ(A, D, Σ_1) is block tridiagonal:
- Θ_11 = Σ_1^{-1} + AᵀD^{-1}A; Θ_tt = D^{-1} + AᵀD^{-1}A for 1 < t < T; Θ_TT = D^{-1};
- Θ_{t,t+1} = −AᵀD^{-1}; all other blocks are 0;
- log det Θ = −log det Σ_1 − (T−1) Σ_j log D_j.

Layering makes acyclicity automatic. There is no acyclicity constraint and no penalty parameter for it.

**Scale.** The input is the sample correlation matrix R (columns standardised), so the method cannot use marginal variances.
- For a stationary window, standardisation rescales every unit by one constant over all slices, so the tied model still holds exactly, with A' = Λ^{-1}AΛ.
- For a fixed start, variances grow over the slices and standardisation breaks the tying. The variant **TiDAL-s** adds one free scale per column, Z = diag(s) Π X, with the gauge s = 1 on slice 1. It has |E_S| + p + (T−1)p + dim Σ_1 parameters; the untied model on the same graph has (T−1)(|E_S| + p) + dim Σ_1. So the tying still removes (T−2)|E_S| − p degrees of freedom, which is positive for T ≥ 3 as soon as there is one edge besides the loops.

## 3. Objective

With S the sample covariance (here R) and P the d × d permutation matrix of Π (P[v, c] = 1 if column v is cell c), let S̃ = PᵀSP be the covariance in cell order. The Gaussian negative log-likelihood per row is, up to constants,

  ℓ(P, A, D, Σ_1) = ½ [ tr(Θ S̃) − log det Θ ]
   = ½ [ tr(Σ_1^{-1} S̃_11) + log det Σ_1 + Σ_{t<T} tr( D^{-1} (S̃_{t+1,t+1} − A S̃_{t,t+1} − S̃_{t+1,t} Aᵀ + A S̃_tt Aᵀ) ) + (T−1) Σ_j log D_j ].

For a fixed P the minimiser is in closed form (`free` law, a given support for each row of A):
- C_00 = Σ_{t<T} S̃_tt, C_10 = Σ_{t<T} S̃_{t+1,t}, C_11 = Σ_{t<T} S̃_{t+1,t+1};
- row j of A: the pooled least-squares coefficients of cell (j, t+1) on the cells (pa(j), t), from C_00 and C_10;
- D_j = (residual variance of that pooled regression) / (T−1); Σ_1 = S̃_11.

The selection criterion is BIC: 2n ℓ̂ + k log n, with k = |supp A| + p + p(p+1)/2 for `free`.
Because the Gaussian likelihood depends on the data only through S, the objective is a quasi-likelihood for non-Gaussian noise and only second moments matter.

Two things are tied, and it helps to separate them:
- **shape tying**: the column DAG must be an unrolling (a grid with the same bipartite graph between all consecutive slices);
- **parameter tying**: the same weights and noise variances on all T−1 copies.
Section 6 shows that, in the population, shape tying does most of the identification, and parameter tying is needed for isolated units. In finite samples parameter tying pools T−1 regressions.

## 4. Optimisation

For fixed (A, D, Σ_1), minimising over P is a quadratic assignment problem, min_P tr(Θ PᵀSP). It is NP-hard in general. The algorithm is a block-coordinate descent on the relaxed problem, followed by exact local improvement. Every step decreases the same penalised likelihood; there are no thresholds and no case rules.

### 4.1 Relaxed alternating minimisation (main loop)

P ranges over the Birkhoff polytope (doubly stochastic matrices). For such P, S̃ = PᵀSP is still positive semidefinite, so the closed-form M-step stays valid.

Repeat until the Frank–Wolfe gap is below 1e-6 or 300 iterations:
1. **M-step.** (A, D, Σ_1) ← closed form of Section 3 at S̃ = PᵀSP, with A dense and a ridge 1e-6 on C_00 and S̃_11.
2. **P-step (one FAQ step, Vogelstein et al. 2015).** Gradient G = 2 S P Θ. Q ← solution of the linear assignment problem min_Q ⟨G, Q⟩ (Hungarian algorithm). P ← P + γ(Q − P), with γ ∈ [0, 1] the exact minimiser of the quadratic γ ↦ tr(Θ (P + γ(Q−P))ᵀ S (P + γ(Q−P))).

### 4.2 Rounding and exact polish

3. Project P on the permutations (one more linear assignment on −P).
4. Hard alternation on the exact profile likelihood: M-step; then the best improving exchange of two columns, evaluated exactly; stop when no exchange lowers ℓ̂. This ends in a point that is optimal with respect to all pairwise exchanges.

### 4.3 Support of A

5. With the alignment fixed, each row of A is chosen by exact best-subset BIC on the pooled regression (2^p subsets per row; p ≤ 12), with the self-loop always in. For larger p: forward–backward BIC.
6. Output: the column DAG of (P, supp A), and its BIC.

### 4.4 Restarts and the choice of T

- Restarts: 30 per (data set, T). One starts from a spectral seriation of |R| (Atkins et al. 1998): the Fiedler order is cut into T blocks of p, and units are matched between consecutive blocks by a linear assignment on |R|. The other 29 start from the barycentre (1/d) 11ᵀ plus a random doubly stochastic perturbation. The barycentre itself is a saddle because of the p! unit relabellings, so a perturbation is needed.
- T is selected by BIC among the divisors of d with T ≥ 3 and p ≥ 2. T = 2 is excluded: the `free` model is not identifiable there (T4a).
- The run with the lowest BIC over restarts and T is returned.

### 4.5 Variants (not the default)

- **Gumbel–Sinkhorn.** P = Sinkhorn((W + Gumbel noise)/τ), with W, A, log D and a Cholesky factor of Σ_1 learned by Adam on ℓ, τ annealed from 1 to 0.05, rounding as in 4.2. This is the form named in the candidate. I recommend the FAQ loop as the default: it has no temperature schedule or learning rate, it is deterministic given the start, and each step has a closed form. The Sinkhorn form is the natural one for TiDAL-s and for nonlinear extensions, where the M-step has no closed form.
- **TiDAL-s** (free column scales): the M-step becomes a few L-BFGS steps on (A, log D, log s, Σ_1).
- **TiDAL-untied** (ablation): same alignment and same support in every slice, but free weights A_t and variances D_t. It isolates shape tying from parameter tying.

### 4.6 Complexity

Per iteration: O(d² p) for the gradient (Θ is block tridiagonal), O(d³) for the assignment, O(T p² + p³) for the M-step. The pairwise polish costs O(d²) candidate exchanges per sweep, each O(p²) if only the affected blocks of C_00, C_10, C_11 are updated.
For d = 50 one iteration is of the order of a millisecond in numpy. With 30 restarts and up to 5 candidate values of T (d = 30: T ∈ {3, 5, 6, 10, 15}), the estimate is 10–60 s per data set. This estimate is not measured.
Memory O(d²). The method is not meant for d in the thousands: the search space is (pT)!.

## 5. What is identifiable: statements

Notation. G_T is the labelled unrolled DAG, height(v) = 1 + the number of edges of a longest directed path ending at v. A unit is **loop-only** if its only parent in G_S is itself, and **isolated** if in addition it has no child other than itself. m_0 = number of loop-only units.

**Assumption F (faithfulness).** The column distribution is faithful to its DAG: to G_T under the `fix` law, and to H = G_T ∪ (a complete DAG on the slice-1 cells of each weakly connected component of G_S) under the `free` and `stat` laws.
F is a real assumption here. For untied Gaussian DAG models faithfulness holds for almost all parameters, but the tied parameters form a set of measure zero in the untied space, and Boege et al. (2025) show that coloured DAG models can contain no faithful distribution at all. Whether F holds for almost all (A, D) in the unrolled model is **open** in this document (Section 7, risk R1).

**Sparsest representations.** Fix p and T. A parameter θ' of the model class with these p and T *represents* Σ if Σ(θ') = Σ. It is *sparsest* if no representation has a smaller |supp A'|. For fixed (p, T) and a fixed law, the number of parameters differs between representations only through |supp A'|, so BIC selects sparsest representations in the limit.

All statements below are for known (p, T). When T is selected by BIC among the divisors of d (Section 4.4), a representation with another T' has a different number of column edges per parameter, and I have no proof that it cannot win; this is **open** (risk R10).

### T1. Slices are a graph invariant; the alignment is not. [proved here; suitable for Lean]

(a) In G_T (self-loop at every unit), height(v(t)) = t. Hence if a DAG is isomorphic to G_T(G_S) and to G_{T'}(G_S'), then T = T', p = p', and both isomorphisms send the same nodes to slice t, for every t.
(b) Two labellings of the same column DAG are related by permutations Q_1..Q_T of the units with A' = Q_{t+1} A Q_tᵀ for all t < T (as 0/1 matrices). The Q_t need not be equal: for the directed ring with loops, Q_t = (rotation by −t) gives G_S' = the reversed ring with loops. So "which columns belong to the same unit" is not determined by the column DAG, but the column DAG is the same for all labellings.
(c) Without self-loops (a) fails: for the single edge 0 → 1 without loops, G_T is a set of disjoint edges and the slice of an edge is not determined.

Proof of (a). Every edge of G_T goes from slice t to slice t+1, so a path ending in v(t) has at most t−1 edges; the self-chain v(1) → … → v(t) has exactly t−1. Isomorphisms preserve height. (b) is a restatement of "both labellings give the same edge set". □
The walk-length lemma already in the paper's Lean development (`edge_score_dist`) contains the first sentence of this proof.

### T2. The Markov equivalence class of G_T. [proved here; graph part suitable for Lean]

In the CPDAG of G_T, an edge is undirected iff it is an edge j(t) → j(t+1) of the self-chain of a loop-only unit j. The class has exactly T^{m_0} members.

Proof. Two parents i(t), k(t) of a node j(t+1) lie in the same slice and are not adjacent, so every node with at least two parents is an unshielded collider and all its incoming edges are compelled. The remaining edges are the self-chain edges of loop-only units. For such a chain, re-rooting it at any j(s) (edges point away from j(s) along the chain) keeps the skeleton, creates no collider (every chain node still has at most one parent), removes none, and creates no cycle (a chain node has no parent outside the chain). So these edges are reversible, and each chain has T orientations without a collider, independently of the other chains. □

Consequence: under the fixed start, a perfect static learner is left with T^{m_0} graphs (m_0 = 1 for a chain or out-tree, the number of leaves for an in-tree, 0 for a ring).

### T3. Fixed start: the column DAG is identifiable. [proved here, given F]

Law `fix`, assumption F, any T ≥ 2, (p, T) known. Let Σ be the covariance of the columns.
(a) Every sparsest representation of Σ has column DAG G*, except that the self-chain of an isolated unit may be reversed as a whole.
(b) For T ≥ 3, the chain of an isolated unit j can be reversed iff its start is stationary, D_{0,j} (1 − A_jj²) = D_j. For T = 2 it can always be reversed.
(c) Hence for T ≥ 3 the column DAG is identified iff no isolated unit has a stationary start. Under G1 (D_0 = D, A_jj ≠ 0) the condition holds.

Proof sketch. (a) By F and Chickering's (2002) theorem, a DAG to which Σ is Markov and which has no more edges than G_T is Markov equivalent to G_T. By T2 it is G_T with some chains of loop-only units re-rooted. It must also be an unrolling with loops, so every edge joins consecutive heights, and every node of height ≥ 2 has its own self-parent (the self-parent map is injective). Re-rooting at an interior node j(s) gives two nodes, j(s−1) and j(s+1), whose only parent is j(s): impossible. Re-rooting at j(T) gives height'(j(1)) = T; if j has a child k ≠ j, the edge j(1) → k(2) is compelled and forces height'(k(2)) = T + 1 > T: impossible. So only isolated units can be reversed.
(b) Let v_t be the variance of x_j(t), v_{t+1} = a² v_t + d. The reversed chain is in the class iff the backward regression coefficient a v_t / v_{t+1} and the backward residual variance d v_t / v_{t+1} do not depend on t ∈ {1..T−1}, i.e. iff v_t / v_{t+1} = r is constant. With T ≥ 3, v_3 − v_2 = a²(v_2 − v_1) gives (1/r)(1/r − 1) = a²(1/r − 1), so r = 1 (stationary) or 1/r = a² (which forces d = 0). □

Remark. If one imposes D_0 = D (as G1 does), an isolated unit is oriented by its growing variance even at T = 2. This is var-sortability, the artefact of the paper; the `free` and `fix` laws with free D_0 do not use it.

### T4. Three exact non-identifiability results. [proved here]

(a) **T = 2, law `free`.** Not identifiable, even with F and sparsity. Example: units a, b, summary edge a → b with loops. Truth: slices {a_1, b_1}, {a_2, b_2}; column DAG a_1 → a_2, a_1 → b_2, b_1 → b_2. The labelling with slices {a_1, b_2}, {a_2, b_1} is also in the class: a_2 = a·a_1 + ε_a and b_1 = β a_1 + γ b_2 + η, where η is a function of (a_1, b_1, ε_b) and therefore independent of ε_a, so D' is diagonal; both self-loops are non-zero; and it has the same number of edges. Its column DAG, a_1 → a_2, a_1 → b_1, b_2 → b_1, is different. With one transition there is nothing to tie.
(b) **Isolated units, stationary start, any T.** A stationary Gaussian AR(1) has a symmetric autocovariance, so reversing the chain of an isolated unit leaves Σ unchanged. In particular p = 1 is never identifiable from a stationary window. (Classical: Weiss 1975; Peters et al. 2009.)
(c) **Symmetric dynamics, any T.** If a weakly connected component has A = Aᵀ and D = σ²I on it, and the start is stationary, then Γ = σ²(I − A²)^{-1} commutes with A, the reversed process has transition Γ AᵀΓ^{-1} = A and innovation covariance D, and reversing all slices of that component leaves Σ unchanged. The reversed column DAG is a sparsest representation. So second moments cannot orient time for an undirected summary graph with symmetric weights and equal noise.

In general the time-reversed window of a stationary component is in the model class iff D̃ = Γ − Γ AᵀΓ^{-1} A Γ is diagonal (and Ã = Γ AᵀΓ^{-1} has a non-zero diagonal). This is where the method differs from the Gaussian non-identifiability of the arrow of time (Peters et al. 2009; Bauer et al. 2016): the model class requires uncorrelated innovations, and the reversed innovations of a generic VAR(1) with p ≥ 2 are correlated. I have not shown that D̃ is non-diagonal for almost all (A, D) with a given support; for A = [[a, 0], [b, c]] a direct argument suggests it, and a = c = 0 (no self-loops) gives a diagonal D̃. **[conjecture]**

### C5. Window law, T ≥ 3. [conjecture]

Law `free` or `stat`, (p, T) known, G_S weakly connected, p ≥ 2, T ≥ 3, assumption F, and D̃ of T4 not diagonal. Then every sparsest representation of Σ has column DAG G*.

What I can show:
- Under F, a sparsest representation has a DAG H' = G'_T ∪ K' (K' a complete DAG on its first slice; a complete DAG imposes no constraint, so Σ is Markov to H') with (T−1)|supp A'| + p(p−1)/2 edges, no more than H. By Chickering's theorem H' is Markov equivalent to H, hence has the same skeleton U. In H the edges between slices 1 and 2 are *not* compelled (the parents of a slice-2 node lie in the slice-1 clique), so the Markov equivalence class alone does not give them; this is the population reason why a static learner does worse on a burn-in window than under the fixed start.
- If the layering of U is unique, i.e. the only partition of the nodes into T layers of size p with edges inside layer 1 or between consecutive layers, and with a perfect matching between consecutive layers, is the true one, then G'_T = G_T. So the conjecture reduces to a **finite combinatorial statement about U** (unique layering), plus the exclusion of the reversed layering by the skeleton (slice T is an independent set, slice 1 is not).
- I checked by hand that the layering is unique for G_S = (a → b with loops) and T = 3, and T4(a) shows it is not unique for T = 2.

What could fail: summary graphs for which U has a second layering at T = 3 (I expect symmetric graphs to be the candidates), and graphs where slice 1 is not a clique in U because Γ^{-1} has zeros. In those cases parameter tying would have to exclude the alternative, and I have no argument for that yet. The plan is to enumerate layerings for all summary graphs with p ≤ 4 and T ∈ {3, 4} before trying a proof (`PILOT_PLAN.md`, Stage 0); this is a graph computation without data.

### T6. Consistency of the penalised MLE. [standard, given identifiability]

Fix p, T and a true parameter such that every sparsest representation of Σ* has column DAG G* (T3 or C5). Let the rows be i.i.d. with covariance Σ* and finite fourth moments. Then the column DAG of the global minimiser of BIC over the model class equals G* with probability tending to 1 as n → ∞.

Proof idea. The discrete part (alignment, support, T) ranges over a finite set. For each element the profile likelihood is a continuous function of S near Σ*. Elements that cannot represent Σ* have a positive Kullback–Leibler gap and lose by a term of order n. Elements that represent Σ* with more parameters lose by the log n penalty, since their likelihood gain is O_p(1). With sub-Gaussian rows the error probability decays exponentially in n times the squared gap.
What it does not give: any guarantee for the *computed* solution. The alignment step is a quadratic assignment problem; the algorithm of Section 4 finds a point that is stationary for the relaxation and optimal for pairwise exchanges, not a certified global minimum.

### What is suitable for Lean

- T1 (a)–(c): yes, short; reuses the existing unrolled-graph definitions and the walk-length lemma.
- T2 and the graph part of T3(a): yes as statements about skeletons and unshielded colliders ("same skeleton and same colliders" as the definition of equivalence). Linking this to conditional independence would need a d-separation library and is out of scope.
- The layering-uniqueness statement of C5, once its correct form is known from the enumeration.
- T3(b), T4: elementary algebra; T4(a) is a finite example.
- Not suitable: T6, and anything that uses F.

## 6. What the theory predicts for the benchmark

- Fixed start: a static learner is already close to the truth in the population (T2: only T^{m_0} graphs). The gain of TiDAL should be modest and come from pooling. FLOP is at 0.19.
- Burn-in window: the Markov equivalence class leaves the edges between the first two slices unoriented and the truth graph contains no edge for the dependence inside slice 1. Shape tying resolves the first, and the model treats the second as a nuisance. FLOP is at 0.52 (0.37 on in-trees with first-slice pairs excluded). This is where a large gain is predicted.
- T = 3 is the hardest horizon that is identifiable at all; rings with symmetric-looking weights and graphs with many isolated or loop-only units are the hardest families.

## 7. Risks and failure modes

| # | Risk | Consequence | What to do |
|---|---|---|---|
| R1 | Assumption F may fail on a non-negligible parameter set, or everywhere for some summary graphs (coloured models can lack faithful distributions) | T3 and C5 lose their hypothesis | Stage 0 of the pilot checks F on exact covariances for the pilot graphs; state F as an assumption in the paper |
| R2 | C5 false for some graphs at T = 3 | Identifiability only for T ≥ 4, or only for graph classes | Enumeration in Stage 0; report the exact exception list |
| R3 | Local optima of the alignment (quadratic assignment; p! unit symmetries; near-symmetric units with similar loops) | Method works with oracle start but not from random starts | The pilot reports the BIC of the oracle alignment next to the found one; if found > oracle in many data sets the optimiser, not the model, is the problem |
| R4 | A simple two-stage pipeline (recover slices by seriation, then run PCMCI+ on the recovered labels) does as well | The likelihood method adds nothing but formality; the identifiability result still stands | Included as a baseline; reported whatever the outcome |
| R5 | Setting seen as artificial (`NOVELTY.md`, Section 4) | No gain in Contribution | Frame as a benchmark statement; do not claim a general method |
| R6 | Misspecification: nonlinear dynamics, lags > 1, contemporaneous edges, latent drivers, time-varying A | Tying is wrong; alignment may still work, the support of A may not | Out of scope for the pilot; one B1-NL configuration could be added later |
| R7 | Standardised fixed-start data break parameter tying | TiDAL (no scales) is misspecified under G1 | TiDAL-s; G1 is a secondary outcome only |
| R8 | The method is told nothing about T but must assume T ≥ 3 and a VAR(1) shape | Cannot be used on T = 2 unrollings; a reviewer may call the grid shape "label information" | Say so. The grid shape is the benchmark's definition, and FLOP is not told it; the comparison is therefore "what the construction leaks", not a fair contest between methods |
| R9 | Cost grows as (pT)!; d ≤ 50 in the pilot | No scaling claim | State the range |
| R10 | T selected by BIC among divisors of d is not covered by T3, C5 or T6 | A wrong T gives a wrong grid | The pilot reports how often the selected T is the true one, and a second arm with T given |

R8 matters for the wording of any claim. TiDAL knows that the data are an unrolling; FLOP, DAGMA and NOTEARS do not. A result "TiDAL beats FLOP" therefore does not say that TiDAL is a better causal-discovery method. It says that the unrolled construction contains enough replicated structure to reconstruct the time order without labels.

## 8. Baselines

Label-free, same input (columns shuffled, standardised unless the baseline needs raw data):
- FLOP (flopsearch, B'1 settings): the method to beat. B'1 levels: SHD/|E| 0.19 (G1), 0.52 (G2).
- DAGMA and NOTEARS, raw and standardised (B'1 settings): 0.35 / 0.90 and 0.47 / 0.89 raw.
- PC (B1 settings).
- Empty graph (SHD/|E| = 1).
- Two-stage baseline: spectral seriation of |R| into slices, both directions of time tried, PCMCI+ on the recovered labels, the direction with fewer contemporaneous links kept (R4).

References that use information TiDAL does not have:
- PCMCI+ with the true labels (ceiling; 0.02 under both starts in B'1).
- TiDAL with the oracle alignment (ceiling of the model; separates optimisation error from model error).
- TiDAL-untied (ablation of parameter tying).
