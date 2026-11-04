# PLAN5: deepening theory2 (targets X1-X3), written before any derivation

Scope: files only in research/theory2/ext/ and new files under research/theory2/lean/RelSort/ (existing Lean files edited only to add imports in RelSort.lean).
Labels used: proved / sketched / conjectured. "Verified" = exact computation, never a proof.

## X1 Random summary graphs
Ensembles: (a) directed Erdos-Renyi G(p,c/p) plus a self-loop at every node; (b) random in-forests (river model: each node picks one downstream neighbour; recursive-tree and Galton-Watson shapes); (c) layered DAG (Poisson(c) parents from previous layer).
Quantities: L = lim_T s_rho(G_T) = 1/2 + |E_x|/(2|E_S|) (Thm 1.5) and S = 1/2 + |E_up|/(2|E_S|) (Thm 1.9).
Plan: for ER, express |E_x|/p and |E_up|/p asymptotically through survival probabilities of Poisson(c) branching processes (x = 1 - e^{-cx}), after checking the SCC statements against the literature (Karp 1990, Dorogovtsev-Mendes-Samukhin 2001, Blanchard-Curien-Krause-Reisach). For |E_up| use the local weak limit of the edge-rooted graph; expected to need an extra fixed point (sources whose whole forward tree is finite).
Success: closed formulas that match exact computations on large seeded graphs (p up to 1e5, several seeds) within Monte Carlo error (|diff| < 3 sd and shrinking with p). Failure: any systematic mismatch; then the formula is reported as refuted and what survives is stated. Status honesty: the limits are "sketched" (local weak convergence argument, no full proof of concentration) unless a full proof is achieved; exact finite-p identities (e.g. L - S = n_1/(2|E_S|) for in-forests) are proved.
## X2 No self-loops everywhere
Target: exact value of s_rho(G_T) for arbitrary summary graph (loops on any subset), the limit, and the iff condition for 1/2; stationary version too. Expected mechanism: walk-length sets L(u,j) instead of distances (periodicity), and "fresh" nodes (in-degree 0 in E_S) that inject a root at every time slice.
Success: a theorem with proof, checked by exhaustive enumeration of all summary graphs on p<=4 (loop patterns included, p=4: 2^16 graphs) and random graphs for p=5,6 against brute-force relatives in G_T. Failure outcome: a counterexample to a conjectured formula is itself a result; the final statement says exactly where the loop assumption is needed.
## X3 Predictive meaning
Prove or refute: accuracy of the oracle relatives-count sort-and-orient rule on edges equals s_rho (expected: true by definition = trivial); and what is the best accuracy of any rule using only marginal relatives counts. Expected: optimal among antisymmetric pairwise-count rules (ties forced to 1/2, strict edges never wrong by Reisach Thm 1); false if the rule may also see node identities/skeleton (counterexample expected). Report honestly if trivial.
## Lean
New files under lean/RelSort/ (e.g. NoLoop.lean, Predict.lean): exact finite-T value without loop assumption, iff-type conditions that are deterministic, X3 identity and pairwise-rule optimality. Rules: no sorry/admit/axiom; #print axioms only propext/Classical.choice/Quot.sound; clean rebuild with LEAN_NUM_THREADS=2; update README.md, AXIOMS.txt, BUILD_LOG.txt. Random-graph limits are not formalised (stated explicitly).
## Deliverables
ext/EXT.md (full proofs), ext/ scripts + result files, summary for ICML paper (main text vs appendix), honest depth assessment.
