# EXT: random summary graphs (X1), summary graphs without self-loops (X2), predictive meaning (X3)

Plan: `PLAN5.md` (commit a80d35c, written before any derivation). Code and result files: `verify/`. Lean: `../lean/RelSort/NoLoop.lean`, `../lean/RelSort/Predict.lean`.
Labels: **proved** (full proof here; "Lean" if also machine-checked), **sketched** (argument given, one analytic step not written out), **conjectured**. "Verified" means checked by exact computation; it never replaces a proof.
Notation as in `THEORY.md` (Section 0). `B_n` is the boolean n-th power of the adjacency matrix of G_S (B_0 = I), `L(u,v) = {n : B_n[u,v] = 1}` is the set of lengths of walks u to v; a node u is **fresh** if it has no incoming edge in E_S (a self-loop counts as an incoming edge). L = lim_T s_rho(G_T) and S = stationary value (Thm 1.9).

Scorecard (details below):

| id | statement | status |
|---|---|---|
| X2.2 | exact value of every edge of G_T for ANY summary graph (no loop assumption) | proved, Lean (`edge_score_NL`, `sRho_unrolled_eq_NL`), exhaustive check p<=4 |
| X2.3-X2.7 | limit exists and is rational; same-SCC edges; crossing edges into cyclic nodes; Reisach Thm 4 without loops; converse false without loops | proved (Lean for the algebraic core), checked |
| X2.9-X2.10 | stationary (infinite-past) value without loops, via difference sets Delta(v,w) | proved for G_Z; windowed version sketched; checked |
| X1.1 | ER limit L = 1/2 + c(1-x^2)/(2(1+c)) | sketched, verified to 4 digits up to p = 10^6 |
| X1.2 | ER stationary S = 1/2 + c F_up/(2(1+c)) with a second fixed point d | sketched, verified to 3-4 digits up to p = 10^5 |
| X1.3-X1.4 | in-forest identities (exact), tree ensembles, layered DAG | proved (identities), numerics for the ensembles |
| X3.1-X3.3 | accuracy of the oracle sort = s_rho; optimal among pairwise count rules; not optimal with node identities | proved, Lean (`accRule_*`), checked; elementary |

---------------------------------------------------------------------------------------------------

## 0. Literature check for X1

- The paper cited by Reisach et al., Blanchard, Curien, Krause, Reisach, "A phase transition in Barak-Erdos random graphs" (arXiv 2512.05756 v2, Jan 2026; v1 title "Erdos-Barak"), was opened and read (abstract and Section 1 as extracted text). Its model is the Barak-Erdos graph on numbered vertices (edge i -> j for i < j with probability p) and its theorem is the critical window of the event "there is a monotone path from 1 to n" at p_n = (log n - log log n + x)/n: P(1 ~> n) -> 1 - e^{-e^{-x}...} (Theorem 1.1, Remark 1.2). It contains **no statement about strongly connected components of random directed graphs**, and it concerns the dense scale p ~ log n / n, not the sparse scale c/p used by the summary-graph ensembles. It is therefore not used for X1. (It is relevant to Reisach et al.'s ordered ER DAGs, i.e. to *DAG-level* reachability, not to summary graphs.)
- The SCC input used here is the classical one for directed ER graphs G(p, c/p): below c = 1 all SCCs are small, above c = 1 there is one giant SCC of size ~ x^2 p, where x is the survival probability of the Poisson(c) branching process, x = 1 - e^{-cx} (the giant in-component and the giant out-component both have size x p, and their intersection is the giant SCC). Checked against the abstract of Dorogovtsev, Mendes, Samukhin, "Giant strongly connected component of directed networks", Phys. Rev. E 64, 025101 (arXiv cond-mat/0103629): for a factorising joint in/out-degree distribution (Poisson in and out degrees are independent in ER) the GSCC size is the product of the giant in- and out-component sizes. Karp (1990) and Luczak (1990) for the ER case are recalled from memory and not opened. Every formula below is in addition checked against exact computation on large seeded graphs (Section 1.5).

---------------------------------------------------------------------------------------------------

## 2. X2: summary graphs without (all) self-loops

(Section 2 comes before X1 in the derivation because X1 uses nothing from it, but X2 is the cleaner theorem; sections are numbered X1, X2, X3 as in the plan.)

### 2.1 Exact value of every edge (any E_S)

**Lemma X2.0 (ancestors; proved, Lean `unrolled_reach_iff`, no loop hypothesis).** u(a) in anc(v(t)) iff a <= t and there is a walk of length exactly t - a from u to v in G_S.
*Proof.* Every edge of G_T raises time by one; a path from u(a) to v(t) is a lift of a walk of length t - a, and every walk lifts. Without loops one cannot pad, so the length is exactly t - a. ∎

**Lemma X2.1 (roots; proved).** u(a) is a root of G_T (has no parent) iff a = 1 or u is fresh. Hence roots(v(t)) = {u(a) : (a = 1 or u fresh), n = t - a >= 0, n in L(u,v)}.
*Proof.* u(a) with a >= 2 has a parent iff some edge (w,u) in E_S exists. ∎

**Theorem X2.2 (edge value without loop assumption; proved, Lean `edge_score_NL`).** Let (i,j) in E_S and e_t = (i(t), j(t+1)) in E_T (paper time t = 1..T-1). Then s_rho(e_t) in {1/2, 1}, independent of T, and s_rho(e_t) = 1 iff Strict_ij(t), where
 Strict_ij(t) :<=> [ exists u: t in L(u,j) and t-1 not in L(u,i) ]  or  [ exists fresh u, exists n in {1..t}: n in L(u,j) and n-1 not in L(u,i) ].
*Proof.* rel(i(t)) is contained in rel(j(t+1)) (Reisach et al., Thm 1, valid for every DAG; Lean `relSet_subset_of_edge`), so the score is 1/2 or 1, and it is 1 iff the inclusion is strict, iff (Reisach et al., Lemma 1, eq. (10)) roots(j(t+1)) minus roots(i(t)) is nonempty. By Lemma X2.1 an element of the difference is u(a) with (a = 1 or u fresh), n = t+1-a in L(u,j), and u(a) not in roots(i(t)); the case a = t+1 would force u = j fresh, impossible since j has the parent i; so a <= t and, by Lemma X2.1 again, u(a) not in roots(i(t)) iff n - 1 = t - a not in L(u,i). The branch a = 1 is n = t (no condition on u), the other branch is fresh u with 1 <= n <= t. ∎
(The Lean proof does not use Lemma 1: it shows directly that the extra relative is u(0) in the first case and u(s-m) in the second, and that a common ancestor of j(s+1) and w(r) can be pushed back along parents until slice 0 or a fresh node, `reduce_ancestor`.)

**Corollary X2.3 (exact finite-T value; proved, Lean `sRho_unrolled_eq_NL`, `sRho_unrolled_ge_half_NL`).** For every summary graph with |E_S| >= 1 and every T >= 2,
 s_rho(G_T) = 1/2 + (1/(2(T-1)|E_S|)) sum_{(i,j) in E_S} N_T(i,j),  N_T(i,j) = #{t in 1..T-1 : Strict_ij(t)},
and in particular s_rho(G_T) >= 1/2.

**Corollary X2.3' (loops everywhere recovers Thm 1.3).** If every node has a self-loop, no node is fresh and L(u,j) = [d(u,j), inf), L(u,i)+1 = [d(u,i)+1, inf), so Strict_ij(t) iff exists u with d(u,j) <= t <= d(u,i): exactly THEORY.md Theorem 1.3.

### 2.2 Limit, same-SCC edges, crossing edges

**Proposition X2.4 (limit exists and is rational; proved).** The sequence B_n of boolean matrices is eventually periodic: B_{n+1} is a function of B_n and there are finitely many boolean matrices, so there are n0, pi >= 1 with B_{n+pi} = B_n for all n >= n0 (pre-period n0, period pi; found by the first repeat). For an edge e = (i,j) let W_e = 1 if some fresh u has n >= 1 with n in L(u,j) and n-1 not in L(u,i) ("fresh witness"), and otherwise delta_e = (1/pi) #{r in one window of pi consecutive t > n0 : exists u, B_t[u,j] = 1 and B_{t-1}[u,i] = 0}. Put delta_e = 1 in the first case. Then
 L = lim_T s_rho(G_T) = 1/2 + (1/(2|E_S|)) sum_e delta_e,  and |s_rho(G_T) - L| <= (n0 + pi + 1)/(2(T-1)).
*Proof.* A fresh witness, once n is allowed, is monotone in t (the range 1..t only grows), and its first n is at most n0 + 1 + pi because D(u) = L(u,j) minus (L(u,i)+1) is eventually periodic with pre-period n0 + 1; so such an edge is strict for all t >= n0 + 1 + pi. Without a witness, Strict_ij(t) depends for t > n0 only on t mod pi. Count the strict t in 1..T-1: error against (T-1) delta_e at most n0 + pi + 1. Average over edges. ∎ Verified (`x2_limit_results.txt`): for 1,115 graphs (all 15 + 511 graphs with p = 2, 3, 390 random p = 4, 199 random p = 5, 6) the exact finite-T value at T = 600 is within the stated bound, and (L - 1/2) * 2|E_S| * pi is an integer in every case.

**Proposition X2.5 (edges inside an SCC; proved).** Let (i,j) in E_S with j reaching i (this includes self-loops and every edge inside an SCC), and suppose it has no fresh witness. Then Strict_ij(t) is false for every t >= n0 + 1. So such an edge contributes only to finitely many layers (at most n0 of them).
*Proof.* Fix u and put f(n) = B_n[u,j], h(n) = B_{n-1}[u,i]. The edge gives h(n) <= f(n). Let m be the length of a walk from j to i and c = m + 1; appending it to a walk u to j of length n gives a walk u to i of length n + c - 1, so f(n) <= h(n+c). Take k >= 1 with kc a multiple of pi. For n >= n0 + 1 (so n - 1 >= n0) periodicity gives h(n + kc) = h(n), and
 h(n) <= f(n) <= h(n+c) <= f(n+c) <= ... <= h(n+kc) = h(n),
so f(n) = h(n) for every u. Hence no u satisfies the first bracket of Strict_ij at t = n. ∎ Verified: no violation in 1,115 graphs (`x2_limit_results.txt`, test L2).

**Proposition X2.6 (crossing edge into a cyclic node; proved, Lean part `strictNL_of_closed_walk`).** If (i,j) is crossing (j does not reach i) and j lies on a cycle, let g be the gcd of the closed-walk lengths at j (the period of its SCC). Then Strict_ij(t) holds for every t in L(j,j), and L(j,j) contains all large multiples of g, so delta_e >= 1/g; delta_e = 1 if g = 1 (in particular if j has a self-loop).
*Proof.* Take u = j: t in L(j,j) and t-1 not in L(j,i) = empty. The closed-walk lengths at j form a numerical semigroup in gZ, which contains all large multiples of g. ∎

**Corollary X2.7 (Reisach et al. Theorem 4 without loops; converse false; proved).**
(a) If every weakly connected component of G_S is strongly connected (any loop pattern, any periods), then L = 1/2 and s_rho(G_T) - 1/2 <= n0/(2(T-1)).
(b) The converse "E_x nonempty implies L > 1/2" and the closed form L = 1/2 + |E_x|/(2|E_S|) are **false without self-loops**: for the single edge 0 -> 1 without loops, s_rho(G_T) = 1/2 for every T (Lean `edgeNoLoops_sRho`); for the chain 0 -> 1 -> 2 without loops, L = 1/2; for the 2-cycle {0,1} plus the edge 0 -> 2 without loops, L = 1/2 although the edge (0,2) is crossing (with a loop at 2, L = 5/8). A crossing edge whose head is not on a cycle can have any density, including 0.
(c) Under the hypothesis "every crossing edge has a head on a cycle", L > 1/2 iff E_x is nonempty (X2.6).
*Proof.* (a) A weak component that is strongly connected and has an edge has no fresh node, except an isolated node, which reaches only itself and has no witness. Every edge is inside an SCC, so X2.5 applies to all of them. (b), (c) from X2.6 and the examples (computed in `x2_examples_results.txt`). ∎
Self-loops are therefore **necessary** exactly for: the closed form for L in terms of |E_x|, the converse of Theorem 4, and the independence of the answer from the periodic structure. They are not needed for the exact finite-T value, the existence and rationality of L, or Theorem 4.

**Proposition X2.8 (acyclic loop-free graphs; proved).** If G_S is acyclic without loops, then L(u,v) is finite, B_n = 0 for n >= p, no non-fresh witness exists for t >= p, and L = 1/2 + #{e : fresh witness}/(2|E_S|) (denominator |E_S| without the p loops). For a **loop-free in-forest** (each node has at most one downstream neighbour), edge (i,j) has a fresh witness iff j has another parent, so L = 1/2 + (m - n_1)/(2m), with m = |E_S| and n_1 the number of nodes with exactly one parent. The stationary value of X2.10 is the same number, so with no loops the finite and the stationary values coincide; with loops they differ by n_1/(2(m+p)) (X1.4). Examples: collider without loops, L = S = 1 (with loops 0.7; Lean `colliderNL_sRho`: s_rho = 1 for all T >= 2); edge with a loop only at its head: L = S = 1 (Lean `edgeLoopHead_sRho`); edge with loops at both ends: L = 2/3, S = 1/2.
*Proof.* First claims: no cycle gives B_n = 0 beyond length p - 1. In an in-forest: if j has another parent k, take a fresh u upstream of k (a leaf of k's subtree, or k itself); its path to j has length n through k, and u is not upstream of i, so n-1 not in L(u,i). If i is the only parent of j, every walk u to j passes through i, so n in L(u,j) gives n-1 in L(u,i). ∎ Verified: `x2_examples_results.txt` (362 random acyclic loop-free graphs: L = S for all; 298 in-forests: formula holds for all).

### 2.3 Stationary version without loops

Let G_Z be the infinite-past unrolled DAG on V_S x Z and Delta(v,w) = {m - n : exists u, n in L(u,v), m in L(u,w)} (subset of Z).

**Proposition X2.9 (relatives in G_Z; proved).** rel_{G_Z}(v(t)) = {w(t + d) : d in Delta(v,w)}. The graph G_Z is invariant under time shifts, so the relation "rel(i(t)) is strictly inside rel(j(t+1))" does not depend on t: with E_up^Z = {(i,j) in E_S : exists w with Delta(j,w) + 1 != Delta(i,w)} the edge (i(t), j(t+1)) is strict iff (i,j) in E_up^Z. Always Delta(i,w) is contained in Delta(j,w) + 1 for an edge (i,j).
*Proof.* w(s) in rel(v(t)) iff a common ancestor u(r) exists, i.e. n = t - r >= 0 and m = s - r >= 0 with n in L(u,v), m in L(u,w); r is free (infinite past), so only m - n = s - t matters. The rest is the shift. ∎
When every node has a loop, Delta(v,w) is Z if w in rel_S(v) and empty otherwise, and E_up^Z is the set E_up of Theorem 1.9.

**Theorem X2.10 (stationary value without loops; windowed form sketched).** Put S^Z = 1/2 + |E_up^Z|/(2|E_S|). Let K0 = n0 + 2 pi and B0 = 2 n0 + 3 pi + 1.
(a) Delta(v,w) intersected with {|d| >= n0 + pi} is pi-periodic; hence E_up^Z is detected with witnesses of |d| <= K0. [proved: for d >= n0 + pi, replace the pair (n, n+d) by (n-pi, n+d-pi), both entries of the pair stay >= n0 where B is pi-periodic; the negative side is symmetric.]
(b) Window version (**sketched**): in G_{B+T} with B >= B0, relatives restricted to the window W = slices B+1..B+T coincide with those of G_Z (a witness pair can be reduced to n <= n0 + pi + K0 <= B by the same shifting). For edges (i(t), j(t+1)) with t at distance more than K0 from both ends of W, the windowed count rho_W strictly increases along the edge iff (i,j) in E_up^Z. The other edges (a fraction at most 2 K0/(T-1)) are boundary effects, so s_rho^{W,B} = S^Z + O(K0/T). With loops everywhere Delta is Z or empty and the boundary effect vanishes (Theorem 1.9 holds exactly for all T).
(c) S^Z = 1/2 iff Delta(j,w) + 1 = Delta(i,w) for all edges and all w. This holds if every weak component of G_S is strongly connected, whatever its periods or loops: in a strongly connected component of period g with a g-colouring lab, Delta(v,w) = lab(w) - lab(v) + gZ, and lab(j) = lab(i) + 1 on an edge. [proved]
Verified (`x2_limit_results.txt`): brute-force relatives in G_{B+T} (B = 40, T = 90, edges at distance >= 30 from the window ends, rho counted inside the window) agree with E_up^Z on every edge of all 15 + 511 graphs with p = 2, 3 (exhaustive) and 248 random p = 4 graphs; no edge ever had rho(i) > rho(j).

**Note on the Lean part of X2.** Formalised without sorry: X2.2 and X2.3 for an arbitrary relation `E` (`isRel_unrolled_iff_NL`, `reduce_ancestor`, `relSet_unrolled_eq_iff_NL`, `edge_score_NL`, `sRho_unrolled_eq_NL`, `sRho_unrolled_ge_half_NL`), the lemmas `strictNL_of_closed_walk` (the u = j witness of X2.6, for the layer s + 1 in L(j,j)), `strictNL_of_fresh_parent` (a fresh parent u != i of j makes every layer strict) and the two examples `colliderNL_sRho`, `edgeLoopHead_sRho`. Not formalised: X2.4 (eventual periodicity), X2.5, X2.7(a), X2.9, X2.10 (they need the pigeonhole on boolean matrices and the chain argument; routine but not done).

---------------------------------------------------------------------------------------------------

## 1. X1: random summary graphs

Always: G_S = non-loop random graph plus a self-loop at every node, so |E_S| = p + m with m the number of non-loop edges. By Theorem 1.5 and Theorem 1.9 only two numbers matter, |E_x| and |E_up|, and for an acyclic graph with loops |E_x| = m.

### 1.1 Exact identities (deterministic)

**Proposition X1.4 (in-forests; proved).** Let the non-loop part be an in-forest (each node has at most one non-loop downstream neighbour, acyclic), m non-loop edges, n_1 the number of nodes with exactly one parent. Then
 L = 1/2 + m/(2(m+p)),  S = 1/2 + (m - n_1)/(2(m+p)),  L - S = n_1/(2(m+p)).
*Proof.* Corollary 1.7 for L. Corollary 1.10 (iii): E_up is the set of edges into confluences, and the number of edges into nodes with indegree >= 2 is m minus the edges into nodes of indegree 1, i.e. m - n_1. ∎ (Checked exactly, to machine precision, in 18 random forests in `x1_ens_results.txt` by the assertion `L - S == n_1/(2|E_S|)`.)
For a sequence of in-forests with empirical parent-count distribution pi and mean mu: L -> 1/2 + mu/(2(1+mu)), S -> 1/2 + (mu - pi_1)/(2(1+mu)). Rivers have mu <= 1; the **finite value is shape-independent (3/4 for trees as p grows) while the stationary value is governed by pi_1, the fraction of pass-through nodes**. Exact computation at p = 20000 (6 seeds each): uniform recursive in-tree, pi_k = 2^{-(k+1)}, pi_1 = 1/4: S -> 11/16 = 0.6875, measured 0.68739 +- 0.00026; preferential attachment (weight = children + 1), measured pi_1 = 0.1656 (the classical limit law 4/((k+1)(k+2)(k+3)) has pi_1 = 1/6, recalled not re-derived): S = 0.70859 +- 0.00028; uniform random labelled tree rooted and oriented to the root (Poisson(1) parents): pi_1 -> e^{-1}, S -> 1/2 + (1-e^{-1})/4 = 0.65803, measured 0.65783 +- 0.00020. In all three L = 0.74999.

### 1.2 Directed ER: finite-T limit

**Proposition X1.1 (sketched; verified).** Let x be the largest root of x = 1 - e^{-cx} (x = 0 for c <= 1). For G_S = G(p, c/p) plus loops, in probability as p -> infinity,
 L = lim_T s_rho(G_T) -> 1/2 + c(1 - x^2) / (2(1 + c)).
*Sketch.* |E_S| = p + m with m/p -> c, so it suffices to show that the fraction of non-loop edges inside an SCC tends to x^2. Local limit around a uniformly chosen non-loop edge (i,j): i has an independent Galton-Watson (GW) Poisson(c) tree of ancestors, j an independent GW Poisson(c) tree of descendants, i has Poisson(c) further children and j Poisson(c) further parents, all with independent GW subtrees. Write In(i) for "the ancestor tree of i, edge excluded, is infinite" (probability x) and Out(j) for the descendant tree of j (probability x); these are independent. A node lies in the giant SCC iff both its ancestor and its descendant tree are infinite (giant in- and out-components intersect in the giant SCC; checked against Dorogovtsev et al. in Section 0). Because i -> j we have In(i) implies In(j) and Out(j) implies Out(i), so both endpoints lie in the giant SCC iff In(i) and Out(j): probability x^2. Outside the giant, the digraph is subcritical in the dual sense (parameter c(1-x) < 1) and its non-trivial SCCs carry o(p) edges; for c <= 1 all SCCs are small and the number of edges inside non-trivial SCCs is o(p) (the expected number of cycles of length k is c^k/k). Concentration of the edge counts follows from a second-moment bound. The two analytic inputs (the identification of the giant SCC and the o(p) bound for the rest) are classical; I did not write out the concentration step. ∎
Consequences: L is increasing up to c = 1 (L = 3/4 at c = 1), reaches its maximum 0.7542 at c ≈ 1.07, and then decays as L - 1/2 ~ e^{-c} (since 1 - x ~ e^{-c}): dense ER summary graphs are rel-unsortable, as Theorem 4 predicts for strongly connected graphs. For c <= 1: L -> 1/2 + c/(2(1+c)).

### 1.3 Directed ER: stationary value

**Proposition X1.2 (sketched; verified).** With x as above let d be the smallest fixed point in [0,1] of d = 1 + e^{-c} - exp(-c(1-x)d) (d = 1 for c <= 1, obtained as the limit of the iteration started at e^{-c}), and
 F_up = (1-x)(1 - e^{-c}) + x(1-x)(1 - exp(-c(1-x)d)).
Then S -> 1/2 + c F_up / (2(1+c)). For c <= 1 this is 1/2 + c(1-e^{-c})/(2(1+c)); at c = 1 it equals the Poisson(1) tree value 0.65803.
*Sketch.* Source SCCs are, up to O(1) many, the nodes without non-loop parents, and Theorem 1.9 says (i,j) in E_up iff some source s reaches j but not i, iff some other parent k of j has a source ancestor s that does not reach i. In the local limit a path from s to i is either a tree path, which is impossible (the unique tree path from s to i uses the edge i -> j backwards), or goes through the giant: s reaches i iff D(s) and A(i), where D(s) = s has an infinite descendant tree and A(i) = i has an infinite ancestor tree (probability x each).
 Case not A(i) (probability 1 - x): any other parent works, since every node has a source ancestor. Contribution (1-x)(1-e^{-c}).
 Case A(i) (probability x): a witness s needs not D(s), hence not D(j) (probability 1 - x, independent of A(i)). For a node v let Q(v) be the event "v has in-degree 0, or some parent u has all children other than v with finite descendant trees (probability e^{-cx} = 1 - x) and Q(u)". The probability d = P(Q(v)) solves 1 - d = e^{-c(1-x)d} - e^{-c}, from the Poisson(c) number of parents each good with probability (1-x)d; the probability is the smallest fixed point because Q is the increasing union of depth-truncated events. The number of other parents k of j with all children other than j finite and Q(k) is Poisson(c(1-x)d), so the contribution is x(1-x)(1 - exp(-c(1-x)d)).
 The sum is F_up, and the total of edges is (1+c)p, of which cp are non-loop. ∎
Large c: 1 - x ~ e^{-c}, F_up ~ e^{-c}, so S - 1/2 ~ e^{-c}/2 and L - S ~ e^{-c}/2. S is maximal (0.665) at c ≈ 1.09.

### 1.4 Layered DAG

**Proposition X1.3 (L proved given edge concentration; S sketched, verified).** Layered DAG with M layers of width w, each node of layer l+1 receives an edge from each node of layer l independently with probability c/w; loops at all nodes. As w -> infinity with M, c fixed, with mu = c(M-1)/M (edges per node):
 L -> 1/2 + mu/(2(1+mu))  (Cor. 1.7: every edge is crossing),
 S -> 1/2 + mu(1 - e^{-c})/(2(1+mu))  (tree-like local limit: (i,j) is in E_up iff j has another parent, as in X1.4).
For fixed w and deep M the cones saturate and S falls below the tree-like value; this finite-width correction was observed (0.7824 against 0.7878 at w = 50, M = 400, c = 2) and is **not** derived (conjectured to vanish as w/ c^{depth} grows).

### 1.5 Verification (exact computation, fixed seeds; `verify/x1_er.py`, `x1_ens.py`)

Exact SCC (scipy), and exact source-set comparison by bitsets (Src(i) is contained in Src(j) on an edge, so E_up is the set of edges where the source counts differ). Prediction versus exact (mean over seeds +- standard deviation):

| c | p | L exact | L formula | S exact | S formula |
|---|---|---|---|---|---|
| 0.5 | 20000 | 0.6666 +- 0.0011 | 0.6667 | 0.5658 +- 0.0014 | 0.5656 |
| 1.0 | 20000 | 0.7486 +- 0.0008 | 0.7500 | 0.6557 +- 0.0013 | 0.6580 |
| 1.5 | 50000 | 0.6975 +- 0.0020 | 0.6981 | 0.6159 +- 0.0018 | 0.6163 |
| 2.0 | 50000 | 0.6214 +- 0.0022 | 0.6217 | 0.5629 +- 0.0014 | 0.5632 |
| 3.0 | 100000 | 0.5434 +- 0.0010 | 0.5433 | 0.5215 +- 0.0006 | 0.5214 |
| 5.0 | 20000 | 0.5059 +- 0.0005 | 0.5058 | 0.5028 +- 0.0003 | 0.5029 |

(full tables: `x1_er_p20000.txt`, `x1_er_p50000.txt`, `x1_er_scaling.txt`; the S formula for c = 2, 16 seeds at p = 50000, differs by 0.0003 with a standard error of 0.0004.) L at c = 1 converges as p grows: 0.7466 (p = 2000), 0.7486 (2e4), 0.74976 (2e5), 0.74992 (1e6); at c = 2: 0.62162 +- 0.00023 at p = 10^6 against 0.62170. The giant SCC fraction matches x^2 (e.g. 0.6353 against 0.6349 at c = 2). The extra fixed point d matters: with d = 1 (Q always true) the formula would give S = 0.6312, 0.5766, 0.5246 at c = 1.5, 2, 3, far outside the error bars; with the fixed point d = 0.485, 0.221, 0.061 it gives 0.6163, 0.5632, 0.5214.
No failure: **no systematic mismatch was found**, so the status "sketched" is limited by the missing concentration proof, not by a numerical discrepancy. Layered and in-forest tables: `x1_ens_results.txt` (S matches the tree-like value within 1-2 standard errors for w >= 2000).

---------------------------------------------------------------------------------------------------

## 3. X3: predictive meaning

Setting. An edge {x,y} of the true DAG is shown in a uniformly random order (first, second) together with the relatives counts (rho_first, rho_second). A **pairwise count rule** is f(a,b) in [0,1], the probability of declaring "first is the tail". The oracle **sort rule** is f_sort(a,b) = 1 if a < b, 1/2 if a = b, 0 if a > b. For the true edge x -> y the probability of a correct orientation is edgeAcc_f(rho_x, rho_y) = (1/2) f(rho_x, rho_y) + (1/2)(1 - f(rho_y, rho_x)). Accuracy of f on a DAG G is the average over edges of edgeAcc_f(rho_x, rho_y).

**Proposition X3.1 (identity; proved, Lean `accRule_sortRule`; trivial).** accuracy(f_sort) = s_rho(G) exactly. *Proof.* Case a < b gives (1/2)(1) + (1/2)(1 - 0) = 1; a = b gives 1/4 + 1/4 = 1/2; a > b gives 0. These are the edge scores. ∎ This is a restatement of the definition of s_rho; its content is only that the fair coin is what the score 1/2 on ties means. Verified by enumeration (`x3_bayes.py`).

**Proposition X3.2 (optimality among pairwise count rules; proved, Lean `accRule_le_sRho`).** On every DAG G, every pairwise count rule has accuracy <= s_rho(G), and f_sort attains it; the sort rule is never wrong on an edge it decides: ρ_x < ρ_y implies accuracy 1, and ρ_y < ρ_x never happens on an edge x -> y (Lean `rho_not_gt_of_edge`).
*Proof.* Reisach et al.'s Theorem 1 gives rho_x <= rho_y on every edge x -> y. If rho_x < rho_y, edgeAcc <= 1 = score. If rho_x = rho_y = a, edgeAcc = (1/2) f(a,a) + (1/2)(1 - f(a,a)) = 1/2 = score for every f: the random order forces the tie decision to 1/2 whatever f does. ∎
So the accuracy of the ideal count-based orientation is 1/2 + (fraction of strictly sorted edges)/2 with zero error on the decided edges and a coin on the rest. **Bayes statement over a DAG ensemble**: for any distribution over DAGs, the Bayes-optimal pairwise count rule has edge-pooled accuracy E[sum_e score]/E|E|, achieved by f_sort. Combined with Theorem 1.9: for the stationary unrolled DAG of G_S the best accuracy of any pairwise count rule is exactly 1/2 + |E_up|/(2|E_S|), which for G(p, c/p) is S of X1.2: at most 0.665 for every c, at most 0.658 for c <= 1, and tends to 1/2 like e^{-c}/2; for river-like in-forests it is 1/2 + (mu - pi_1)/(2(1+mu)), at most 0.75. The finite-G_T value (Theorem 1.5) is the same statement for Reisach et al.'s generator.

**Proposition X3.3 (the restriction to counts is necessary; verified by exhaustive computation).** Over the uniform ensemble of all labelled DAGs on p nodes, the Bayes-optimal rule with features (rho_first, rho_second) has accuracy exactly equal to the edge-pooled s_rho (p = 2..5: 0.5, 0.5625, 0.592262, 0.605547), as X3.2 predicts, whereas the Bayes-optimal rule that also sees the two node degrees reaches 0.5, 0.625, 0.660714, 0.669238. Example: three nodes, skeleton a - c - b: the collider has counts (2,2,3) (all edges strict) while the chain and fork all have counts (3,3,3) (ties); given the skeleton, the center c is the tail more often than not among the tied graphs (probability 2/3 under the uniform prior on {a -> c -> b, b -> c -> a, a <- c -> b}), so a degree-aware rule beats the coin. So s_rho is the Bayes accuracy of pairwise count rules only; richer features (degrees, the whole count vector with the skeleton) can exceed it, and any non-exchangeable prior on ties (e.g. "ties flow downstream") can too.
Honest assessment: X3.1 is a definition, X3.2 a three-line consequence of Reisach et al.'s Theorem 1; X3.3 is a numerical illustration that the optimality is sharp. The value of X3 for the paper is interpretive (s_rho = accuracy of the ideal sort-and-orient rule, with a precision-1/coverage reading), not a new theorem.

---------------------------------------------------------------------------------------------------

## 4. What goes into the ICML paper, and an honest depth assessment

**Main text (one short paragraph or a proposition each):**
1. Interpretation of s_rho as the accuracy of the ideal count-sort-orient rule, precision 1 on decided edges, coverage 2(s_rho - 1/2) (X3.1/X3.2, two sentences; the optimality among pairwise count rules is one line).
2. The ER result in one display with a figure: L(c) and S(c) from X1.1/X1.2 against exact simulation, peak near the percolation threshold c = 1 (L = 3/4 at c = 1, maximum 0.754 at c = 1.07; S maximum 0.665) and exponential decay beyond. Message for benchmark design: random summary graphs are sortable only near or below criticality, because the giant SCC kills sortability (Theorem 4's converse in random form).
3. For river networks: S = 1/2 + (mu - pi_1)/(2(1+mu)); the finite value is shape-independent but the stationary one counts pass-through nodes (X1.4).
**Appendix:** X2 (Theorem X2.2 with proof, X2.4-X2.7, the examples table, X2.9-X2.10 statement), the full derivation of X1.2 including the fixed point d, the layered DAG and the in-forest tables, the Lean map (`NoLoop.lean`, `Predict.lean`), X3.3 numbers.

**Depth.** X2 is real mathematics but modest: the walk-length formula, "fresh nodes inject a root at every slice", the chain argument of X2.5 and the periodic density are new and non-obvious in their consequences (the closed form and the converse of Theorem 4 genuinely need the loops; with loops absent the answer depends on periods and on source nodes), yet the proof technique is the same as before and the exact formula is still a per-edge criterion. X1 adds the most for a reader: it turns a deterministic edge-counting theorem into predictions for the standard random-graph benchmark, the only non-routine step being the second fixed point d in the stationary value; it is not fully proved (concentration step omitted), but is confirmed numerically to 3-4 digits. X3 is routine (a definition plus a one-line optimality). Overall the three targets add a moderate amount: X1.1/X1.2 and the X2 converse counterexamples are worth an appendix section; none of this is a deep new theorem, and a referee will still see the paper's core as a clean completion of Reisach et al.'s Section 5.

## 5. Files

`verify/x2_enum.py` (brute-force relatives versus Theorem X2.2: exhaustive p = 2, 3 (T = 2..9), p = 4 (65535 graphs, T = 3, 5, 8), random p = 5, 6; results `x2_enum_p*.txt`: 0 mismatches in 120, 3,577, 196,605, 8,928 and 4,494 (graph, T) cases); `verify/x2_limit.py` (limit density, X2.5-X2.7, stationary window check; `x2_limit_results.txt`); `verify/x2_examples.py` (`x2_examples_results.txt`); `verify/x1_er.py`, `x1_ens.py`; `verify/x3_bayes.py`. Run with `/usr/bin/python3` (numpy, scipy) from `verify/`.
