# PLAN6: from "sketched" to a proof of X1 (random summary graphs), written before any derivation in x1proof/

Scope: files only in research/theory2/x1proof/ and new Lean files under research/theory2/lean/RelSort/ (existing Lean files edited only for imports). Labels: **proved** / **proved-assuming-cited-theorem (name)** / **sketched** / **refuted**. "Verified" = exact computation, never a proof. Literature is quoted only if opened; otherwise "from memory, unverified".

## Model
Non-loop part: directed Erdos-Renyi D(p, c/p) (each ordered pair i != j an edge independently with probability c/p), plus a self-loop at every node. m = number of non-loop edges, |E_S| = p + m. By THEORY.md Thm 1.5 / 1.9:
 L = 1/2 + |E_x|/(2|E_S|),  S = 1/2 + |E_up|/(2|E_S|),
E_x = edges joining different SCCs, E_up = {(i,j) : Src(j) != Src(i)} (a source SCC reaches j but not i). x solves x = 1 - e^{-cx} (largest root; x = 0 for c <= 1).

## Targets (precise statements)
**T1.** For every fixed c > 0, N/p -> c x^2 in probability and in L1, where N = number of non-loop edges whose endpoints lie in the same SCC. Hence |E_x|/m -> 1 - x^2 and L -> 1/2 + c(1-x^2)/(2(1+c)) in probability and L1 (L is bounded). Supporting claims proved on the way, each with its own numerical test: (a) P(K <= |out(v)| < eps p) <= C e^{-kappa K} uniformly in p (dichotomy); (b) w.h.p. every node has |out| < C log p or >= eps p, same for |in|; (c) w.h.p. any v with large out-set reaches any w with large in-set; hence the giant SCC is {large out} intersect {large in}, of size ~ x^2 p, and every other SCC has size < C log p (c > 1); (d) local limit of K-explorations to Galton-Watson Poisson(c) trees and asymptotic independence for two distinct edges (second moment).
**T2.** For c > 1, |E_up|/p -> c F_up in probability and L1, F_up = (1-x)(1-e^{-c}) + x(1-x)(1 - exp(-c(1-x)d)), d = smallest fixed point of d = 1 + e^{-c} - exp(-c(1-x)d). Then S -> 1/2 + c F_up/(2(1+c)). Plan: deterministic characterisation (witness lemma: (i,j) in E_up iff a source reaches j and not i; E_up is contained in the edges whose head has another non-loop parent), reduction to a local functional Psi_K (finite out-trees of indegree-0 nodes), error terms via (a)-(c) and first-moment counts of finite out-sets of size in [K, C log p], then the GW computation of E Psi_K and its limit K -> infinity. If the formula is wrong in detail, derive the correct one and re-check numerically.
**T3.** c < 1: N is O_P(1) (E N <= c^2/(1-c), elementary), E_up = edges into heads with another parent up to o(p) edges: L -> 1/2 + c/(2(1+c)), S -> 1/2 + c(1-e^{-c})/(2(1+c)). c = 1: same limits, proved by K-truncation (critical GW dies a.s.) and monotone coupling for N; state exactly what is NOT proved (rates, critical window c = 1 + lambda p^{-1/3}).

## Known theorems to rely on (to be opened and verified, else flagged)
Karp 1990 (transitive closure of a random digraph), Luczak 1990 / Luczak-Seierstad 2009 (critical behaviour of random digraphs), Cooper-Frieze 2004 (CPC, giant SCC for given degree sequences), Janson-Luczak-Rucinski, van der Hofstad vol. 2 (local convergence), Bordenave notes. Aim: the proof is self-contained apart from textbook facts (Chernoff bound, Le Cam/Poisson coupling, GW survival probability and total-progeny law); the cited theorems are used only for comparison.

## What counts as failure
- A step of the argument that needs more than the textbook facts above (then label sketched and say which step).
- Any intermediate quantity whose exact computation disagrees with its claimed limit beyond 3 sd and does not shrink with p (then refuted; state what survives).
- A deterministic lemma that fails an exhaustive check.

## Numerical tests (fixed seeds, p up to 1e5 and beyond, 2 cores max)
Dichotomy mass; max small-set size vs log p; second-largest SCC; giant fraction vs x^2; N vs Ntilde (local functional) for K = 5..40; E_up vs Psi_K edge by edge; E_up subset of confluence edges (exact); witnesses of large out-set never occur when in(i) is large; fraction of heads with another parent (c <= 1); Monte Carlo of the GW recursion for d; E N for c < 1 vs c^2/(1-c).

## Lean (deterministic lemmas only, genuinely used)
Witness lemma for E_up (edge into a node with another parent; strictness from a source reaching the other parent but not the tail; strictness from disjoint ancestor sets). No sorry/admit/axiom, #print axioms, build with LEAN_NUM_THREADS=2, update README, AXIOMS.txt, BUILD_LOG.txt.

## Deliverables
x1proof/X1PROOF.md (proofs, labelled, dependence explicit), scripts + result files in x1proof/, short note on replacing Appendix L's "sketch" labels.
