# Note: what can replace the "sketch" labels in Appendix L (appendix.tex, section "Random Summary Graphs: Derivations and Checks")

Source of the new statements: `X1PROOF.md` (this directory). Status after PLAN6: both ER limits are proved; nothing in Appendix L's formulas changes.

## Paragraph "Directed ER, finite limit (sketch)" -> "(proved)"

Replace the heading and the last two sentences ("The two analytic inputs are classical; the concentration step ... is not written out") by:

> **Directed ER, finite limit (proved).** Let $N$ be the number of non-loop edges inside an SCC. For every fixed $c>0$, $N/p\to c\,x^2$ in $L^1$ and in probability, hence $L\to\frac12+\frac{c(1-x^2)}{2(1+c)}$. For $c<1$, $\mathbb E N\le c^2/(1-c)$ (an edge inside an SCC lies on a directed cycle, and the expected number of such (cycle, edge) pairs of length $k$ is at most $c^k$); $c=1$ follows from $c<c'$ by monotone coupling. For $c>1$ the proof (i) shows that each node has an out-set of size $<C\log p$ or $\ge\epsilon p$, by comparing breadth-first exploration with a supercritical random walk, and likewise for in-sets; (ii) shows by a two-round exploration that w.h.p.\ every node with a large out-set reaches every node with a large in-set, so the giant SCC is $\{\text{large out}\}\cap\{\text{large in}\}$ and all other SCCs have $O(\log p)$ nodes (this is the Karp--Łuczak structure; we re-prove what is needed instead of citing it); (iii) approximates $N$ by the edge count of the bounded-size event $\{|\mathrm{out}(j)|\ge K,|\mathrm{in}(i)|\ge K\}$ (error $\le C(K+p\,e^{-\kappa K})$ in expectation), whose concentration follows from local convergence of the sparse ER digraph to the Poisson tree and a second-moment bound. Details: X1PROOF.md, Props.~5, 9.

Remove the sentence about having checked only the abstract of Dorogovtsev et al. (no longer used). Keep the remark on the Barak--Erd\H{o}s paper.

## Paragraph "Directed ER, stationary value (sketch)" -> "(proved)"

Replace "This fixed point is derived heuristically and not proved." by the statement and proof outline:

> **Directed ER, stationary value (proved).** $|E_\uparrow|/p\to cF_\uparrow$ in $L^1$ with $F_\uparrow=(1-x)(1-e^{-c})+x(1-x)\bigl(1-e^{-c(1-x)d}\bigr)$, $d$ the smallest fixed point of $d=1+e^{-c}-e^{-c(1-x)d}$. Deterministic input (Lean-checked): $(i,j)\in E_\uparrow$ iff some source node reaches a parent $k\ne i$ of $j$ and does not reach $i$ (this also gives $E_\uparrow\subseteq$ edges into confluences). Probabilistic input: for $c>1$ the witness can be taken to be a node of in-degree $0$ whose whole forward set has fewer than $K$ nodes whenever $i$ has a large in-set, because a source with a large out-set reaches every node with a large in-set (step (ii) above); the remaining witness types (forward sets of size in $[K,C\log p)$, nontrivial source SCCs, expected number $O(1)$ since $ce^{1-c}<1$) contribute $o(p)$. The resulting functional is local, so its limit is the tree probability; the recursion for $d$ is the exact probability of the event "some ancestor chain ends at a node of in-degree 0 and every node on the chain has only finite side subtrees", whose depth-truncations increase to it (smallest fixed point). For $c\le1$, $F_\uparrow=1-e^{-c}$ (confluence edges). Details: X1PROOF.md, Props.~6--10.

Large-$c$ asymptotics (already in the appendix) follow from the closed form.

## Other lines to touch

* Table \ref{tab:er} caption: add "proved limits" and the new rows if wanted: $c=1.5$, $p=10^5$, 32 graphs, $L=0.69819\pm0.00024$ (formula $0.69810$); $c=2$, $p=3.2\times10^5$, $L=0.62170\pm0.00012$ (0.62170).
* The sentence "so the status 'sketch' is limited by the missing concentration proof, not by a numerical discrepancy" -> delete. Add: "Intermediate claims (dichotomy of set sizes, structure of the SCCs, edge-by-edge equality of $E_\uparrow$ with its local approximation) were also tested exactly; X1PROOF.md, Section 8."
* Scorecard in EXT.md (X1.1, X1.2): "sketched" -> "proved (X1PROOF.md)". X1.3 (layered DAG) stays "sketched": not treated here. X1.4 (in-forests) unchanged.
* Honest caveats to keep in the text: rates are not given; the critical window $c=1+\lambda p^{-1/3}$ is not treated; the limit is $T\to\infty$ at fixed $p$ then $p\to\infty$ (fixed $T$, large $p$ is a different regime, not covered); Karp, Łuczak and Cooper--Frieze were not opened (their content is not used).
* Main text: "proved" can replace "verified numerically" for the two ER formulas, with a pointer to this appendix.
