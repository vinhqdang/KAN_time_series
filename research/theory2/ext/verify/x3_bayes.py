"""X3: (i) oracle sort-and-orient accuracy equals s_rho; (ii) Bayes-optimal rule over the uniform ensemble of labelled DAGs
for features (rho_first, rho_second) [pairwise count rule] equals the edge-pooled s_rho; (iii) rules that also see the node
degrees do strictly better (so the optimality needs the restriction to counts).
Edge-pooled: sum over DAGs of (number of edges * s_rho) / total edges. Rule sees an edge in a uniformly random order."""
import itertools, sys
from collections import defaultdict
from fractions import Fraction

def all_dags(p):
    seen = set()
    pairs = [(i, j) for i in range(p) for j in range(p) if i != j]
    for perm in itertools.permutations(range(p)):
        pos = {v: k for k, v in enumerate(perm)}
        fwd = [(i, j) for (i, j) in pairs if pos[i] < pos[j]]
        for mask in range(1 << len(fwd)):
            es = frozenset(fwd[k] for k in range(len(fwd)) if mask >> k & 1)
            seen.add(es)
    return seen

def rho_vec(es, p):
    anc = [1 << v for v in range(p)]
    changed = True
    while changed:
        changed = False
        for (i, j) in es:
            n = anc[j] | anc[i]
            if n != anc[j]: anc[j] = n; changed = True
    desc = [1 << v for v in range(p)]
    for v in range(p):
        for u in range(p):
            if anc[v] >> u & 1: desc[u] |= 1 << v
    rho = []
    for y in range(p):
        r = 0
        for x in range(p):
            if anc[y] >> x & 1: r |= desc[x]
        rho.append(bin(r).count("1"))
    return rho

def run(p):
    dags = all_dags(p)
    N1 = defaultdict(lambda: [0, 0]); N2 = defaultdict(lambda: [0, 0])  # features -> [first is tail, first is head]
    spool = Fraction(0); tot = 0; strict = wrong = 0
    for es in dags:
        rho = rho_vec(es, p)
        deg = [sum(1 for (i, j) in es if v in (i, j)) for v in range(p)]
        for (x, y) in es:
            tot += 1
            a, b = rho[x], rho[y]
            spool += Fraction(1) if a < b else (Fraction(1, 2) if a == b else Fraction(0))
            if a < b: strict += 1
            if a > b: wrong += 1
            # order (x,y): first = tail
            N1[(a, b)][0] += 1; N2[(deg[x], deg[y], a, b)][0] += 1
            # order (y,x): first = head
            N1[(b, a)][1] += 1; N2[(deg[y], deg[x], b, a)][1] += 1
    # each edge presented in two orders with prob 1/2 each: total weight per edge 1 -> halve counts
    def bayes(N): return Fraction(sum(max(v) for v in N.values()), 2 * tot)
    return len(dags), tot, spool / tot, bayes(N1), bayes(N2), strict / tot, wrong

if __name__ == "__main__":
    for p in [2, 3, 4, 5]:
        nd, tot, s, b1, b2, cov, wrong = run(p)
        print(f"p={p} DAGs={nd} edges={tot} pooled s_rho={float(s):.6f} Bayes(rho pair)={float(b1):.6f} equal={s==b1} "
              f"Bayes(rho pair + degrees)={float(b2):.6f} coverage(strict)={cov:.4f} 1/2+cov/2={0.5+cov/2:.6f} edges with rho_x>rho_y={wrong}")
