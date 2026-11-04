"""X2: named examples (limit L from the density formula, stationary S^Z), and check L = S^Z for acyclic loop-free graphs."""
import random
from fractions import Fraction
from x2_limit import analyse, pred_up_Z, edges_of
from x2_enum import graph_from_bits

def adj_from_edges(p, es):
    a = [0] * p
    for (i, j) in es: a[i] |= 1 << j
    return a

def LS(p, es):
    adj = adj_from_edges(p, es)
    res, R = analyse(adj, p)
    up = pred_up_Z(adj, p); nE = len(edges_of(adj, p))
    return res["L"], Fraction(1, 2) + Fraction(sum(up.values()), 2 * nE), res

ex = {
 "edge 0->1, no loops": (2, [(0, 1)]),
 "edge 0->1, loop at 0": (2, [(0, 1), (0, 0)]),
 "edge 0->1, loop at 1": (2, [(0, 1), (1, 1)]),
 "edge 0->1, loops at both": (2, [(0, 1), (0, 0), (1, 1)]),
 "collider a->c<-b, no loops": (3, [(0, 2), (1, 2)]),
 "collider, loops everywhere": (3, [(0, 2), (1, 2), (0, 0), (1, 1), (2, 2)]),
 "fork c->a, c->b, no loops": (3, [(2, 0), (2, 1)]),
 "chain 0->1->2, no loops": (3, [(0, 1), (1, 2)]),
 "directed 3-cycle, no loops": (3, [(0, 1), (1, 2), (2, 0)]),
 "2-cycle + fresh 2->0, no loops": (3, [(0, 1), (1, 0), (2, 0)]),
 "2-cycle + 2->0, loop at 2": (3, [(0, 1), (1, 0), (2, 0), (2, 2)]),
 "2-cycle + 0->2, no loops": (3, [(0, 1), (1, 0), (0, 2)]),
 "2-cycle + 0->2, loop at 2": (3, [(0, 1), (1, 0), (0, 2), (2, 2)]),
}
for k, (p, es) in ex.items():
    L, S, res = LS(p, es)
    print(f"{k:34s} L = {str(L):6s} ({float(L):.4f})   S^Z = {str(S):6s} ({float(S):.4f})   period={res['pi']} preperiod={res['n0']}")
# acyclic loop-free: L == S^Z ?
rng = random.Random(11); n = bad = 0
for _ in range(400):
    p = rng.choice([3, 4, 5]); perm = list(range(p)); rng.shuffle(perm)
    es = [(perm[a], perm[b]) for a in range(p) for b in range(a + 1, p) if rng.random() < 0.4]
    if not es: continue
    L, S, _ = LS(p, es); n += 1
    if L != S: bad += 1; print("L != S^Z", p, es, L, S)
print(f"acyclic loop-free random graphs: {n}, L != S^Z in {bad}")

# loop-free in-forests: L = S^Z = 1/2 + (m - n_1)/(2m)
rng = random.Random(12); n = bad = 0
for _ in range(300):
    p = rng.choice([3, 4, 5, 6, 7])
    es = [(v, rng.randrange(v + 1, p)) for v in range(p - 1) if rng.random() < 0.8]   # v flows to a later node: in-forest, acyclic
    if not es: continue
    L, S, _ = LS(p, es); n += 1
    m = len(es); indeg = [sum(1 for e in es if e[1] == v) for v in range(p)]
    n1 = sum(1 for d in indeg if d == 1)
    pred = Fraction(1, 2) + Fraction(m - n1, 2 * m)
    if not (L == S == pred): bad += 1; print("forest formula FAIL", p, es, L, S, pred)
print(f"loop-free in-forests: {n} graphs, formula 1/2+(m-n_1)/(2m) failures: {bad}")
