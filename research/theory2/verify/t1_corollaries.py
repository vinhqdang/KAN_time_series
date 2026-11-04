"""Brute-force checks of the T1 corollaries (in-forests, out-forests, chains, colliders, forks, DAG summaries,
Theorem 4 of Reisach et al. and its converse), using the exact enumerator of t1_enum.py.
Run: OMP_NUM_THREADS=1 /usr/bin/python3 t1_corollaries.py"""
import random, sys
from fractions import Fraction as Fr
from collections import Counter
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from t1_enum import unrolled_rel, edge_val, dist_matrix, sccs, INF

def s_finite(p, E, T):
    rel, idx = unrolled_rel(p, E, T); rho = [bin(r).count("1") for r in rel]
    return sum(edge_val(rho[idx(i, t)], rho[idx(j, t + 1)]) for t in range(1, T) for (i, j) in E) / ((T - 1) * len(E))

def s_stat(p, E, T):
    B = p - 1; Tt = B + T; rel, idx = unrolled_rel(p, E, Tt); W = 0
    for v in range(p):
        for t in range(B + 1, Tt + 1): W |= 1 << idx(v, t)
    rho = lambda v, t: bin(rel[idx(v, t)] & W).count("1")
    return sum(edge_val(rho(i, t), rho(j, t + 1)) for t in range(B + 1, Tt) for (i, j) in E) / ((T - 1) * len(E))

rng = random.Random(7); out = []; c = Counter()
def P_(s): print(s); out.append(s)
for g in range(3000):
    p = rng.randint(2, 8); perm = list(range(p)); rng.shuffle(perm); T = rng.randint(2, 6)
    # in-forest: each node has at most one out-neighbour (downstream), acyclic
    E = [(v, v) for v in range(p)]
    for a in range(p - 1):
        if rng.random() < 0.85: E.append((perm[a], perm[rng.randrange(a + 1, p)]))
    m = len(E) - p
    indeg = Counter(j for (i, j) in E if i != j)
    pred_stat = Fr(1, 2) + Fr(sum(1 for (i, j) in E if i != j and indeg[j] >= 2), 2 * len(E))
    pred_fin_lim = Fr(1, 2) + Fr(m, 2 * (m + p))
    c["inforest_stat_fail"] += s_stat(p, E, T) != pred_stat
    c["inforest_stat_gt_half_iff_confluence_fail"] += (s_stat(p, E, T) > Fr(1, 2)) != (max(indeg.values(), default=0) >= 2)
    c["inforest_fin_fail"] += abs(s_finite(p, E, 60) - pred_fin_lim) > Fr(p, 2 * 59)  # within the F3 bound
    # out-forest: each node has at most one parent
    E2 = [(v, v) for v in range(p)]
    for b in range(1, p):
        if rng.random() < 0.85: E2.append((perm[rng.randrange(0, b)], perm[b]))
    c["outforest_stat_not_half"] += s_stat(p, E2, T) != Fr(1, 2)
    m2 = len(E2) - p
    c["outforest_fin_fail"] += abs(s_finite(p, E2, 60) - (Fr(1, 2) + Fr(m2, 2 * (m2 + p)))) > Fr(p, 2 * 59)
    # random summary graph: lim 1/2 iff every weak component strongly connected (Thm 4 + converse)
    E3 = [(v, v) for v in range(p)] + [(i, j) for i in range(p) for j in range(p) if i != j and rng.random() < rng.uniform(0.05, 0.6)]
    d = dist_matrix(p, E3); comp = sccs(p, d)
    cond = all(comp[i] == comp[j] for (i, j) in E3)
    sf = s_finite(p, E3, 80)
    if cond:   # Theorem 4 direction: s(G_T) - 1/2 <= (p-1)/(2(T-1))
        c["thm4_fail"] += not (Fr(1, 2) <= sf <= Fr(1, 2) + Fr(p - 1, 2 * 79))
    else:      # converse: s(G_T) >= 1/2 + 1/(2|E_S|) for every T, so the limit is not 1/2
        c["thm4_converse_fail"] += not (sf >= Fr(1, 2) + Fr(1, 2 * len(E3)))
    c["cond_true"] += cond
    # stationary: 1/2 iff each weak component has exactly one source SCC
    nsrc_ok = True
    # weak components
    und = {v: {v} for v in range(p)}
    for (i, j) in E3:
        if und[i] is not und[j]:
            s = und[i] | und[j]
            for v in s: und[v] = s
    for compset in {frozenset(s) for s in und.values()}:
        srcs = {comp[v] for v in compset if not any(d[u][v] < INF and comp[u] != comp[v] for u in range(p))}
        if len(srcs) != 1: nsrc_ok = False
    c["stat_half_iff_unique_source_fail"] += nsrc_ok != (s_stat(p, E3, T) == Fr(1, 2))
    c["graphs"] += 1
P_(f"corollary checks, 3000 random graphs per family (seed 7): {dict(c)}")
# named examples
ex = {"chain 0->1->2": (3, [(0, 1), (1, 2)]), "collider 0->2<-1": (3, [(0, 2), (1, 2)]), "fork 0<-2->1": (3, [(2, 0), (2, 1)]),
      "Fig4-like 3-cycle": (3, [(0, 1), (1, 2), (2, 0)]), "2 nodes 0->1": (2, [(0, 1)])}
for name, (p, E) in ex.items():
    E = [(v, v) for v in range(p)] + E
    P_(f"{name}: finite s(G_T) T=2,5,20,100: {[str(s_finite(p, E, T)) for T in (2, 5, 20, 100)]}; stationary: {s_stat(p, E, 5)}")
# self-loops are essential: 0->1 without self-loops
P_(f"no self-loops, 0->1: finite s(G_T) T=5: {s_finite(2, [(0, 1)], 5)} (with self-loops limit 2/3)")
with open(__file__.replace(".py", "_results.txt"), "w") as f: f.write("\n".join(out) + "\n")
