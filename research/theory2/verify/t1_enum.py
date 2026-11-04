"""T1 verification by exact enumeration of relatives in time-unrolled DAGs.

Definitions exactly as Reisach et al. (arXiv 2605.06288): P_aa={a}, anc(y) includes y, rel(y)=desc(anc(y)),
rho(v)=|rel(v)|, s_rho = mean over edges of 1 / 1/2 / 0.  G_T: nodes v(t), t=1..T; edges (i(t), j(t+1)) for
(i,j) in E_S (self-loops included), t=1..T-1.

Checks (all exact, integer/Fraction arithmetic):
 F1  per-edge value in G_T equals 1/2 + 1/2*1[exists u: d(u,j) <= t <= d(u,i)]   (d = shortest-path length, inf if none)
 F2  rho(v(t)) = sum_w max(T - m_t(v,w), 0),  m_t(v,w) = min{ d(u,w) : d(u,v) <= t-1 }
 F3  limit L = 1/2 + |E_cross|/(2|E_S|), E_cross = edges between different SCCs; 0 <= s(G_T)-L <= (p-1)|E_within|/(2(T-1)|E_S|)
 S1  stationary (latent burn-in of B slices, window of T slices; relatives = marginal d-connection in the full DAG,
     restricted to the window): for B >= p-1, rho(v(t)) = T*|rel_S(v)| and per-edge value = 1/2+1/2*1[rel_S(i)!=rel_S(j)]
     = 1/2 + 1/2*1[Src(j) != Src(i)] (Src = source SCCs of the condensation reaching the node)
 C0  the plan's original conjecture applied to the FINITE G_T limit (edge strict iff |rel_S(j)|>|rel_S(i)|): counted as
     counterexamples when it disagrees with the exact limit.
Run: OMP_NUM_THREADS=1 /usr/bin/python3 t1_enum.py  (seeds fixed)."""
import itertools, random, sys
from fractions import Fraction as Fr
INF = 10 ** 9

def dist_matrix(p, E):
    d = [[INF] * p for _ in range(p)]
    for i in range(p): d[i][i] = 0
    for (i, j) in E:
        if i != j: d[i][j] = min(d[i][j], 1)
    for k in range(p):
        for i in range(p):
            for j in range(p):
                if d[i][k] + d[k][j] < d[i][j]: d[i][j] = d[i][k] + d[k][j]
    return d

def sccs(p, d):
    comp = [-1] * p; c = 0
    for i in range(p):
        if comp[i] < 0:
            for j in range(p):
                if d[i][j] < INF and d[j][i] < INF: comp[j] = c
            c += 1
    return comp

def unrolled_rel(p, E, Ttot):
    """bitset relatives in G_{Ttot}; node index v*Ttot + (t-1)"""
    N = p * Ttot; idx = lambda v, t: v * Ttot + (t - 1)
    par = [[] for _ in range(N)]; ch = [[] for _ in range(N)]
    for t in range(1, Ttot):
        for (i, j) in E:
            par[idx(j, t + 1)].append(idx(i, t)); ch[idx(i, t)].append(idx(j, t + 1))
    anc = [0] * N; desc = [0] * N
    for t in range(1, Ttot + 1):
        for v in range(p):
            x = idx(v, t); a = 1 << x
            for q in par[x]: a |= anc[q]
            anc[x] = a
    for t in range(Ttot, 0, -1):
        for v in range(p):
            x = idx(v, t); a = 1 << x
            for q in ch[x]: a |= desc[q]
            desc[x] = a
    rel = [0] * N
    for x in range(N):
        a = anc[x]; r = 0
        while a:
            lb = a & -a; q = lb.bit_length() - 1; r |= desc[q]; a ^= lb
        rel[x] = r
    return rel, idx

def edge_val(rx, ry):
    return Fr(1) if rx < ry else (Fr(1, 2) if rx == ry else Fr(0))

def summary_rel(p, d):
    anc = [{u for u in range(p) if d[u][v] < INF} for v in range(p)]
    rel = [{w for w in range(p) if any(d[u][w] < INF for u in anc[v])} for v in range(p)]
    return anc, rel

def source_sccs(p, d, comp):
    src_comps = set()
    for c in set(comp):
        mem = [v for v in range(p) if comp[v] == c]
        if not any(d[u][mem[0]] < INF and comp[u] != c for u in range(p)): src_comps.add(c)
    return [frozenset(comp[u] for u in range(p) if comp[u] in src_comps and d[u][v] < INF) for v in range(p)]

def check_graph(p, E, Ts, stats):
    d = dist_matrix(p, E); comp = sccs(p, d)
    ancS, relS = summary_rel(p, d); Src = source_sccs(p, d, comp)
    Ecross = sum(1 for (i, j) in E if comp[i] != comp[j]); Ewith = len(E) - Ecross
    L = Fr(1, 2) + Fr(Ecross, 2 * len(E))
    Lstat = Fr(1, 2) + Fr(sum(1 for (i, j) in E if relS[i] != relS[j]), 2 * len(E))
    # source-SCC characterisation of stationary strictness
    for (i, j) in E:
        if (relS[i] != relS[j]) != (Src[i] != Src[j]): stats["S1_src_fail"] += 1
    # plan conjecture (count comparison on summary relatives) vs exact finite limit
    Lconj = Fr(1, 2) + Fr(sum(1 for (i, j) in E if len(relS[j]) > len(relS[i])), 2 * len(E))
    if Lconj != L: stats["C0_conj_vs_finite_limit_disagree"] += 1
    if Lconj != Lstat: stats["C0_conj_vs_stationary_disagree"] += 1
    for T in Ts:
        rel, idx = unrolled_rel(p, E, T)
        rho = [bin(r).count("1") for r in rel]
        tot = Fr(0)
        for t in range(1, T):
            for (i, j) in E:
                v = edge_val(rho[idx(i, t)], rho[idx(j, t + 1)]); tot += v
                pred = Fr(1, 2) + (Fr(1, 2) if any(d[u][j] <= t <= d[u][i] for u in range(p)) else 0)
                if v != pred: stats["F1_fail"] += 1
                stats["F1_edges"] += 1
        # F2
        for v in range(p):
            for t in range(1, T + 1):
                B = [u for u in range(p) if d[u][v] <= t - 1]
                pr = sum(max(T - min(d[u][w] for u in B), 0) for w in range(p))
                if pr != rho[idx(v, t)]: stats["F2_fail"] += 1
        s = tot / ((T - 1) * len(E))
        bound = Fr((p - 1) * Ewith, 2 * (T - 1) * len(E))
        if not (L <= s <= L + bound): stats["F3_fail"] += 1
        stats["maxgap"] = max(stats["maxgap"], float(s - L))
        # stationary: burn-in B slices latent
        for Bb in sorted({max(p - 1, 0), 2 * p + 1}):
            Tt = Bb + T; rel2, idx2 = unrolled_rel(p, E, Tt)
            W = 0
            for v in range(p):
                for t in range(Bb + 1, Tt + 1): W |= 1 << idx2(v, t)
            rho2 = {}
            for v in range(p):
                for t in range(Bb + 1, Tt + 1):
                    rho2[(v, t)] = bin(rel2[idx2(v, t)] & W).count("1")
                    if rho2[(v, t)] != T * len(relS[v]): stats["S1_rho_fail"] += 1
            tot2 = Fr(0)
            for t in range(Bb + 1, Tt):
                for (i, j) in E:
                    tot2 += edge_val(rho2[(i, t)], rho2[(j, t + 1)])
            if tot2 / ((T - 1) * len(E)) != Lstat: stats["S1_fail"] += 1
        stats["checks"] += 1
    return L, Lstat

def rand_graph(rng, p):
    kind = rng.choice(["er", "er", "dag", "inforest", "outforest", "chainish", "blocks"])
    E = set((v, v) for v in range(p))
    if kind == "er":
        q = rng.uniform(0.05, 0.7)
        for i in range(p):
            for j in range(p):
                if i != j and rng.random() < q: E.add((i, j))
    elif kind == "dag":
        q = rng.uniform(0.1, 0.8); perm = list(range(p)); rng.shuffle(perm)
        for a in range(p):
            for b in range(a + 1, p):
                if rng.random() < q: E.add((perm[a], perm[b]))
    elif kind == "inforest":
        perm = list(range(p)); rng.shuffle(perm)
        for a in range(p - 1):
            if rng.random() < 0.85: E.add((perm[a], perm[rng.randrange(a + 1, p)]))
    elif kind == "outforest":
        perm = list(range(p)); rng.shuffle(perm)
        for b in range(1, p):
            if rng.random() < 0.85: E.add((perm[rng.randrange(0, b)], perm[b]))
    elif kind == "chainish":
        perm = list(range(p)); rng.shuffle(perm)
        for a in range(p - 1): E.add((perm[a], perm[a + 1]))
        if rng.random() < 0.5: E.add((perm[-1], perm[rng.randrange(0, p - 1)]))  # back edge
    else:  # strongly connected blocks wired feed-forward
        k = rng.randint(1, p); lab = [rng.randrange(k) for _ in range(p)]
        for i in range(p):
            for j in range(p):
                if i != j and lab[i] == lab[j] and rng.random() < 0.6: E.add((i, j))
                if lab[i] < lab[j] and rng.random() < 0.3: E.add((i, j))
    return sorted(E), kind

if __name__ == "__main__":
    from collections import Counter
    stats = Counter(maxgap=0.0, F1_fail=0, F2_fail=0, F3_fail=0, S1_fail=0, S1_rho_fail=0, S1_src_fail=0, C0_conj_vs_stationary_disagree=0)
    out = []
    def P_(s): print(s); sys.stdout.flush(); out.append(s)
    # exhaustive: all summary graphs (with all self-loops) on p=2,3,4 nodes
    for p, Ts in [(2, [2, 3, 4, 5, 6]), (3, [2, 3, 4, 6, 9]), (4, [2, 3, 5, 8])]:
        pairs = [(i, j) for i in range(p) for j in range(p) if i != j]
        n = 0
        for mask in range(1 << len(pairs)):
            E = [(v, v) for v in range(p)] + [pairs[k] for k in range(len(pairs)) if mask >> k & 1]
            check_graph(p, E, Ts, stats); n += 1
        P_(f"exhaustive p={p}: {n} graphs, T in {Ts}; cumulative {dict(stats)}")
    rng = random.Random(20261104)
    kinds = Counter(); n = 0
    for g in range(10000):
        p = rng.randint(2, 7); E, kind = rand_graph(rng, p); kinds[kind] += 1
        Ts = [2, rng.randint(3, 6), rng.randint(7, 14)]
        check_graph(p, E, Ts, stats); n += 1
    P_(f"random: {n} graphs (seed 20261104, p in 2..7, kinds {dict(kinds)}); cumulative {dict(stats)}")
    with open(__file__.replace("t1_enum.py", "t1_enum_results.txt"), "w") as f: f.write("\n".join(out) + "\n")
