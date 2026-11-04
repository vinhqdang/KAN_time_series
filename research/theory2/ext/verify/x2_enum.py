"""X2: summary graphs without (all) self-loops. Brute-force relatives in G_T vs the walk-length formula.
Exhaustive: all 2^(p*p) digraphs (loop patterns included) for p<=3 (T=2..9), p=4 (T in {3,5,8}); random p=5,6.
Predicted per-edge strictness (paper time t=1..T-1, edge (i(t), j(t+1))):
  strict iff exists u: B_t[u,j] and not B_{t-1}[u,i]                     (u(1) is a root)
         or exists fresh u (in-degree 0 in E_S), 1<=n<=t: B_n[u,j] and not B_{n-1}[u,i]   (u(s) is a root for every s)
B_n = boolean n-th power of the adjacency matrix (B_0 = I)."""
import sys, random, itertools
from fractions import Fraction

def powers(adj, p, N):
    # B[n][u] = bitmask of v with walk length n u->v
    B = [[1 << u for u in range(p)]]
    for n in range(N):
        prev = B[-1]
        nxt = []
        for u in range(p):
            m = 0; x = prev[u]
            for v in range(p):
                if x >> v & 1: m |= adj[v]
            nxt.append(m)
        B.append(nxt)
    return B

def brute_edge_scores(adj, p, T):
    # nodes id = t*p+v, t=0..T-1 (slice t+1)
    n = p * T
    parents = [0] * n
    for t in range(T - 1):
        for i in range(p):
            for j in range(p):
                if adj[i] >> j & 1:
                    parents[(t + 1) * p + j] |= 1 << (t * p + i)
    anc = [0] * n
    for t in range(T):          # topological = by time
        for v in range(p):
            x = t * p + v
            a = 1 << x
            m = parents[x]
            while m:
                b = m & -m; m ^= b
                a |= anc[b.bit_length() - 1]
            anc[x] = a
    desc = [1 << x for x in range(n)]
    for x in range(n):
        a = anc[x]
        while a:
            b = a & -a; a ^= b
            desc[b.bit_length() - 1] |= 1 << x
    rel = []
    for y in range(n):
        r = 0; a = anc[y]
        while a:
            b = a & -a; a ^= b
            r |= desc[b.bit_length() - 1]
        rel.append(bin(r).count("1"))
    out = {}
    for t in range(T - 1):
        for i in range(p):
            for j in range(p):
                if adj[i] >> j & 1:
                    a, b = rel[t * p + i], rel[(t + 1) * p + j]
                    out[(i, j, t + 1)] = Fraction(1) if a < b else (Fraction(1, 2) if a == b else Fraction(0))
    return out

def pred_edge_scores(adj, p, T):
    B = powers(adj, p, T)
    indeg = [0] * p
    for i in range(p):
        for j in range(p):
            if adj[i] >> j & 1: indeg[j] += 1
    fresh = [u for u in range(p) if indeg[u] == 0]
    out = {}
    def has(n, u, v): return n >= 0 and (B[n][u] >> v & 1)
    for t in range(1, T):
        for i in range(p):
            for j in range(p):
                if not adj[i] >> j & 1: continue
                s = any(has(t, u, j) and not has(t - 1, u, i) for u in range(p))
                if not s:
                    s = any(has(n, u, j) and not has(n - 1, u, i) for u in fresh for n in range(1, t + 1))
                out[(i, j, t)] = Fraction(1) if s else Fraction(1, 2)
    return out

def check(adj, p, T):
    a = brute_edge_scores(adj, p, T); b = pred_edge_scores(adj, p, T)
    return a == b, (a, b)

def graph_from_bits(bits, p):
    return [sum(((bits >> (i * p + j)) & 1) << j for j in range(p)) for i in range(p)]

if __name__ == "__main__":
    which = sys.argv[1]
    bad = 0; cnt = 0
    if which == "exh":
        p = int(sys.argv[2]); Ts = [int(x) for x in sys.argv[3].split(",")]
        for bits in range(1 << (p * p)):
            adj = graph_from_bits(bits, p)
            if all(a == 0 for a in adj): continue
            for T in Ts:
                ok, ex = check(adj, p, T); cnt += 1
                if not ok:
                    bad += 1
                    if bad <= 3: print("MISMATCH", adj, T)
    else:
        p = int(sys.argv[2]); Ts = [int(x) for x in sys.argv[3].split(",")]; n = int(sys.argv[4])
        rng = random.Random(20261105 + p)
        for _ in range(n):
            q = rng.choice([0.15, 0.25, 0.4])
            adj = [sum((1 << j) for j in range(p) if rng.random() < q) for i in range(p)]
            if all(a == 0 for a in adj): continue
            for T in Ts:
                ok, ex = check(adj, p, T); cnt += 1
                if not ok:
                    bad += 1
                    if bad <= 3: print("MISMATCH", adj, T)
    print(f"mode={which} p={sys.argv[2]} Ts={sys.argv[3]} cases={cnt} mismatches={bad}")
