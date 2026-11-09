"""X2: limit, structure and stationary (infinite-past) statements for summary graphs with arbitrary loop patterns.
Checks (all exact, small p):
 (L1) eventual period/preperiod of boolean powers; limit density formula delta_e, L = 1/2 + sum delta/(2|E|);
      finite-T value (fast formula, T=600) within (n0+pi+p)/T of L.
 (L2) edges with head j reaching tail i (same SCC), no fresh witness: strict only for t <= n0+1.
 (L3) crossing edge whose head lies on a cycle with SCC period g: density >= 1/g; =1 if g=1.
 (L4) every weak component strongly connected (any loop pattern) => L = 1/2.
 (L5) counterexamples to 'crossing edge exists => L > 1/2' (reported).
 (S)  stationary: Delta(v,w) = {m-n : exists u, n in L(u,v), m in L(u,w)}; E_up^Z = {(i,j): exists w, Delta(j,w)+1 != Delta(i,w)};
      compared with brute force in G_{B+T} windows (interior edges, rho restricted to window).
 (S2) stationary iff: S=1/2 <=> E_up^Z empty; strongly connected comps (any period) give empty."""
import sys, random, math
from fractions import Fraction
from math import gcd
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))   # sibling modules also under python -I
from x2_enum import powers, graph_from_bits

def seq_info(adj, p, N=200):
    B = powers(adj, p, N)
    seen = {}
    for n in range(N + 1):
        key = tuple(B[n])
        if key in seen:
            n0 = seen[key]; return B, n0, n - n0
        seen[key] = n
    raise RuntimeError("no repeat")

def reach_matrix(adj, p):
    R = [adj[i] | (1 << i) for i in range(p)]
    for k in range(p):
        for i in range(p):
            if R[i] >> k & 1: R[i] |= R[k]
    return R

def edges_of(adj, p): return [(i, j) for i in range(p) for j in range(p) if adj[i] >> j & 1]

def strict_t(B, p, i, j, t, fresh, extended):
    # B has enough powers
    h = lambda n, u, v: n >= 0 and (B[n][u] >> v & 1)
    if any(h(t, u, j) and not h(t - 1, u, i) for u in range(p)): return True
    return any(h(n, u, j) and not h(n - 1, u, i) for u in fresh for n in range(1, t + 1))

def analyse(adj, p):
    B, n0, pi = seq_info(adj, p)
    N = len(B) - 1
    # extend powers to cover t up to n0+pi+3 plus T test
    E = edges_of(adj, p); nE = len(E)
    fresh = [u for u in range(p) if not any(adj[w] >> u & 1 for w in range(p))]
    R = reach_matrix(adj, p)
    res = {"n0": n0, "pi": pi, "E": E, "edges": {}}
    for (i, j) in E:
        witness = any(any(B[n][u] >> j & 1 and not (n >= 1 and B[n - 1][u] >> i & 1) for n in range(1, N + 1)) for u in fresh)
        if witness: dens = Fraction(1)
        else:
            cnt = sum(1 for t in range(n0 + 2, n0 + 2 + pi) if any((B[t][u] >> j & 1) and not (B[t - 1][u] >> i & 1) for u in range(p)))
            dens = Fraction(cnt, pi)
        same = (R[j] >> i & 1) != 0
        # last strict t for non-witness same-SCC edges
        last = max([t for t in range(1, n0 + 2 + pi) if strict_t(B, p, i, j, t, fresh, True)] or [0])
        # period g of head's SCC (gcd of closed walk lengths at j), 0 if on no cycle
        g = 0
        for n in range(1, N + 1):
            if B[n][j] >> j & 1: g = gcd(g, n)
        res["edges"][(i, j)] = dict(dens=dens, wit=witness, same=same, last=last, g=g)
    res["L"] = Fraction(1, 2) + sum(v["dens"] for v in res["edges"].values()) / (2 * nE)
    # finite T = 600 value via fast formula
    T = 600
    BB = powers(adj, p, T)
    tot = 0
    for t in range(1, T):
        for (i, j) in E:
            tot += 1 if strict_t(BB, p, i, j, t, fresh, True) else 0
    sT = Fraction(1, 2) + Fraction(tot, 2 * (T - 1) * nE)
    res["sT"] = sT; res["T"] = T
    return res, R

def weak_strong(adj, p, R):
    # every weakly connected component strongly connected
    und = [0] * p
    for i in range(p):
        for j in range(p):
            if (adj[i] >> j & 1) or (adj[j] >> i & 1): und[i] |= 1 << j; und[j] |= 1 << i
    W = [und[i] | (1 << i) for i in range(p)]
    for k in range(p):
        for i in range(p):
            if W[i] >> k & 1: W[i] |= W[k]
    return all(((R[i] >> j & 1) and (R[j] >> i & 1)) for i in range(p) for j in range(p) if W[i] >> j & 1)

def run_limit(graphs, label):
    cnt = bad = 0; nL4 = 0; nL5 = 0; ex5 = []; denom_ok = 0
    for adj, p in graphs:
        if all(a == 0 for a in adj): continue
        res, R = analyse(adj, p); cnt += 1; nE = len(res["E"])
        # L1
        if abs(res["sT"] - res["L"]) > Fraction(res["n0"] + res["pi"] + 1, 2 * (res["T"] - 1)): bad += 1; print("L1 FAIL", adj)
        if (res["L"] - Fraction(1, 2)) * 2 * nE * res["pi"] != int((res["L"] - Fraction(1, 2)) * 2 * nE * res["pi"]): bad += 1; print("denominator FAIL", adj)
        for e, v in res["edges"].items():
            if v["same"] and not v["wit"] and v["dens"] != 0: bad += 1; print("L2 density FAIL", adj, e)
            if v["same"] and not v["wit"] and v["last"] > res["n0"] + 1: bad += 1; print("L2 last FAIL", adj, e, v["last"], res["n0"])
            if (not v["same"]) and v["g"] > 0 and v["dens"] < Fraction(1, v["g"]): bad += 1; print("L3 FAIL", adj, e)
            if (not v["same"]) and v["g"] == 1 and v["dens"] != 1: bad += 1; print("L3 g=1 FAIL", adj, e)
        if weak_strong(adj, p, R):
            nL4 += 1
            if res["L"] != Fraction(1, 2): bad += 1; print("L4 FAIL", adj)
        hascross = any(not v["same"] for v in res["edges"].values())
        if hascross and res["L"] == Fraction(1, 2):
            nL5 += 1
            if len(ex5) < 3: ex5.append(adj)
    print(f"[{label}] graphs={cnt} failures={bad}  (L4 applicable: {nL4};  crossing edge but L=1/2: {nL5}, e.g. {ex5})", flush=True)

# ---------------- stationary ----------------
def delta_sets(B, p, Nmax, K):
    # Delta[v][w] = set of d in [-K,K]: exists u, n in L(u,v), m = n+d in L(u,w), 0<=n,m<=Nmax
    D = [[set() for _ in range(p)] for _ in range(p)]
    for u in range(p):
        for v in range(p):
            Lv = [n for n in range(Nmax + 1) if B[n][u] >> v & 1]
            for w in range(p):
                Lw = set(m for m in range(Nmax + 1) if B[m][u] >> w & 1)
                for n in Lv:
                    for d in range(-K, K + 1):
                        if n + d in Lw: D[v][w].add(d)
    return D

def pred_up_Z(adj, p, K=20, Nmax=90):
    B = powers(adj, p, Nmax)
    D = delta_sets(B, p, Nmax, K + 1)
    up = {}
    for (i, j) in edges_of(adj, p):
        s = False
        for w in range(p):
            a = set(d for d in D[i][w] if abs(d) <= K - 1)
            b = set(d + 1 for d in D[j][w] if abs(d + 1) <= K - 1)
            if a != b: s = True
        up[(i, j)] = s
    return up

def brute_stationary(adj, p, Bn, Tw):
    # nodes slices 0..Bn+Tw-1; window slices Bn..Bn+Tw-1; returns per interior edge strictness via rho restricted to window
    S = Bn + Tw; n = p * S
    parents = [0] * n
    for t in range(S - 1):
        for i in range(p):
            for j in range(p):
                if adj[i] >> j & 1: parents[(t + 1) * p + j] |= 1 << (t * p + i)
    anc = [0] * n
    for t in range(S):
        for v in range(p):
            x = t * p + v; a = 1 << x; m = parents[x]
            while m:
                b = m & -m; m ^= b; a |= anc[b.bit_length() - 1]
            anc[x] = a
    desc = [1 << x for x in range(n)]
    for x in range(n):
        a = anc[x]
        while a:
            b = a & -a; a ^= b; desc[b.bit_length() - 1] |= 1 << x
    mask = 0
    for x in range(Bn * p, n): mask |= 1 << x
    rho = {}
    for y in range(Bn * p, n):
        r = 0; a = anc[y]
        while a:
            b = a & -a; a ^= b; r |= desc[b.bit_length() - 1]
        rho[y] = bin(r & mask).count("1")
    return rho

def run_stationary(graphs, label, Bn, Tw, margin):
    cnt = bad = 0; nonsat = 0; ex = []; tot_up = 0; tot_e = 0
    for adj, p in graphs:
        if all(a == 0 for a in adj): continue
        up = pred_up_Z(adj, p); tot_up += sum(up.values()); tot_e += len(up); rho = brute_stationary(adj, p, Bn, Tw); cnt += 1
        for (i, j), s in up.items():
            for t in range(margin, Tw - 1 - margin):
                a, b = rho[(Bn + t) * p + i], rho[(Bn + t + 1) * p + j]
                obs = a < b
                if a > b: bad += 1; print("ORDER FAIL", adj)
                if obs != s:
                    bad += 1
                    if bad < 4: print("STAT FAIL", adj, (i, j), t, a, b, s)
                    break
        # S = 1/2 iff no up edges ; and strongly connected => none
        R = reach_matrix(adj, p)
        if weak_strong(adj, p, R) and any(up.values()): bad += 1; print("S2 FAIL", adj)
        if any(up.values()) and False: pass
    print(f"[{label}] graphs={cnt} edges={tot_e} strict-edges(predicted)={tot_up} failures={bad}", flush=True)

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "limit":
        run_limit([(graph_from_bits(b, 2), 2) for b in range(1 << 4)], "limit p=2 exhaustive")
        run_limit([(graph_from_bits(b, 3), 3) for b in range(1 << 9)], "limit p=3 exhaustive")
        rng = random.Random(5)
        gs = []
        for _ in range(int(sys.argv[2])):
            p = 4; q = rng.choice([0.15, 0.25, 0.4])
            gs.append(([sum((1 << j) for j in range(p) if rng.random() < q) for i in range(p)], p))
        run_limit(gs, "limit p=4 random")
        gs = []
        for _ in range(int(sys.argv[3])):
            p = rng.choice([5, 6]); q = rng.choice([0.12, 0.2, 0.3])
            gs.append(([sum((1 << j) for j in range(p) if rng.random() < q) for i in range(p)], p))
        run_limit(gs, "limit p=5,6 random")
    else:
        run_stationary([(graph_from_bits(b, 2), 2) for b in range(1 << 4)], "stationary p=2 exhaustive", 40, 90, 30)
        run_stationary([(graph_from_bits(b, 3), 3) for b in range(1 << 9)], "stationary p=3 exhaustive", 40, 90, 30)
        rng = random.Random(6); gs = []
        for _ in range(int(sys.argv[2])):
            p = 4; q = rng.choice([0.15, 0.25, 0.4])
            gs.append(([sum((1 << j) for j in range(p) if rng.random() < q) for i in range(p)], p))
        run_stationary(gs, "stationary p=4 random", 40, 90, 30)
