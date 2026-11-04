"""X1: directed Erdos-Renyi summary graph G(p,c/p) (non-loop edges) plus a self-loop at every node.
Exact computation of L = 1/2 + |E_x|/(2|E_S|) (Thm 1.5) and S = 1/2 + |E_up|/(2|E_S|) (Thm 1.9),
E_up = edges (i,j) with Src(j) != Src(i) (source SCCs reaching j vs i; exact via bitsets),
against the branching-process formulas (EXT.md, Prop X1.1/X1.2)."""
import sys, math
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def x_surv(c):
    if c <= 1: return 0.0
    x = 0.9
    for _ in range(10000): x = 1 - math.exp(-c * x)
    return x

def d_fix(c, x):
    d = math.exp(-c)
    for _ in range(100000):
        nd = 1 + math.exp(-c) - math.exp(-c * (1 - x) * d)
        if abs(nd - d) < 1e-15: break
        d = nd
    return d

def predict(c):
    x = x_surv(c); d = d_fix(c, x)
    Fx = 1 - x * x
    Fup = (1 - x) * (1 - math.exp(-c)) + x * (1 - x) * (1 - math.exp(-c * (1 - x) * d))
    return (0.5 + c * Fx / (2 * (1 + c)), 0.5 + c * Fup / (2 * (1 + c)), x, d, Fx, Fup)

def gen(p, c, rng):
    m = rng.binomial(p * (p - 1), c / p)
    i = rng.integers(0, p, size=int(m * 1.02) + 10); j = rng.integers(0, p, size=len(i))
    k = i != j
    key = np.unique(i[k].astype(np.int64) * p + j[k])
    rng.shuffle(key); key = key[:m]
    return (key // p).astype(np.int64), (key % p).astype(np.int64)

def measure(p, c, seed, want_S=True):
    rng = np.random.default_rng(seed)
    ei, ej = gen(p, c, rng)
    return LS_from_edges(p, ei, ej, want_S, c=c, seed=seed)

def LS_from_edges(p, ei, ej, want_S=True, c=None, seed=None):
    m = len(ei)
    A = csr_matrix((np.ones(m, dtype=np.int8), (ei, ej)), shape=(p, p))
    nc, lab = connected_components(A, directed=True, connection="strong")
    ci, cj = lab[ei], lab[ej]
    cross = ci != cj
    ncross = int(cross.sum()); ES = p + m
    L = 0.5 + ncross / (2 * ES)
    giant = np.bincount(lab).max() / p
    out = dict(p=p, c=c, seed=seed, m=m, giant=giant, Fx=ncross / m if m else float('nan'), L=L)
    if not want_S: return out
    # condensation DAG, topological order (Kahn)
    xi, xj = ci[cross], cj[cross]
    indeg = np.bincount(xj, minlength=nc)
    src = np.where(indeg == 0)[0]; k = len(src)
    sid = -np.ones(nc, dtype=np.int64); sid[src] = np.arange(k)
    order_idx = np.argsort(xi, kind="stable"); xi_s, xj_s = xi[order_idx], xj[order_idx]
    start = np.searchsorted(xi_s, np.arange(nc + 1))
    deg = indeg.copy(); stack = list(src); topo = []
    while stack:
        u = stack.pop(); topo.append(u)
        for v in xj_s[start[u]:start[u + 1]]:
            deg[v] -= 1
            if deg[v] == 0: stack.append(v)
    assert len(topo) == nc
    nb = (k + 7) // 8
    R = np.zeros((nc, nb), dtype=np.uint8)
    for s_ in src:
        R[s_, sid[s_] >> 3] |= (1 << (sid[s_] & 7))
    for u in topo:  # push to successors
        ru = R[u]
        for v in xj_s[start[u]:start[u + 1]]:
            R[v] |= ru
    pc = np.unpackbits(R, axis=1).sum(axis=1)
    up = pc[ci] != pc[cj]
    out.update(S=0.5 + int(up.sum()) / (2 * ES), Fup=int(up.sum()) / m, nsrc=k / p)
    return out

if __name__ == "__main__":
    p = int(sys.argv[1]); cs = [float(x) for x in sys.argv[2].split(",")]; nseed = int(sys.argv[3])
    wantS = (len(sys.argv) < 5 or sys.argv[4] != "L")
    print("p c | L_exact(mean+-sd) L_pred | S_exact(mean+-sd) S_pred | giant x^2 | Fx Fup (exact vs pred)")
    for c in cs:
        res = [measure(p, c, 1000 * int(c * 100) + s, wantS) for s in range(nseed)]
        Lp, Sp, x, d, Fx, Fup = predict(c)
        La = np.array([r["L"] for r in res]); ga = np.mean([r["giant"] for r in res])
        line = f"{p} {c:4.2f} | {La.mean():.5f}+-{La.std(ddof=1) if nseed>1 else 0:.5f} {Lp:.5f} | "
        if wantS:
            Sa = np.array([r["S"] for r in res]); Fu = np.mean([r["Fup"] for r in res])
            line += f"{Sa.mean():.5f}+-{Sa.std(ddof=1) if nseed>1 else 0:.5f} {Sp:.5f} | "
        else: line += "- | "
        line += f"{ga:.4f} {x*x:.4f} | Fx {np.mean([r['Fx'] for r in res]):.4f} {Fx:.4f}"
        if wantS: line += f" Fup {Fu:.4f} {Fup:.4f}"
        print(line, flush=True)
