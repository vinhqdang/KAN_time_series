"""Convergence-in-p check for the ER table row c = 1 (appendix tab:er): S exact 0.6557+-0.0013 at p = 20,000 vs
formula 0.65803.  Same generator, seeds and SCC/source-set definition as ../ext/verify/x1_er.py (imported), but the
source-set comparison is done in chunks of sources and level by level (np.bitwise_or.at), so that memory stays
O(n_scc * chunk/8) and p up to 10^6 is feasible on one core.  Exact (no hashing): an edge (i,j) between SCCs is an
up-edge iff Src(j) != Src(i); since Src(i) is a subset of Src(j), this holds iff some chunk's popcount differs.
Usage: OMP_NUM_THREADS=1 /usr/bin/python3 er_c1_conv.py p nseed [c]   (seeds 1000*int(100c)+s, as in x1_er.py)"""
import sys, os, time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ext", "verify"))
import x1_er as X

def measure(p, c, seed, chunk_bytes=1024):
    rng = np.random.default_rng(seed)
    ei, ej = X.gen(p, c, rng); m = len(ei)
    A = csr_matrix((np.ones(m, dtype=np.int8), (ei, ej)), shape=(p, p))
    nc, lab = connected_components(A, directed=True, connection="strong")
    ci, cj = lab[ei], lab[ej]; cross = ci != cj; ES = p + m
    L = 0.5 + int(cross.sum()) / (2 * ES)
    xi, xj = ci[cross], cj[cross]
    # deduplicate condensation edges for propagation (duplicates do not change unions)
    key = np.unique(xi.astype(np.int64) * nc + xj); ui, uj = key // nc, key % nc
    indeg = np.bincount(uj, minlength=nc); src = np.where(indeg == 0)[0]; k = len(src)
    # longest-path level of each SCC (Kahn by layers)
    level = np.zeros(nc, dtype=np.int64); deg = indeg.copy(); frontier = src; lv = 0
    order = np.argsort(ui, kind="stable"); ui_s, uj_s = ui[order], uj[order]
    start = np.searchsorted(ui_s, np.arange(nc + 1))
    layers = []
    while len(frontier):
        layers.append(frontier); level[frontier] = lv
        # edges out of the frontier
        cnt = start[frontier + 1] - start[frontier]
        idx = np.repeat(start[frontier], cnt) + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
        tgt = uj_s[idx]
        np.subtract.at(deg, tgt, 1)
        nxt = np.unique(tgt[deg[tgt] == 0]); frontier = nxt; lv += 1
    assert sum(len(l) for l in layers) == nc
    # edges grouped by level of their tail
    elev = level[ui]; eord = np.argsort(elev, kind="stable"); ui2, uj2, elev2 = ui[eord], uj[eord], elev[eord]
    estart = np.searchsorted(elev2, np.arange(lv + 1))
    up = np.zeros(len(xi), dtype=bool)
    sid = -np.ones(nc, dtype=np.int64); sid[src] = np.arange(k)
    nbits = 8 * chunk_bytes
    for c0 in range(0, k, nbits):
        nb = chunk_bytes
        R = np.zeros((nc, nb), dtype=np.uint8)
        s_ = src[c0:c0 + nbits]; b = np.arange(len(s_))
        R[s_, b >> 3] = (1 << (b & 7)).astype(np.uint8)
        for l in range(lv):
            a, z = estart[l], estart[l + 1]
            if z > a: np.bitwise_or.at(R, uj2[a:z], R[ui2[a:z]])
        pc = np.unpackbits(R, axis=1).sum(axis=1, dtype=np.int64)
        up |= pc[xi] != pc[xj]
        del R
    S = 0.5 + int(up.sum()) / (2 * ES)
    return L, S, k / p, lv

if __name__ == "__main__":
    p = int(sys.argv[1]); nseed = int(sys.argv[2]); c = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    Lp, Sp = X.predict(c)[:2]
    Ls, Ss = [], []
    for s in range(nseed):
        t = time.time(); L, S, fs, lv = measure(p, c, 1000 * int(c * 100) + s)
        Ls.append(L); Ss.append(S)
        print(f"p={p} c={c} seed={1000*int(c*100)+s} L={L:.5f} S={S:.5f} src/p={fs:.3f} levels={lv} t={time.time()-t:.1f}s", flush=True)
    Ls, Ss = np.array(Ls), np.array(Ss)
    sd = lambda a: a.std(ddof=1) if len(a) > 1 else 0.0
    print(f"SUMMARY p={p} c={c} nseed={nseed} L={Ls.mean():.5f}+-{sd(Ls):.5f} (pred {Lp:.5f}) "
          f"S={Ss.mean():.5f}+-{sd(Ss):.5f} (pred {Sp:.5f}) se(S)={sd(Ss)/np.sqrt(len(Ss)):.5f}", flush=True)
