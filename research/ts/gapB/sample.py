"""Sampling of CausalRivers subgraphs, seeds 6000..6029 (copied from stage0/prep.py, tau_e part removed).
Writes subgraphs.json (node ids only, no raw data)."""
import numpy as np, json
from common import *
SEEDS = list(range(6000, 6030)); SIZES = [5, 10]
MIN_WINDOW_DAYS = 365

def run_len_best(mask):
    best = (0, 0); s = None
    for i, v in enumerate(np.append(mask, False)):
        if v and s is None: s = i
        if not v and s is not None:
            if i - s > best[1] - best[0]: best = (s, i)
            s = None
    return best

def day_valid(reg):
    ok = None
    for rn, k in RES.items():
        a = np.load(f"{CACHE}/{reg}_agg_{rn}.npy")
        per = 96 // k
        v = ~np.isnan(a).reshape(-1, per, a.shape[1]).any(1)
        ok = v if ok is None else ok & v
    return ok

def sample_subgraph(rng, Gu, pool, d, dv, ix):
    pool = set(pool); nodes_l = sorted(pool); rej = 0
    while True:
        s = nodes_l[rng.integers(len(nodes_l))]
        S = [s]; front = set(n for n in Gu[s] if n in pool)
        while len(S) < d and front:
            fl = sorted(front); n = fl[rng.integers(len(fl))]
            S.append(n); front.discard(n)
            front |= set(m for m in Gu[n] if m in pool and m not in S)
        if len(S) < d: rej += 1; continue
        m = dv[:, [ix[n] for n in S]].all(1)
        a, b = run_len_best(m)
        if b - a >= MIN_WINDOW_DAYS: return sorted(S), (a, b), rej
        rej += 1

if __name__ == "__main__":
    out = []
    for reg in REGIONS:
        G = load_graph(reg); x, ids = load_x(reg); ix = {n: i for i, n in enumerate(ids)}
        dv = day_valid(reg)
        miss20 = np.ones(len(ids), bool)
        for rn, k in RES.items():
            a = aggregate(x, k); miss20 &= np.isnan(a).mean(0) <= 0.20
        pool = [int(n) for n in ids[miss20]]
        Gu = G.to_undirected()
        for d in SIZES:
            used = set()
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                while True:
                    S, (a, b), rej = sample_subgraph(rng, Gu, pool, d, dv, ix)
                    if tuple(S) not in used: break
                used.add(tuple(S))
                E = [(int(u), int(v)) for u, v in G.subgraph(S).edges]
                out.append({"region": reg, "d": d, "seed": seed, "nodes": [int(n) for n in S], "day0": int(a), "day1": int(b), "edges": E})
                print(reg, d, seed, S, b - a, len(E), flush=True)
    json.dump({"subgraphs": out}, open("subgraphs.json", "w"))
