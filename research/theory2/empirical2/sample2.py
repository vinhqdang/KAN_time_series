"""Phase E step 1 (copy of empirical/sample.py, only the seeds and file names changed): sample CausalRivers subgraphs with the stage0 sampler (copied from research/ts/stage0/prep.py and
common.py, unchanged logic), new seeds 8000..8029 (Phase E, PLAN2.md) per region and size; compute the stationary T1 prediction from the
full regional river graph.  Output: subgraphs_8000.json (node ids, windows, induced edges, predictions; no raw data).
Run: OMP_NUM_THREADS=1 /usr/bin/python3 sample2.py"""
import numpy as np, json, pickle, networkx as nx
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from datapaths import CR_CACHE as CACHE, CR_PRODUCT as PRODUCT   # see ../DATA.md
REGIONS = ["east_germany", "bavaria"]; RES = {"6h": 24, "12h": 48, "24h": 96}; MIN_FRAC = 0.5
SEEDS = list(range(8000, 8030)); SIZES = [5, 10]; MIN_WINDOW_DAYS = 365

def load_x(reg): return np.load(f"{CACHE}/{reg}_x.npy", mmap_mode="r"), np.load(f"{CACHE}/{reg}_ids.npy")
def aggregate(x, k):
    T, N = x.shape; r = np.asarray(x[: T // k * k]).reshape(T // k, k, N)
    cnt = np.isfinite(r).sum(1); s = np.nansum(r, 1)
    return np.where(cnt >= MIN_FRAC * k, s / np.maximum(cnt, 1), np.nan).astype(np.float64)
def load_graph(reg): return pickle.load(open(f"{PRODUCT}/rivers_{reg}.p", "rb"))   # networkx-only pickle (checked in stage0)
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
        a = np.load(f"{CACHE}/{reg}_agg_{rn}.npy"); per = 96 // k
        v = ~np.isnan(a).reshape(-1, per, a.shape[1]).any(1); ok = v if ok is None else ok & v
    return ok
def sample_subgraph(rng, Gu, pool, d, dv, ix):
    pool = set(pool); nodes_l = sorted(pool); rej = 0
    while True:
        s = nodes_l[rng.integers(len(nodes_l))]; S = [s]; front = set(n for n in Gu[s] if n in pool)
        while len(S) < d and front:
            fl = sorted(front); n = fl[rng.integers(len(fl))]; S.append(n); front.discard(n)
            front |= set(m for m in Gu[n] if m in pool and m not in S)
        if len(S) < d: rej += 1; continue
        m = dv[:, [ix[n] for n in S]].all(1); a, b = run_len_best(m)
        if b - a >= MIN_WINDOW_DAYS: return sorted(S), (a, b), rej
        rej += 1

def prediction(G, S, E):
    anc = {v: nx.ancestors(G, v) | {v} for v in S}
    rho = {v: sum(1 for w in S if anc[v] & anc[w]) for v in S}
    vals = [1.0 if rho[u] < rho[v] else (0.5 if rho[u] == rho[v] else 0.0) for (u, v) in E]
    parents_in_S = {v: sum(1 for (a, b) in E if b == v) for v in S}
    return rho, (float(np.mean(vals)) if vals else None), vals, max(parents_in_S.values()) >= 2

def main():
    out = []
    for reg in REGIONS:
        G = load_graph(reg); x, ids = load_x(reg); ix = {int(n): i for i, n in enumerate(ids)}
        outdeg = np.bincount([G.out_degree(n) for n in G.nodes])
        print(reg, "nodes", G.number_of_nodes(), "edges", G.number_of_edges(), "is DAG", nx.is_directed_acyclic_graph(G), "out-degree histogram", outdeg.tolist(), flush=True)
        dv = day_valid(reg); miss20 = np.ones(len(ids), bool)
        for rn, k in RES.items():
            a = aggregate(x, k); miss20 &= np.isnan(a).mean(0) <= 0.20
        pool = [int(n) for n in ids[miss20]]; Gu = G.to_undirected()
        for d in SIZES:
            used = set()
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                while True:
                    S, (a, b), rej = sample_subgraph(rng, Gu, pool, d, dv, ix)
                    if tuple(S) not in used: break
                used.add(tuple(S))
                E = [(int(u), int(v)) for u, v in G.subgraph(S).edges]
                rho, pred, vals, conf = prediction(G, S, E)
                out.append({"region": reg, "d": d, "seed": seed, "nodes": [int(n) for n in S], "day0": int(a), "day1": int(b),
                            "edges": E, "rho_pred": {str(k): v for k, v in rho.items()}, "edge_pred": vals,
                            "s_pred": pred, "confluence": bool(conf), "rejections": int(rej)})
                print(reg, d, seed, len(E), "edges", "conf" if conf else "noconf", pred, b - a, flush=True)
    json.dump(out, open(__file__.replace("sample2.py", "subgraphs_8000.json"), "w"))

if __name__ == "__main__":
    main()
