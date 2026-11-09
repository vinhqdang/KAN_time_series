"""Phase E, exploratory only (not pre-registered; run after test2.py results were seen).  Tests the Prop 3.6
explanation (driver of rank > 1): condition on the first k principal components of the same leave-out candidate
stations instead of their mean, k = 1, 3, 5, 10; same primary test (Fisher z with residual Bartlett n_eff, k
conditioning variables).  Writes explore2.json.  Run: OMP_NUM_THREADS=1 /usr/bin/python3 explore2.py > explore2.log"""
import numpy as np, json, pickle, networkx as nx
from scipy.stats import norm, spearmanr
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from datapaths import CR_CACHE as CACHE, CR_PRODUCT as PRODUCT   # see ../DATA.md
RES = {"12h": 48, "24h": 96}; KLAG = {"12h": 100, "24h": 50}; ALPHA = 0.01
HERE = __file__.rsplit("/", 1)[0] if "/" in __file__ else "."
SG = json.load(open(f"{HERE}/subgraphs_8000.json")); KS = [1, 3, 5, 10]
G_, anc_c = {}, {}
def graph(reg):
    if reg not in G_: G = pickle.load(open(f"{PRODUCT}/rivers_{reg}.p", "rb")); G_[reg] = (G, G.to_undirected())
    return G_[reg]
def anc(reg, v):
    if (reg, v) not in anc_c: anc_c[(reg, v)] = nx.ancestors(graph(reg)[0], v) | {v}
    return anc_c[(reg, v)]
def acf(x, K):
    x = x - x.mean(); v = x @ x; return np.array([x[:-k] @ x[k:] / v for k in range(1, K + 1)])
out = {}
for rn in RES:
    per = {k: [] for k in KS}
    for sg in SG:
        reg = sg["region"]; a = np.load(f"{CACHE}/{reg}_agg_{rn}.npy", mmap_mode="r"); ids = np.load(f"{CACHE}/{reg}_ids.npy")
        ix = {int(n): i for i, n in enumerate(ids)}; p = 96 // RES[rn]; rows = slice(sg["day0"] * p, sg["day1"] * p); S = sg["nodes"]; d = len(S)
        X = np.asarray(a[rows][:, [ix[n] for n in S]], float); G, Gu = graph(reg); excl = set(S)
        for s in S:
            if s in Gu: excl |= set(nx.single_source_shortest_path_length(Gu, s, cutoff=2))
        cand = [int(n) for n in ids if int(n) not in excl]
        W = np.asarray(a[rows][:, [ix[n] for n in cand]], float); W = W[:, (np.isfinite(W).mean(0) >= 0.9) & (np.nanstd(W, 0) > 0)]
        W = (W - np.nanmean(W, 0)) / np.nanstd(W, 0); W = np.nan_to_num(W)
        U, sv, _ = np.linalg.svd(W - W.mean(0), full_matrices=False)
        X = (X - X.mean(0)) / X.std(0); n = len(X); iu = np.triu_indices(d, 1)
        nonrel = np.array([not (anc(reg, S[i]) & anc(reg, S[j])) for i, j in zip(*iu)])
        for k in KS:
            Zk = np.column_stack([np.ones(n), U[:, :k]]); E = X - Zk @ np.linalg.lstsq(Zk, X, rcond=None)[0]
            Rr = np.corrcoef(E.T); A = [acf(E[:, j], KLAG[rn]) for j in range(d)]
            Bres = np.array([max(1.0, 1 + 2 * A[i] @ A[j]) for i, j in zip(*iu)]); m = d * (d - 1) / 2
            z = np.arctanh(np.clip(np.abs(Rr[iu]), 0, 0.999999)) * np.sqrt(np.maximum(n / Bres - 3 - k, 1)); dec = 2 * norm.sf(z) < ALPHA / m
            M = np.zeros((d, d), bool); M[iu] = dec; M |= M.T; cnt = 1 + M.sum(1); pos = {s: q for q, s in enumerate(S)}
            acc = [1.0 if cnt[pos[u]] < cnt[pos[v]] else (0.5 if cnt[pos[u]] == cnt[pos[v]] else 0.0) for u, v in sg["edges"]]
            per[k].append({"nonrel_dec": dec[nonrel].tolist(), "acc": acc, "conf": sg["confluence"], "s_pred": sg["s_pred"],
                           "var_share_k": float((sv[:k] ** 2).sum() / (sv ** 2).sum())})
    out[rn] = {}
    for k in KS:
        P = per[k]; nd = np.concatenate([q["nonrel_dec"] for q in P]); conf = [q for q in P if q["conf"]]
        accs = np.concatenate([q["acc"] for q in conf]); rng = np.random.default_rng(0); bs = []
        for _ in range(2000):
            idx = rng.integers(0, len(conf), len(conf)); bs.append(np.concatenate([conf[i]["acc"] for i in idx]).mean())
        rho = spearmanr([q["s_pred"] for q in P if q["acc"]], [np.mean(q["acc"]) for q in P if q["acc"]])[0]
        out[rn][k] = {"frac_nonrel_declared": float(nd.mean()), "conf_acc": float(accs.mean()), "conf_acc_ci": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                      "spearman": float(rho), "median_var_share_topk": float(np.median([q["var_share_k"] for q in P]))}
        print(rn, "k =", k, out[rn][k], flush=True)
json.dump(out, open(f"{HERE}/explore2.json", "w"), indent=1)
