"""Exploratory (not pre-registered): marginal correlation of block-mean series for pairs that are summary relatives
in the river graph versus pairs that are not.  Run: OMP_NUM_THREADS=1 /usr/bin/python3 explore_corr.py"""
import numpy as np, json, pickle, networkx as nx
_tp = __file__.replace("explore_corr.py", "test_p.py"); exec(open(_tp).read().split("res = {")[0].replace("__file__", repr(_tp)))
G = {r: pickle.load(open(f"/tmp/claude-0/dl/pz/product/rivers_{r}.p", "rb")) for r in ["east_germany", "bavaria"]}
out = {}
for rn in RES:
    rel, non = [], []
    for s in S:
        anc = {v: nx.ancestors(G[s["region"]], v) | {v} for v in s["nodes"]}
        R = np.corrcoef(series(s, rn).T); nd = s["nodes"]
        for a in range(len(nd)):
            for b in range(a + 1, len(nd)):
                (rel if anc[nd[a]] & anc[nd[b]] else non).append(R[a, b])
    out[rn] = {"relatives": [len(rel), float(np.mean(rel)), float(np.quantile(rel, 0.1))],
               "non_relatives": [len(non), float(np.mean(non)), float(np.quantile(non, 0.1))]}
    print(rn, "pairs (n, mean r, 10% quantile r): relatives", out[rn]["relatives"], "non-relatives", out[rn]["non_relatives"])
json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1)
