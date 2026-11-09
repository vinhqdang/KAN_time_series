"""Exploratory (not pre-registered): marginal correlation of block-mean series for pairs that are summary relatives
in the river graph versus pairs that are not.  Run: OMP_NUM_THREADS=1 /usr/bin/python3 explore_corr.py"""
import numpy as np, json, pickle, networkx as nx
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_p import S, RES, series   # subgraphs and loader of test_p.py (its main analysis runs only as a script)
from datapaths import CR_PRODUCT    # importable once test_p has put .. on sys.path; see ../DATA.md
G = {r: pickle.load(open(f"{CR_PRODUCT}/rivers_{r}.p", "rb")) for r in ["east_germany", "bavaria"]}
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
