import numpy as np, json, pandas as pd
from common import *

_cache = {}
def agg_arr(reg, rn):
    if (reg, rn) not in _cache:
        _cache[(reg, rn)] = (np.load(f"{CACHE}/{reg}_agg_{rn}.npy"), {int(n): i for i, n in enumerate(np.load(f"{CACHE}/{reg}_ids.npy"))})
    return _cache[(reg, rn)]

def get_series(sg, rn, standardize=True):
    a, ix = agg_arr(sg["region"], rn)
    per = 96 // RES[rn]
    X = a[sg["day0"] * per: sg["day1"] * per][:, [ix[n] for n in sg["nodes"]]].astype(np.float64)
    assert not np.isnan(X).any()
    if standardize:
        X = (X - X.mean(0)) / X.std(0)
    return X

def load_sub():
    return json.load(open("subgraphs.json"))["subgraphs"]

def edge_table():
    return pd.read_csv("edges.csv")

def label_edges(e, rn, factor=0.5):
    """contemporaneous iff tau_e < factor * Delta (Delta in 15-min steps); factor 0.5 is the pre-registered rule"""
    return e["tau_e"].values < factor * RES[rn]

# ---- scale-only controls -------------------------------------------------
def var_order_score(X):
    return X.var(0)

def r2_order_score(X):
    """R^2 of each variable regressed (OLS) on all other variables at the same time step"""
    d = X.shape[1]; Xc = X - X.mean(0); r2 = np.zeros(d)
    for j in range(d):
        o = [i for i in range(d) if i != j]
        b, *_ = np.linalg.lstsq(Xc[:, o], Xc[:, j], rcond=None)
        res = Xc[:, j] - Xc[:, o] @ b
        r2[j] = 1 - res.var() / Xc[:, j].var()
    return r2

def pair_orient(score, i_up, i_dn, tol=1e-9):
    """score ordering: causal order = increasing score. returns 1 correct, 0.5 tie, 0 wrong for true edge up->dn"""
    a, b = score[i_up], score[i_dn]
    if abs(a - b) <= tol * max(1.0, abs(a), abs(b)): return 0.5
    return 1.0 if a < b else 0.0
