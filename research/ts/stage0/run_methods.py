"""usage: run_methods.py val|test   (val: seeds 5000-5004; test: seeds 5005-5029)"""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[k] = "1"
import sys, time, json, numpy as np, pandas as pd
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score, average_precision_score
from lib import *; from methods import *

phase = sys.argv[1]
VAL, TEST = range(5000, 5005), range(5005, 5030)
VALP = ("val", "val2")
SEL = json.load(open("selected_hparams.json")) if phase == "test" else None

def configs(res, phase):
    c = []
    if phase == "val2":     # grid extension: DYNOTEARS / NOTEARS-SVAR optimum sat at the edge of the first grid
        c += [("dynotears", dict(p=1, lam=l)) for l in (0.3, 1.0)]
        c += [("notears_svar2", dict(p=1, lam=l)) for l in (0.3, 1.0)]
    elif phase == "val":
        c += [("var", dict(p=1)), ("corr", {})]
        c += [("pcmci+", dict(p=1, alpha=a)) for a in (0.01, 0.05)]
        c += [("varlingam", dict(p=1, prune=b)) for b in (True, False)]
        c += [("dynotears", dict(p=1, lam=l)) for l in (0.001, 0.01, 0.1)]
        c += [("notears_svar2", dict(p=1, lam=l)) for l in (0.01, 0.1)]
        c += [("var_sortnregress", {}), ("r2_sortnregress", {}), ("random", {})]
    else:
        sel = SEL[res]
        c += [("var", dict(p=1)), ("var", dict(p=2)), ("corr", {})]
        for m in ("pcmci+", "varlingam", "dynotears", "notears_svar2"):
            for p in (1, 2): c.append((m, dict(sel[m], p=p)))
        c += [("var_sortnregress", {}), ("r2_sortnregress", {}), ("random", {})]
    return c

def score(name, cfg, X, rng):
    if name == "var": return score_var(X, cfg["p"])
    if name == "corr": return score_corr(X)
    if name == "pcmci+": return score_pcmciplus(X, cfg["p"], cfg["alpha"])
    if name == "varlingam": return score_varlingam(X, cfg["p"], cfg["prune"])
    if name == "dynotears": return score_dynotears(X, cfg["p"], cfg["lam"])
    if name == "notears_svar2": return score_notears_svar2(X, cfg["p"], cfg["lam"])
    if name == "var_sortnregress": return score_var_sortnregress(X, rng)
    if name == "r2_sortnregress": return score_r2_sortnregress(X, rng)
    if name == "random": return score_random(X, rng)

def task(a):
    sg, rn, std = a
    nodes = sg["nodes"]; d = len(nodes); idx = {n: i for i, n in enumerate(nodes)}
    X = get_series(sg, rn, standardize=std)
    Y = np.zeros((d, d), int)
    for u, v in sg["edges"]: Y[idx[u], idx[v]] = 1
    off = ~np.eye(d, dtype=bool)
    rows, erows = [], []
    for name, cfg in configs(rn, phase):
        if not std and name == "random": continue
        rng = np.random.default_rng(sg["seed"] * 31 + list(RES).index(rn))
        t0 = time.time()
        try:
            S = score(name, cfg, X, rng); err = ""
        except Exception as ex:
            S = np.zeros((d, d)); err = repr(ex)[:80]
        S = np.nan_to_num(S)
        key = dict(phase=phase, region=sg["region"], d=sg["d"], seed=sg["seed"], res=rn, scale="std" if std else "raw",
                   method=name, cfg=json.dumps(cfg, sort_keys=True))
        rows.append(dict(key, auroc=roc_auc_score(Y[off], S[off]), auprc=average_precision_score(Y[off], S[off]),
                         secs=time.time() - t0, err=err))
        for u, v in sg["edges"]:
            a_, b_ = S[idx[u], idx[v]], S[idx[v], idx[u]]
            o = 0.5 if a_ == b_ else float(a_ > b_)
            erows.append(dict(key, u=u, v=v, s_uv=a_, s_vu=b_, orient=o))
    return rows, erows

if __name__ == "__main__":
    seeds = VAL if phase in VALP else TEST
    sub = [s for s in load_sub() if s["seed"] in seeds]
    jobs = [(s, rn, True) for s in sub for rn in RES]
    if phase == "test": jobs += [(s, rn, False) for s in sub for rn in RES]
    R, E = [], []
    t0 = time.time()
    with Pool(2) as pool:
        for i, (r, e) in enumerate(pool.imap_unordered(task, jobs, chunksize=1)):
            R += r; E += e
            if i % 10 == 0: print(i, len(jobs), round(time.time() - t0), flush=True)
    pd.DataFrame(R).to_csv(f"results_{phase}.csv", index=False)
    pd.DataFrame(E).to_csv(f"edgeres_{phase}.csv", index=False)
    print("done", time.time() - t0)
