"""Phase 2 step 2: test P1-P3 as fixed in THEORY.md Section 6.  Reads subgraphs_7000.json and the stage0 block-mean
caches (outside the repo).  Writes results_p.json and prints the verdict table.
Run: OMP_NUM_THREADS=1 /usr/bin/python3 test_p.py"""
import numpy as np, json
from scipy.stats import norm, spearmanr
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from datapaths import CR_CACHE as CACHE   # see ../DATA.md
RES = {"12h": 48, "24h": 96}
S = json.load(open(__file__.replace("test_p.py", "subgraphs_7000.json")))
arr = {}
def series(sg, rn):
    key = (sg["region"], rn)
    if key not in arr:
        arr[key] = (np.load(f"{CACHE}/{sg['region']}_agg_{rn}.npy", mmap_mode="r"),
                    {int(n): i for i, n in enumerate(np.load(f"{CACHE}/{sg['region']}_ids.npy"))})
    a, ix = arr[key]; per = 96 // RES[rn]
    X = np.asarray(a[sg["day0"] * per: sg["day1"] * per][:, [ix[n] for n in sg["nodes"]]], dtype=np.float64)
    assert not np.isnan(X).any()
    return (X - X.mean(0)) / X.std(0)

def acf(x, K):
    x = x - x.mean(); v = x @ x
    return np.array([x[:-k] @ x[k:] / v for k in range(1, K + 1)])

def rel_counts(X, alpha=0.01, neff=False, K=20):
    n, d = X.shape; R = np.corrcoef(X.T); m = d * (d - 1) / 2
    A = [acf(X[:, j], K) for j in range(d)] if neff else None
    sig = np.zeros((d, d), bool)
    for i in range(d):
        for j in range(i + 1, d):
            nn = n / max(1.0, 1 + 2 * float(A[i] @ A[j])) if neff else n
            z = np.arctanh(np.clip(R[i, j], -0.999999, 0.999999)) * np.sqrt(max(nn - 3, 1))
            sig[i, j] = sig[j, i] = 2 * norm.sf(abs(z)) < alpha / m
    return 1 + sig.sum(1), sig

def edge_acc(cnt, nodes, E):
    ix = {n: k for k, n in enumerate(nodes)}
    return [1.0 if cnt[ix[u]] < cnt[ix[v]] else (0.5 if cnt[ix[u]] == cnt[ix[v]] else 0.0) for (u, v) in E]

def boot_ci(per_sg_vals, B=2000, seed=0):
    rng = np.random.default_rng(seed); k = len(per_sg_vals); out = []
    for _ in range(B):
        idx = rng.integers(0, k, k); v = np.concatenate([per_sg_vals[i] for i in idx]); out.append(v.mean())
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))

if __name__ == "__main__":   # main analysis; the helpers above are imported by explore_corr.py
    res = {"P1": {}, "P2": {}, "P3": {}, "explore": {}}
    conf = [s for s in S if s["confluence"]]
    res["P1"] = {"n_conf": len(conf), "n_conf_pred_gt_half": sum(s["s_pred"] > 0.5 for s in conf),
                 "exceptions": [(s["region"], s["d"], s["seed"]) for s in conf if s["s_pred"] <= 0.5],
                 "n_noconf": len(S) - len(conf), "noconf_pred_values": sorted({s["s_pred"] for s in S if not s["confluence"]})}
    print("P1:", res["P1"])
    for rn in RES:
        for mode in ["prereg", "neff"]:
            per = []
            for s in S:
                X = series(s, rn); cnt, sig = rel_counts(X, neff=(mode == "neff"))
                v = edge_acc(cnt, s["nodes"], s["edges"])
                iu = np.triu_indices(len(s["nodes"]), 1)
                per.append({"region": s["region"], "d": s["d"], "seed": s["seed"], "conf": s["confluence"], "s_pred": s["s_pred"],
                            "vals": v, "acc": float(np.mean(v)), "frac_sig": float(sig[iu].mean()), "n": X.shape[0]})
            out = {}
            for grp, flag in [("conf", True), ("noconf", False)]:
                P = [p for p in per if p["conf"] == flag]
                allv = np.concatenate([p["vals"] for p in P])
                out[grp] = {"n_subgraphs": len(P), "n_edges": int(len(allv)), "acc": float(allv.mean()),
                            "ci": boot_ci([np.array(p["vals"]) for p in P]), "mean_frac_sig": float(np.mean([p["frac_sig"] for p in P]))}
            rho, pval = spearmanr([p["s_pred"] for p in per], [p["acc"] for p in per])
            out["spearman"] = {"rho": float(rho), "p": float(pval), "n": len(per)}
            # breakdown
            out["by_cell"] = {}
            for reg in ["east_germany", "bavaria"]:
                for d in [5, 10]:
                    P = [p for p in per if p["region"] == reg and p["d"] == d and p["conf"]]
                    if P:
                        allv = np.concatenate([p["vals"] for p in P])
                        out["by_cell"][f"{reg}_d{d}"] = {"acc": float(allv.mean()), "n_sg": len(P), "frac_sig": float(np.mean([p['frac_sig'] for p in P]))}
            out["per_subgraph"] = [{k: p[k] for k in ["region", "d", "seed", "conf", "s_pred", "acc", "frac_sig", "n"]} for p in per]
            if mode == "prereg":
                c, nc = out["conf"], out["noconf"]
                res["P2"][rn] = {"conf": c, "noconf": nc, "pass_conf": c["ci"][0] > 0.5, "pass_noconf": abs(nc["acc"] - 0.5) <= 0.05}
                res["P3"][rn] = {**out["spearman"], "pass": out["spearman"]["rho"] >= 0.3}
                res["explore"][rn + "_prereg_by_cell"] = out["by_cell"]; res["explore"][rn + "_prereg_per_subgraph"] = out["per_subgraph"]
            else:
                res["explore"][rn + "_neff"] = {k: out[k] for k in ["conf", "noconf", "spearman", "by_cell"]}
            print(rn, mode, "conf", out["conf"], "noconf", out["noconf"], "spearman", out["spearman"])
            print("   by cell", out["by_cell"])
    res["kill_rule_triggered"] = not any(res["P2"][rn]["pass_conf"] for rn in RES)
    print("P2 pass_conf", {rn: res["P2"][rn]["pass_conf"] for rn in RES}, "pass_noconf", {rn: res["P2"][rn]["pass_noconf"] for rn in RES})
    print("P3 pass", {rn: res["P3"][rn]["pass"] for rn in RES}, "kill rule triggered:", res["kill_rule_triggered"])
    json.dump(res, open(__file__.replace("test_p.py", "results_p.json"), "w"), indent=1)
