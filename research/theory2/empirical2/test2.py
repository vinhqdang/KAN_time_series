"""Phase E step 2: test P4-P6 exactly as fixed in finite/FINITE.md Section 7 (pre-registered in PLAN2.md).
Reads subgraphs_8000.json, the stage0 block-mean caches and the regional river graphs (outside the repo).
Writes results2.json (derived statistics only) and prints the verdicts.
Run: OMP_NUM_THREADS=1 /usr/bin/python3 test2.py > test2.log"""
import numpy as np, json, pickle, networkx as nx
from scipy.stats import norm, spearmanr
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from datapaths import CR_CACHE as CACHE, CR_PRODUCT as PRODUCT   # see ../DATA.md
RES = {"12h": 48, "24h": 96}; KLAG = {"12h": 100, "24h": 50}; ALPHA = 0.01; MINVALID = 0.9
HERE = __file__.rsplit("/", 1)[0] if "/" in __file__ else "."
SG = json.load(open(f"{HERE}/subgraphs_8000.json"))

graphs, anc_cache, arrays = {}, {}, {}
def graph(reg):
    if reg not in graphs:
        G = pickle.load(open(f"{PRODUCT}/rivers_{reg}.p", "rb")); graphs[reg] = (G, G.to_undirected())
    return graphs[reg]
def anc(reg, v):
    if (reg, v) not in anc_cache: anc_cache[(reg, v)] = nx.ancestors(graph(reg)[0], v) | {v}
    return anc_cache[(reg, v)]
def arr(reg, rn):
    if (reg, rn) not in arrays:
        arrays[(reg, rn)] = (np.load(f"{CACHE}/{reg}_agg_{rn}.npy", mmap_mode="r"), np.load(f"{CACHE}/{reg}_ids.npy"))
    return arrays[(reg, rn)]

def acf(x, K):
    x = x - x.mean(); v = x @ x; return np.array([x[:-k] @ x[k:] / v for k in range(1, K + 1)])

def fisher_dec(c, nprime, d, k):
    m = d * (d - 1) / 2; z = np.arctanh(np.clip(np.abs(c), 0, 0.999999)) * np.sqrt(np.maximum(nprime - 3 - k, 1))
    return 2 * norm.sf(z) < ALPHA / m

def edge_acc(cnt, nodes, E):
    ix = {n: k for k, n in enumerate(nodes)}
    return [1.0 if cnt[ix[u]] < cnt[ix[v]] else (0.5 if cnt[ix[u]] == cnt[ix[v]] else 0.0) for (u, v) in E]

def boot(per, stat, B=2000, seed=0):
    rng = np.random.default_rng(seed); k = len(per); out = []
    for _ in range(B):
        idx = rng.integers(0, k, k); out.append(stat([per[i] for i in idx]))
    return float(np.nanquantile(out, 0.025)), float(np.nanquantile(out, 0.975))

def analyse(sg, rn):
    reg = sg["region"]; a, ids = arr(reg, rn); ix = {int(n): i for i, n in enumerate(ids)}; per = 96 // RES[rn]
    rows = slice(sg["day0"] * per, sg["day1"] * per); S = sg["nodes"]; d = len(S)
    X = np.asarray(a[rows][:, [ix[n] for n in S]], dtype=np.float64); assert not np.isnan(X).any()
    G, Gu = graph(reg)
    excl = set(S)
    for s in S:
        if s in Gu: excl |= set(nx.single_source_shortest_path_length(Gu, s, cutoff=2).keys())
    cand = [int(n) for n in ids if int(n) not in excl]
    W = np.asarray(a[rows][:, [ix[n] for n in cand]], dtype=np.float64)
    ok = np.isfinite(W).mean(0) >= MINVALID; W = W[:, ok]; cand = [c for c, o in zip(cand, ok) if o]
    info = {"n_cand": len(cand)}
    if len(cand) < 3: return None, info
    W = (W - np.nanmean(W, 0)) / np.nanstd(W, 0); Z = np.nanmean(W, 1); keep = np.isfinite(Z)
    info["dropped_t"] = int((~keep).sum()); X = X[keep]; Z = Z[keep]; n = len(Z)
    info["contam_frac"] = float(np.mean([any(anc(reg, c) & anc(reg, s) for s in S) for c in cand if c in G]) if cand else np.nan)
    X = (X - X.mean(0)) / X.std(0); Z = (Z - Z.mean()) / Z.std()
    R = np.corrcoef(np.column_stack([X, Z]).T); rz = R[:d, d]; Rxx = R[:d, :d]
    P = (Rxx - np.outer(rz, rz)) / np.sqrt(np.outer(1 - rz ** 2, 1 - rz ** 2))
    E = X - np.outer(Z, rz)                                   # OLS residuals on Z (standardised, so slope = r_uZ)
    K = KLAG[rn]; Araw = [acf(X[:, j], K) for j in range(d)]; Ares = [acf(E[:, j], K) for j in range(d)]
    iu = np.triu_indices(d, 1)
    Braw = np.array([max(1.0, 1 + 2 * Araw[i] @ Araw[j]) for i, j in zip(*iu)])
    Bres = np.array([max(1.0, 1 + 2 * Ares[i] @ Ares[j]) for i, j in zip(*iu)])
    nonrel = np.array([not (anc(reg, S[i]) & anc(reg, S[j])) for i, j in zip(*iu)])
    dec = {"marg_nominal": fisher_dec(Rxx[iu], n, d, 0), "marg_neff": fisher_dec(Rxx[iu], n / Braw, d, 0),
           "cond_nominal": fisher_dec(P[iu], n, d, 1), "cond_neff": fisher_dec(P[iu], n / Bres, d, 1)}
    acc = {}
    for k, v in dec.items():
        M = np.zeros((d, d), bool); M[iu] = v; M = M | M.T; acc[k] = edge_acc(1 + M.sum(1), S, sg["edges"])
    # F-theory: triad reliability, loadings, predicted partial correlations and detection probabilities
    tri = [rz[i] * rz[j] / Rxx[i, j] for i, j, m in zip(*iu, nonrel) if m and abs(Rxx[i, j]) >= 0.1]
    r_hat = float(np.clip(np.median(tri), 0, 1)) if tri else np.nan
    pred = {}
    if np.isfinite(r_hat) and r_hat > 0:
        ell = np.minimum(rz ** 2 / r_hat, 0.999)
        lu, lv = ell[iu[0]], ell[iu[1]]
        pi_pred = (1 - r_hat) * np.sqrt(lu * lv) / np.sqrt((1 - r_hat * lu) * (1 - r_hat * lv))
        zc = norm.isf(ALPHA / (2 * d * (d - 1) / 2))
        for mode, nn in [("nominal", np.full(len(Bres), n)), ("neff", n / Bres)]:
            tz = zc / np.sqrt(np.maximum(nn - 4, 1)); sd = np.sqrt(Bres / n); mu = np.arctanh(pi_pred)
            pred[mode] = (norm.sf((tz - mu) / sd) + norm.cdf((-tz - mu) / sd))
        ncrit = 4 + (zc / np.arctanh(np.maximum(pi_pred, 1e-12))) ** 2
        taue = np.tanh(zc / np.sqrt(np.maximum(n / Bres - 4, 1)))
        r_crit = 1 - taue * (1 - np.sqrt(lu * lv)) / (np.sqrt(lu * lv) * (1 - taue))
    else:
        pi_pred = ncrit = r_crit = np.full(len(Bres), np.nan); pred = {"nominal": pi_pred, "neff": pi_pred}
    out = {"n": n, "nonrel": nonrel.tolist(), "dec": {k: v.tolist() for k, v in dec.items()}, "acc": acc,
           "r_hat": r_hat, "Braw_med": float(np.median(Braw)), "Bres_med": float(np.median(Bres)),
           "neff_raw_med": float(np.median(n / Braw)), "neff_res_med": float(np.median(n / Bres)),
           "pi_obs_nonrel": np.abs(P[iu][nonrel]).tolist(), "pi_pred_nonrel": np.asarray(pi_pred)[nonrel].tolist(),
           "kappa_obs_nonrel": np.abs(Rxx[iu][nonrel]).tolist(), "rz": rz.tolist(),
           "pred_nominal_nonrel": np.asarray(pred["nominal"])[nonrel].tolist(), "pred_neff_nonrel": np.asarray(pred["neff"])[nonrel].tolist(),
           "ncrit_nonrel": np.asarray(ncrit)[nonrel].tolist(), "rcrit_nonrel": np.asarray(r_crit)[nonrel].tolist()}
    return out, info

res = {"per_subgraph": {}, "excluded": {}, "P4": {}, "P5": {}, "P6": {}, "theory": {}}
for rn in RES:
    per = []
    for sg in SG:
        o, info = analyse(sg, rn)
        key = f"{sg['region']}_d{sg['d']}_s{sg['seed']}"
        if o is None: res["excluded"].setdefault(rn, []).append({"key": key, **info}); continue
        o.update(info); o.update({"key": key, "region": sg["region"], "d": sg["d"], "conf": sg["confluence"], "s_pred": sg["s_pred"], "n_edges": len(sg["edges"])})
        per.append(o)
    res["per_subgraph"][rn] = per
    def frac(P_, k):
        num = sum(int(np.sum(np.array(p["dec"][k])[np.array(p["nonrel"])])) for p in P_); den = sum(int(np.sum(p["nonrel"])) for p in P_)
        return num / den if den else np.nan
    P_nr = [p for p in per if any(p["nonrel"])]
    res["P4"][rn] = {"n_subgraphs_with_nonrel": len(P_nr), "n_nonrel_pairs": int(sum(np.sum(p["nonrel"]) for p in P_nr))}
    for k in ["marg_nominal", "marg_neff", "cond_nominal", "cond_neff"]:
        res["P4"][rn][k] = {"frac": frac(P_nr, k), "ci": boot(P_nr, lambda q, k=k: frac(q, k))}
    res["P4"][rn]["pass_primary"] = res["P4"][rn]["cond_neff"]["frac"] < 0.5
    res["P4"][rn]["pass_secondary"] = res["P4"][rn]["cond_nominal"]["frac"] < 0.5
    conf = [p for p in per if p["conf"]]
    res["P5"][rn] = {"n_subgraphs": len(conf), "n_edges": int(sum(p["n_edges"] for p in conf))}
    for k in ["marg_nominal", "marg_neff", "cond_nominal", "cond_neff"]:
        st = lambda q, k=k: float(np.mean(np.concatenate([q_["acc"][k] for q_ in q])))
        res["P5"][rn][k] = {"acc": st(conf), "ci": boot(conf, st)}
    for k in ["cond_neff", "cond_nominal"]:
        a5 = res["P5"][rn][k]; a5["pass"] = a5["acc"] > 0.5 and a5["ci"][0] > 0.5
    withE = [p for p in per if p["n_edges"] >= 1]
    for k in ["cond_neff", "cond_nominal", "marg_nominal"]:
        rho, pv = spearmanr([p["s_pred"] for p in withE], [np.mean(p["acc"][k]) for p in withE])
        res["P6"].setdefault(rn, {})[k] = {"rho": float(rho), "p": float(pv), "n": len(withE), "pass": bool(rho >= 0.3)}
    # theory checks
    cat = lambda f: np.concatenate([np.asarray(p[f], float) for p in P_nr])
    rh = np.array([p["r_hat"] for p in per], float)
    res["theory"][rn] = {
        "n_median": float(np.median([p["n"] for p in per])), "n_range": [int(min(p["n"] for p in per)), int(max(p["n"] for p in per))],
        "neff_raw_median": float(np.median([p["neff_raw_med"] for p in per])), "neff_res_median": float(np.median([p["neff_res_med"] for p in per])),
        "Braw_median": float(np.median([p["Braw_med"] for p in per])), "Bres_median": float(np.median([p["Bres_med"] for p in per])),
        "r_hat_median": float(np.nanmedian(rh)), "r_hat_q": np.nanquantile(rh, [0.1, 0.25, 0.75, 0.9]).tolist(),
        "kappa_obs_nonrel_median": float(np.median(cat("kappa_obs_nonrel"))),
        "pi_obs_nonrel_median": float(np.median(cat("pi_obs_nonrel"))), "pi_pred_nonrel_median": float(np.nanmedian(cat("pi_pred_nonrel"))),
        "pred_frac_cond_nominal": float(np.nanmean(cat("pred_nominal_nonrel"))), "pred_frac_cond_neff": float(np.nanmean(cat("pred_neff_nonrel"))),
        "ncrit_nonrel_median": float(np.nanmedian(cat("ncrit_nonrel"))), "rcrit_nonrel_median": float(np.nanmedian(cat("rcrit_nonrel"))),
        "frac_pairs_n_above_ncrit": float(np.nanmean(np.concatenate([np.asarray(p["ncrit_nonrel"]) < p["n"] for p in P_nr]))),
        "frac_pairs_rhat_above_rcrit": float(np.nanmean(np.concatenate([p["r_hat"] > np.asarray(p["rcrit_nonrel"]) for p in P_nr]))),
        "contam_frac_median": float(np.nanmedian([p["contam_frac"] for p in per])), "n_cand_median": float(np.median([p["n_cand"] for p in per])),
    }
    print(f"\n==== {rn}: {len(per)} subgraphs analysed, excluded {len(res['excluded'].get(rn, []))}")
    print(" P4", json.dumps(res["P4"][rn]))
    print(" P5", json.dumps(res["P5"][rn]))
    print(" P6", json.dumps(res["P6"][rn]))
    print(" theory", json.dumps(res["theory"][rn]))
res["kill_rule_triggered"] = not any(res["P4"][rn]["pass_primary"] for rn in RES)
res["P4_holds"] = all(res["P4"][rn]["pass_primary"] for rn in RES)
res["P5_holds"] = any(res["P5"][rn]["cond_neff"]["pass"] for rn in RES)
res["P6_holds"] = any(res["P6"][rn]["cond_neff"]["pass"] for rn in RES)
print("\nP4 holds (primary, both resolutions):", res["P4_holds"], "| P5 holds:", res["P5_holds"], "| P6 holds:", res["P6_holds"], "| kill rule triggered:", res["kill_rule_triggered"])
json.dump(res, open(f"{HERE}/results2.json", "w"), indent=1)
