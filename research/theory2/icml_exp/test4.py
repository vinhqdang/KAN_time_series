"""PLAN4.md analyses: E1 (discriminating test of Thm 1.9 against competitor predictors) and E2 (lagged relatives
estimator), plus F-theory out-of-sample checks.  The estimator is the PLAN3 effective-n Fisher-z test, extended
to lags |h| <= L with Bonferroni over pairs and lags (L = 0 is exactly the PLAN3 test).

Usage:  OMP_NUM_THREADS=1 python3 -I test4.py <this_dir> <mode> <out.json>
  mode = dev   : development run on the PLAN3 data (WT actuators_random_walk_2..16, LT actuators_white) only
  mode = main  : the pre-registered PLAN4 analyses on held-out data
Data locations (raw data are not in the repository): see DATA below."""
import sys, os, json, itertools, time
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

HERE, MODE, OUT = sys.argv[1:4]
sys.path.insert(0, HERE)
import graphs4 as G

DATA = {"crl": "/tmp/claude-0/dl4/crl", "cam": "/tmp/claude-0/dl4/cam",
        "wt": "/tmp/claude-0/dl3/cc/wt_walks_v1/wt_walks_v1", "lt": "/tmp/claude-0/dl3/cc/lt_walks_v1/lt_walks_v1"}
ALPHA = 0.01; NBOOT = 2000; FAMILY_ALPHA = 0.05

# ----------------------------------------------------------------------------------------------- preprocessing
def std(X):
    return (X - X.mean(0)) / X.std(0)

def blockdiff(X, D):
    nb = len(X) // D; B = X[:nb * D].reshape(nb, D, -1).mean(1)
    return np.diff(B, axis=0)

def acf(X, K):
    Xc = X - X.mean(0); n = len(X); g0 = (Xc ** 2).sum(0)
    return np.array([(Xc[h:] * Xc[:n - h]).sum(0) / g0 for h in range(1, K + 1)])

# ----------------------------------------------------------------------------------------------- estimator
def unit_stats(X, K, L):
    """X: (n, d) standardised.  Returns Rh (2L+1, d, d) with Rh[L+h][u, v] = corr(X_u(t+h), X_v(t)) on the
    overlapping samples, the Bartlett factor B (d, d), ACFs, and n."""
    n, d = X.shape
    A = acf(X, K)
    B = np.maximum(1.0, 1 + 2 * np.einsum('ku,kv->uv', A, A))
    Rh = np.zeros((2 * L + 1, d, d))
    for h in range(0, L + 1):
        C = np.corrcoef(np.hstack([X[h:], X[:n - h]]), rowvar=False)
        Rh[L + h] = C[:d, d:]
        Rh[L - h] = C[:d, d:].T
    return {"Rh": Rh, "B": B, "A": A, "n": n, "L": L}

def zcrit(m, L):
    return norm.isf(ALPHA / (2 * m * (2 * L + 1)))

def tstat(st, sub, k=0):
    """max over lags of |atanh r(h)| sqrt(n_eff(h) - 3 - k), n_eff(h) = (n - |h|) / B, on index list sub."""
    sub = list(sub); L = st["L"]; n = st["n"]
    Bs = st["B"][np.ix_(sub, sub)]
    T = np.zeros((len(sub), len(sub)))
    for j, h in enumerate(range(-L, L + 1)):
        R = np.clip(st["Rh"][j][np.ix_(sub, sub)], -0.999999, 0.999999)
        ne = (n - abs(h)) / Bs
        T = np.maximum(T, np.abs(np.arctanh(R)) * np.sqrt(np.maximum(ne - 3 - k, 1e-9)))
    T = np.maximum(T, T.T); np.fill_diagonal(T, 0)
    return T

def declare(st, sub):
    d = len(list(sub)); m = d * (d - 1) / 2
    return tstat(st, sub) > zcrit(m, st["L"])

def edge_values(Dm, sub, edges):
    pos = {v: i for i, v in enumerate(sub)}; rho = 1 + Dm.sum(1)
    return [G.cmp(rho[pos[a]], rho[pos[b]]) for a, b in edges]

# ----------------------------------------------------------------------------------------------- statistics
def boot_ci(per_unit, stat, seed=0):
    rng = np.random.default_rng(seed); u = len(per_unit); bs = []
    for _ in range(NBOOT):
        idx = rng.integers(0, u, u); bs.append(stat([per_unit[i] for i in idx]))
    bs = np.array(bs, float)
    return [float(np.nanquantile(bs, 0.025)), float(np.nanquantile(bs, 0.975))]

def pooled_mean(xs):
    a = np.concatenate([np.asarray(x, float) for x in xs]) if len(xs) else np.array([])
    return float(a.mean()) if a.size else float("nan")

def signflip_p(deltas, seed=1, nmc=100000):
    """one-sided p for H1: mean(delta) < 0, by sign flips of the per-unit deltas (exact if <= 16 units)."""
    x = np.asarray(deltas, float); u = len(x); obs = x.mean()
    if u <= 16:
        signs = np.array(list(itertools.product([1, -1], repeat=u)), float)
    else:
        rng = np.random.default_rng(seed); signs = rng.choice([1.0, -1.0], size=(nmc, u)); signs[0] = 1
    stats = (signs * x).mean(1)
    return float(np.mean(stats <= obs + 1e-12))

def holm(ps):
    keys = list(ps); order = sorted(keys, key=lambda k: ps[k]); m = len(order); adj = {}; run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[k])); adj[k] = run
    return adj

# ----------------------------------------------------------------------------------------------- loading
def load_lt(path, nodes):
    df = pd.read_csv(path)
    return df[nodes].to_numpy(float)

def lt_units(exp, nodes, seglen):
    folder = DATA["crl"] if exp in ("citris_1", "buchholz_1_obs") else DATA["cam"]
    X = load_lt(os.path.join(folder, exp + ".csv"), nodes)
    nseg = len(X) // seglen
    return [X[j * seglen:(j + 1) * seglen] for j in range(nseg)]

def wt_regime_units(nodes, D, drop=100):
    out, ids = [], []
    for e, name in enumerate(["regime_jumps_single", "regime_jumps_multi"]):
        df = pd.read_csv(os.path.join(DATA["wt"], name + ".csv"))
        for r in range(32):
            X = df.loc[df.flag == r, nodes].to_numpy(float)[drop:]
            out.append(blockdiff(X, D)); ids.append(f"{name}:{r}")
    return out, ids

# ----------------------------------------------------------------------------------------------- per-unit core
def prepare(X):
    """drop rows with a missing value; drop nodes constant in this unit; standardise."""
    ok = np.all(np.isfinite(X), 1); X = X[ok]
    keep = [i for i in range(X.shape[1]) if X[:, i].std() > 0]
    return std(X[:, keep]), keep, int((~ok).sum())

def unit_nodesets(p, avail, score_edges, d_list, k, seed_of_d):
    sets = [("full", tuple(avail))]
    if len(avail) == p:
        for d in d_list:
            for S in G.sample_subgraphs(p, score_edges, d, k, seed_of_d(d)):
                sets.append((f"sub{d}", S))
    else:   # some node constant in this unit: sample among available nodes (relabelled)
        loc = {v: i for i, v in enumerate(avail)}
        e2 = [(loc[a], loc[b]) for a, b in score_edges if a in loc and b in loc]
        for d in d_list:
            for S in G.sample_subgraphs(len(avail), e2, d, k, seed_of_d(d)):
                sets.append((f"sub{d}", tuple(sorted(avail[i] for i in S))))
    return sets

def run_dataset(name, units, unit_ids, nodes, rel_edges, score_edges, inputs, K, L, seedbase, subsets=True):
    """Apply the estimator with lag window L to every unit and node set; return per-instance rows and per-unit
    full-set summaries."""
    p = len(nodes); R = G.related(p, rel_edges)
    rows, fullsum, removed, neffs = [], [], {}, []
    for j, Xraw in enumerate(units):
        X, keep, nna = prepare(Xraw)
        if len(keep) < p: removed[unit_ids[j]] = [nodes[i] for i in range(p) if i not in keep]
        st = unit_stats(X, K, L)
        loc = {v: i for i, v in enumerate(keep)}
        sets = unit_nodesets(p, keep, score_edges, (5, 8) if subsets else (), 10,
                             lambda d: 10 * (seedbase + j) + d)
        iu = np.triu_indices(len(keep), 1); neffs.append(float(np.median((X.shape[0] / st["B"])[iu])))
        for sid, (kind, S) in enumerate(sets):
            S = list(S); Dm = declare(st, [loc[v] for v in S])
            indS, preds, _ = G.edge_predictions(S, p, rel_edges, R, inputs, score_edges)
            obs = edge_values(Dm, S, indS); pos = {v: i for i, v in enumerate(S)}
            for e, (a, b) in enumerate(indS):
                rows.append({"unit": j, "set": sid, "kind": kind, "edge": (nodes[a], nodes[b]), "obs": obs[e],
                             "edge_det": bool(Dm[pos[a], pos[b]]), **{k: preds[k][e] for k in G.PREDICTORS}})
            if kind == "full":
                pr = [(S[i], S[k2]) for i, k2 in itertools.combinations(range(len(S)), 2)]
                dec = {(nodes[a], nodes[b]): bool(Dm[i, k2]) for (i, k2), (a, b) in
                       zip(itertools.combinations(range(len(S)), 2), pr)}
                fullsum.append({"nonrel": [float(Dm[i, k2]) for (i, k2), (a, b) in
                                           zip(itertools.combinations(range(len(S)), 2), pr) if not R[a, b]],
                                "rel": [float(Dm[i, k2]) for (i, k2), (a, b) in
                                        zip(itertools.combinations(range(len(S)), 2), pr) if R[a, b]],
                                "decl": dec})
    return {"rows": rows, "full": fullsum, "removed_nodes": removed, "median_neff": neffs, "R": R}

# ----------------------------------------------------------------------------------------------- E1 evaluation
def e1_eval(res, nunits, competitors=None, robustness=True):
    out = e1_core(res["rows"], nunits, competitors)
    if robustness:
        out["detected_edge_subset"] = e1_core([r for r in res["rows"] if r["edge_det"]], nunits, competitors,
                                              magnitude=None, spearman=False)
        out["theory_strict_subset"] = e1_core([r for r in res["rows"] if r["theory"] != 0.5], nunits, competitors,
                                              magnitude=None, spearman=False)
    return out

def e1_core(rows, nunits, competitors=None, magnitude=-0.10, spearman=True):
    comps = competitors or G.PREDICTORS[1:]
    out = {"n_instances": len(rows)}
    if not rows: return out
    loss = lambda r, k: abs(r["obs"] - r[k])
    out["exact_match"] = {k: float(np.mean([r["obs"] == r[k] for r in rows])) for k in G.PREDICTORS}
    out["mean_loss_all"] = {k: float(np.mean([loss(r, k) for r in rows])) for k in G.PREDICTORS}
    obs = np.array([r["obs"] for r in rows])
    out["obs_value_counts"] = {"1": int(np.sum(obs == 1)), "1/2": int(np.sum(obs == .5)), "0": int(np.sum(obs == 0))}
    th = np.array([r["theory"] for r in rows])
    out["obs_given_theory"] = {str(v): (float(obs[th == v].mean()) if np.any(th == v) else None,
                                       float(np.mean(obs[th == v] == v)) if np.any(th == v) else None,
                                       int(np.sum(th == v))) for v in (0.0, 0.5, 1.0)}
    tests, ps = {}, {}
    for c in comps:
        per_unit_th, per_unit_c, deltas = [], [], []
        ninst = 0
        for u in range(nunits):
            ru = [r for r in rows if r["unit"] == u and r["theory"] != r[c]]
            if not ru: continue
            lt = [loss(r, "theory") for r in ru]; lc = [loss(r, c) for r in ru]
            per_unit_th.append(lt); per_unit_c.append(lc); deltas.append(np.mean(lt) - np.mean(lc)); ninst += len(ru)
        if not deltas:
            tests[c] = {"n_disagree": 0}; continue
        pooled_th = pooled_mean(per_unit_th); pooled_c = pooled_mean(per_unit_c)
        pairs = [np.array(a) - np.array(b) for a, b in zip(per_unit_th, per_unit_c)]
        ci = boot_ci(pairs, pooled_mean)
        p = signflip_p(deltas); ps[c] = p
        dis = [r for r in rows if r["theory"] != r[c]]
        null_delta = float(np.mean([abs(0.5 - r["theory"]) - abs(0.5 - r[c]) for r in dis]))
        tests[c] = {"n_disagree": ninst, "units_with_disagreement": len(deltas), "loss_theory": pooled_th,
                    "delta_if_all_ties": null_delta,
                    "loss_comp": pooled_c, "delta": pooled_th - pooled_c, "delta_ci": ci,
                    "units_theory_better": int(np.sum(np.array(deltas) < 0)), "p_onesided": p,
                    "p_reverse": signflip_p([-x for x in deltas])}
    adj = holm(ps)
    for c in ps:
        tests[c]["p_holm"] = adj[c]
        tests[c]["pass"] = bool(adj[c] <= FAMILY_ALPHA and tests[c]["delta"] < (magnitude if magnitude is not None else 0))
    rev = holm({c: tests[c]["p_reverse"] for c in ps})
    for c in ps:
        tests[c]["p_reverse_holm"] = rev[c]
        tests[c]["competitor_significantly_better"] = bool(rev[c] <= FAMILY_ALPHA and tests[c]["delta"] > 0)
    out["tests"] = tests
    out["E1_pass_all"] = bool(ps and all(tests[c].get("pass", False) for c in ps))
    # P9-style Spearman per predictor over subgraphs
    sp = {}
    keys = sorted(set((r["unit"], r["set"]) for r in rows if r["kind"] != "full"))
    if keys and spearman:
        grp = {}
        for r in rows:
            if r["kind"] != "full": grp.setdefault((r["unit"], r["set"]), []).append(r)
        acc = np.array([np.mean([r["obs"] for r in grp[k]]) for k in keys])
        for k in G.PREDICTORS:
            pred = np.array([np.mean([r[k] for r in grp[kk]]) for kk in keys])
            sp[k] = float(spearmanr(pred, acc)[0]) if pred.std() > 0 else None
        out["spearman_subgraphs"] = sp; out["n_subgraphs"] = len(keys)
    return out

def nonrel_and_detection(res):
    nr = [np.array(f["nonrel"]) for f in res["full"]]; rl = [np.array(f["rel"]) for f in res["full"]]
    o = {"nonrel_pairs_per_unit": int(np.median([len(x) for x in nr])), "nonrel_dep_frac": pooled_mean(nr),
         "rel_pairs_per_unit": int(np.median([len(x) for x in rl])), "rel_detect_frac": pooled_mean(rl)}
    o["nonrel_dep_ci"] = boot_ci(nr, pooled_mean) if sum(len(x) for x in nr) else None
    o["rel_detect_ci"] = boot_ci(rl, pooled_mean)
    return o

# ----------------------------------------------------------------------------------------------- E2 evaluation
def e2_eval(res, nodes, edges, nunits):
    """Q7-Q9 as in PLAN3 P7-P9, on the full-set rows and subgraph rows of run_dataset."""
    rows = res["rows"]; o = nonrel_and_detection(res)
    acc_u = [[r["obs"] for r in rows if r["unit"] == u and r["kind"] == "full" and r["theory"] == 1.0]
             for u in range(nunits)]
    acc_u = [np.array(a) for a in acc_u if len(a)]
    o["acc_pred1"] = pooled_mean(acc_u); o["acc_pred1_ci"] = boot_ci(acc_u, pooled_mean)
    o["n_pred1_instances"] = int(sum(len(a) for a in acc_u))
    half = [r["obs"] for r in rows if r["kind"] == "full" and r["theory"] == 0.5]
    o["acc_pred_half_edges"] = float(np.mean(half)) if half else None
    grp = {}
    for r in rows:
        if r["kind"] != "full": grp.setdefault((r["unit"], r["set"]), []).append(r)
    keys = sorted(grp)
    s = np.array([np.mean([r["theory"] for r in grp[k]]) for k in keys])
    a = np.array([np.mean([r["obs"] for r in grp[k]]) for k in keys]); us = np.array([k[0] for k in keys])
    rho, pv = spearmanr(s, a)
    rng = np.random.default_rng(0); bs = []
    idx_by_u = {u: np.where(us == u)[0] for u in range(nunits)}
    for _ in range(NBOOT):
        idx = rng.integers(0, nunits, nunits); m = np.concatenate([idx_by_u[i] for i in idx])
        bs.append(spearmanr(s[m], a[m])[0])
    o.update({"spearman": float(rho), "spearman_p": float(pv), "n_subgraphs": len(keys),
              "spearman_ci": [float(np.nanquantile(bs, .025)), float(np.nanquantile(bs, .975))]})
    o["Q7_pass"] = bool(o["nonrel_dep_frac"] < 0.5)
    o["Q8_pass"] = bool(o["acc_pred1"] > 0.5 and o["acc_pred1_ci"][0] > 0.5)
    o["Q9_pass"] = bool(o["spearman"] >= 0.3)
    return o

def per_edge_detection(res, nodes, edges):
    out = {}
    for a, b in edges:
        k = (nodes[a], nodes[b]); v = [f["decl"][k] if k in f["decl"] else f["decl"].get((k[1], k[0]))
                                       for f in res["full"]]
        v = [x for x in v if x is not None]
        out[f"{k[0]}->{k[1]}"] = float(np.mean(v)) if v else None
    return out

def detection_difference(res_lag, res_0):
    """paired (same units) difference in the share of true relative pairs detected, lagged minus lag-0."""
    a = [np.array(f["rel"]) for f in res_lag["full"]]; b = [np.array(f["rel"]) for f in res_0["full"]]
    d = [x - y for x, y in zip(a, b)]
    nra = [np.array(f["nonrel"]) for f in res_lag["full"]]; nrb = [np.array(f["nonrel"]) for f in res_0["full"]]
    dn = [x - y for x, y in zip(nra, nrb)]
    return {"detect_lag": pooled_mean(a), "detect_lag0": pooled_mean(b), "diff": pooled_mean(d),
            "diff_ci": boot_ci(d, pooled_mean), "nonrel_lag": pooled_mean(nra), "nonrel_lag0": pooled_mean(nrb),
            "nonrel_diff": pooled_mean(dn), "nonrel_diff_ci": boot_ci(dn, pooled_mean) if sum(len(x) for x in dn) else None,
            "Q10_pass": bool(pooled_mean(d) > 0 and boot_ci(d, pooled_mean)[0] > 0)}

# ----------------------------------------------------------------------------------------------- F-theory checks
def fcheck(units, nodes, edges, K, L, seed=1, ndraw=2000):
    """First half of each unit -> predicted second-half decisions (Fisher-z asymptotics, FINITE Prop 1.7 and
    THEORY_ADD Prop A2): per lag p(h); pair prediction p_ind = 1 - prod_h (1 - p(h)) (lags independent) and
    p_max = max_h p(h) (lower bound)."""
    p = len(nodes); R = G.related(p, edges)
    _, pr_full, _ = G.edge_predictions(range(p), p, edges, R, set(), edges)
    e1 = [e for e, v in zip(edges, pr_full["theory"]) if v == 1.0]
    zc = zcrit(p * (p - 1) / 2, L); iu = np.triu_indices(p, 1); rng = np.random.default_rng(seed)
    acc = {k: [] for k in ["pn_ind", "pn_max", "on", "pr_ind", "pr_max", "or", "pacc", "oacc"]}
    used = 0
    for Xraw in units:
        X, keep, _ = prepare(Xraw)
        if len(keep) < p: continue
        used += 1; h2 = len(X) // 2
        X1, X2 = std(X[:h2]), std(X[h2:])
        s1 = unit_stats(X1, K, L); s2 = unit_stats(X2, K, L)
        n2 = s2["n"]; Bhat = np.maximum(1.0, 1 + 2 * np.einsum('ku,kv->uv', s1["A"], s1["A"]))
        Pq = np.ones((p, p)); Pm = np.zeros((p, p))
        for j, h in enumerate(range(-L, L + 1)):
            mu = np.abs(np.arctanh(np.clip(s1["Rh"][j], -.999999, .999999))) * \
                np.sqrt(np.maximum((n2 - abs(h)) / Bhat - 3, 1e-9))
            Ph = norm.sf(zc - mu) + norm.cdf(-zc - mu)
            Ph = np.maximum(Ph, Ph.T)
            Pq *= (1 - Ph); Pm = np.maximum(Pm, Ph)
        Pi = 1 - Pq; np.fill_diagonal(Pi, 0); np.fill_diagonal(Pm, 0)
        D2 = declare(s2, range(p)); nr = ~R[iu]
        acc["pn_ind"] += list(Pi[iu][nr]); acc["pn_max"] += list(Pm[iu][nr]); acc["on"] += list(D2[iu][nr].astype(float))
        acc["pr_ind"] += list(Pi[iu][~nr]); acc["pr_max"] += list(Pm[iu][~nr]); acc["or"] += list(D2[iu][~nr].astype(float))
        sims = []
        for _ in range(ndraw):
            U = rng.random((p, p)); Dm = np.triu(U < Pi, 1); Dm = Dm | Dm.T
            sims.append(np.mean(edge_values(Dm, list(range(p)), e1)))
        acc["pacc"].append(float(np.mean(sims))); acc["oacc"].append(float(np.mean(edge_values(D2, list(range(p)), e1))))
    m = lambda k: float(np.mean(acc[k])) if acc[k] else None
    ok = (m("or") is not None and m("pr_max") - 0.03 <= m("or") <= m("pr_ind") + 0.03)
    return {"units_used": used, "F1_check_pass": bool(ok), "F2_pred_nonrel_ind": m("pn_ind"), "F2_pred_nonrel_max": m("pn_max"),
            "F2_obs_nonrel_half2": m("on"), "F1_pred_rel_ind": m("pr_ind"), "F1_pred_rel_max": m("pr_max"),
            "F1_obs_rel_half2": m("or"), "pred_acc_pred1": m("pacc"), "obs_acc_pred1_half2": m("oacc")}

# ----------------------------------------------------------------------------------------------- drivers
def strip(res):
    return {k: v for k, v in res.items() if k not in ("rows", "full", "R")}

def lt_dataset(exp, seglen, K, L, seedbase, linear):
    nodes, edges, edesign, inputs = G.lt_system(exp, linear=linear)
    units = lt_units(exp, nodes, seglen)
    ids = [f"{exp}:{j}" for j in range(len(units))]
    return nodes, edges, edesign, inputs, units, ids

def main_dev():
    """Development only (PLAN4 states this): run the E1/E2 code on PLAN3's data to test it."""
    out = {}
    import importlib.util
    spec = importlib.util.spec_from_file_location("cc_graph", os.path.join(HERE, "..", "realdata", "cc_graph.py"))
    cc = importlib.util.module_from_spec(spec); spec.loader.exec_module(cc)
    nodes, edges = cc.system("wt"); inputs = set(range(5))
    units = []
    for k in range(2, 17):
        df = pd.read_csv(os.path.join(DATA["wt"], f"actuators_random_walk_{k}.csv"))
        units.append(blockdiff(df[nodes].to_numpy(float), 10))
    ids = [f"rw{k}" for k in range(2, 17)]
    for L in (0, 3):
        r = run_dataset("wt_dev", units, ids, nodes, edges, edges, inputs, 20, L, 9000)
        out[f"wt_dev_L{L}"] = {"E1": e1_eval(r, len(units)), "nonrel_det": nonrel_and_detection(r)}
        print("dev WT L", L, json.dumps(out[f"wt_dev_L{L}"]["nonrel_det"]), flush=True)
        print("   E1 tests:", {c: (round(t.get('delta', np.nan), 3), t.get('p_holm')) for c, t in out[f'wt_dev_L{L}']['E1']['tests'].items()}, flush=True)
    return out

def main_held_out():
    out = {"meta": {"alpha": ALPHA, "nboot": NBOOT, "family_alpha": FAMILY_ALPHA}}
    t0 = time.time()
    # ---------------- E1 primary: citris_1, G_lin, lag-0 estimator, K = 50, 10 units of 10,000 levels
    for lin, tag in [(True, "E1_citris_Glin_L0"), (False, "E1_citris_GD_L0")]:
        nodes, edges, edesign, inputs, units, ids = lt_dataset("citris_1", 10000, 50, 0, 9300, lin)
        r = run_dataset(tag, units, ids, nodes, edges, edesign, inputs, 50, 0, 9300)
        out[tag] = {"E1": e1_eval(r, len(units)), "nonrel_det": nonrel_and_detection(r), **strip(r),
                    "per_edge_detection": per_edge_detection(r, nodes, edesign)}
        print(tag, "done", round(time.time() - t0), "s", flush=True)
    # citris with the lagged estimator (E1 secondary, E2 secondary), L = 5
    nodes, edges, edesign, inputs, units, ids = lt_dataset("citris_1", 10000, 50, 5, 9300, True)
    r5 = run_dataset("E1_citris_Glin_L5", units, ids, nodes, edges, edesign, inputs, 50, 5, 9300)
    r0 = run_dataset("E1_citris_Glin_L0b", units, ids, nodes, edges, edesign, inputs, 50, 0, 9300)
    out["E1_citris_Glin_L5"] = {"E1": e1_eval(r5, len(units)), "nonrel_det": nonrel_and_detection(r5), **strip(r5),
                                "per_edge_detection": per_edge_detection(r5, nodes, edesign)}
    out["E2_citris_detection"] = detection_difference(r5, r0)
    out["F_citris_L0"] = fcheck(units, nodes, edges, 50, 0)
    out["F_citris_L5"] = fcheck(units, nodes, edges, 50, 5)
    print("citris lagged done", round(time.time() - t0), "s", flush=True)
    # ---------------- E1 secondary: i.i.d. SCM sets (snapshot), G_D, lag-0, K = 20, segments of 1,000
    seeds = {"buchholz_1_obs": 9400, "scm_1_reference": 9500, "scm_2_reference": 9600, "scm_4_reference": 9700,
             "scm_5_reference": 9800}
    allrows, nun = [], 0
    for exp, sb in seeds.items():
        nodes, edges, edesign, inputs, units, ids = lt_dataset(exp, 1000, 20, 0, sb, False)
        r = run_dataset(exp, units, ids, nodes, edges, edesign, inputs, 20, 0, sb)
        out[f"E1_iid_{exp}"] = {"E1": e1_eval(r, len(units)), "nonrel_det": nonrel_and_detection(r), **strip(r)}
        for row in r["rows"]:
            row = dict(row); row["unit"] += nun; allrows.append(row)
        nun += len(units)
        print(exp, "done", round(time.time() - t0), "s", flush=True)
    out["E1_iid_pooled"] = {"E1": e1_eval({"rows": allrows}, nun), "units": nun}
    del allrows
    # ---------------- WT regime_jumps: E2 primary (D = 10, L = 3 vs L = 0), E1 tertiary (bipartite)
    nodes, edges, _, inputs = G.wt_regime_system()
    units, ids = wt_regime_units(nodes, 10)
    res = {}
    for L in (0, 3):
        res[L] = run_dataset(f"wt_D10_L{L}", units, ids, nodes, edges, edges, inputs, 20, L, 9200)
        out[f"E2_wt_D10_L{L}"] = {"Q": e2_eval(res[L], nodes, edges, len(units)), "E1": e1_eval(res[L], len(units)),
                                  **strip(res[L]), "per_edge_detection": per_edge_detection(res[L], nodes, edges)}
        print("WT D10 L", L, "done", round(time.time() - t0), "s", flush=True)
    out["E2_wt_D10_detection"] = detection_difference(res[3], res[0])
    out["F_wt_D10_L0"] = fcheck(units, nodes, edges, 20, 0)
    out["F_wt_D10_L3"] = fcheck(units, nodes, edges, 20, 3)
    # secondary: D = 1, L = 30 vs L = 0 (no subgraphs needed for detection; Q computed anyway)
    units1, _ = wt_regime_units(nodes, 1)
    res1 = {}
    for L in (0, 30):
        res1[L] = run_dataset(f"wt_D1_L{L}", units1, ids, nodes, edges, edges, inputs, 20, L, 9200)
        out[f"E2_wt_D1_L{L}"] = {"Q": e2_eval(res1[L], nodes, edges, len(units1)), "E1": e1_eval(res1[L], len(units1)),
                                 **strip(res1[L]),
                                 "per_edge_detection": per_edge_detection(res1[L], nodes, edges)}
        print("WT D1 L", L, "done", round(time.time() - t0), "s", flush=True)
    out["E2_wt_D1_detection"] = detection_difference(res1[30], res1[0])
    out["F_wt_D1_L0"] = fcheck(units1, nodes, edges, 20, 0)
    out["F_wt_D1_L30"] = fcheck(units1, nodes, edges, 20, 30)
    return out

def jsonable(o):
    if isinstance(o, dict): return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return o

if __name__ == "__main__":
    res = main_dev() if MODE == "dev" else main_held_out()
    with open(OUT, "w") as fh:
        json.dump(jsonable(res), fh, indent=1)
    print("wrote", OUT)
