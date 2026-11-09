"""B3 pre-registered analyses (PLAN.md).  Estimator, loss, sign-flip test, Holm and bootstrap are copied unchanged
from ../icml_exp/test4.py (E1); only the predictor set, the instance subsets and the verdict rules are new.
Usage: OMP_NUM_THREADS=1 python3 -I test_b3.py <data_dir> <results_dir>
Resumable: per-dataset instance tables are cached in <results_dir>/cache_<name>.json.gz and reused."""
import sys, os, json, gzip, itertools, time
import numpy as np, pandas as pd
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graphs_b3 as G

DATA, OUTDIR = sys.argv[1:3]
ALPHA = 0.01; NBOOT = 2000; FAMILY_ALPHA = 0.025; MAG = -0.10
OLD8 = G.PREDICTORS[1:9]
COMPS = OLD8 + ["const_half", "wc_rule", "theory_GD"]

# ------------------------------------------------------------------ estimator (test4.py, unchanged, L = 0 only)
def std(X): return (X - X.mean(0)) / X.std(0)
def acf(X, K):
    Xc = X - X.mean(0); n = len(X); g0 = (Xc ** 2).sum(0)
    return np.array([(Xc[h:] * Xc[:n - h]).sum(0) / g0 for h in range(1, K + 1)])
def unit_stats(X, K):
    A = acf(X, K); B = np.maximum(1.0, 1 + 2 * np.einsum('ku,kv->uv', A, A))
    return {"R": np.corrcoef(X, rowvar=False), "B": B, "n": len(X)}
def declare(st, sub):
    sub = list(sub); d = len(sub); m = d * (d - 1) / 2; zc = norm.isf(ALPHA / (2 * m))
    R = np.clip(st["R"][np.ix_(sub, sub)], -0.999999, 0.999999); ne = st["n"] / st["B"][np.ix_(sub, sub)]
    T = np.abs(np.arctanh(R)) * np.sqrt(np.maximum(ne - 3, 1e-9)); np.fill_diagonal(T, 0)
    return T > zc
def prepare(X):
    ok = np.all(np.isfinite(X), 1); X = X[ok]
    keep = [i for i in range(X.shape[1]) if X[:, i].std() > 0]
    return std(X[:, keep]), keep, int((~ok).sum())

# ------------------------------------------------------------------ statistics (test4.py, unchanged)
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
    x = np.asarray(deltas, float); u = len(x); obs = x.mean()
    if u <= 16: signs = np.array(list(itertools.product([1, -1], repeat=u)), float)
    else:
        rng = np.random.default_rng(seed); signs = rng.choice([1.0, -1.0], size=(nmc, u)); signs[0] = 1
    return float(np.mean((signs * x).mean(1) <= obs + 1e-12))
def holm(ps):
    order = sorted(ps, key=lambda k: ps[k]); m = len(order); adj = {}; run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[k])); adj[k] = run
    return adj

# ------------------------------------------------------------------ data
def units_of(exp, nodes, seglen):
    X = pd.read_csv(os.path.join(DATA, exp + ".csv"), usecols=nodes)[nodes].to_numpy(float)
    return [X[j * seglen:(j + 1) * seglen] for j in range(len(X) // seglen)]

def run_dataset(name, units, nodes, rel_edges, score_edges, inputs, K, seedbase):
    cache = os.path.join(OUTDIR, f"cache_{name}.json.gz")
    if os.path.exists(cache):
        return json.load(gzip.open(cache, "rt"))
    p = len(nodes); R = G.related(p, rel_edges); RD = G.related(p, score_edges)
    comp = G.weak_components(p, rel_edges); srcs = [v for v in inputs if not any(b == v for a, b in rel_edges)]
    rows, full, removed, neff = [], [], {}, []
    for j, Xraw in enumerate(units):
        X, keep, nna = prepare(Xraw)
        if len(keep) < p: removed[j] = [nodes[i] for i in range(p) if i not in keep]
        st = unit_stats(X, K); loc = {v: i for i, v in enumerate(keep)}
        iu = np.triu_indices(len(keep), 1); neff.append(float(np.median((X.shape[0] / st["B"])[iu])))
        sets = [("full", tuple(keep))]
        if len(keep) == p:
            for d in (5, 8):
                sets += [(f"sub{d}", S) for S in G.sample_subgraphs(p, score_edges, d, 10, 10 * (seedbase + j) + d)]
        else:
            l2 = {v: i for i, v in enumerate(keep)}; e2 = [(l2[a], l2[b]) for a, b in score_edges if a in l2 and b in l2]
            for d in (5, 8):
                sets += [(f"sub{d}", tuple(sorted(keep[i] for i in S)))
                         for S in G.sample_subgraphs(len(keep), e2, d, 10, 10 * (seedbase + j) + d)]
        for sid, (kind, S) in enumerate(sets):
            S = list(S); Dm = declare(st, [loc[v] for v in S]); pos = {v: i for i, v in enumerate(S)}
            rho = 1 + Dm.sum(1)
            indS, pr = G.edge_predictions(S, p, rel_edges, R, inputs, score_edges, RD, comp)
            srcS = [v for v in S if v in srcs]
            for e, (a, b) in enumerate(indS):
                others = [w for w in srcS if w not in (a, b) and not R[a, w] and R[b, w]]   # Src(b) \ Src(a), in S
                rows.append({"unit": j, "set": sid, "kind": kind, "edge": [nodes[a], nodes[b]],
                             "obs": G.cmp(rho[pos[a]], rho[pos[b]]), "edge_det": bool(Dm[pos[a], pos[b]]),
                             "second_source": bool(others),
                             "mech": bool(others) and any(Dm[pos[b], pos[w]] for w in others)
                                     and not any(Dm[pos[a], pos[w]] for w in others),
                             **{k: pr[k][e] for k in G.PREDICTORS}})
            if kind == "full":
                pairs = list(itertools.combinations(range(len(S)), 2))
                full.append({"nonrel": [float(Dm[i, k]) for i, k in pairs if not R[S[i], S[k]]],
                             "rel": [float(Dm[i, k]) for i, k in pairs if R[S[i], S[k]]],
                             "decl": {f"{nodes[S[i]]}|{nodes[S[k]]}": bool(Dm[i, k]) for i, k in pairs},
                             "r": {f"{nodes[S[i]]}|{nodes[S[k]]}": float(st["R"][loc[S[i]], loc[S[k]]]) for i, k in pairs}})
    res = {"rows": rows, "full": full, "removed_nodes": removed, "median_neff": neff, "nunits": len(units)}
    json.dump(res, gzip.open(cache + ".part", "wt")); os.replace(cache + ".part", cache)
    return res

# ------------------------------------------------------------------ tests
def compare(rows, nunits, comps, mag=MAG, alpha=FAMILY_ALPHA):
    """theory vs each competitor on the rows where they disagree; per-unit delta = mean loss(theory) - mean
    loss(comp); one-sided sign-flip test; Holm over distinct competitors (identical predictions on `rows` are merged)."""
    loss = lambda r, k: abs(r["obs"] - r[k])
    out = {"n_instances": len(rows)}
    if not rows: return out
    out["exact_match"] = {k: float(np.mean([r["obs"] == r[k] for r in rows])) for k in ["theory"] + comps}
    obs = np.array([r["obs"] for r in rows])
    out["obs_value_counts"] = {"1": int(np.sum(obs == 1)), "1/2": int(np.sum(obs == .5)), "0": int(np.sum(obs == 0))}
    sig = {}
    for c in comps: sig.setdefault(tuple(r[c] for r in rows), []).append(c)
    groups = list(sig.values()); tests, ps = {}, {}
    for grp in groups:
        c = grp[0]; pt, pc, deltas, ninst = [], [], [], 0
        for u in range(nunits):
            ru = [r for r in rows if r["unit"] == u and r["theory"] != r[c]]
            if not ru: continue
            lt = [loss(r, "theory") for r in ru]; lc = [loss(r, c) for r in ru]
            pt.append(lt); pc.append(lc); deltas.append(np.mean(lt) - np.mean(lc)); ninst += len(ru)
        name = "=".join(grp)
        if not deltas:
            tests[name] = {"n_disagree": 0, "testable": False}; continue
        pairs = [np.array(a) - np.array(b) for a, b in zip(pt, pc)]
        p = signflip_p(deltas); ps[name] = p
        tests[name] = {"testable": True, "n_disagree": ninst, "units_with_disagreement": len(deltas),
                       "loss_theory": pooled_mean(pt), "loss_comp": pooled_mean(pc),
                       "delta": pooled_mean(pt) - pooled_mean(pc), "delta_ci": boot_ci(pairs, pooled_mean),
                       "units_theory_better": int(np.sum(np.array(deltas) < 0)), "p_onesided": p,
                       "p_reverse": signflip_p([-x for x in deltas])}
    adj = holm(ps); rev = holm({c: tests[c]["p_reverse"] for c in ps})
    for c in ps:
        tests[c]["p_holm"] = adj[c]; tests[c]["p_reverse_holm"] = rev[c]
        tests[c]["pass"] = bool(adj[c] <= alpha and tests[c]["delta"] < (mag if mag is not None else 0))
        tests[c]["competitor_significantly_better"] = bool(rev[c] <= alpha and tests[c]["delta"] > 0)
    out["tests"] = tests
    out["n_testable"] = len(ps)
    out["pass_all"] = bool(ps and all(tests[c]["pass"] for c in ps))
    out["any_competitor_better"] = bool(any(tests[c]["competitor_significantly_better"] for c in ps))
    return out

def prereq(res):
    nr = [np.array(f["nonrel"]) for f in res["full"]]; rl = [np.array(f["rel"]) for f in res["full"]]
    o = {"nonrel_dep_frac": pooled_mean(nr), "rel_detect_frac": pooled_mean(rl),
         "rel_detect_ci": boot_ci(rl, pooled_mean),
         "nonrel_dep_ci": boot_ci(nr, pooled_mean) if sum(len(x) for x in nr) else None}
    o["pass"] = bool(o["nonrel_dep_frac"] < 0.5 and o["rel_detect_frac"] >= 0.8)
    return o

def switch_contrast(rows, nunits):
    """S3: colour-source -> colour-sensor edges; per unit, mean obs with a second source in S minus without."""
    d = []
    for u in range(nunits):
        ru = [r for r in rows if r["unit"] == u and r["edge"][0] in G.COL and r["edge"][1] in G.COLSENS]
        a = [r["obs"] for r in ru if r["second_source"]]; b = [r["obs"] for r in ru if not r["second_source"]]
        if a and b: d.append(np.mean(a) - np.mean(b))
    return {"units": len(d), "mean_contrast": float(np.mean(d)) if d else None,
            "units_positive": int(np.sum(np.array(d) > 0)), "p_onesided": signflip_p([-x for x in d]) if d else None,
            "pass": bool(d and np.mean(d) >= 0.25 and signflip_p([-x for x in d]) <= FAMILY_ALPHA)}

def per_edge(rows):
    out = {}
    for r in rows:
        k = f"{r['edge'][0]}->{r['edge'][1]}|{'2src' if r['second_source'] else 'nosrc'}"
        out.setdefault(k, []).append(r["obs"])
    return {k: {"n": len(v), "mean_obs": float(np.mean(v)), "share_1": float(np.mean(np.array(v) == 1)),
                "share_half": float(np.mean(np.array(v) == .5))} for k, v in sorted(out.items())}

def analyse(res, comps, label):
    rows = res["rows"]; n = res["nunits"]
    Dstar = [r for r in rows if r["theory"] != r["wc_rule"]]
    half_vs_sort = [r for r in rows if r["theory"] == 0.5 and r["edge_det"]]
    o = {"label": label, "prereq": prereq(res), "removed_nodes": res["removed_nodes"],
         "median_neff_per_unit": res["median_neff"], "n_instances": len(rows), "n_Dstar": len(Dstar),
         "primary_Dstar": compare(Dstar, n, comps),
         "S1_theory_half_detected_edge": compare(half_vs_sort, n, comps, mag=None),
         "S1u_theory_half_all": compare([r for r in rows if r["theory"] == 0.5], n, comps, mag=None),
         "S2_all_instances": compare(rows, n, comps),
         "S3_switch": switch_contrast(rows, n),
         "S6_mechanism_share": (float(np.mean([r["mech"] for r in Dstar if r["obs"] == 1]))
                                if any(r["obs"] == 1 for r in Dstar) else None),
         "Dstar_obs_share_1": float(np.mean([r["obs"] == 1 for r in Dstar])) if Dstar else None,
         "per_edge": per_edge(rows)}
    if any(r["theory"] != r["theory_GD"] for r in rows):
        o["S5_Glin_vs_GD"] = compare([r for r in rows if r["theory"] != r["theory_GD"]], n, ["theory_GD"], mag=None)
    if any(r["edge"][1] == "green" for r in rows):
        col = [r for r in rows if r["edge"][1] == "green" and r["edge"][0] in ("red", "blue")]
        o["collider_edges_into_green"] = {
            "with_other_source": {"n": sum(r["second_source"] for r in col),
                                  "share_obs_1": float(np.mean([r["obs"] == 1 for r in col if r["second_source"]])) if any(r["second_source"] for r in col) else None},
            "without": {"n": sum(not r["second_source"] for r in col),
                        "share_obs_half": float(np.mean([r["obs"] == .5 for r in col if not r["second_source"]])) if any(not r["second_source"] for r in col) else None}}
    return o

def pool(results):
    rows, n, full, rem, ne = [], 0, [], {}, []
    for r in results:
        for row in r["rows"]:
            row = dict(row); row["unit"] += n; rows.append(row)
        full += r["full"]; ne += r["median_neff"]
        rem.update({str(int(k) + n): v for k, v in r["removed_nodes"].items()}); n += r["nunits"]
    return {"rows": rows, "full": full, "removed_nodes": rem, "median_neff": ne, "nunits": n}

def strip_for_print(o):
    pr = o["primary_Dstar"]
    return {"prereq": o["prereq"], "n_Dstar": o["n_Dstar"], "Dstar_obs_share_1": o["Dstar_obs_share_1"],
            "primary": {k: (round(v.get("delta", float("nan")), 3), v.get("p_holm"), v.get("pass")) for k, v in pr.get("tests", {}).items()},
            "S1": {k: (round(v.get("delta", float("nan")), 3), v.get("p_holm"), v.get("pass")) for k, v in o["S1_theory_half_detected_edge"].get("tests", {}).items()},
            "S3": o["S3_switch"], "S6": o["S6_mechanism_share"]}

if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True); t0 = time.time(); out = {"meta": {"alpha": ALPHA, "family_alpha": FAMILY_ALPHA,
                                                                          "magnitude": MAG, "nboot": NBOOT}}
    # H1 primary: ar_1 time series, 3 experiments x 10 segments of 3,000 rows, K = 50
    nodes, EL, ED, inp = G.ar1_system(); ar = []
    units = []
    for exp in ["ar_1_uniform_ref", "ar_1_uniform_pol_1_90", "ar_1_uniform_pol_12_30"]:
        units += units_of(exp, nodes, 3000)
    resA = run_dataset("ar_1", units, nodes, EL, ED, inp, 50, G.SEEDBASE["ar_1"])
    out["H1_ar_1"] = analyse(resA, COMPS, "H1 primary: lt_camera_walks_v1 ar_1_uniform_*, G_lin")
    print("H1", json.dumps(strip_for_print(out["H1_ar_1"])), round(time.time() - t0), "s", flush=True)
    # per experiment (reported)
    for e, exp in enumerate(["ar_1_uniform_ref", "ar_1_uniform_pol_1_90", "ar_1_uniform_pol_12_30"]):
        sub = {"rows": [dict(r, unit=r["unit"] - 10 * e) for r in resA["rows"] if 10 * e <= r["unit"] < 10 * e + 10],
               "full": resA["full"][10 * e:10 * e + 10], "removed_nodes": {}, "median_neff": resA["median_neff"][10 * e:10 * e + 10],
               "nunits": 10}
        out[f"H1_per_experiment_{exp}"] = {"primary_Dstar": compare([r for r in sub["rows"] if r["theory"] != r["wc_rule"]], 10, COMPS),
                                           "prereq": prereq(sub)}
    # snapshot sets: 10 segments of 1,000 rows, K = 20; G_lin = G_D (no polariser varies)
    scm = {}
    for exp in ["scm_5_red", "scm_5_blue", "scm_5_green", "scm_4_green", "scm_4_blue", "scm_4_red"]:
        nodes, E, E2, inp = G.scm_system(exp)
        scm[exp] = run_dataset(exp, units_of(exp, nodes, 1000), nodes, E, E2, inp, 20, G.SEEDBASE[exp])
        out[f"per_set_{exp}"] = analyse(scm[exp], COMPS, exp)
        print(exp, json.dumps(strip_for_print(out[f"per_set_{exp}"])), round(time.time() - t0), "s", flush=True)
    out["H2_scm_5_collider"] = analyse(pool([scm["scm_5_red"], scm["scm_5_blue"]]), COMPS,
                                       "H2 co-primary (collider R->G<-B): scm_5_red + scm_5_blue")
    print("H2", json.dumps(strip_for_print(out["H2_scm_5_collider"])), flush=True)
    out["S4_scm_multisource"] = analyse(pool([scm["scm_5_green"], scm["scm_4_green"], scm["scm_4_blue"]]), COMPS,
                                        "S4: scm_5_green + scm_4_green + scm_4_blue")
    print("S4", json.dumps(strip_for_print(out["S4_scm_multisource"])), flush=True)
    out["S4_control_scm_4_red"] = out.pop("per_set_scm_4_red")
    # verdicts (PLAN.md section 6)
    h1, h2 = out["H1_ar_1"], out["H2_scm_5_collider"]
    v = {"H1_pass": h1["primary_Dstar"]["pass_all"], "H1_prereq_pass": h1["prereq"]["pass"],
         "H2_pass": h2["primary_Dstar"]["pass_all"], "H2_prereq_pass": h2["prereq"]["pass"],
         "K2_H1_competitor_better_on_Dstar": h1["primary_Dstar"]["any_competitor_better"],
         "K2_H2_competitor_better_on_Dstar": h2["primary_Dstar"]["any_competitor_better"],
         "S1_H1_pass": h1["S1_theory_half_detected_edge"].get("pass_all"),
         "S1_H2_pass": h2["S1_theory_half_detected_edge"].get("pass_all"),
         "K4_H1_mechanism_below_half": (h1["S6_mechanism_share"] is not None and h1["S6_mechanism_share"] < 0.5),
         "K4_H2_mechanism_below_half": (h2["S6_mechanism_share"] is not None and h2["S6_mechanism_share"] < 0.5)}
    v["K1_H1_uninformative"] = bool((not v["H1_pass"]) and (not v["H1_prereq_pass"]))
    v["K3_H1_restricted_claim"] = bool(v["H1_pass"] and not v["S1_H1_pass"])
    v["full_separation_H1"] = bool(v["H1_pass"] and v["S1_H1_pass"] and not v["K2_H1_competitor_better_on_Dstar"]
                                   and not v["K4_H1_mechanism_below_half"])
    out["verdicts"] = v
    print("VERDICTS", json.dumps(v), flush=True)
    json.dump(out, open(os.path.join(OUTDIR, "results_b3.json"), "w"), indent=1, default=float)
    print("wrote results_b3.json", round(time.time() - t0), "s")
