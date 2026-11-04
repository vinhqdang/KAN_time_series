"""Stage B of PLAN3.md (commit cfcab28): pre-registered tests P7-P9 (Causal Chambers wind tunnel, random-walk
runs 2-16), replication R7-R9 (light tunnel, actuators_white, 10 segments), secondary S1-S4 and F-theory checks.
Usage: OMP_NUM_THREADS=1 python3 -I test3.py <this_dir> <wt_dir> <lt_dir>  -> results3.json (and stdout log)."""
import sys, os, csv, json, itertools
import numpy as np
from scipy.stats import norm, spearmanr
HERE, WT_DIR, LT_DIR = sys.argv[1:4]
sys.path.insert(0, HERE); import cc_graph as G

ALPHA = 0.01; NBOOT = 2000

def load(path, nodes):
    with open(path) as fh:
        r = csv.reader(fh); head = next(r); rows = list(r)
    c = {h: i for i, h in enumerate(head)}
    X = np.array([[float(row[c[v]]) if row[c[v]] not in ("", "nan", "NaN") else np.nan for v in nodes] for row in rows])
    ok = np.all(np.isfinite(X), 1)
    return X[ok], int((~ok).sum())

def std(X): return (X - X.mean(0)) / X.std(0)

def blockdiff(X, D):
    nb = len(X) // D; B = X[:nb * D].reshape(nb, D, -1).mean(1)
    return np.diff(B, axis=0)

def acf(X, K):
    Xc = X - X.mean(0); n = len(X); g0 = (Xc ** 2).sum(0)
    return np.array([(Xc[h:] * Xc[:n - h]).sum(0) / g0 for h in range(1, K + 1)])   # (K, d)

def neff_matrix(n, A):
    s = 1 + 2 * np.einsum('ku,kv->uv', A, A)
    return n / np.maximum(1.0, s)

def zcrit(d): m = d * (d - 1) / 2; return norm.isf(ALPHA / (2 * m))

def resid_on(X, z):
    Z = np.column_stack([np.ones(len(z)), z]); beta, *_ = np.linalg.lstsq(Z, X, rcond=None)
    return X - Z @ beta

def stats(X, K, z=None):
    """Correlation matrix, ACFs and n_eff for the series X (rows = time); if z is given, of the residuals on z."""
    Y = resid_on(X, z) if z is not None else X
    sd = Y.std(0); keep = sd > 0
    Y = Y[:, keep]; R = np.corrcoef(Y, rowvar=False); A = acf(Y, K)
    return R, A, neff_matrix(len(Y), A), keep

def declare(R, NE, sub, k=0, nominal=False, n=None):
    """Declared-dependence matrix on index list sub (Bonferroni over its pairs)."""
    sub = list(sub); d = len(sub); zc = zcrit(d)
    Rs = np.clip(R[np.ix_(sub, sub)], -0.999999, 0.999999)
    ne = np.full((d, d), float(n)) if nominal else NE[np.ix_(sub, sub)]
    Z = np.abs(np.arctanh(Rs)) * np.sqrt(np.maximum(ne - 3 - k, 1e-9))
    Dm = Z > zc; np.fill_diagonal(Dm, False); return Dm

def edge_vals(Dm, sub, edges):
    pos = {v: i for i, v in enumerate(sub)}; rho = 1 + Dm.sum(1)
    out = []
    for a, b in edges:
        if a in pos and b in pos:
            ra, rb = rho[pos[a]], rho[pos[b]]; out.append(1.0 if ra < rb else (0.5 if ra == rb else 0.0))
    return out

def nonrel_frac(Dm, sub, Rel):
    pos = list(sub); v = []
    for i, j in itertools.combinations(range(len(pos)), 2):
        if not Rel[pos[i], pos[j]]: v.append(float(Dm[i, j]))
    return v

def boot_ci(per_unit, stat=lambda xs: np.mean(np.concatenate(xs)), seed=0):
    rng = np.random.default_rng(seed); u = len(per_unit); bs = []
    for _ in range(NBOOT):
        idx = rng.integers(0, u, u); bs.append(stat([per_unit[i] for i in idx]))
    return [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]

def pooled(per_unit): return float(np.mean(np.concatenate([np.asarray(x, float) for x in per_unit])))

def analyse(units, nodes, edges, K, subseed, label, nominal=False, cond=None, extra_edges_half=None):
    """units: list of preprocessed (n, p) arrays (already standardised). Returns dict of results."""
    p = len(nodes); Rel = G.related(p, edges)
    _, predv = G.predicted_edges(range(p), edges, Rel)
    e1 = [e for e, v in zip(edges, predv) if v == 1.0]; ehalf = [e for e, v in zip(edges, predv) if v == 0.5]
    nonrel_u, acc_u, half_u, sub_rows, rel_rec_u, neffs = [], [], [], [], [], []
    for ui, X in enumerate(units):
        z = None; full = list(range(p))
        if cond is not None:
            z = X[:, cond]; full = [i for i in range(p) if i != cond]
        R, A, NE, keep = stats(X[:, full], K, z)
        if not keep.all(): raise RuntimeError(f"{label}: constant node in unit {ui}")
        # map full node ids -> positions in R
        loc = {v: i for i, v in enumerate(full)}
        sub = full; Dm = declare(R, NE, [loc[v] for v in sub], k=0 if z is None else 1, nominal=nominal, n=len(X))
        nonrel_u.append(nonrel_frac(Dm, sub, Rel))
        acc_u.append(edge_vals(Dm, sub, e1)); half_u.append(edge_vals(Dm, sub, ehalf))
        rr = [float(Dm[i, j]) for i, j in itertools.combinations(range(len(sub)), 2) if Rel[sub[i], sub[j]]]
        rel_rec_u.append(rr)
        iu = np.triu_indices(len(sub), 1); neffs.append(float(np.median((NE if not nominal else np.full_like(NE, len(X)))[iu])))
        if subseed is not None and cond is None:
            for d in (5, 8):
                for S in G.sample_subgraphs(p, edges, d, 10, 10 * (subseed + ui_id(ui, label)) + d):
                    ind, pv = G.predicted_edges(S, edges, Rel)
                    Ds = declare(R, NE, list(S), nominal=nominal, n=len(X))
                    ev = edge_vals(Ds, list(S), ind)
                    sub_rows.append((ui, d, float(np.mean(pv)), float(np.mean(ev))))
    out = {"label": label, "units": len(units), "n_per_unit": [len(X) for X in units],
           "median_neff_per_unit": neffs,
           "nonrel_dep_frac": pooled(nonrel_u), "nonrel_dep_ci": boot_ci([np.asarray(x) for x in nonrel_u]),
           "nonrel_pairs_per_unit": len(nonrel_u[0]),
           "acc_pred1": pooled(acc_u), "acc_pred1_ci": boot_ci([np.asarray(x) for x in acc_u]),
           "n_edges_pred1": len(e1), "acc_pred_half_edges": pooled(half_u) if ehalf else None,
           "pred_half_edges": [(nodes[a], nodes[b]) for a, b in ehalf],
           "rel_recovery": pooled(rel_rec_u),
           "per_unit_nonrel": [float(np.mean(x)) for x in nonrel_u], "per_unit_acc": [float(np.mean(x)) for x in acc_u]}
    if sub_rows:
        s = np.array([r[2] for r in sub_rows]); a = np.array([r[3] for r in sub_rows]); us = np.array([r[0] for r in sub_rows])
        rho, pv = spearmanr(s, a)
        rng = np.random.default_rng(0); bs = []
        for _ in range(NBOOT):
            idx = rng.integers(0, len(units), len(units)); m = np.concatenate([np.where(us == i)[0] for i in idx])
            bs.append(spearmanr(s[m], a[m])[0])
        out.update({"spearman": float(rho), "spearman_p": float(pv), "spearman_ci": [float(np.nanquantile(bs, .025)), float(np.nanquantile(bs, .975))],
                    "n_subgraphs": len(s), "sub_pred_mean": float(s.mean()), "sub_acc_mean": float(a.mean()),
                    "sub_acc_pred_gt_half": float(a[s > 0.5].mean()), "sub_acc_pred_eq_half": float(a[s == 0.5].mean()) if np.any(s == 0.5) else None})
    return out

UNIT_IDS = {}
def ui_id(ui, label): return UNIT_IDS[label][ui]

def fcheck(units, nodes, edges, K, seed=1, ndraw=2000):
    """F-theory out-of-sample checks: first half -> predicted detection probabilities on the second half."""
    p = len(nodes); Rel = G.related(p, edges); _, predv = G.predicted_edges(range(p), edges, Rel)
    e1 = [e for e, v in zip(edges, predv) if v == 1.0]; zc = zcrit(p); iu = np.triu_indices(p, 1)
    rng = np.random.default_rng(seed)
    pn, on, pr, orr, kap, rmin, nh, pacc, oacc, neff2 = [], [], [], [], [], [], [], [], [], []
    for X in units:
        h = len(X) // 2; X1, X2 = std(X[:h]), std(X[h:])
        R1, A1, _, _ = stats(X1, K); R2, A2, NE2, _ = stats(X2, K)
        ne_hat = neff_matrix(len(X2), A1)
        mu = np.abs(np.arctanh(np.clip(R1, -.999999, .999999))) * np.sqrt(np.maximum(ne_hat - 3, 1e-9))
        P = norm.sf(zc - mu) + norm.cdf(-zc - mu); np.fill_diagonal(P, 0)
        D2 = declare(R2, NE2, range(p))
        nr = ~Rel[iu]
        pn += list(P[iu][nr]); on += list(D2[iu][nr].astype(float)); pr += list(P[iu][~nr]); orr += list(D2[iu][~nr].astype(float))
        kap.append(float(np.median(np.abs(R1[iu][nr])))); rmin.append(float(np.min(np.abs(R1[iu][~nr]))))
        neff2.append(float(np.median(NE2[iu])))
        # Monte Carlo predicted accuracy on the predicted-1 edges
        accs = []
        for _ in range(ndraw):
            U = rng.random((p, p)); Dm = np.triu(U < P, 1); Dm = Dm | Dm.T
            accs.append(np.mean(edge_vals(Dm, list(range(p)), e1)))
        pacc.append(float(np.mean(accs))); oacc.append(float(np.mean(edge_vals(D2, list(range(p)), e1))))
    kmed = float(np.median(kap))
    return {"F2_pred_nonrel_frac": float(np.mean(pn)), "F2_obs_nonrel_frac_half2": float(np.mean(on)),
            "F2_kappa_hat_median": kmed, "F2_n_half": float(3 + (zc / np.arctanh(kmed)) ** 2) if kmed > 0 else None,
            "neff_half2_median": float(np.median(neff2)),
            "F1_pred_rel_recovery": float(np.mean(pr)), "F1_obs_rel_recovery_half2": float(np.mean(orr)),
            "F1_rho_min_half1_median": float(np.median(rmin)),
            "pred_acc_pred1_edges": float(np.mean(pacc)), "obs_acc_pred1_edges_half2": float(np.mean(oacc)),
            "per_unit_pred_acc": pacc, "per_unit_obs_acc": oacc}

def triad(units, nodes, edges, zi):
    p = len(nodes); Rel = G.related(p, edges); out = []
    for X in units:
        R = np.corrcoef(X, rowvar=False)
        for u, v in itertools.combinations([i for i in range(p) if i != zi], 2):
            if not Rel[u, v] and abs(R[u, v]) >= 0.1: out.append(R[u, zi] * R[v, zi] / R[u, v])
    return {"n_pairs": len(out), "r_hat": float(np.clip(np.median(out), 0, 1)) if out else None}

def main():
    res = {}
    # ------------------------------------------------------------------ WT
    nodes, edges = G.system("wt"); runs = list(range(2, 17))
    raw = []; dropped = {}
    for k in runs:
        X, nd = load(os.path.join(WT_DIR, f"actuators_random_walk_{k}.csv"), nodes); raw.append(X); dropped[k] = nd
    res["wt_dropped_rows"] = dropped
    prim = [std(blockdiff(X, 10)) for X in raw]
    UNIT_IDS["wt"] = runs; UNIT_IDS["wt_nom"] = runs
    res["P"] = analyse(prim, nodes, edges, 20, 9000, "wt")
    res["S2_wt_nominal"] = analyse(prim, nodes, edges, 20, 9000, "wt_nom", nominal=True)
    lev = [std(X) for X in raw]
    UNIT_IDS["wt_lev"] = runs
    res["S1_wt_levels"] = analyse(lev, nodes, edges, 100, 9000, "wt_lev")
    for D in (1, 30):
        UNIT_IDS[f"wt_D{D}"] = runs
        res[f"S3_wt_D{D}"] = analyse([std(blockdiff(X, D)) for X in raw], nodes, edges, 20, 9000, f"wt_D{D}")
    zi = nodes.index("pressure_ambient"); baro = [nodes.index(v) for v in ["pressure_upwind", "pressure_downwind", "pressure_intake"]]
    for name, U, K in [("levels", lev, 100), ("primary", prim, 20)]:
        UNIT_IDS[f"s4_{name}"] = runs
        r = analyse(U, nodes, edges, K, None, f"s4_{name}", cond=zi)
        # barometer-pair fractions before/after (pairs among the 3 in-tunnel barometers are relatives; report |r| with Z)
        bz = [float(np.mean([abs(np.corrcoef(X[:, b], X[:, zi])[0, 1]) for b in baro])) for X in U]
        r["mean_abs_corr_barometers_with_ambient_per_unit"] = bz
        r["triad"] = triad(U, nodes, edges, zi)
        res[f"S4_wt_cond_ambient_{name}"] = r
    res["F_wt"] = fcheck(prim, nodes, edges, 20)
    # ------------------------------------------------------------------ LT
    nodes, edges = G.system("lt")
    X, nd = load(os.path.join(LT_DIR, "actuators_white.csv"), nodes); res["lt_dropped_rows"] = nd
    segs = [std(X[j * 2000:(j + 1) * 2000]) for j in range(10)]
    UNIT_IDS["lt"] = list(range(10)); UNIT_IDS["lt_nom"] = list(range(10))
    res["R"] = analyse(segs, nodes, edges, 20, 9100, "lt")
    res["S2_lt_nominal"] = analyse(segs, nodes, edges, 20, 9100, "lt_nom", nominal=True)
    res["F_lt"] = fcheck(segs, nodes, edges, 20)
    # Malus caveat: detection of pol_k with ir_3 / vis_3 (relative pairs predicted to be missed)
    nl = {v: i for i, v in enumerate(nodes)}; mal = []
    for Xs in segs:
        R, A, NE, _ = stats(Xs, 20); Dm = declare(R, NE, range(len(nodes)))
        mal.append({f"{a}-{b}": [round(float(R[nl[a], nl[b]]), 3), bool(Dm[nl[a], nl[b]])] for a in ("pol_1", "pol_2") for b in ("ir_3", "vis_3")})
    res["lt_malus_pairs_per_segment"] = mal
    with open(os.path.join(HERE, "results3.json"), "w") as f: json.dump(res, f, indent=1)
    # ------------------------------------------------------------------ log
    def line(tag, r):
        s = (f"{tag}: nonrel dep {r['nonrel_dep_frac']:.3f} {np.round(r['nonrel_dep_ci'],3).tolist()} | acc(pred 1) {r['acc_pred1']:.3f} "
             f"{np.round(r['acc_pred1_ci'],3).tolist()} | pred-1/2 edges {r['acc_pred_half_edges']} | rel recovery {r['rel_recovery']:.3f} | median n_eff {np.median(r['median_neff_per_unit']):.0f}")
        if "spearman" in r: s += f" | Spearman {r['spearman']:.3f} (p={r['spearman_p']:.2g}) CI {np.round(r['spearman_ci'],3).tolist()} n={r['n_subgraphs']}"
        print(s)
    for k in ["P", "S2_wt_nominal", "S1_wt_levels", "S3_wt_D1", "S3_wt_D30", "S4_wt_cond_ambient_levels", "S4_wt_cond_ambient_primary", "R", "S2_lt_nominal"]:
        line(k, res[k])
    for k in ["F_wt", "F_lt"]:
        print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in res[k].items() if not a.startswith("per_unit")})
    print("triad", res["S4_wt_cond_ambient_levels"]["triad"], res["S4_wt_cond_ambient_primary"]["triad"])
    print("dropped rows", res["wt_dropped_rows"], res["lt_dropped_rows"])

if __name__ == "__main__":
    main()
