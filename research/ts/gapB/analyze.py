"""usage: analyze.py val|test  -> results_<phase>.md (rules fixed in PLAN.md)"""
import sys, numpy as np, pandas as pd, os
from scipy.stats import spearmanr
import gapb
phase = sys.argv[1]
FAMS = gapb.FAMS
rng = np.random.default_rng(6000)
NB, NP = 2000, 20000

def load(prefix, w):
    f = f"{prefix}_{phase}_{w}.csv"
    return pd.read_csv(f) if os.path.exists(f) and os.path.getsize(f) > 5 else None

def per_dataset(R):
    """one row per (dataset, family) with rho, gains, regrets"""
    out = []
    key = ["data", "cell", "cluster", "d", "seed"] + [c for c in ("sim", "subject", "res", "region") if c in R.columns]
    R = R[R.setting >= 0]
    for k, g in R.groupby(key + ["family"], dropna=False):
        meta = dict(zip(key + ["family"], k))
        c = g[(g.nedges >= 1) & g.val_mse.notna() & g.auroc.notna()]
        row = dict(meta, n_cand=len(c), rho=np.nan, rho_ap=np.nan)
        if len(c) >= 4 and c.auroc.nunique() > 1 and c.val_mse.nunique() > 1:
            row["rho"] = spearmanr(-c.val_mse, c.auroc)[0]
        if len(c) >= 4 and c.auprc.nunique() > 1 and c.val_mse.nunique() > 1:
            row["rho_ap"] = spearmanr(-c.val_mse, c.auprc)[0]
        if len(c):
            vals, ne = c.val_mse.values, c.nedges.values
            d = meta["d"]; te = c.true_edges.iloc[0]
            rnd, orc = c.auroc.mean(), c.auroc.max()
            for rule, v, tgt in (("fc", vals, None), ("bic", c.bic.values, None), ("sp", None, d), ("spt", None, te)):
                if rule in ("sp", "spt"):
                    i = min(range(len(c)), key=lambda i: (abs(ne[i] - tgt), ne[i], i))
                else:
                    i = min(range(len(c)), key=lambda i: (v[i] if np.isfinite(v[i]) else np.inf, ne[i], i))
                a = c.auroc.values[i]
                row[f"auc_{rule}"] = a; row[f"gain_{rule}"] = a - rnd; row[f"regret_{rule}"] = a - orc
            row["auc_rnd"] = rnd; row["auc_orc"] = orc
        out.append(row)
    return pd.DataFrame(out)

def boot_median(df, col):
    """cluster bootstrap CI of the median of col; df has 'cluster'"""
    d = df.dropna(subset=[col]); cl = d.cluster.unique()
    if len(d) == 0: return np.nan, np.nan, np.nan
    groups = {c: g[col].values for c, g in d.groupby("cluster")}
    meds = []
    for _ in range(NB):
        pick = rng.choice(len(cl), len(cl))
        meds.append(np.median(np.concatenate([groups[cl[i]] for i in pick])))
    return np.median(d[col]), np.percentile(meds, 2.5), np.percentile(meds, 97.5)

def signflip_p(diffs):
    """one-sided p for mean(diffs) > 0 by sign-flip permutation on cluster-level diffs"""
    d = np.asarray(diffs); d = d[np.isfinite(d)]
    if len(d) == 0: return np.nan
    obs = d.mean(); s = rng.choice([-1, 1], size=(NP, len(d)))
    return (1 + (((s * d).mean(1)) >= obs - 1e-15).sum()) / (NP + 1)

def holm(ps):
    ps = np.array(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1, run)
    return adj

def cluster_mean(df, col):
    return df.groupby("cluster")[col].mean()

def verdict_block(D, N, label, lines):
    """D per-dataset table, N necessity table for one dataset group; returns dict with pass flags"""
    lines.append(f"\n### {label}\n")
    lines.append(f"datasets (dataset x family rows / 5): {len(D) // 5}, clusters: {D.cluster.nunique()}\n")
    rows = []; p_gain, p_sp, p_bic, p_nec = [], [], [], []
    for f in FAMS:
        g = D[D.family == f]
        med, lo, hi = boot_median(g, "rho"); nan_frac = g.rho.isna().mean()
        medg = np.nanmedian(g.rho.fillna(0)); 
        cm = lambda col: cluster_mean(g, col)
        gain = cm("gain_fc"); pg = signflip_p(gain.values)
        psp = signflip_p((cm("gain_fc") - cm("gain_sp")).dropna().values)
        pbic = signflip_p((cm("gain_fc") - cm("gain_bic")).dropna().values)
        p_gain.append(pg); p_sp.append(psp); p_bic.append(pbic)
        n = N[N.family == f] if N is not None and len(N) else pd.DataFrame()
        if len(n):
            nmed, nlo, nhi = boot_median(n, "auc_nec")
            cmn = n.groupby("cluster")
            pn = signflip_p((cmn["auc_nec"].mean() - cmn["auc_S"].mean()).dropna().values)
        else: nmed = nlo = nhi = pn = np.nan
        p_nec.append(pn)
        rows.append(dict(family=f, n=len(g), rho_med=med, rho_lo=lo, rho_hi=hi, rho_nan_frac=nan_frac, rho_med_nan0=medg,
                         rho_ap_med=np.nanmedian(g.rho_ap), auc_fc=g.auc_fc.mean(), auc_sp=g.auc_sp.mean(), auc_bic=g.auc_bic.mean(), auc_spt=g.auc_spt.mean(),
                         auc_rnd=g.auc_rnd.mean(), auc_orc=g.auc_orc.mean(),
                         gain_fc=g.gain_fc.mean(), gain_sp=g.gain_sp.mean(), gain_bic=g.gain_bic.mean(), gain_spt=g.gain_spt.mean(),
                         regret_fc=g.regret_fc.mean(), regret_sp=g.regret_sp.mean(), regret_bic=g.regret_bic.mean(),
                         p_gain=pg, p_fc_gt_sp=psp, p_fc_gt_bic=pbic,
                         nec_med=nmed, nec_lo=nlo, nec_hi=nhi, nec_auc_mean=n.auc_nec.mean() if len(n) else np.nan, nec_auc_S_mean=n.auc_S.mean() if len(n) else np.nan,
                         nec_auc_G_med=np.nanmedian(n.auc_nec_G) if len(n) else np.nan, S_auc_G_med=np.nanmedian(n.auc_S_G) if len(n) else np.nan,
                         p_nec_gt_S=pn))
    T = pd.DataFrame(rows)
    T["p_gain_holm"] = holm(p_gain); T["p_sp_holm"], T["p_bic_holm"] = np.split(holm(p_sp + p_bic), 2)
    T["p_nec_holm"] = holm(np.nan_to_num(p_nec, nan=1.0))
    T["H1_pass"] = (T.rho_med >= 0.3) & (T.rho_lo > 0) & (T.gain_fc > 0) & (T.p_gain_holm < 0.05) & (T.rho_nan_frac <= 0.3)
    T["H2_pass"] = (T.gain_fc > T.gain_sp) & (T.gain_fc > T.gain_bic) & (T.p_sp_holm < 0.05) & (T.p_bic_holm < 0.05)
    T["H3_pass"] = (T.nec_lo > 0.5) & (T.nec_auc_mean > T.nec_auc_S_mean) & (T.p_nec_holm < 0.05)
    lines.append(T.round(3).T.to_string() + "\n")
    h1 = int(T.H1_pass.sum()); h2 = int(T.H2_pass.sum()); h3 = int(T.H3_pass.sum())
    lines.append(f"H1 families passing: {h1}/5 -> {'SUPPORTED' if h1 >= 3 else 'NEGATIVE'}; H2: {h2}/5 -> {'supported' if h2 >= 3 else 'not supported'}; H3: {h3}/5 -> {'supported' if h3 >= 3 else 'not supported'}\n")
    return T

lines = [f"# Gap B results ({phase} seeds)\n", "Generated by analyze.py; rules from PLAN.md. rho = Spearman(-val MSE, directed AUROC) over non-empty grid settings.\n"]
for w, label in (("synth", "Synthetic SVAR sanity"), ("rivers", "CausalRivers (primary)"), ("netsim", "netsim (secondary)")):
    R = load("results", w)
    if R is None: continue
    N = load("nec", w)
    D = per_dataset(R)
    D.to_csv(f"perdataset_{phase}_{w}.csv", index=False)
    errs = R[R.setting < 0]
    lines.append(f"\n## {label}\n")
    if len(errs): lines.append(f"fit errors: {len(errs)} (families: {errs.family.value_counts().to_dict()}), first: {errs.err.iloc[0]}\n")
    T = verdict_block(D, N, "all datasets", lines)
    if w == "synth":
        ok = all(T.set_index("family").loc[f, "rho_med"] >= 0.3 for f in ("lasso", "thrvar"))
        lines.append(f"Gate S (median rho >= 0.3 for lasso and thrvar): {'PASS' if ok else 'FAIL'}\n")
    if w == "rivers":
        for col in ("cell", "res"):
            tb = D.groupby([col, "family"]).rho.median().unstack().round(3)
            lines.append(f"\nmedian rho by {col}:\n" + tb.to_string() + "\n")
        tb = D.groupby(["cell", "family"])[["gain_fc", "gain_sp", "gain_bic"]].mean().round(3)
        lines.append("\nmean gains by cell and family (FC / SP / BIC):\n" + tb.to_string() + "\n")
    if w == "netsim":
        lines.append("\n(netsim: clusters = sims)\n")
open(f"results_{phase}.md", "w").write("\n".join(lines))
print("\n".join(lines))
