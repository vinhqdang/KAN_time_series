"""B'3 (post hoc; population computation, no new data): detectability oracle for the B1 mechanism.

For every B1 test graph and parameter set (same seeds and RNG streams as b1/run_b1.py, regime R-het) and T in
{3, 5, 10}, compute the exact population covariance of the G1 window (x_1 = e_1) and of the G2 window (burn-in
B = 100 from x_1 = e_1, i.e. exactly what G2 simulates), then:
  * graph check: the population relatives (|rho| > 1e-20) reproduce the exact graph s_rho (Thm 1.5 / 1.9);
  * oracle-threshold sort: counts c_i = #{j : |rho_ij| > eps}, eps the Alg. 1 threshold at n = 1000, alpha = 0.05;
    predicted OA = edge-sortability of c (ties 1/2) = the order accuracy rel-SnR would have if each sample
    correlation were replaced by its population value;
  * detection-probability oracle: P_ij = P(|r_ij| > eps) under Fisher z, atanh(r) ~ N(atanh(rho), 1/(n - 3));
    for an edge (i, j) the count difference c_j - c_i = sum_{k != i, j} (I_jk - I_ik) is treated as a sum of
    independent Bernoulli differences (the shared I_ij cancels); predicted OA = P(c_i < c_j) + P(c_i = c_j)/2,
    averaged over edges (expected order accuracy of rel-SnR up to the dependence between indicators).
Compare the G1 - G2 predictions with the observed rel-SnR gain in b1/results/b1_test.csv.gz (cell and unit level).

Run: OMP_NUM_THREADS=1 /usr/bin/python3 oracle_b3.py results
"""
import sys, os, json
import numpy as np
import pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "b1"))
from common import FAMILIES, NONSC_FAMILIES, SC_FAMILIES, summary_graph, weights, true_dag, srho_G1_exact, \
    srho_stationary, sortability, B_BURN

N = 1000
ALPHA = 0.05
TCRIT = stats.t.ppf(1 - ALPHA / 2, N - 2)
EPS = TCRIT / np.sqrt(N - 2 + TCRIT ** 2)
out_lines = []


def P(*a):
    s = " ".join(str(x) for x in a); out_lines.append(s); print(s, flush=True)


def window_cov(A, sd, T, gen):
    """Covariance of (x_1, ..., x_T) stacked as column t*p+v. Step x_{t+1} = A^T x_t + e_{t+1}."""
    p = A.shape[0]; D = np.diag(sd ** 2)
    S0 = D.copy()
    if gen == "G2":
        for _ in range(B_BURN):
            S0 = A.T @ S0 @ A + D
    Sig = [S0]
    for t in range(1, T):
        Sig.append(A.T @ Sig[-1] @ A + D)
    C = np.zeros((T * p, T * p))
    for s in range(T):
        Ak = np.eye(p)
        for t in range(s, T):
            blk = Sig[s] @ Ak                        # Cov(x_s, x_t) = Sigma_s A^{t-s}
            C[s * p:(s + 1) * p, t * p:(t + 1) * p] = blk
            C[t * p:(t + 1) * p, s * p:(s + 1) * p] = blk.T
            Ak = Ak @ A
    return C


def corr(C):
    s = np.sqrt(np.diag(C)); return C / np.outer(s, s)


def poibin(ps):
    """pmf of a sum of independent Bernoulli differences with probabilities given as list of (p_plus, p_minus):
    returns dict offset -> pmf array on support [-K, K]."""
    K = len(ps); pmf = np.zeros(2 * K + 1); pmf[K] = 1.0
    for pp, pm in ps:
        new = pmf * (1 - pp) * (1 - pm) + pmf * pp * pm     # 0 change (both or neither)
        new[1:] += pmf[:-1] * pp * (1 - pm)                 # +1
        new[:-1] += pmf[1:] * pm * (1 - pp)                 # -1
        pmf = new
    return pmf, K


def detect_prob(R):
    z = np.arctanh(np.clip(np.abs(R), 0, 1 - 1e-15)); se = 1 / np.sqrt(N - 3); ze = np.arctanh(EPS)
    Pd = stats.norm.sf((ze - z) / se) + stats.norm.cdf((-ze - z) / se)
    np.fill_diagonal(Pd, 1.0)
    return Pd


def predicted_oa_prob(Pd, G):
    E = np.argwhere(G); d = Pd.shape[0]; vals = []
    for i, j in E:
        ks = [k for k in range(d) if k != i and k != j]
        # c_j - c_i = sum_k (I_jk - I_ik)
        pmf, K = poibin(list(zip(Pd[j, ks], Pd[i, ks])))
        vals.append(pmf[K + 1:].sum() + 0.5 * pmf[K])
    return float(np.mean(vals))


def main():
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    rows = []
    for fam in FAMILIES:
        for p in (5, 10):
            for s in range(1000, 1030):
                rng_g = np.random.default_rng([FAMILIES.index(fam), p, s])
                S = summary_graph(fam, p, rng_g); A, sd = weights(S, rng_g, var_regime="het")
                for T in (3, 5, 10):
                    G = true_dag(S, T)
                    row = dict(family=fam, p=p, seed=s, T=T, s_exact_G1=srho_G1_exact(S, T), s_exact_stat=srho_stationary(S))
                    for gen in ("G1", "G2"):
                        R = corr(window_cov(A, sd, T, gen))
                        absR = np.abs(R)
                        row[f"graph_srho_{gen}"] = sortability((absR > 1e-20).sum(0).astype(float), G)
                        row[f"orc_thr_{gen}"] = sortability((absR > EPS).sum(0).astype(float), G)
                        Pd = detect_prob(R)
                        row[f"orc_prob_{gen}"] = predicted_oa_prob(Pd, G)
                        Gx = G.copy(); Gx[:p, :] = False        # edges not leaving the first window slice
                        row[f"graph_srho_x1_{gen}"] = sortability((absR > 1e-20).sum(0).astype(float), Gx)
                        row[f"orc_prob_x1_{gen}"] = predicted_oa_prob(Pd, Gx)
                        off = ~np.eye(len(R), dtype=bool); rel = absR > 1e-20
                        row[f"pop_recall_{gen}"] = float(((absR > EPS) & rel & off).sum() / max((rel & off).sum(), 1))
                        row[f"exp_recall_{gen}"] = float((Pd * (rel & off)).sum() / max((rel & off).sum(), 1))
                    rows.append(row)
        P(f"done {fam}")
    O = pd.DataFrame(rows)
    O.to_csv(f"{out}/oracle_b3.csv.gz", index=False)
    analyse(O, out)


def analyse(O, out):
    b1 = pd.read_csv(os.path.join(HERE, "..", "b1", "results", "b1_test.csv.gz"))
    b1 = b1[b1.regime == "het"]
    key = ["family", "p", "seed", "T"]
    a = b1[b1.gen == "G1"].set_index(key).sort_index(); b = b1[b1.gen == "G2"].set_index(key).loc[a.index]
    O = O.set_index(key).loc[a.index]
    D = pd.DataFrame(index=a.index)
    D["obs"] = a.M1_oa - b.M1_oa
    D["obs_rel"] = a.est_rel_sort - b.est_rel_sort
    D["gap"] = O.s_exact_G1 - O.s_exact_stat
    D["thr"] = O.orc_thr_G1 - O.orc_thr_G2
    D["prob"] = O.orc_prob_G1 - O.orc_prob_G2
    for g in ("G1", "G2"):
        D[f"obsOA_{g}"] = (a if g == "G1" else b).M1_oa
        D[f"prob_{g}"] = O[f"orc_prob_{g}"]; D[f"thr_{g}"] = O[f"orc_thr_{g}"]
        D[f"recall_{g}"] = O[f"exp_recall_{g}"]
    chk = max(float(np.abs(O.graph_srho_G1 - O.s_exact_G1).max()), float(np.abs(O.graph_srho_G2 - O.s_exact_stat).max()))
    P(f"B'3 detectability oracle (post hoc, population computation). eps = {EPS:.4f} (n = {N}, alpha = {ALPHA}).")
    P(f"check: population relatives (|rho| > 1e-20) reproduce the exact graph s_rho: max |diff| = {chk:.2e}")
    D = D.reset_index()
    C = D.groupby(["family", "p", "T"]).mean(numeric_only=True).reset_index()
    res = {"check_max_diff": chk}
    P("\n== Levels: predicted vs observed rel-SnR OA, mean over p, T, graphs ==")
    P("  family  | exact s G1/G2 | oracle-thr G1/G2 | oracle-prob G1/G2 | observed OA G1/G2 | exp. recall G1/G2")
    for fam in FAMILIES:
        x = D[D.family == fam]; xo = O[O.index.get_level_values(0) == fam]
        P(f"  {fam:7s} | {xo.s_exact_G1.mean():.3f}/{xo.s_exact_stat.mean():.3f} | {x.thr_G1.mean():.3f}/{x.thr_G2.mean():.3f} | "
          f"{x.prob_G1.mean():.3f}/{x.prob_G2.mean():.3f} | {x.obsOA_G1.mean():.3f}/{x.obsOA_G2.mean():.3f} | "
          f"{x.recall_G1.mean():.2f}/{x.recall_G2.mean():.2f}")
    P("\n== Which predictor explains the observed G1 - G2 gain? ==")
    for lvl, X in (("cell (42)", C), ("unit (1260 graph x T)", D)):
        r = {}
        for pred in ("gap", "thr", "prob"):
            sp = stats.spearmanr(X[pred], X.obs).statistic; pe = stats.pearsonr(X[pred], X.obs).statistic
            sl, ic = np.polyfit(X[pred], X.obs, 1)
            mae = float(np.abs(X.obs - X[pred]).mean())
            r2_id = 1 - float(((X.obs - X[pred]) ** 2).sum() / ((X.obs - X.obs.mean()) ** 2).sum())
            r[pred] = dict(spearman=float(sp), pearson=float(pe), r2_linear=float(pe ** 2), slope=float(sl),
                           intercept=float(ic), mae=mae, r2_identity=r2_id)
            P(f"  {lvl:22s} {pred:5s}: Spearman {sp:.3f}  Pearson {pe:.3f} (R^2 {pe**2:.3f})  OLS obs = {sl:.2f}*pred {ic:+.3f}  "
              f"MAE(obs - pred) {mae:.3f}  R^2 about identity {r2_id:.3f}")
        # partial: does prob add to gap and vice versa?
        Z = np.column_stack([np.ones(len(X)), X.gap, X.prob])
        beta, *_ = np.linalg.lstsq(Z, X.obs, rcond=None)
        rss_full = float(((X.obs - Z @ beta) ** 2).sum())
        rss_gap = float(((X.obs - np.polyval(np.polyfit(X.gap, X.obs, 1), X.gap)) ** 2).sum())
        rss_prob = float(((X.obs - np.polyval(np.polyfit(X.prob, X.obs, 1), X.prob)) ** 2).sum())
        tss = float(((X.obs - X.obs.mean()) ** 2).sum())
        r["joint"] = dict(beta=beta.tolist(), R2_full=1 - rss_full / tss, R2_gap=1 - rss_gap / tss, R2_prob=1 - rss_prob / tss)
        P(f"  {lvl:22s} joint OLS obs ~ gap + prob: coef gap {beta[1]:+.3f}, prob {beta[2]:+.3f}; R^2 full {1 - rss_full / tss:.3f}, "
          f"gap only {1 - rss_gap / tss:.3f}, prob only {1 - rss_prob / tss:.3f}")
        res[lvl] = r
    # graph-cluster bootstrap for the unit-level R^2 difference (prob minus gap)
    g = D.family + "|" + D.p.astype(str) + "|" + D.seed.astype(str)
    gs = g.unique(); idx = {k: np.flatnonzero(g.values == k) for k in gs}
    rng = np.random.default_rng(0); diffs = []
    for _ in range(2000):
        pick = np.concatenate([idx[k] for k in rng.choice(gs, len(gs))])
        X = D.iloc[pick]
        diffs.append(stats.pearsonr(X.prob, X.obs).statistic ** 2 - stats.pearsonr(X.gap, X.obs).statistic ** 2)
    res["unit_R2_prob_minus_gap_ci"] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    P(f"  unit-level R^2(prob) - R^2(gap): graph-cluster bootstrap 95% CI [{res['unit_R2_prob_minus_gap_ci'][0]:+.3f}, "
      f"{res['unit_R2_prob_minus_gap_ci'][1]:+.3f}]")
    P("\n== Per cell (SC families and selected), observed gain vs predictions ==")
    for _, r in C.iterrows():
        if r.family in SC_FAMILIES or (r.family in ("chain", "intree") and r["T"] == 10):
            P(f"  {r.family:7s} p={r.p:2d} T={r['T']:2d}: observed {r.obs:+.3f} | exact gap {r.gap:+.3f} | oracle-thr {r.thr:+.3f} | "
              f"oracle-prob {r.prob:+.3f} | exp. recall G1/G2 {r.recall_G1:.2f}/{r.recall_G2:.2f}")
    ring = C[(C.family == "ring") & (C.p == 10) & (C["T"] == 10)].iloc[0]
    res["ring_10_10"] = dict(obs=ring.obs, gap=ring.gap, thr=ring.thr, prob=ring.prob, obsG1=ring.obsOA_G1,
                             obsG2=ring.obsOA_G2, probG1=ring.prob_G1, probG2=ring.prob_G2,
                             recallG1=ring.recall_G1, recallG2=ring.recall_G2)
    # edges not leaving the first window slice (links to B'2 item 5)
    O2 = O.copy()
    xs = pd.DataFrame(dict(gap_x1=O2.graph_srho_x1_G1 - O2.graph_srho_x1_G2, prob_x1=O2.orc_prob_x1_G1 - O2.orc_prob_x1_G2,
                           gap=O2.s_exact_G1 - O2.s_exact_stat, prob=O2.orc_prob_G1 - O2.orc_prob_G2))
    xs["family"] = O2.index.get_level_values(0)
    P("\n== Exact gap and oracle prediction on all edges vs edges not leaving slice 1 (mean over p, T, graphs) ==")
    for fam in FAMILIES:
        x = xs[xs.family == fam]
        P(f"  {fam:7s} exact gap all {x.gap.mean():+.3f} / excl. slice-1 edges {x.gap_x1.mean():+.3f} | "
          f"oracle-prob all {x.prob.mean():+.3f} / excl. {x.prob_x1.mean():+.3f}")
    x = xs[xs.family.isin(NONSC_FAMILIES)]
    P(f"  non-SC  exact gap all {x.gap.mean():+.3f} / excl. {x.gap_x1.mean():+.3f} | oracle-prob all {x.prob.mean():+.3f} / excl. {x.prob_x1.mean():+.3f}")
    res["x1"] = {fam: dict(gap=float(xs[xs.family == fam].gap.mean()), gap_x1=float(xs[xs.family == fam].gap_x1.mean()),
                           prob=float(xs[xs.family == fam].prob.mean()), prob_x1=float(xs[xs.family == fam].prob_x1.mean()))
                 for fam in FAMILIES}
    res["cells"] = C.to_dict(orient="records")
    open(f"{out}/oracle_b3.txt", "w").write("\n".join(out_lines) + "\n")
    json.dump(res, open(f"{out}/oracle_b3.json", "w"), indent=1, default=float)
    figure(C, out)


def figure(C, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = dict(symER="#1b9e77", ring="#66a61e", chain="#d95f02", outtree="#7570b3", intree="#e7298a",
                ER1="#a6761d", ER2="#666666")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for k, (pred, lab) in enumerate((("gap", "exact graph gap (Thm 1.5 vs 1.9)"),
                                     ("prob", "detectability-oracle prediction"))):
        for fam in FAMILIES:
            c = C[C.family == fam]
            ax[k].scatter(c[pred], c.obs, s=12 + 3 * c["T"], color=cols[fam], label=fam,
                          marker="s" if fam in SC_FAMILIES else "o", alpha=0.85)
        ax[k].plot([0, 0.52], [0, 0.52], "k:", lw=0.8); ax[k].axhline(0, color="k", lw=0.5)
        ax[k].set_xlabel(lab + ", G1 $-$ G2")
        sp = stats.spearmanr(C[pred], C.obs).statistic; pe = stats.pearsonr(C[pred], C.obs).statistic
        ax[k].set_title(f"Spearman {sp:.2f}, Pearson {pe:.2f}", fontsize=9)
    ax[0].set_ylabel("observed rel-SnR OA gain, G1 $-$ G2"); ax[0].legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(f"{out}/fig_oracle.png", dpi=160); fig.savefig(f"{out}/fig_oracle.pdf")


if __name__ == "__main__":
    main()
