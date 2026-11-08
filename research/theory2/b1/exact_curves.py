"""B1 exact (population, graph-only) computations. No data, no algorithm.

  predict : exact s_rho of the generator ground truth for every test summary graph (G1: Thm 1.3/1.5;
            G2, G2s: Thm 1.9 stationary value), per family, p, T, and the predicted G1-G2 gap.
  curves  : exact s_rho(G_T) for T = 2..20 per family (mean over test graphs) + limit + stationary; figure.
  delta   : check of the sharpened bound 0 <= s(G_T) - L <= min(Delta, T-1)|E_o| / (2(T-1)|E_S|) by ENUMERATION
            (bitset relatives, independent of the theorem) on random graphs; and the burn-in sharpening
            (B >= Delta suffices for Thm 1.9; B = Delta - 1 does not in general).
Run: OMP_NUM_THREADS=1 /usr/bin/python3 exact_curves.py {predict|curves|delta} OUTDIR
"""
import sys, json
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *

TEST_SEEDS = list(range(1000, 1030))
VAL_SEEDS = list(range(9000, 9005))


def graph_for(family, p, seed):
    rng = np.random.default_rng([FAMILIES.index(family), p, seed])
    S = summary_graph(family, p, rng)
    return S, rng


def predict(out):
    rows = []
    for fam in FAMILIES:
        for p in (5, 10):
            for s in TEST_SEEDS:
                S, _ = graph_for(fam, p, s)
                st = srho_stationary(S); L = srho_limit(S); D = delta_max(S)
                for T in (3, 5, 10):
                    g1 = srho_G1_exact(S, T)
                    rows.append(dict(family=fam, p=p, seed=s, T=T, s_G1=g1, s_G2=st, L=L, Delta=D, gap=g1 - st))
    json.dump(rows, open(f"{out}/exact_predictions.json", "w"))
    lines = ["family  p   T   mean s_G1  mean s_G2  mean gap  min gap  max gap  mean Delta"]
    for fam in FAMILIES:
        for p in (5, 10):
            for T in (3, 5, 10):
                r = [x for x in rows if x["family"] == fam and x["p"] == p and x["T"] == T]
                g = np.array([x["gap"] for x in r])
                lines.append(f"{fam:7s} {p:2d} {T:3d}   {np.mean([x['s_G1'] for x in r]):.4f}     "
                             f"{np.mean([x['s_G2'] for x in r]):.4f}    {g.mean():.4f}   {g.min():.4f}   {g.max():.4f}   "
                             f"{np.mean([x['Delta'] for x in r]):.2f}")
    open(f"{out}/exact_predictions.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def curves(out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ts = list(range(2, 21))
    res = {}
    for fam in FAMILIES:
        for p in (5, 10):
            G1 = np.array([[srho_G1_exact(graph_for(fam, p, s)[0], T) for T in Ts] for s in TEST_SEEDS])
            st = np.array([srho_stationary(graph_for(fam, p, s)[0]) for s in TEST_SEEDS])
            L = np.array([srho_limit(graph_for(fam, p, s)[0]) for s in TEST_SEEDS])
            res[f"{fam}_{p}"] = dict(T=Ts, G1_mean=G1.mean(0).tolist(), G1_min=G1.min(0).tolist(),
                                    G1_max=G1.max(0).tolist(), stat_mean=float(st.mean()), L_mean=float(L.mean()))
    json.dump(res, open(f"{out}/exact_curves.json", "w"), indent=1)
    lines = ["exact s_rho(G_T), mean over the 30 test summary graphs; last two columns: limit L and stationary S"]
    lines.append("family_p  " + " ".join(f"T={T:<4d}" for T in Ts) + "  L       S")
    for k, v in res.items():
        lines.append(f"{k:10s}" + " ".join(f"{x:.4f}" for x in v["G1_mean"]) + f"  {v['L_mean']:.4f}  {v['stat_mean']:.4f}")
    open(f"{out}/exact_curves.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    cols = dict(symER="#1b9e77", ring="#66a61e", chain="#d95f02", outtree="#7570b3", intree="#e7298a",
                ER1="#a6761d", ER2="#666666")
    for ax, p in zip(axes, (5, 10)):
        for fam in FAMILIES:
            v = res[f"{fam}_{p}"]
            ax.plot(Ts, v["G1_mean"], "-", color=cols[fam], label=fam, lw=1.6)
            ax.plot([Ts[0], Ts[-1]], [v["stat_mean"]] * 2, ":", color=cols[fam], lw=1.2)
        ax.axvline(p, color="k", lw=0.5, alpha=0.4)
        ax.set_title(f"p = {p}"); ax.set_xlabel("T (slices)"); ax.set_xticks([2, 5, 10, 15, 20])
    axes[0].set_ylabel(r"exact $s_\rho$")
    axes[1].legend(fontsize=7, ncol=2, frameon=False)
    fig.suptitle(r"solid: fixed start $G_T$ (Thm 1.5 exact); dotted: stationary window (Thm 1.9)", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{out}/fig_srho_vs_T.png", dpi=160); fig.savefig(f"{out}/fig_srho_vs_T.pdf")


def delta(out):
    """Enumeration-based check on random summary graphs (with self-loops)."""
    rng = np.random.default_rng(424242)
    n_graph = 0; n_case = 0; viol_delta = []; viol_lower = []; tight_delta = 0; strict_better = 0
    burn_ok = 0; burn_fail_dm1 = 0; burn_cases = 0
    for rep in range(3000):
        p = int(rng.integers(2, 9))
        kind = rng.integers(0, 3)
        if kind == 0:                       # directed ER with random density
            q = rng.uniform(0.05, 0.6); S = rng.random((p, p)) < q
        elif kind == 1:                     # sparse ER near c = 1..2
            S = rng.random((p, p)) < rng.uniform(1, 2) / p
        else:                               # random family member
            S = summary_graph(FAMILIES[rng.integers(0, len(FAMILIES))], max(p, 3), rng); p = S.shape[0]
        S = S.copy(); np.fill_diagonal(S, True)
        ES, Ex, Eo, Eup = edge_counts(S); L = 0.5 + Ex / (2 * ES); D = delta_max(S)
        n_graph += 1
        for T in (2, 3, 4, 5, 7, 10, 15):
            G = true_dag(S, T)
            s = sortability(rel_window_enum(S, T, 0).astype(float), G)
            exc = s - L
            bD = min(D, T - 1) * Eo / (2 * (T - 1) * ES)
            bP = min(p - 1, T - 1) * Eo / (2 * (T - 1) * ES)
            n_case += 1
            if exc < -1e-12: viol_lower.append((p, T, S.astype(int).tolist()))
            if exc > bD + 1e-12: viol_delta.append((p, T, s, L, bD, S.astype(int).tolist()))
            if abs(exc - bD) < 1e-12 and bD > 0: tight_delta += 1
            if bD < bP - 1e-12: strict_better += 1
        if p <= 6 and rep % 3 == 0:          # burn-in sharpening
            T = 3; G = true_dag(S, T); st = 0.5 + Eup / (2 * ES)
            burn_cases += 1
            if abs(sortability(rel_window_enum(S, T, D).astype(float), G) - st) < 1e-12: burn_ok += 1
            if D >= 1 and abs(sortability(rel_window_enum(S, T, D - 1).astype(float), G) - st) > 1e-12: burn_fail_dm1 += 1
    res = dict(graphs=n_graph, cases=n_case, lower_violations=len(viol_lower), delta_violations=len(viol_delta),
               delta_bound_attained=tight_delta, cases_delta_strictly_sharper_than_p_minus_1=strict_better,
               burn_cases=burn_cases, burn_B_eq_Delta_gives_stationary=burn_ok,
               burn_B_eq_Delta_minus_1_differs=burn_fail_dm1,
               first_violation=(viol_delta[0] if viol_delta else None))
    json.dump(res, open(f"{out}/delta_check.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    dict(predict=predict, curves=curves, delta=delta)[sys.argv[1]](sys.argv[2])
