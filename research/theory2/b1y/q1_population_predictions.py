"""Q1 graph-only predictions (no data drawn, no learner run): for the Q1 test graphs (seeds 4000-4029) compute
  * exact s_rho gap (Thm 1.5 minus Thm 1.9),
  * the threshold-oracle order accuracy (population correlations, eps at n = 1000) of every arm,
  * lam_star and the number of pairs whose detectability (|rho| > eps) differs between CF2_m1 and G1.
Oracle retention of an arm X is (O(X) - O(G2)) / (O(G1) - O(G2)); it is reported so that the measured retentions in the
test run can be compared with a prediction made before any data were drawn.
Run: OMP_NUM_THREADS=1 /usr/bin/python3 q1_population_predictions.py > results/q1_population_predictions.txt
"""
import numpy as np, json
import common_y as cy
from common_y import *
import seeds as sd_

POP_ARMS = ["G1", "G2s", "CF1_0.25", "CF1_0.5", "CF1_2", "CF1_4", "CF2_m1", "CF2_m3", "CF2_m10"]


def main():
    rows = []
    eps = cy.eps_threshold()
    for fam in FAMILIES:
        for p in (5, 10):
            for s in sd_.SEEDS["q1_test"]:
                rng = sd_.graph_rng(fam, p, s)
                S = summary_graph(fam, p, rng); A, sd = weights(S, rng, "het")
                for T in (3, 5, 10):
                    G = true_dag(S, T)
                    o = {arm: oracle_oa(window_cov(A, sd, T, cy.start_cov(arm, A, sd, T)), G) for arm in POP_ARMS}
                    lam, degen = cy.lam_star(A, sd, T)
                    C1 = window_cov(A, sd, T, np.diag(sd ** 2))
                    C2 = window_cov(A, sd, T, cy.start_cov("CF2_m1", A, sd, T))
                    flips = int((((np.abs(corr_of(C1)) > eps) != (np.abs(corr_of(C2)) > eps)).sum()) // 2)
                    rows.append(dict(family=fam, p=p, seed=s, T=T, gap=srho_G1_exact(S, T) - srho_stationary(S),
                                     lam=lam, flips=flips, **{f"O_{a}": v for a, v in o.items()}))
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv("results/q1_population_predictions.csv.gz", index=False)
    nonsc = df[df.family.isin(NONSC_FAMILIES)]
    print("graph-only predictions, Q1 test graphs (seeds 4000-4029), non-SC families pooled (", nonsc.groupby(['family','p','seed']).ngroups, "graphs )")
    g = nonsc.groupby(["family", "p", "seed"]).mean(numeric_only=True)            # graph-level: mean over T
    gain = (g.O_G1 - g.O_G2s).mean()
    print(f"exact s_rho gap (mean over T,graphs): {g.gap.mean():.3f}; oracle OA gain G1-G2s: {gain:.3f}")
    for a in POP_ARMS[2:]:
        r = (g[f"O_{a}"] - g.O_G2s).mean() / gain
        print(f"oracle retention {a:9s}: {r:6.3f}   (mean oracle OA {g[f'O_{a}'].mean():.3f}; G1 {g.O_G1.mean():.3f}; G2s {g.O_G2s.mean():.3f})")
    print(f"lam_star: median {df.lam.median():.2e}, min {df.lam.min():.2e}, max {df.lam.max():.2e}; share lam_star < 1e-6 (grid floor): {(df.lam < 1e-6).mean():.3f}")
    print(f"detectability flips between CF2_m1 and G1 (pairs): mean {df.flips.mean():.2f}, 95th pct {df.flips.quantile(.95):.0f}, max {df.flips.max()}")
    for fam in FAMILIES:
        h = df[df.family == fam].groupby(["p", "seed"]).mean(numeric_only=True)
        og = max((h["O_G1"] - h["O_G2s"]).mean(), 1e-9)
        r = lambda a: (h[f"O_{a}"] - h["O_G2s"]).mean() / og
        print(f"  {fam:8s} gap {h.gap.mean():.3f} oracleGain {og:.3f} retCF2_m1 {r('CF2_m1'):.3f} retCF1_4 {r('CF1_4'):.3f} retCF1_0.25 {r('CF1_0.25'):.3f}")


if __name__ == "__main__":
    main()
