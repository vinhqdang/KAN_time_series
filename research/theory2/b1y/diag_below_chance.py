"""Q2 step-0 diagnosis (PLAN.md section 4.2; VALIDATION seeds 9810-9812 only; produces no verdict).
B1x found ancestral order accuracy of standardised DAGMA and GOLEM at 0.31-0.42 (below 1/2). Here: on validation data sets, standardised
DAGMA (package defaults) and standardised GOLEM-EV/NV (p = 5 cells only), generators G1 and G2: shares of true edges (i, j) that the estimate
orders forwards (i an ancestor of j and not vice versa), backwards, or incomparable/cyclic. If backwards exceeds forwards the below-chance level is a
genuine reversed orientation and not a scoring error (the unit tests show truth -> 1 and transpose -> 0).
  python diag_below_chance.py OUT.csv
"""
import sys, csv
import numpy as np
import common_y as cy, common_y2 as c2, seeds as sd_, methods_y as my
from methods_x import golem as golem_fn

CELLS = [(5, 3), (5, 5), (10, 3), (10, 5)]
rows = []
for fam in ("chain", "intree", "ring", "ER2"):
    for (p, T) in CELLS:
        for s in sd_.SEEDS["q2_val"]:
            rng_g = sd_.graph_rng(fam, p, s); S = cy.summary_graph(fam, p, rng_g); A, sd = cy.weights_cap(S, rng_g, 0.8); G = cy.true_dag(S, T)
            for arm in ("G1", "G2"):
                rng = sd_.data_rng(fam, p, s, T, cy.ARMS.index(arm), 1000, 30)
                X = cy.simulate_arm(arm, A, sd, T, 1000, rng); d = X.shape[1]
                perm = rng.permutation(d); Xp = X[:, perm]; Gp = G[np.ix_(perm, perm)]; sl = cy.slice_of_cols(perm, p)
                Xs = (Xp - Xp.mean(0)) / Xp.std(0); z = np.zeros((d, d), bool)
                outs = {"DAGMA_std": my.dagma_hp(Xs)[0]}
                if p == 5:
                    Bev, Bnv = golem_fn(Xs); outs["GOLEMEV_std"] = Bev != 0; outs["GOLEMNV_std"] = Bnv != 0
                for name, M in outs.items():
                    e = c2.eval_graph(Gp, M, z, sl)
                    rows.append(dict(method=name, family=fam, p=p, T=T, seed=s, arm=arm, fwd=e["fwd"], bwd=e["bwd"], incomparable=1 - e["fwd"] - e["bwd"],
                                     aoa=e["aoa"], nedge=e["nedge"], dag=e["dag"]))
        print(fam, "done", flush=True)
with open(sys.argv[1], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
import pandas as pd
d = pd.DataFrame(rows)
print(d.groupby(["method", "arm"])[["fwd", "bwd", "incomparable", "aoa", "nedge"]].mean().round(3).to_string())
