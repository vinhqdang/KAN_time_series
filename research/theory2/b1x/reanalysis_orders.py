"""B'2 (post hoc): order accuracy excluding the edges that leave the first window slice.

True window edges always go from slice t to slice t+1, so no true edge lies inside the first slice; "excluding
first-slice pairs" therefore means here: drop the true edges whose tail is in the first window slice (the slice whose
nodes are latently confounded under G2). Orders of rel-, R2- and Var-SortnRegress are recomputed exactly as in the
committed runs (same RNG streams; Alg. 2 is not needed for the order) and checked against the committed M*_oa.

Run: OMP_NUM_THREADS=1 /usr/bin/python3 reanalysis_orders.py results/orders_x1.csv.gz
"""
import sys, os, gzip, csv
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "b1")); sys.path.insert(0, os.path.join(HERE, "..", "b1nl"))
from common import (FAMILIES, summary_graph, weights, simulate, true_dag, rel_counts, r2_scores, var_scores,
                    order_from_scores, order_accuracy)


def oa_parts(Gp, order, slice_of):
    pos = np.empty(len(order), dtype=int); pos[order] = np.arange(len(order))
    E = np.argwhere(Gp)
    ok = pos[E[:, 0]] < pos[E[:, 1]]
    keep = slice_of[E[:, 0]] > 0
    return float(ok.mean()), float(ok[keep].mean())


def b1_rows():
    GENS = ["G1", "G2", "G2s"]; REGIMES = ["het", "eq"]
    for fam in FAMILIES:
        for p in (5, 10):
            for s in range(1000, 1030):
                for T in (3, 5, 10):
                    for gen in ("G1", "G2"):
                        reg = "het"
                        rng_g = np.random.default_rng([FAMILIES.index(fam), p, s])
                        S = summary_graph(fam, p, rng_g)
                        A, sd = weights(S, rng_g, var_regime=reg)
                        rng = np.random.default_rng([FAMILIES.index(fam), p, s, T, GENS.index(gen), REGIMES.index(reg), 7])
                        X = simulate(A, sd, T, 1000, gen, rng)
                        d = X.shape[1]; G = true_dag(S, T)
                        perm = rng.permutation(d); Xp = X[:, perm]; Gp = G[np.ix_(perm, perm)]
                        slice_of = perm // p
                        rel = rel_counts(Xp); r2 = r2_scores(Xp); var = var_scores(Xp)
                        o = dict(M1=order_from_scores(rel, rng), M2=order_from_scores(r2, rng), M3=order_from_scores(var, rng))
                        row = dict(exp="B1", cfg="LG", family=fam, p=p, seed=s, T=T, gen=gen)
                        for k, oo in o.items():
                            row[f"{k}_oa"], row[f"{k}_oa_x1"] = oa_parts(Gp, oo, slice_of)
                        yield row


def nl_rows():
    from common_nl import params, simulate_nl, DATA_KEY
    GENS = ["G1", "G2", "G2L"]
    seeds = dict(NL=range(2000, 2025), SGN=range(2000, 2025), SGNU=range(2000, 2010))
    for cfg in ("NL", "SGN", "SGNU"):
        for fam in FAMILIES:
            for p in (5, 10):
                for s in seeds[cfg]:
                    P = params(cfg, fam, p, s); S = P["S"]
                    for T in (3, 5, 10):
                        for gen in ("G1", "G2"):
                            rng = np.random.default_rng([DATA_KEY[cfg], FAMILIES.index(fam), p, s, T, GENS.index(gen), 11])
                            X = simulate_nl(P, T, 1000, gen, rng)
                            d = X.shape[1]; G = true_dag(S, T)
                            perm = rng.permutation(d); Xp = X[:, perm]; Gp = G[np.ix_(perm, perm)]
                            slice_of = perm // p
                            rel = rel_counts(Xp); r2 = r2_scores(Xp); var = var_scores(Xp)
                            o = dict(M1=order_from_scores(rel, rng), M2=order_from_scores(r2, rng), M3=order_from_scores(var, rng))
                            row = dict(exp="B1NL", cfg=cfg, family=fam, p=p, seed=s, T=T, gen=gen)
                            for k, oo in o.items():
                                row[f"{k}_oa"], row[f"{k}_oa_x1"] = oa_parts(Gp, oo, slice_of)
                            yield row


def main():
    rows = list(b1_rows()) + list(nl_rows())
    pd.DataFrame(rows).to_csv(sys.argv[1], index=False)


if __name__ == "__main__":
    main()
