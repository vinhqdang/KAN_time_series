"""Q1c graph-only predictions (no data, no learner): oracle (threshold) order-accuracy gain G1 - G2s as a function of
n (eps = eps(n)) and of the spectral-radius cap, on the dose graphs (seeds 4200-4219), families ring, chain, ER2,
p in {5, 10}, T in {3, 5}. Reported relative to the exact s_rho gap (Thm 1.5 - Thm 1.9).
Run: OMP_NUM_THREADS=1 /usr/bin/python3 q1_dose_population.py > results/q1_dose_population.txt
"""
import numpy as np
import common_y as cy
from common_y import *
import seeds as sd_

FAMS = ("ring", "chain", "ER2")
NS = (200, 1000, 5000, 25000, 10 ** 6)
CAPS = (0.5, 0.8, 0.95)


def gain(S, A, sd, T, n):
    G = true_dag(S, T)
    C1 = window_cov(A, sd, T, np.diag(sd ** 2)); C2 = window_cov(A, sd, T, stationary_cov(A, sd))
    return oracle_oa(C1, G, n) - oracle_oa(C2, G, n)


def main():
    print("oracle gain / exact gap by n (cap 0.8) and by cap (n = 1000); mean over graphs and T in (3,5)")
    res = {}
    for fam in FAMS:
        for p in (5, 10):
            acc = {("n", n): [] for n in NS}; acc.update({("cap", c): [] for c in CAPS}); gaps = {c: [] for c in CAPS}
            for s in sd_.SEEDS["q1_dose"]:
                for cap in CAPS:
                    rng = sd_.graph_rng(fam, p, s)
                    S = summary_graph(fam, p, rng); A, sd = weights_cap(S, rng, cap)
                    for T in (3, 5):
                        gaps[cap].append(srho_G1_exact(S, T) - srho_stationary(S))
                        if cap == 0.8:
                            for n in NS: acc[("n", n)].append(gain(S, A, sd, T, n))
                        acc[("cap", cap)].append(gain(S, A, sd, T, 1000))
            g = np.mean(gaps[0.8])
            print(f"{fam:6s} p={p:2d} exact gap {g:.3f} | " + " ".join(f"n={n}:{np.mean(acc[('n', n)]) / g:.2f}" for n in NS) + " | " +
                  " ".join(f"cap={c}:{np.mean(acc[('cap', c)]) / np.mean(gaps[c]):.2f}" for c in CAPS))


if __name__ == "__main__":
    main()
