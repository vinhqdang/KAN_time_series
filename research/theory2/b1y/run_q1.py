"""Q1 simulation (PLAN.md). STATUS: written before any run; NOT executed on study seeds at the time of the plan commit.
One shard = one (study, family, p); resumable (one row appended per finished data set).

  OMP_NUM_THREADS=1 /usr/bin/python3 run_q1.py {main|ndose|rhodose} OUT.csv FAMILY P [--val]
--val uses the validation seeds (debugging and timing only; nothing from them is evidence).

Rows (long format, see analyse_y.py schema): study, family, p, seed, T, n, cap, arm, d, oa_full, oa_x1, shd_x1n, gap, specrad.
Methods: rel-SortnRegress only (order = ascending Alg. 1 count, ties random; SHD from B1's Alg. 2 on SHD arms). Time labels
hidden: columns are randomly permuted per data set (as in B1/b1x); scoring is in the permuted frame.
"""
import sys, os, csv, time
import numpy as np
import common_y as cy
from common_y import *
import seeds as sd_

MAIN_ARMS = ["G1", "G1rep", "G2", "G2s", "CF1_0.25", "CF1_0.5", "CF1_2", "CF1_4", "CF2_m1", "CF2_m3", "CF2_m10"]
SHD_ARMS = {"G1", "G2", "CF1_0.25", "CF1_4", "CF2_m1"}          # Alg. 2 + SHD_x1 only here (cost)
MAIN_T = (3, 5, 10)
SHD_T = (3, 5)
DOSE_FAMS = ("ring", "chain", "ER2")
DOSE_T = (3, 5)
DOSE_N = (200, 1000, 5000, 25000)
DOSE_CAPS = {0.8: 0, 0.5: 1, 0.95: 2}                         # cap -> rho_index in the data key
FIELDS = ["study", "family", "p", "seed", "T", "n", "cap", "arm", "d", "oa_full", "oa_x1", "shd_x1n", "gap", "specrad"]


def one(study, fam, p, s, T, arm, n, cap, shd):
    rng_g = sd_.graph_rng(fam, p, s)
    S = summary_graph(fam, p, rng_g)
    A, sd = weights_cap(S, rng_g, cap)
    rng = sd_.data_rng(fam, p, s, T, ARMS.index(arm), n, DOSE_CAPS.get(cap, 0))
    X = simulate_arm(arm, A, sd, T, n, rng)
    d = X.shape[1]; G = true_dag(S, T)
    perm = rng.permutation(d); Xp = X[:, perm]; Gp = G[np.ix_(perm, perm)]; sl = slice_of_cols(perm, p)
    order = order_from_scores(rel_counts(Xp), rng)
    row = dict(study=study, family=fam, p=p, seed=s, T=T, n=n, cap=cap, arm=arm, d=d,
               oa_full=order_accuracy(Gp, order), oa_x1=oa_x1(Gp, order, sl), shd_x1n=float("nan"),
               gap=srho_G1_exact(S, T) - srho_stationary(S), specrad=float(max(abs(np.linalg.eigvals(A)))))
    if shd:
        Mdir = regress_in_order(Xp, order)
        row["shd_x1n"] = shd_fn(Gp, Mdir, np.zeros_like(Mdir), exclude=mask_inside_first_slice(sl)) / max(int(Gp.sum()), 1)
    return row


from common import shd as shd_fn   # noqa: E402  (B1's SHD: each unordered pair counts 1 if its state differs)


def tasks(study, fam, p, val):
    seeds = sd_.SEEDS["q1_val" if val else ("q1_test" if study == "main" else "q1_dose")]
    for s in seeds:
        if study == "main":
            for T in MAIN_T:
                for arm in MAIN_ARMS:
                    yield (study, fam, p, s, T, arm, 1000, 0.8, arm in SHD_ARMS and T in SHD_T)
        elif study == "ndose":
            for T in DOSE_T:
                for n in DOSE_N:
                    for arm in ("G1", "G2s"):
                        yield (study, fam, p, s, T, arm, n, 0.8, False)
        else:
            for T in DOSE_T:
                for cap in (0.5, 0.95):
                    for arm in ("G1", "G2s"):
                        yield (study, fam, p, s, T, arm, 1000, cap, False)


def main():
    study, out, fam, p = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]); val = "--val" in sys.argv
    if study != "main" and fam not in DOSE_FAMS:
        raise SystemExit(f"{study} runs only for {DOSE_FAMS}")
    done = set()
    if os.path.exists(out):
        for r in csv.DictReader(open(out)):
            done.add((r["study"], r["family"], int(r["p"]), int(r["seed"]), int(r["T"]), r["arm"], int(r["n"]), float(r["cap"])))
    new = not os.path.exists(out)
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new: w.writeheader()
        for t in tasks(study, fam, p, val):
            key = (t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7])
            if key in done: continue
            w.writerow(one(*t)); f.flush()


if __name__ == "__main__":
    main()
