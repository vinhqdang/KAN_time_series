"""B1 simulation: generators G1 / G2 / G2s, methods with time labels hidden, metrics (PLAN.md Sections 1.3-1.6).

Run: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /usr/bin/python3 run_b1.py {val|test} OUT.csv.gz [n_proc] [families]
(the test run was done in shards by family and concatenated: see README)
One row per data set (family, p, seed, T, gen, regime). The 'het' rows also carry the R-std outputs (columns with
prefix std_), since R-std only changes Var-SortnRegress and the estimated var-sortability.
"""
import sys, os, time, csv, gzip
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

TEST_SEEDS = list(range(1000, 1030))
VAL_SEEDS = list(range(9000, 9005))
GENS = ["G1", "G2", "G2s"]
REGIMES = ["het", "eq"]
N = 1000


def tasks(split):
    seeds = TEST_SEEDS if split == "test" else VAL_SEEDS
    out = []
    for fam in FAMILIES:
        for p in (5, 10):
            for s in seeds:
                for T in (3, 5, 10):
                    for gen in GENS:
                        for reg in REGIMES:
                            if gen == "G2s" and reg != "het":
                                continue
                            out.append((fam, p, s, T, gen, reg))
    return out


def mixing_error(A, sd, B=B_BURN):
    Sig = stationary_cov(A, sd); D = np.diag(sd ** 2); C = D.copy()
    for _ in range(B):
        C = A.T @ C @ A + D
    return float(np.abs(C - Sig).max() / np.abs(Sig).max())


def run_pc(X):
    from causallearn.search.ConstraintBased.PC import pc
    cg = pc(X, 0.01, "fisherz", stable=True, show_progress=False)
    g = cg.G.graph
    Mdir = (g == -1) & (g.T == 1)
    Mund = ((g == -1) & (g.T == -1)) | ((g == 1) & (g.T == 1))
    return Mdir, Mund


def run_flop(Xs):
    import flopsearch
    F = np.asarray(flopsearch.flop(np.ascontiguousarray(Xs), 2.0, restarts=10))
    return F == 1, F == 2


def eval_graph(row, key, Gp, Mdir, Mund, excl1):
    row[f"{key}_shd"] = shd(Gp, Mdir, Mund)
    row[f"{key}_shdx1"] = shd(Gp, Mdir, Mund, exclude=excl1)
    row[f"{key}_orient"] = orient_score(Gp, Mdir, Mund)
    row[f"{key}_nedge"] = int(Mdir.sum() + np.triu(Mund, 1).sum())
    try:
        sn, sc = sid(Gp, Mdir & ~Mund)
    except Exception:
        sn, sc = float("nan"), float("nan")
    row[f"{key}_sid"] = sc; row[f"{key}_sidn"] = sn


def one(task):
    fam, p, s, T, gen, reg = task
    t0 = time.time()
    rng_g = np.random.default_rng([FAMILIES.index(fam), p, s])
    S = summary_graph(fam, p, rng_g)
    A, sd = weights(S, rng_g, var_regime=reg)
    rng = np.random.default_rng([FAMILIES.index(fam), p, s, T, GENS.index(gen), REGIMES.index(reg), 7])
    X = simulate(A, sd, T, N, gen, rng)
    d = X.shape[1]
    G = true_dag(S, T)
    perm = rng.permutation(d)
    Xp = X[:, perm]; Gp = G[np.ix_(perm, perm)]
    slice_of = perm // p
    excl1 = (slice_of[:, None] == 0) & (slice_of[None, :] == 0)
    Xs = (Xp - Xp.mean(0)) / Xp.std(0)

    row = dict(family=fam, p=p, seed=s, T=T, gen=gen, regime=reg, d=d, n_true=int(G.sum()))
    row["s_exact"] = srho_G1_exact(S, T) if gen == "G1" else srho_stationary(S)
    row["s_exact_G1"] = srho_G1_exact(S, T); row["s_exact_stat"] = srho_stationary(S)
    row["Delta"] = delta_max(S); row["specrad"] = float(max(abs(np.linalg.eigvals(A))))
    row["mixerr"] = mixing_error(A, sd)

    rel = rel_counts(Xp); r2 = r2_scores(Xp); var = var_scores(Xp); var_std = var_scores(Xs)
    row["est_rel_sort"] = sortability(rel, Gp)
    row["est_r2_sort"] = sortability(r2, Gp)
    row["est_var_sort"] = sortability(var, Gp, tol=1e-9 * var.max())
    row["std_est_var_sort"] = sortability(var_std, Gp, tol=1e-9 * var_std.max())

    orders = dict(M1=order_from_scores(rel, rng), M2=order_from_scores(r2, rng), M3=order_from_scores(var, rng),
                  M4=rng.permutation(d), R0=order_from_scores(slice_of.astype(float), rng))
    orders_std = dict(M3=order_from_scores(var_std, rng))
    for key, o in list(orders.items()) + [("std_M3", orders_std["M3"])]:
        Mdir = regress_in_order(Xp, o)
        row[f"{key}_oa"] = order_accuracy(Gp, o)
        eval_graph(row, key, Gp, Mdir, np.zeros_like(Mdir), excl1)
    Mdir, Mund = run_flop(Xs)
    eval_graph(row, "M5", Gp, Mdir, Mund, excl1)
    if reg == "het" and gen in ("G1", "G2") and d <= 50:
        Mdir, Mund = run_pc(Xs)
        eval_graph(row, "M6", Gp, Mdir, Mund, excl1)
    row["secs"] = round(time.time() - t0, 2)
    return row


def main():
    split, out = sys.argv[1], sys.argv[2]
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    tl = tasks(split)
    if len(sys.argv) > 4:                     # optional shard: comma-separated families (to keep each job < 2 h)
        fams = sys.argv[4].split(",")
        tl = [t for t in tl if t[0] in fams]
    # largest first for load balance
    tl.sort(key=lambda t: -(t[1] * t[3]))
    from multiprocessing import Pool
    rows = []; t0 = time.time()
    with Pool(nproc) as pool:
        for k, r in enumerate(pool.imap_unordered(one, tl, chunksize=1)):
            rows.append(r)
            if (k + 1) % 100 == 0 or k + 1 == len(tl):
                print(f"{k + 1}/{len(tl)} {time.time() - t0:.0f}s", flush=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("family", "p", "seed", "T", "gen", "regime"), k))
    rows.sort(key=lambda r: (r["family"], r["p"], r["seed"], r["T"], r["gen"], r["regime"]))
    with gzip.open(out, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    main()
