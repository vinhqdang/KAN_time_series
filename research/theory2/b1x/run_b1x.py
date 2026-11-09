"""B'1 simulation (PLAN.md): B1 generators G1 / G2, time labels hidden; continuous-optimisation learners (DAGMA,
NOTEARS, GOLEM-EV/NV) on raw and standardised data, B1's sort baselines and FLOP, PCMCI+ with labels (reference).

Run (one shard = one (family, p); resumable):
  OMP_NUM_THREADS=1 /usr/bin/python3 run_b1x.py {val|test} OUT.csv.gz FAMILY P [--timing]
--timing fits every method and records only the seconds (no metric is computed); used on validation seeds only.
"""
import sys, os, time, csv, gzip
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "b1")); sys.path.insert(0, HERE)
from common import (FAMILIES, summary_graph, weights, simulate, true_dag, srho_G1_exact, srho_stationary,
                    rel_counts, r2_scores, var_scores, order_from_scores, regress_in_order, shd, order_accuracy,
                    orient_score, sid, sortability)
from methods_x import dagma_linear, notears_linear, golem, pcmciplus_window

TEST_SEEDS = list(range(3000, 3012))
VAL_SEEDS = [9600]
TS = (3, 5)
GENS = ["G1", "G2"]
N = 1000
GOLEM_MAX_P = 5          # GOLEM only on p = 5 cells (run-time budget, PLAN §1.4)


def ancestors(M):
    """Reflexive-free transitive closure of a directed bool matrix: R[i,j] = path i -> ... -> j."""
    R = M.copy()
    while True:
        R2 = R | ((R.astype(np.int32) @ R.astype(np.int32)) > 0)
        if (R2 == R).all():
            return R
        R = R2


def anc_order_accuracy(G_true, Mdir):
    """Mean over true edges (i, j): 1 if i is an ancestor of j in the estimate and not vice versa, 0 if the reverse,
    1/2 otherwise (incomparable or both, i.e. on a cycle). Uses the directed part only."""
    R = ancestors(Mdir)
    E = np.argwhere(G_true)
    f, b = R[E[:, 0], E[:, 1]], R[E[:, 1], E[:, 0]]
    return float(np.where(f & ~b, 1.0, np.where(b & ~f, 0.0, 0.5)).mean())


def is_dag(M):
    return not ancestors(M).diagonal().any()


def eval_graph(row, key, Gp, Mdir, Mund, excl1):
    row[f"{key}_shd"] = shd(Gp, Mdir, Mund)
    row[f"{key}_shdx1"] = shd(Gp, Mdir, Mund, exclude=excl1)
    row[f"{key}_orient"] = orient_score(Gp, Mdir, Mund)
    row[f"{key}_nedge"] = int(Mdir.sum() + np.triu(Mund, 1).sum())
    D = Mdir & ~Mund
    row[f"{key}_aoa"] = anc_order_accuracy(Gp, D)
    row[f"{key}_dag"] = int(is_dag(D))
    try:
        sn, sc = sid(Gp, D)
    except Exception:
        sn, sc = float("nan"), float("nan")
    row[f"{key}_sid"] = sc; row[f"{key}_sidn"] = sn


def run_flop(Xs):
    import flopsearch
    F = np.asarray(flopsearch.flop(np.ascontiguousarray(Xs), 2.0, restarts=10))
    return F == 1, F == 2


def one(task, timing=False):
    fam, p, s, T, gen = task
    t0 = time.time()
    fi = FAMILIES.index(fam)
    rng_g = np.random.default_rng([301, fi, p, s])
    S = summary_graph(fam, p, rng_g)
    A, sd = weights(S, rng_g, var_regime="het")
    rng = np.random.default_rng([302, fi, p, s, T, GENS.index(gen), 13])
    X = simulate(A, sd, T, N, gen, rng)                  # time order, column t*p+v
    d = X.shape[1]
    G = true_dag(S, T)
    perm = rng.permutation(d)
    Xp = X[:, perm]; Gp = G[np.ix_(perm, perm)]
    slice_of = perm // p
    excl1 = (slice_of[:, None] == 0) & (slice_of[None, :] == 0)
    Xs = (Xp - Xp.mean(0)) / Xp.std(0)
    row = dict(family=fam, p=p, seed=s, T=T, gen=gen, d=d, n_true=int(G.sum()),
               s_exact_G1=srho_G1_exact(S, T), s_exact_stat=srho_stationary(S))
    secs = {}
    if not timing:
        row["est_var_sort"] = sortability(var_scores(Xp), Gp, tol=1e-9 * var_scores(Xp).max())
        row["est_rel_sort"] = sortability(rel_counts(Xp), Gp)
        # empty-graph reference
        eval_graph(row, "EMPTY", Gp, np.zeros_like(Gp), np.zeros_like(Gp), excl1)

    def rec(key, fn):
        t = time.time(); out = fn(); secs[key] = time.time() - t
        return out

    # sort baselines (B1 implementations); orders drawn in a fixed sequence from rng
    rel = rel_counts(Xp); r2 = r2_scores(Xp); var = var_scores(Xp)
    orders = dict(M1=order_from_scores(rel, rng), M2=order_from_scores(r2, rng), M3=order_from_scores(var, rng),
                  std_M3=order_from_scores(np.ones(d), rng))   # standardised variances are all 1: random order
    for key, o in orders.items():
        Mdir = rec(key, lambda: regress_in_order(Xp, o))
        if not timing:
            row[f"{key}_oa"] = order_accuracy(Gp, o)
            eval_graph(row, key, Gp, Mdir, np.zeros_like(Mdir), excl1)
    Mdir, Mund = rec("M5", lambda: run_flop(Xs))
    if not timing:
        eval_graph(row, "M5", Gp, Mdir, Mund, excl1)
    # continuous optimisation, raw and standardised
    for tag, Z in (("raw", Xp), ("std", Xs)):
        W = rec(f"DAGMA_{tag}", lambda: dagma_linear(Z))
        if not timing:
            eval_graph(row, f"DAGMA_{tag}", Gp, W != 0, np.zeros((d, d), bool), excl1)
        W = rec(f"NOTEARS_{tag}", lambda: notears_linear(Z))
        if not timing:
            eval_graph(row, f"NOTEARS_{tag}", Gp, W != 0, np.zeros((d, d), bool), excl1)
        if p <= GOLEM_MAX_P:
            Bev, Bnv = rec(f"GOLEM_{tag}", lambda: golem(Z))
            if not timing:
                eval_graph(row, f"GOLEMEV_{tag}", Gp, Bev != 0, np.zeros((d, d), bool), excl1)
                eval_graph(row, f"GOLEMNV_{tag}", Gp, Bnv != 0, np.zeros((d, d), bool), excl1)
    # PCMCI+ on the time-indexed raw data (labels used; reference only)
    Md, Mu = rec("PCMCI", lambda: pcmciplus_window(X, p, T))
    if not timing:
        eval_graph(row, "PCMCI", Gp, Md[np.ix_(perm, perm)], Mu[np.ix_(perm, perm)], excl1)
    for k, v in secs.items():
        row[f"secs_{k}"] = round(v, 2)
    row["secs"] = round(time.time() - t0, 2)
    return row


def main():
    split, out, fam, p = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    timing = "--timing" in sys.argv
    seeds = TEST_SEEDS if split == "test" else VAL_SEEDS
    tl = [(fam, p, s, T, g) for s in seeds for T in TS for g in GENS]
    part = out + ".rows.csv"                             # resumable: append one row per finished data set
    done = set()
    if os.path.exists(part):
        with open(part) as f:
            for r in csv.DictReader(f):
                done.add((r["family"], int(r["p"]), int(r["seed"]), int(r["T"]), r["gen"]))
    t0 = time.time(); k = 0
    for task in tl:
        if task in done:
            continue
        r = one(task, timing)
        new = not os.path.exists(part)
        # fixed column set per shard: header from the first row
        with open(part, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(r.keys()) if new else next(csv.reader(open(part))))
            if new:
                w.writeheader()
            w.writerow(r)
        k += 1
        print(f"{task} {r['secs']}s total {time.time() - t0:.0f}s", flush=True)
    rows = list(csv.DictReader(open(part)))
    with gzip.open(out + ".tmp", "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    os.replace(out + ".tmp", out)


if __name__ == "__main__":
    main()
