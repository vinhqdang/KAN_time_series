"""B1-NL simulation (PLAN.md Sections 1.3-1.6). One row per data set (cfg, family, p, seed, T, gen in {G1, G2}).
Each G2 row also carries the rel-SnR order accuracy on an independent G2L data set (burn-in 500; sanity rule S1).
R-het only; the R-std outputs (Var-SnR and estimated var-sortability on standardised data) are columns std_*.

Run (resumable, one shard per (cfg, family, p), written atomically, completed shards skipped):
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /usr/bin/python3 run_b1nl.py {val|test} OUTDIR [cfg,...] [timing]
"""
import sys, os, time, csv, gzip
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_nl import *
from common_nl import FAMILIES, CONFIGS, DATA_KEY

TEST_SEEDS = dict(NL=list(range(2000, 2025)), SGN=list(range(2000, 2025)), SGNU=list(range(2000, 2010)))
VAL_SEEDS = dict(NL=list(range(9100, 9103)), SGN=list(range(9100, 9103)), SGNU=list(range(9100, 9103)))
GENS = ["G1", "G2", "G2L"]
N = 1000


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


def rel_detect(Xp, perm, S, T, gen, alpha=0.05):
    """Recall / false-positive rate of the correlation-count relatives estimator against the graph relatives."""
    from scipy import stats
    n = Xp.shape[0]
    C = np.corrcoef(Xp, rowvar=False)
    tcrit = stats.t.ppf(1 - alpha / 2, n - 2); eps = tcrit / np.sqrt(n - 2 + tcrit ** 2)
    det = np.abs(C) > eps
    R = window_relatives(S, T, gen)[np.ix_(perm, perm)]
    off = ~np.eye(len(perm), dtype=bool)
    rec = float((det & R & off).sum() / max((R & off).sum(), 1))
    fpr = float((det & ~R & off).sum() / max((~R & off).sum(), 1)) if (~R & off).any() else float("nan")
    return rec, fpr


def one(task):
    cfg, fam, p, s, T, gen = task
    t0 = time.time()
    P = params(cfg, fam, p, s); S = P["S"]
    rng = np.random.default_rng([DATA_KEY[cfg], FAMILIES.index(fam), p, s, T, GENS.index(gen), 11])
    X = simulate_nl(P, T, N, gen, rng)
    d = X.shape[1]
    G = true_dag(S, T)
    perm = rng.permutation(d)
    Xp = X[:, perm]; Gp = G[np.ix_(perm, perm)]
    slice_of = perm // p
    excl1 = (slice_of[:, None] == 0) & (slice_of[None, :] == 0)
    Xs = (Xp - Xp.mean(0)) / Xp.std(0)

    row = dict(cfg=cfg, family=fam, p=p, seed=s, T=T, gen=gen, d=d, n_true=int(G.sum()))
    row["s_exact"] = srho_G1_exact(S, T) if gen == "G1" else srho_stationary(S)
    row["s_exact_G1"] = srho_G1_exact(S, T); row["s_exact_stat"] = srho_stationary(S)
    row["Delta"] = delta_max(S)
    row["specrad_W"] = float(max(abs(np.linalg.eigvals(P["W"]))))
    row["coupling_err"] = coupling_error(P, seed=s)
    row["rel_recall"], row["rel_fpr"] = rel_detect(Xp, perm, S, T, gen)

    rel = rel_counts(Xp); r2 = r2_scores(Xp); var = var_scores(Xp); var_std = var_scores(Xs)
    row["est_rel_sort"] = sortability(rel, Gp)
    row["est_r2_sort"] = sortability(r2, Gp)
    row["est_var_sort"] = sortability(var, Gp, tol=1e-9 * var.max())
    row["std_est_var_sort"] = sortability(var_std, Gp, tol=1e-9 * var_std.max())

    orders = dict(M1=order_from_scores(rel, rng), M2=order_from_scores(r2, rng), M3=order_from_scores(var, rng),
                  M4=rng.permutation(d), R0=order_from_scores(slice_of.astype(float), rng))
    orders["std_M3"] = order_from_scores(var_std, rng)
    for key, o in orders.items():
        Mdir = regress_in_order(Xp, o)
        row[f"{key}_oa"] = order_accuracy(Gp, o)
        eval_graph(row, key, Gp, Mdir, np.zeros_like(Mdir), excl1)
    Mdir, Mund = run_flop(Xs)
    eval_graph(row, "M5", Gp, Mdir, Mund, excl1)

    if gen == "G2":            # S1 replication with a long burn-in: rel-SnR order accuracy only
        rngL = np.random.default_rng([DATA_KEY[cfg], FAMILIES.index(fam), p, s, T, GENS.index("G2L"), 11])
        XL = simulate_nl(P, T, N, "G2L", rngL)
        permL = rngL.permutation(d)
        row["G2L_M1_oa"] = order_accuracy(G[np.ix_(permL, permL)], order_from_scores(rel_counts(XL[:, permL]), rngL))
    row["secs"] = round(time.time() - t0, 3)
    return row


def shard_tasks(split, cfg, fam, p):
    seeds = (TEST_SEEDS if split == "test" else VAL_SEEDS)[cfg]
    return [(cfg, fam, p, s, T, gen) for s in seeds for T in (3, 5, 10) for gen in ("G1", "G2")]


def write(rows, path):
    keys = sorted({k for r in rows for k in r},
                  key=lambda k: (k not in ("cfg", "family", "p", "seed", "T", "gen"), k))
    rows.sort(key=lambda r: (r["cfg"], r["family"], r["p"], r["seed"], r["T"], r["gen"]))
    with gzip.open(path + ".part", "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(path + ".part", path)


def main():
    split, outdir = sys.argv[1], sys.argv[2]
    cfgs = sys.argv[3].split(",") if len(sys.argv) > 3 else CONFIGS
    os.makedirs(outdir, exist_ok=True)
    t00 = time.time()
    for cfg in cfgs:
        for fam in FAMILIES:
            for p in (5, 10):
                path = f"{outdir}/{split}_{cfg}_{fam}_{p}.csv.gz"
                if os.path.exists(path):
                    print(f"skip {path}", flush=True); continue
                t0 = time.time()
                rows = [one(t) for t in shard_tasks(split, cfg, fam, p)]
                write(rows, path)
                print(f"done {cfg} {fam} p={p}: {len(rows)} rows, {time.time() - t0:.0f}s "
                      f"(total {time.time() - t00:.0f}s)", flush=True)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
