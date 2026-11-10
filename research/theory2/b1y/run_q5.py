"""Q5 (PLAN.md section 7 + Addendum 1): extra baselines on the B1x design. One shard = one (family, p, T) cell, resumable per graph seed.

  OMP_NUM_THREADS=1 python run_q5.py {val|test} OUT.csv FAMILY P T

Data: B1 generators G1 / G2 (arm names as in Q1), time labels hidden, n = 1000, families ring, chain, intree, ER2; p in {5, 10}; T in {3, 5};
fresh streams KEY_Q5 (403/404), test seeds 4100-4111, validation seed 9820.
Label-free methods: FLOP, rel-SnR, Var-SnR, EMPTY, GES (Tetrad FGES, penalty 2), GRaSP and BOSS (causal-learn, penalty 2, depth 3), DirectLiNGAM,
  and as robustness rows Tetrad GRaSP and BOSS.
Time-labelled references: PCMCI+ (tigramite, as in B1x), pooled lag-1 OLS threshold (VAR_OLS), pooled VAR-LiNGAM (VARLiNGAM_pooled), DYNOTEARS.
Rows as run_q2.py (arm = G1 / G2).
"""
import sys, os, csv, time
import numpy as np
import common_y as cy
import common_y2 as c2
import seeds as sd_
import methods_y as my
import methods_y2 as m2
from common import rel_counts, var_scores, order_from_scores, regress_in_order

FIELDS = ["family", "p", "seed", "T", "arm", "method", "d", "n_true", "shd_n", "shd_x1n", "aoa", "aoa_x1", "fwd", "bwd", "nedge",
          "dag", "sidn", "oa_full", "oa_x1", "secs", "status"]
LIMIT = 600
ARM_INDEX = {"G1": cy.ARMS.index("G1"), "G2": cy.ARMS.index("G2")}


def rec(rows, base, method, secs, status="ok", **vals):
    r = dict(base, method=method, secs=round(secs, 2), status=status); r.update(vals); rows.append(r)


def learn(rows, base, method, fn, Gp, sl, perm=None, fork=True):
    """fn() -> (Mdir, Mund) in the frame of fn's input. perm: if given the result is time-ordered and is permuted to the data frame."""
    t0 = time.time()
    if fork:
        val, st = c2.with_timeout(fn, LIMIT)
    else:
        try:
            val, st = fn(), "ok"
        except Exception as e:                          # noqa: BLE001
            val, st = None, "err:" + repr(e)[:200]
    secs = time.time() - t0
    if st != "ok":
        rec(rows, base, method, secs, st, **c2.NAN_METRICS); return
    Md, Mu = val
    if perm is not None:
        Md = Md[np.ix_(perm, perm)]; Mu = Mu[np.ix_(perm, perm)]
    rec(rows, base, method, secs, "ok", **c2.eval_graph(Gp, Md, Mu, sl))


def run_graph(fam, p, s, T):
    rng_g = sd_.graph_rng(fam, p, s, key=sd_.KEY_Q5_GRAPH); S = cy.summary_graph(fam, p, rng_g); A, sd = cy.weights_cap(S, rng_g, 0.8)
    G = cy.true_dag(S, T); d = G.shape[0]; rows = []; z = np.zeros((d, d), bool)
    import flopsearch
    for arm in ("G1", "G2"):
        rng = sd_.data_rng(fam, p, s, T, ARM_INDEX[arm], 1000, key=sd_.KEY_Q5_DATA)
        X = cy.simulate_arm(arm, A, sd, T, 1000, rng)
        perm = rng.permutation(d); Xp = X[:, perm]; Gp = G[np.ix_(perm, perm)]; sl = cy.slice_of_cols(perm, p)
        Xs = (Xp - Xp.mean(0)) / Xp.std(0)
        base = dict(family=fam, p=p, seed=s, T=T, arm=arm, d=d, n_true=int(G.sum()))
        rng2 = sd_.data_rng(fam, p, s, T, ARM_INDEX[arm], 1000, 20, key=sd_.KEY_Q5_DATA)     # order tie-breaking stream
        ro = order_from_scores(rel_counts(Xp), rng2); vo = order_from_scores(var_scores(Xp), rng2)
        rec(rows, base, "RelSnR", 0.0, **c2.eval_order(Gp, ro, sl), **c2.eval_graph(Gp, regress_in_order(Xp, ro), z, sl))
        rec(rows, base, "VarSnR", 0.0, **c2.eval_order(Gp, vo, sl), **c2.eval_graph(Gp, regress_in_order(Xp, vo), z, sl))
        rec(rows, base, "EMPTY", 0.0, **c2.eval_graph(Gp, z, z, sl))
        learn(rows, base, "FLOP", lambda: (lambda F: (F == 1, F == 2))(np.asarray(flopsearch.flop(np.ascontiguousarray(Xs), 2.0, restarts=10))), Gp, sl)
        learn(rows, base, "GRaSP", lambda: my.grasp_cpdag(Xp), Gp, sl)
        learn(rows, base, "BOSS", lambda: my.boss_cpdag(Xp), Gp, sl)
        learn(rows, base, "DirectLiNGAM", lambda: my.directlingam_dag(Xp), Gp, sl)
        learn(rows, base, "GES", lambda: m2.tetrad_call("fges", Xp), Gp, sl, fork=False)
        learn(rows, base, "GRaSP_tetrad", lambda: m2.tetrad_call("grasp", Xp), Gp, sl, fork=False)
        learn(rows, base, "BOSS_tetrad", lambda: m2.tetrad_call("boss", Xp), Gp, sl, fork=False)
        # time-labelled references (time-ordered window X)
        learn(rows, base, "PCMCI", lambda: __import__("methods_x").pcmciplus_window(X, p, T), Gp, sl, perm=perm)
        learn(rows, base, "VAR_OLS", lambda: my.var_ols_labelled(X, p, T), Gp, sl, perm=perm)
        learn(rows, base, "VARLiNGAM_pooled", lambda: m2.varlingam_pooled(X, p, T), Gp, sl, perm=perm)
        learn(rows, base, "DYNOTEARS", lambda: m2.dynotears(X, p, T), Gp, sl, perm=perm, fork=False)
    return rows


def main():
    mode, out, fam, p, T = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    seeds = sd_.SEEDS["q5_test" if mode == "test" else "q5_val"]
    done = set()
    if os.path.exists(out):
        done = {int(r["seed"]) for r in csv.DictReader(open(out))}
    new = not os.path.exists(out)
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader(); f.flush()
        for s in seeds:
            if s in done:
                continue
            for r in run_graph(fam, p, s, T):
                w.writerow({k: r.get(k, "") for k in FIELDS})
            f.flush(); print(mode, fam, p, T, s, "ok", flush=True)


if __name__ == "__main__":
    main()
