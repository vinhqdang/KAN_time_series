"""Q2 (PLAN.md section 4 + Addendum 1): variance-matched control and tuned learners. One shard = one (family, p, T) cell, resumable per graph seed.

  OMP_NUM_THREADS=1 python run_q2.py tune OUT.csv FAMILY P T     # validation seeds, DAGMA/NOTEARS grids on raw G1 and G2 (hyper-parameter choice)
  OMP_NUM_THREADS=1 python run_q2.py val  OUT.csv FAMILY P T     # validation seeds, all methods with the tuned values (debugging/timing only)
  OMP_NUM_THREADS=1 python run_q2.py test OUT.csv FAMILY P T     # test seeds 4300-4311; needs results/q2_tuned.json

Arms: G1, G2 (simulated), VM_G1 = G1 data x sqrt(v_G2/v_G1), VM_G2 = G2 data x sqrt(v_G1/v_G2) (population factors; no new simulation).
Methods: FLOP (standardised input), DAGMA, NOTEARS (tuned, raw), GOLEM-EV / GOLEM-NV (defaults, raw; only d <= GOLEM_MAX_D), Var-SnR, rel-SnR, EMPTY.
FLOP and rel-SnR are invariant to column rescaling: computed once per base arm, copied to the VM arm after asserting identical standardised data.
Time labels hidden (columns permuted per data set). Rows (long): family,p,seed,T,arm,method,d,n_true,shd_n,shd_x1n,aoa,aoa_x1,fwd,bwd,nedge,dag,sidn,oa_full,oa_x1,secs.
"""
import sys, os, csv, json, time
import numpy as np
import common_y as cy
import common_y2 as c2
import seeds as sd_
import methods_y as my
from common import rel_counts, var_scores, order_from_scores, regress_in_order
from methods_x import golem as golem_fn

HERE = os.path.dirname(os.path.abspath(__file__))
GOLEM_MAX_D = 50                 # PLAN.md Addendum 1: GOLEM runs where d <= 50 (the (p, T) = (20, 3) cell, d = 60, is excluded)
DAGMA_GRID = [(l1, th) for l1 in (0.01, 0.03, 0.1) for th in (0.1, 0.3)]
NOTEARS_GRID = [(l1, th) for l1 in (0.01, 0.1) for th in (0.1, 0.3)]
DAGMA_DEFAULT, NOTEARS_DEFAULT = (0.03, 0.3), (0.1, 0.3)
FIELDS = ["family", "p", "seed", "T", "arm", "method", "d", "n_true", "shd_n", "shd_x1n", "aoa", "aoa_x1", "fwd", "bwd", "nedge",
          "dag", "sidn", "oa_full", "oa_x1", "secs", "status"]
LIMIT = 1500                     # seconds per fit; a longer fit is recorded as status "timeout" (never dropped silently)
ARM_INDEX = {"G1": cy.ARMS.index("G1"), "G2": cy.ARMS.index("G2"), "VM_G1": 11, "VM_G2": 12}


def build(fam, p, s, T):
    rng_g = sd_.graph_rng(fam, p, s); S = cy.summary_graph(fam, p, rng_g); A, sd = cy.weights_cap(S, rng_g, 0.8)
    G = cy.true_dag(S, T); data = {}
    for arm in ("G1", "G2"):
        rng = sd_.data_rng(fam, p, s, T, ARM_INDEX[arm], 1000)
        X = cy.simulate_arm(arm, A, sd, T, 1000, rng)
        perm = rng.permutation(X.shape[1])
        data[arm] = (X, perm)
    data["VM_G1"] = (data["G1"][0] * cy.vm_factors(A, sd, T, "G1", "G2"), data["G1"][1])
    data["VM_G2"] = (data["G2"][0] * cy.vm_factors(A, sd, T, "G2", "G1"), data["G2"][1])
    return S, G, data


def rec(rows, base, method, secs, status="ok", **vals):
    r = dict(base, method=method, secs=round(secs, 2), status=status); r.update(vals); rows.append(r)


def learn(rows, base, method, fn, Gp, sl):
    """fn() -> (Mdir, Mund); run with a wall-clock limit; failures become rows with NaN metrics and a status."""
    t0 = time.time(); val, st = c2.with_timeout(fn, LIMIT); secs = time.time() - t0
    if st != "ok":
        rec(rows, base, method, secs, st, **c2.NAN_METRICS); return
    rec(rows, base, method, secs, "ok", **c2.eval_graph(Gp, val[0], val[1], sl))


def run_graph(fam, p, s, T, tuned):
    S, G, data = build(fam, p, s, T); d = G.shape[0]; rows = []
    import flopsearch
    cache = {}                                           # base arm -> (flop matrix, standardised data, flop secs, rel order)
    z = np.zeros((d, d), bool)
    for arm in ("G1", "G2", "VM_G1", "VM_G2"):
        X, perm = data[arm]; Xp = X[:, perm]; Gp = G[np.ix_(perm, perm)]; sl = cy.slice_of_cols(perm, p)
        Xs = (Xp - Xp.mean(0)) / Xp.std(0)
        base = dict(family=fam, p=p, seed=s, T=T, arm=arm, d=d, n_true=int(G.sum()))
        rng = sd_.data_rng(fam, p, s, T, ARM_INDEX[arm], 1000, 10)       # tie-breaking stream for the orders
        if arm in ("G1", "G2"):
            t0 = time.time(); F = np.asarray(flopsearch.flop(np.ascontiguousarray(Xs), 2.0, restarts=10))
            relo = order_from_scores(rel_counts(Xp), rng)
            cache[arm] = (F, Xs, time.time() - t0, relo)
        else:
            F, Xs0, _, relo = cache[arm.replace("VM_", "")]
            assert np.allclose(Xs, Xs0, atol=1e-9), "VM rescaling changed the standardised data"
        rec(rows, base, "FLOP", cache[arm.replace("VM_", "")][2], **c2.eval_graph(Gp, F == 1, F == 2, sl))
        rec(rows, base, "RelSnR", 0.0, **c2.eval_order(Gp, relo, sl), **c2.eval_graph(Gp, regress_in_order(Xp, relo), z, sl))
        vo = order_from_scores(var_scores(Xp), rng)
        rec(rows, base, "VarSnR", 0.0, **c2.eval_order(Gp, vo, sl), **c2.eval_graph(Gp, regress_in_order(Xp, vo), z, sl))
        rec(rows, base, "EMPTY", 0.0, **c2.eval_graph(Gp, z, z, sl))
        learn(rows, base, "DAGMA", lambda: my.dagma_hp(Xp, *tuned["DAGMA"]), Gp, sl)
        learn(rows, base, "NOTEARS", lambda: my.notears_hp(Xp, *tuned["NOTEARS"]), Gp, sl)
        if d <= GOLEM_MAX_D:
            t0 = time.time(); val, st = c2.with_timeout(lambda: golem_fn(Xp), 2 * LIMIT); tg = time.time() - t0
            for name, k in (("GOLEMEV", 0), ("GOLEMNV", 1)):
                if st != "ok":
                    rec(rows, base, name, tg, st, **c2.NAN_METRICS)
                else:
                    rec(rows, base, name, tg, "ok", **c2.eval_graph(Gp, val[k] != 0, z, sl))
    return rows


def run_tune_graph(fam, p, s, T):
    S, G, data = build(fam, p, s, T); d = G.shape[0]; rows = []
    for arm in ("G1", "G2"):
        X, perm = data[arm]; Xp = X[:, perm]; Gp = G[np.ix_(perm, perm)]; sl = cy.slice_of_cols(perm, p)
        base = dict(family=fam, p=p, seed=s, T=T, arm=arm, d=d, n_true=int(G.sum()))
        for l1, th in DAGMA_GRID:
            learn(rows, base, f"DAGMA|{l1}|{th}", lambda: my.dagma_hp(Xp, l1, th), Gp, sl)
        for l1, th in NOTEARS_GRID:
            learn(rows, base, f"NOTEARS|{l1}|{th}", lambda: my.notears_hp(Xp, l1, th), Gp, sl)
    return rows


def main():
    mode, out, fam, p, T = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    seeds = sd_.SEEDS["q2_test" if mode == "test" else "q2_val"]
    tuned = None
    if mode in ("val", "test"):
        tuned = json.load(open(os.environ.get("B1Y_TUNED", os.path.join(HERE, "results", "q2_tuned.json"))))
        tuned = {k: (v["lambda1"], v["w_threshold"]) for k, v in tuned.items()}
    done = set()
    if os.path.exists(out):
        done = {int(r["seed"]) for r in csv.DictReader(open(out))}
    new = not os.path.exists(out)
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for s in seeds:
            if s in done:
                continue
            rows = run_tune_graph(fam, p, s, T) if mode == "tune" else run_graph(fam, p, s, T, tuned)
            for r in rows:
                w.writerow({k: r.get(k, "") for k in FIELDS})
            f.flush()
            print(mode, fam, p, T, s, "ok", flush=True)


if __name__ == "__main__":
    main()
