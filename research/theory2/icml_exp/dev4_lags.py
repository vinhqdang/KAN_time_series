"""Development only (PLAN4 section 2): lag windows L and block sizes D on PLAN3's WT data
(actuators_random_walk_2..16), using the test4.py estimator.  Not evidence for any verdict.
Usage: OMP_NUM_THREADS=1 python3 -I dev4_lags.py <this dir>"""
import sys, os
HERE = sys.argv[1]; sys.path.insert(0, HERE)
sys.argv = ["test4", HERE, "dev", "/dev/null"]   # test4 reads its arguments at import time
import test4                                     # module-level definitions only; its main block does not run
t4 = vars(test4)
import cc_graph as cc                            # ../realdata/cc_graph.py, put on sys.path by test4
import pandas as pd
nodes, edges = cc.system("wt")
raw = [pd.read_csv(os.path.join(t4["DATA"]["wt"], f"actuators_random_walk_{k}.csv"))[nodes].to_numpy(float) for k in range(2, 17)]
for D, Ls in [(1, (0, 10, 30)), (3, (0, 10)), (10, (0, 1, 3))]:
    units = [t4["blockdiff"](X, D) for X in raw]
    for L in Ls:
        r = t4["run_dataset"]("d", units, list(range(15)), nodes, edges, edges, set(range(5)), 20, L, 9000)
        q = t4["e2_eval"](r, nodes, edges, len(units))
        print(f"D={D} L={L} nonrel={q['nonrel_dep_frac']:.3f} rel_detect={q['rel_detect_frac']:.3f} "
              f"acc_pred1={q['acc_pred1']:.3f} spearman={q['spearman']:.3f}", flush=True)
r = t4["run_dataset"]("d", [t4["blockdiff"](X, 10) for X in raw], list(range(15)), nodes, edges, edges, set(range(5)), 20, 0, 9002)
print("D=10 L=0 with PLAN3 subgraph seeds 10*(9000+run)+d: spearman =", round(t4["e2_eval"](r, nodes, edges, 15)["spearman"], 4))
