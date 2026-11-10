"""Q4, part with raw data (PLAN.md section 6, roadmap b8): sensitivity of the plug-in effective-n test to the truncation lag K on E1 (Causal Chambers
wind tunnel, 15 runs; light tunnel, 10 segments of one file) and on the CausalRivers subgraphs (12 h headline, 24 h robustness).
K in {20, 50, 100, 300}; the base K is the one used in the committed analysis (WT primary 20, WT levels 100, LT 20, rivers 100 at 12 h and 50 at 24 h).
Reports, per K, the pooled non-relative dependence share and edge accuracy, the median Bartlett factor, and how many declared-dependence decisions change
relative to the base K. Descriptive; no verdict. Needs the data of ../DATA.md (THEORY2_DATA_ROOT or the per-dataset variables).
  OMP_NUM_THREADS=1 python q4_e1_k.py > results/analysis_Q4_k.txt
"""
import os, sys, json, importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); T2 = os.path.join(HERE, "..")
sys.path.insert(0, T2)
import datapaths as dp
KS = [20, 50, 100, 300]

# ------------------------------------------------------------------------------------------------ E1 (realdata/test3.py functions)
sys.argv = ["test3.py", os.path.join(T2, "realdata"), dp.CC_WT_WALKS, dp.CC_LT_WALKS]
spec = importlib.util.spec_from_file_location("test3", os.path.join(T2, "realdata", "test3.py")); t3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t3)


def decisions(units, p, K, nominal=False):
    out = []
    for X in units:
        R, A, NE, keep = t3.stats(X, K); Dm = t3.declare(R, NE, list(range(p)), nominal=nominal, n=len(X))
        iu = np.triu_indices(p, 1); out.append((Dm[iu], (np.maximum(1.0, 1 + 2 * np.einsum('ku,kv->uv', A, A)))[iu]))
    return out


def report(label, units, nodes, edges, base):
    p = len(nodes); print(f"== {label}: {len(units)} units, base K = {base}")
    dec0 = decisions(units, p, base)
    for K in KS:
        decK = decisions(units, p, K)
        flips = sum(int((a[0] != b[0]).sum()) for a, b in zip(dec0, decK)); npairs = sum(len(a[0]) for a in dec0)
        B = np.concatenate([b[1] for b in decK]); B0 = np.concatenate([a[1] for a in dec0])
        r = t3.analyse(units, nodes, edges, K, None, f"{label}_K{K}")
        print(f"  K={K:3d}: median B {np.median(B):.2f} (ratio to base median {np.median(B / B0):.3f}); decisions changed vs K={base}: {flips} of {npairs}; "
              f"non-relative pairs declared dependent {r['nonrel_dep_frac']:.3f} {np.round(r['nonrel_dep_ci'], 3).tolist()}; accuracy on edges predicted 1 {r['acc_pred1']:.3f}")


nodes, edges = t3.G.system("wt"); runs = list(range(2, 17))
raw = [t3.load(os.path.join(dp.CC_WT_WALKS, f"actuators_random_walk_{k}.csv"), nodes)[0] for k in runs]
report("E1 WT primary (D = 10, first differences)", [t3.std(t3.blockdiff(X, 10)) for X in raw], nodes, edges, 20)
report("E1 WT levels (secondary S1)", [t3.std(X) for X in raw], nodes, edges, 100)
nodes, edges = t3.G.system("lt"); X, _ = t3.load(os.path.join(dp.CC_LT_WALKS, "actuators_white.csv"), nodes)
report("E1 LT replication (10 segments of one file)", [t3.std(X[j * 2000:(j + 1) * 2000]) for j in range(10)], nodes, edges, 20)

# ------------------------------------------------------------------------------------------------ rivers (empirical2/test2.py head)
path = os.path.join(T2, "empirical2", "test2.py"); src = open(path).read(); head = src[:src.index('res = {"per_subgraph"')]
ns = {"__file__": path, "__name__": "test2_head"}; exec(compile(head, path, "exec"), ns)
print("\n== CausalRivers subgraphs (120), decisions of the Fisher-z test with the Bartlett n_eff; changes relative to the committed K ==")
for rn, base in (("12h", 100), ("24h", 50)):
    ref = {}
    for K in [base] + [k for k in sorted(set(KS)) if k != base]:
        ns["KLAG"] = {"12h": K, "24h": K}; dec_all = {"cond_neff": [], "marg_neff": []}; acc = []; keyd = {}
        for si, sg in enumerate(ns["SG"]):
            o, info = ns["analyse"](sg, rn)
            if o is None: continue
            keyd[si] = o
        if K == base: ref = keyd
        stats = {}
        for m in ("cond_neff", "marg_neff"):
            stats[m] = keyd
        flips = {m: 0 for m in ("cond_neff", "marg_neff")}; npairs = 0
        if ref:
            for k, o in keyd.items():
                for m in flips: flips[m] += int((np.array(o["dec"][m]) != np.array(ref[k]["dec"][m])).sum())
                npairs += len(o["dec"]["cond_neff"])
        nr = [(np.array(o["dec"]["cond_neff"], bool)[np.array(o["nonrel"], bool)]) for o in keyd.values() if any(o["nonrel"])]
        share = float(np.concatenate(nr).mean()); a5 = float(np.mean(np.concatenate([o["acc"]["cond_neff"] for o in keyd.values() if o["acc"]["cond_neff"]])))
        print(f"  {rn} K={K:3d}{' (committed K)' if K == base else ''}: non-relative pairs declared dependent (cond_neff) {share:.3f}; mean edge accuracy (cond_neff, all subgraphs) {a5:.3f}; median Bres {np.median([o['Bres_med'] for o in keyd.values()]):.2f}", end="")
        print(f"; decisions changed vs K={base}: cond_neff {flips['cond_neff']}, marg_neff {flips['marg_neff']} of {npairs}" if K != base else "", flush=True)
