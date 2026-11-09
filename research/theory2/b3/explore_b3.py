"""POST HOC (not pre-registered): per-pair detection on full node sets, from the test_b3.py caches.
Usage: python3 -I explore_b3.py <results_dir>"""
import sys, os, json, gzip
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graphs_b3 as G
R = sys.argv[1]
def pairs(name, nodes, rel_edges, unit_slices):
    res = json.load(gzip.open(os.path.join(R, f"cache_{name}.json.gz"), "rt"))
    p = len(nodes); Rel = G.related(p, rel_edges); idx = {v: i for i, v in enumerate(nodes)}
    for lab, sl in unit_slices:
        full = res["full"][sl]
        print(f"-- {name} {lab}: pairs declared in >= 1 unit although non-relatives, and relatives never/rarely detected")
        keys = full[0]["decl"].keys()
        for k in keys:
            a, b = k.split("|"); rel = Rel[idx[a], idx[b]]
            d = np.mean([f["decl"][k] for f in full]); r = np.median([f["r"][k] for f in full])
            if (not rel and d > 0) or (rel and d < 0.8):
                print(f"   {'REL ' if rel else 'NON '} {a:8s} {b:8s} declared {d:.2f}  median r {r:+.3f}")
    rows = res["rows"]
    bad = {}
    for r in rows:
        if r["obs"] != r["theory"]:
            k = f"{r['edge'][0]}->{r['edge'][1]} theory {r['theory']} obs {r['obs']}"
            bad[k] = bad.get(k, 0) + 1
    print(f"   mismatches theory vs observed ({sum(bad.values())} of {len(rows)}):",
          dict(sorted(bad.items(), key=lambda x: -x[1])[:12]))
nodes, EL, ED, inp = G.ar1_system()
pairs("ar_1", nodes, EL, [("ref", slice(0, 10)), ("pol_1_90", slice(10, 20)), ("pol_12_30", slice(20, 30))])
for exp in ["scm_5_red", "scm_5_blue", "scm_5_green", "scm_4_green", "scm_4_blue", "scm_4_red"]:
    nodes, E, _, _ = G.scm_system(exp)
    pairs(exp, nodes, E, [("all", slice(0, 10))])

# POST HOC: distribution of the actuator columns in the scm sets (descriptive only)
import pandas as pd
D = sys.argv[2] if len(sys.argv) > 2 else None
if D:
    for exp in ["scm_5_red", "scm_5_blue", "scm_5_green", "scm_4_green", "scm_4_blue", "scm_4_red"]:
        df = pd.read_csv(os.path.join(D, exp + ".csv"), usecols=["red", "green", "blue"])
        print(exp, {c: (int(df[c].min()), int(df[c].max()), round(float(df[c].std()), 1), int(df[c].nunique())) for c in df})

# POST HOC: S5 (theory on G_lin vs theory on G_D) split by ar_1 experiment
res = json.load(gzip.open(os.path.join(R, "cache_ar_1.json.gz"), "rt"))
for e, lab in enumerate(["ref", "pol_1_90", "pol_12_30"]):
    rr = [r for r in res["rows"] if 10 * e <= r["unit"] < 10 * e + 10 and r["theory"] != r["theory_GD"]]
    if rr:
        print(f"S5 {lab}: n={len(rr)} exact theory(G_lin) {np.mean([r['obs']==r['theory'] for r in rr]):.3f}"
              f"  theory(G_D) {np.mean([r['obs']==r['theory_GD'] for r in rr]):.3f}")
