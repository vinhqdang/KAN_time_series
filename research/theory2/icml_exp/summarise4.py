"""Print the PLAN4 verdict tables from results4.json.  Usage: python3 summarise4.py results4.json"""
import sys, json
d = json.load(open(sys.argv[1]))
def r(x, k=3): return None if x is None else round(x, k)
def e1(tag, e):
    if "obs_value_counts" not in e: print(f"\n## {tag}: no instances"); return
    print(f"\n## {tag}: instances {e['n_instances']}, obs counts {e['obs_value_counts']}")
    print("   obs | theory value (mean obs, exact-match share, n):", e["obs_given_theory"])
    print("   exact match:", {k: r(v) for k, v in e["exact_match"].items()})
    for c, t in e.get("tests", {}).items():
        if t.get("n_disagree", 0) == 0: print(f"   {c}: no disagreement"); continue
        print(f"   {c}: n={t['n_disagree']} loss T {r(t['loss_theory'])} C {r(t['loss_comp'])} delta {r(t['delta'])} "
              f"CI {[r(x) for x in t['delta_ci']]} units {t['units_theory_better']}/{t['units_with_disagreement']} "
              f"p_holm {r(t['p_holm'],5)} pass {t['pass']} rev_holm {r(t['p_reverse_holm'],4)} alltie {r(t['delta_if_all_ties'])}")
    if "E1_pass_all" in e: print("   E1 pass all:", e["E1_pass_all"])
    if "spearman_subgraphs" in e: print("   spearman:", {k: r(v) for k, v in e["spearman_subgraphs"].items()})
for k, v in d.items():
    if k.startswith("E1_") or (k.startswith("E2_wt") and "E1" in v):
        if "nonrel_det" in v: print(f"\n# {k} nonrel/det:", {a: (r(b) if isinstance(b, float) else b) for a, b in v["nonrel_det"].items()})
        if "removed_nodes" in v and v["removed_nodes"]: print("   removed:", v["removed_nodes"])
        e1(k, v["E1"])
        for sub in ("detected_edge_subset", "theory_strict_subset"):
            if sub in v["E1"]: e1(k + " / " + sub, v["E1"][sub])
    if k.startswith("E2_wt") and "Q" in v:
        print(f"\n# {k} Q:", {a: (r(b) if isinstance(b, float) else b) for a, b in v["Q"].items()})
    if k.endswith("_detection") or k.startswith("F_"):
        print(f"\n# {k}:", {a: (r(b) if isinstance(b, float) else b) for a, b in v.items()})
