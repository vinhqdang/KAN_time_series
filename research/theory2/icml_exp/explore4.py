"""Post hoc (no verdict): compare the theory with a constant 'every edge 1/2' predictor (not pre-registered), from
results4.json.  Usage: python3 explore4.py results4.json"""
import sys, json
d = json.load(open(sys.argv[1]))
for k in ["E1_citris_Glin_L0", "E1_citris_GD_L0", "E1_iid_pooled", "E2_wt_D10_L0", "E2_wt_D1_L30"]:
    e = d[k]["E1"]; c = e["obs_value_counts"]; n = e["n_instances"]
    print(f"{k}: exact match theory {e['exact_match']['theory']:.3f}; constant-1/2 predictor {c['1/2']/n:.3f}; "
          f"mean |obs - theory| {e['mean_loss_all']['theory']:.3f}; mean |obs - 1/2| "
          f"{(c['1']*0.5 + c['0']*0.5)/n:.3f}")
