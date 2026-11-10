"""Holm adjustment over the seven claims of PLAN.md section 11 (family F, alpha 0.05). Reads results/analysis_Q1.json and results/analysis_Q2.json.
C1 V3 (dose at fixed pattern != 0, one-sided in the direction observed), C2 V4, C3 V5, C4 within-family slope > 0, C5 DAGMA precondition, C6 NOTEARS precondition,
C7 FLOP's own G1 - G2 shift != 0 (two-sided p). One-sided bootstrap p-values have floor 1/10,001 (slope: 1/2,001). Verdicts defined by thresholds are not adjusted.
  python holm_family.py > results/holm_family.txt
"""
import json, os
import analyse_y as ay
HERE = os.path.dirname(os.path.abspath(__file__))
q1 = json.load(open(os.path.join(HERE, "results", "analysis_Q1.json")))
q2p = os.path.join(HERE, "results", "analysis_Q2.json")
q2 = json.load(open(q2p)) if os.path.exists(q2p) else {}
p = {"C1_V3_dose_fixed_pattern": q1["Q1_nonSC"]["V3_dose_fixed_pattern"]["p_one_sided"], "C2_V4_n_dose": q1["dose"]["V4_n_dose"]["p_one_sided"],
     "C3_V5_rho_dose": q1["dose"]["V5_rho_dose"]["p_one_sided"], "C4_slope_within_family": q1["Q1_nonSC"]["slope_gain_on_exact_gap_within_family"]["p_one_sided"]}
cl = q2.get("claims_p", {})
for tag, name in (("C5", "C5_DAGMA_precondition_intree"), ("C6", "C6_NOTEARS_precondition_intree"), ("C7", "C7_FLOP_shift_G1_minus_G2")):
    if tag in cl: p[name] = cl[tag]
h = ay.holm(p)
print(f"Holm over {len(p)} of 7 pre-listed claims (missing claims are listed below)\n")
for k, v in sorted(p.items()):
    print(f"  {k:36s} p = {v:.5f}   Holm-adjusted {h[k][0]:.5f}   {'reject' if h[k][1] else 'not rejected'} at 0.05")
miss = [n for n in ("C5", "C6", "C7") if n not in cl]
if miss: print("\n  not available:", miss)
