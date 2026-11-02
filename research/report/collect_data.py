import os, glob, json
import numpy as np, pandas as pd
ROOT = os.path.join(os.path.dirname(__file__), "..", ".."); ER = f"{ROOT}/experimental_results"; PL = f"{ROOT}/research/pilots"
D = {}
r = pd.read_csv(f"{ER}/instdag_raw.csv"); g = r.groupby(["method", "d"]).auroc.mean().unstack()
D["published"] = {m: [round(float(g.loc[m, d]), 3) for d in (6, 10, 20)] for m in g.index}
def rw(path): df = pd.read_csv(path).set_index("method"); return {m: [round(float(df.loc[m, "bk_auroc_mean"]), 3), round(float(df.loc[m, "bk_auroc_std"]), 3)] for m in df.index}
D["bike_daily"] = rw(f"{ER}/realworld_bike_summary.csv"); D["bike_hourly"] = rw(f"{ER}/realworld_bike_hourly_primary_summary.csv"); D["beijing"] = rw(f"{ER}/realworld_beijing_primary_summary.csv")
sc = pd.read_csv(f"{ER}/realworld_large_scale_raw.csv"); D["fullseries"] = {ds: {m: [round(float(x.bk_auroc.mean()), 3), round(float(x.bk_auroc.std()), 3)] for m, x in gg.groupby("method")} for ds, gg in sc.groupby("dataset")}
v = pd.read_csv(f"{ER}/x1_validation_raw.csv"); gv = v.groupby(["d", "variant"]).auroc.mean().unstack(); D["x1_val"] = {str(d): {k: round(float(gv.loc[d, k]), 3) for k in gv.columns} for d in gv.index}
f = pd.read_csv(f"{ER}/x1_final_raw.csv"); ff = f[f.part == "fresh"].groupby(["d", "method"])[["auroc", "auprc"]].mean(); D["x1_fresh"] = {str(d): {m: [round(float(ff.loc[(d, m), "auroc"]), 3), round(float(ff.loc[(d, m), "auprc"]), 3)] for m in ("SPADE", "SPADE-AB")} for d in (6, 10, 20)}
rows = [pd.read_csv(p) for d in (50, 100, 200) for p in sorted(glob.glob(f"{ER}/x4_final_d{d}*.csv"))]; x4 = pd.concat(rows); g4 = x4.groupby(["d", "method"]).auroc.agg(["mean", "std", "count"])
D["x4"] = {m: {str(d): [round(float(g4.loc[(d, m), "mean"]), 3), int(g4.loc[(d, m), "count"])] for d in (50, 100, 200) if (d, m) in g4.index} for m in x4.method.unique()}
s = pd.read_csv(f"{ER}/sachs_raw.csv"); s = s[(s.status == "ok") & (s.part == "primary")]; gs = s.groupby("method")[["auroc", "auprc", "hits"]].mean(); sd = s.groupby("method").auroc.std()
D["sachs"] = {m.replace("SPADE-adaptive_boot", "SPADE-AB"): [round(float(gs.loc[m, "auroc"]), 3), round(float(sd[m]), 3), round(float(gs.loc[m, "auprc"]), 3), round(float(gs.loc[m, "hits"]), 1)] for m in gs.index}
t = pd.read_csv(f"{ER}/x3_fidelity_raw.csv"); t["nl"] = t.linear_oracle_r2 < 0.5
D["x3"] = {m: {"all_r2": round(float(t[(t.method == m) & t.found].r2.mean()), 3), "all_cls": round(float(t[(t.method == m) & t.found].cls_ok.mean()), 3), "nl_r2": round(float(t[(t.method == m) & t.found & t.nl].r2.mean()), 3), "nl_n": int((t[(t.method == m) & t.found & t.nl]).shape[0])} for m in ("MLP-edge", "SPADE", "SPADE-2stage")}
f5 = pd.read_csv(f"{ER}/x5_final_raw.csv"); b5 = pd.read_csv(f"{ER}/x5b_beijing_raw.csv")
D["x5_fin"] = {m: [round(float(x.mse.mean()), 4), round(float(x.time_s.mean()), 1)] for m, x in f5.groupby("model")}; D["x5_bj"] = {m: round(float(x.mse.mean()), 4) for m, x in b5.groupby("model")}
d10 = pd.read_csv(f"{PL}/G1_debiased_selection/raw_test_d10.csv"); d20 = pd.read_csv(f"{PL}/G1_debiased_selection/raw_test_d20.csv")
D["g1"] = {str(dd): {m: {c: round(float(x[c].mean()), 3) for c in ("auroc", "auprc", "curve_r2", "slope") if c in x and x[c].notna().any()} for m, x in df.groupby("method")} for dd, df in (("10", d10), ("20", d20))}
g2 = pd.read_csv(f"{PL}/G2_shape_benchmark/summary_pooled.csv").set_index("method"); D["g2"] = {m: {"AUROC": round(float(g2.loc[m, "AUROC"]), 3), "SF-F1": round(float(g2.loc[m, "SF-F1"]), 3), "condR2": round(float(g2.loc[m, "condR2"]), 3), "SF-recall": round(float(g2.loc[m, "SFrecall"]), 3)} for m in g2.index}
frames = [pd.read_csv(p) for p in glob.glob(f"{PL}/G3_scale/results/test_*_s800_*.csv")]; g3 = pd.concat(frames); g3 = g3[g3.gen == "std"]
D["g3"] = {m: {str(d): [round(float(x.auroc.mean()), 3), (round(float(x.order_agree.mean()), 2) if "order_agree" in x and x.order_agree.notna().any() else None)] for d, x in gg.groupby("d")} for m, gg in g3.groupby("method")}
A = pd.read_csv(f"{PL}/G4_benchmark_audit/results_A_raw.csv"); B = pd.read_csv(f"{PL}/G4_benchmark_audit/results_B_std.csv")
D["g4"] = {k: {m: {str(d): round(float(x.auroc.mean()), 3) for d, x in gg.groupby("d")} for m, gg in X.groupby("method")} for k, X in (("A", A), ("B", B))}
D["g4_vs"] = {str(d): round(float(x.vs_raw.mean()), 2) for d, x in A.groupby("d")}
j = json.load(open(f"{PL}/G5_nonstationary/g5_results.json"))["test"]; rows = []
for k, vv in j.items():
    for m, mm in vv.items(): rows.append(dict(sc=k.split("|")[0], method=m, auprc=mm["auprc"], auroc=mm["auroc"]))
g5 = pd.DataFrame(rows).groupby(["sc", "method"])[["auprc", "auroc"]].mean(); D["g5"] = {sc_: {m: [round(float(g5.loc[(sc_, m), "auroc"]), 3), round(float(g5.loc[(sc_, m), "auprc"]), 3)] for m in g5.loc[sc_].index} for sc_ in ("I1", "I1C", "ST")}
D["curves"] = json.load(open(f"{os.path.dirname(__file__)}/data/curves.json"))
json.dump(D, open(f"{os.path.dirname(__file__)}/data/report_data.json", "w")); print("ok", len(json.dumps(D)))
