"""Q1 analysis driver (PLAN.md section 3.3): merge shards -> results/q1.csv.gz, then verdicts V1-V5, slope, sanity rules S1-S4, SC families,
SHD arms, oracle calibration. Writes results/analysis_Q1.txt and results/analysis_Q1.json. No threshold is set here; all rules are in analyse_y.py.
  python run_analysis_q1.py
"""
import glob, json, os
import numpy as np
import pandas as pd
import analyse_y as ay

HERE = os.path.dirname(os.path.abspath(__file__))
files = sorted(glob.glob(os.path.join(HERE, "results", "shards", "q1", "*.csv")))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.to_csv(os.path.join(HERE, "results", "q1.csv.gz"), index=False)
out = {"n_rows": int(len(df)), "n_shards": len(files)}
lines = [f"Q1 analysis. {len(df)} rows from {len(files)} shards (main {int((df.study == 'main').sum())}, ndose {int((df.study == 'ndose').sum())}, rhodose {int((df.study == 'rhodose').sum())}).",
         "Metric OA_x1 (order accuracy of rel-SortnRegress, edges leaving slice 1 dropped); unit = summary graph; graph-cluster bootstrap CIs.", ""]


def dump(title, obj):
    lines.append("== " + title); lines.append(json.dumps(obj, indent=1, default=float)); lines.append("")


v = ay.q1_verdicts(df, ay.NONSC); out["Q1_nonSC"] = v; dump("Q1 verdicts, non-SC families pooled (V1, V2, V3, slope)", v)
sc = ay.q1_verdicts(df, ay.SC) if "STOP" not in v else {}
out["Q1_SC"] = sc; dump("Same statistics, SC families (symER, ring) pooled (descriptive)", sc)
dose = ay.q1_dose(df); out["dose"] = dose; dump("Dose studies (V4 n-dose, V5 rho-dose)", dose)

# sanity rules
m = df[(df.study == "main") & df.family.isin(ay.NONSC)]
g = ay._graph_table(m, "oa_x1", ["arm"])
S1 = float(abs((g["G2s"] - g["G2"]).mean())); S4 = ay.boot_mean((g["G1"] - g["G2"]).values)
pop = pd.read_csv(os.path.join(HERE, "results", "q1_population_predictions.csv.gz"))
pop["npairs"] = (pop["T"] * pop["p"]) * (pop["T"] * pop["p"] - 1) / 2
pop["flip_share"] = pop["flips"] / pop["npairs"]
S2 = float((pop.lam < 1e-6).mean()); S3 = float((pop.flip_share <= 0.02).mean())
san = dict(S1_burnin_mixed=dict(value=S1, ok=bool(S1 < 0.02)), S2_CF2_definable=dict(share_lam_at_floor=S2, ok=bool(S2 <= 0.10)),
           S3_CF2_m1_detectability=dict(share_cases_flip_le_2pct=S3, ok=bool(S3 >= 0.90)),
           S4_B1_reproduces=dict(gain=S4[0], ci=list(S4[1]), ok=bool(S4[1][0] > 0)))
out["sanity"] = san; dump("Sanity rules S1-S4", san)

# SHD arms: retention on SHD_x1n (descriptive, no threshold)
sh = df[(df.study == "main") & df.shd_x1n.notna() & df.family.isin(ay.NONSC)]
gs = ay._graph_table(sh, "shd_x1n", ["arm"])
res = {}
if {"G1", "G2"} <= set(gs.columns):
    gain = (gs["G2"] - gs["G1"]).values        # SHD: lower is better, so the artefact gain is G2 - G1
    mu, ci, _ = ay.boot_mean(gain); res["SHD_x1n_G2_minus_G1"] = dict(mean=mu, ci=list(ci), G1=float(gs["G1"].mean()), G2=float(gs["G2"].mean()))
    for arm in ("CF1_0.25", "CF1_4", "CF2_m1"):
        if arm in gs.columns:
            R, rci, _ = ay.boot_ratio((gs["G2"] - gs[arm]).values, gain); res[f"retention_{arm}"] = dict(R=R, ci=list(rci), mean=float(gs[arm].mean()))
out["shd_arms"] = res; dump("SHD_x1n arms (non-SC pooled; T in {3, 5}); retention = (G2 - arm)/(G2 - G1)", res)

# oracle calibration (consistency check of the simulation, not a mechanism claim)
arm_map = {"G1": "O_G1", "G2s": "O_G2s", "CF1_0.25": "O_CF1_0.25", "CF1_0.5": "O_CF1_0.5", "CF1_2": "O_CF1_2", "CF1_4": "O_CF1_4",
           "CF2_m1": "O_CF2_m1", "CF2_m3": "O_CF2_m3", "CF2_m10": "O_CF2_m10"}
rows = []
for arm, col in arm_map.items():
    d = df[(df.study == "main") & (df.arm == arm)].merge(pop[["family", "p", "seed", "T", col]].rename(columns={col: "oracle"}), on=["family", "p", "seed", "T"])
    d["arm"] = arm; rows.append(d)
cal = pd.concat(rows)
cell = cal.groupby(["family", "p", "T", "arm"]).agg(obs=("oa_full", "mean"), orc=("oracle", "mean")).reset_index()
x, y = cell.orc.values, cell.obs.values
slope, icpt = np.polyfit(x, y, 1); r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
calib = dict(cells=int(len(cell)), r2=r2, slope=float(slope), intercept=float(icpt))
for fam_set, nm in ((ay.NONSC, "nonSC"), (ay.SC, "SC")):
    c = cell[cell.family.isin(fam_set)]; sl_, ic_ = np.polyfit(c.orc, c.obs, 1); calib[nm] = dict(r2=float(np.corrcoef(c.orc, c.obs)[0, 1] ** 2), slope=float(sl_), intercept=float(ic_))
out["oracle_calibration_oa_full"] = calib; dump("Oracle calibration: observed OA_full vs population threshold-oracle OA (cell level, all arms)", calib)

# arm means table
tab = m.groupby("arm").agg(oa_x1=("oa_x1", "mean"), oa_full=("oa_full", "mean")).round(4)
lines.append("== Mean OA by arm (non-SC, main study, all T)"); lines.append(tab.to_string()); lines.append("")
out["arm_means"] = tab.to_dict()
open(os.path.join(HERE, "results", "analysis_Q1.txt"), "w").write("\n".join(lines))
json.dump(out, open(os.path.join(HERE, "results", "analysis_Q1.json"), "w"), indent=1, default=float)
print("\n".join(lines))
