"""
X1 final evaluation of the frozen candidate SPADE-AB (= 'adaptive_boot' in spade_variants.py), chosen on
validation seeds 200-219 (d=6,10) / 200-204 (d=20) by highest mean AUROC at every width.
Evaluated (a) on fresh seeds 300-319 (d=6,10) and 300-304 (d=20) against the published SPADE ('base'),
and (b) on the official reporting seeds (0-4 / 0-2), where stored SCORE and NoGAM results exist for a paired comparison.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))
import pandas as pd
from src.synthetic import generate_instantaneous_anm
import spade_variants as V

ER = os.path.join(os.path.dirname(__file__), "..", "experimental_results")
SETS = [("fresh", {6: range(300, 320), 10: range(300, 320), 20: range(300, 305)}),
        ("official", {6: range(0, 5), 10: range(0, 5), 20: range(0, 3)})]
rows = []
for part, cfg in SETS:
    for d, ss in cfg.items():
        for s in ss:
            X, A = generate_instantaneous_anm(1500, d, seed=s)
            for name, key in [("SPADE", "base"), ("SPADE-AB", "adaptive_boot")]:
                t0 = time.time(); au, ap, f1 = V.score(V.VARIANTS[key](X, s), A)
                rows.append(dict(part=part, d=d, seed=s, method=name, auroc=au, auprc=ap, f1=f1, time_s=round(time.time() - t0, 1)))
                print(f"{part} d={d} s{s} {name:9s} AUROC={au:.3f} AUPRC={ap:.3f} t={time.time()-t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(ER, "x1_final_raw.csv"), index=False)
df = pd.DataFrame(rows); print(df.groupby(["part", "d", "method"])[["auroc", "auprc", "f1", "time_s"]].mean().round(3))
