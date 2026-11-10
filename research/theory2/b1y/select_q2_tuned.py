"""Choose the DAGMA and NOTEARS hyper-parameters from the Q2 validation-seed tuning rows (PLAN.md section 4.1, fixed rule):
mean SHD_x1n over the validation graphs, pooled over families, cells and both generators (G1, G2) on raw data; one setting per
method, used for all four arms. Ties (|difference| < 1e-12) go to the package default. Writes results/q2_tuned.json.
  python select_q2_tuned.py TUNE1.csv [TUNE2.csv ...]
"""
import sys, json, os
import pandas as pd
DEFAULT = {"DAGMA": (0.03, 0.3), "NOTEARS": (0.1, 0.3)}


def select(df):
    df = df.copy(); df[["m", "l1", "th"]] = df.method.str.split("|", expand=True); df["l1"] = df.l1.astype(float); df["th"] = df.th.astype(float)
    # pool: mean over graphs of the mean over (cell, generator) rows, i.e. plain mean over all rows (equal rows per setting)
    out = {}; table = []
    for m, g in df.groupby("m"):
        s = g.groupby(["l1", "th"]).shd_x1n.mean().sort_values()
        best = s.iloc[0]; cand = [k for k, v in s.items() if abs(v - best) < 1e-12]
        pick = DEFAULT[m] if DEFAULT[m] in cand else cand[0]
        out[m] = dict(lambda1=pick[0], w_threshold=pick[1], mean_shd_x1n=float(s[pick]), n_rows=int(len(g) // len(s)))
        table += [dict(method=m, lambda1=k[0], w_threshold=k[1], mean_shd_x1n=float(v)) for k, v in s.items()]
    return out, pd.DataFrame(table)


if __name__ == "__main__":
    df = pd.concat([pd.read_csv(f) for f in sys.argv[1:]])
    out, table = select(df)
    here = os.path.dirname(os.path.abspath(__file__))
    json.dump(out, open(os.path.join(here, "results", "q2_tuned.json"), "w"), indent=1)
    table.to_csv(os.path.join(here, "results", "q2_tuning_table.csv"), index=False)
    print(table.to_string(index=False)); print(json.dumps(out, indent=1))
