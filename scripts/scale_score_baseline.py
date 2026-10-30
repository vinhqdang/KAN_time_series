"""X4 add-on: SCORE (Rolland et al. 2022), the strongest non-linear order-based baseline that can run at d>=100, on the same
fresh seeds 600-602 and data as scale_instantaneous.py final (NoGAM, DAGMA-nonlinear and GraN-DAG cannot run at this size)."""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))
import pandas as pd
from src.synthetic import generate_instantaneous_anm
from instantaneous_dag_benchmark import score_alg
from spade_variants import score

rows = []
for d in [int(x) for x in os.environ.get("DS", "50,100").split(",")]:
    for s in (600, 601, 602):
        X, A = generate_instantaneous_anm(3000, d, seed=s); t0 = time.time()
        try:
            au, ap, f1 = score(score_alg(X, s), A); dt = time.time() - t0
            rows.append(dict(d=d, seed=s, method="SCORE", auroc=au, auprc=ap, time_s=round(dt, 1)))
            print(f"d={d} s{s} SCORE AUROC={au:.3f} AUPRC={ap:.3f} t={dt:.0f}s", flush=True)
        except Exception as e:
            rows.append(dict(d=d, seed=s, method="SCORE", auroc=float("nan"), auprc=float("nan"), time_s=float("nan"))); print("SCORE FAILED", d, s, e, flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(__file__), "..", "experimental_results", "x4_score.csv"), index=False)
