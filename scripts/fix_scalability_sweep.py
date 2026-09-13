"""
Regenerate experimental_results/scalability_sweep.csv with the SAME protocol
(and same fit_cdkan() call) used for the real, trained-to-convergence timings
in honest_causal_raw.csv (d=5,10,20,50), rather than the previous 10-step
extrapolation used by scripts/ablation_scale.py.

Reviewer-noted bug this fixes: build_scale() in build_results.py only borrows
a width d from this file when d is NOT already present in honest_causal_raw.csv
(currently true only for d=15), so Table "SPADE parameter count and fit time
vs. graph width d" was silently mixing two incompatible timing methodologies,
producing a non-monotonic d=15 spike (47.8s, vs. 1.8s/4.4s at d=10/20). This
script produces a d=15 number under the identical protocol as the rest of the
table.
"""
import os, sys, time
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
import torch
torch.set_num_threads(4)

from scripts.honest_causal_benchmark import make_dataset, fit_cdkan

RES = os.path.join(os.path.dirname(__file__), "..", "experimental_results")

if __name__ == "__main__":
    rows = []
    # d=10/20 included only as a same-protocol sanity cross-check against the
    # independently-measured honest_causal_raw.csv numbers; only d=15 is
    # actually consumed by build_scale() (d=10/20/50 already come from there).
    for d in [10, 15, 20]:
        ts, npar = [], None
        for seed in [42, 43, 44]:
            X, _ = make_dataset("nonlinear", n=2000, d=d, seed=seed)
            _, dt, npar = fit_cdkan(X, seed)
            ts.append(dt)
            print(f"d={d} seed={seed} time={dt:.2f}s params={npar}", flush=True)
        rows.append(dict(d=d, params=npar, fit_time_s=round(float(np.mean(ts)), 1)))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "scalability_sweep.csv"), index=False)
    print("Wrote scalability_sweep.csv:")
    print(pd.DataFrame(rows))
