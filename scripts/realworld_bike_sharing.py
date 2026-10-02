"""
Real-world validation on a NON-financial managerial setting: daily bike-sharing
demand (UCI Bike Sharing dataset, Fanaee-T & Gama 2014; Capital Bikeshare,
Washington DC, 2011-2012, 731 days). Added in Revision 2 in response to
Reviewer 1's repeated concern that real-data validation was confined to a narrow
financial setting.

WHY THIS DATASET. Demand planning is a core operations-management decision
problem (fleet rebalancing, staffing, capacity), and the data come with
well-established background knowledge that gives partial ground truth WITHOUT
any labelled graph: (i) weather and calendar are determined outside the system
(demand cannot cause the weather), (ii) yesterday cannot be caused by today, and
(iii) a few mechanisms are domain-certain (commuters ride on working days,
warmth drives leisure riding, demand is persistent). This lets us score every
method against domain knowledge instead of only reporting "a plausible-looking
graph".

=======================================================================
PRE-REGISTERED PROTOCOL  (written and committed BEFORE any method was run on
this data; nothing below was changed after seeing results)
=======================================================================
Variables (d=12). Row t (t=1..730) contains
  tier 0 ("pre-determined"): yr_t, workingday_t, and the lag-1 values
      temp_{t-1}, hum_{t-1}, windspeed_{t-1}, casual_{t-1}, registered_{t-1}
  tier 1 ("weather"):  temp_t, hum_t, windspeed_t
  tier 2 ("demand"):   casual_t, registered_t
cnt is excluded (it is exactly casual+registered). Every method receives the
same standardized 730x12 matrix and returns a [cause, effect] edge-score matrix.

FORBIDDEN set F (edges that are impossible given the tiers, i.e. from a later tier
to an earlier one): tier1->tier0 (21), tier2->tier0 (14), tier2->tier1 (6) = 41.
POSITIVE set P (7 domain-certain edges):
  casual_{t-1}->casual_t, registered_{t-1}->registered_t   (demand persistence)
  temp_{t-1}->temp_t                                         (weather persistence)
  temp_t->casual_t, temp_t->registered_t                     (warmth drives riding)
  workingday_t->casual_t, workingday_t->registered_t         (commute/leisure split)
All other ordered pairs are NEUTRAL and are not scored.

PRIMARY METRIC: background-knowledge AUROC = AUROC separating P (label 1) from F
(label 0) using each method's edge scores (binary-output methods use their binary
edges). 0.5 = no better than chance at telling real mechanisms from impossible ones.
SECONDARY: among each method's top-12 edges (continuous scores) or all of its edges
(binary-output methods): number in F (violations), number in P (recovered, of 7).
SECONDARY (SPADE only): sign/shape checks of the learned spline edge functions
against domain-certain signs (workingday->registered positive, workingday->casual
negative, temp->casual increasing then flattening).

UNCERTAINTY: resample 0 = the original ordered data; resamples 1..5 = moving-block
bootstrap (block length 30 days, resample index as RNG seed) of the 730 rows. Each
method is run once per resample with seed = resample index. Report mean +- std
over the 6 resamples.

HYPERPARAMETERS: NONE tuned on this data. SPADE uses exactly the settings of the
headline synthetic benchmark (400 epochs, lr 5e-3, grid 8, lambda_g=0.01, chosen on
synthetic held-out seeds 100-119). Every baseline uses its settings from
scripts/instantaneous_dag_benchmark.py unchanged.

FAILURES are recorded in the output (status column) and never silently dropped.
"""
import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data", "bike_sharing", "day.csv")
OUT_DIR = os.path.join(ROOT, "experimental_results")

NAMES = ["yr", "workingday", "temp_lag", "hum_lag", "wind_lag", "casual_lag", "registered_lag",
         "temp", "hum", "wind", "casual", "registered"]
TIER = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2])
IDX = {n: i for i, n in enumerate(NAMES)}
POSITIVE = [("casual_lag", "casual"), ("registered_lag", "registered"), ("temp_lag", "temp"),
            ("temp", "casual"), ("temp", "registered"),
            ("workingday", "casual"), ("workingday", "registered")]
B_RESAMPLES = 5
BLOCK = 30


def load_design():
    df = pd.read_csv(DATA)
    cur = df[["yr", "workingday", "temp", "hum", "windspeed", "casual", "registered"]].to_numpy(float)
    # rows t=1..730: [yr, workingday, lag(temp,hum,wind,casual,registered), temp,hum,wind,casual,registered]
    lag = cur[:-1, 2:7]
    now = cur[1:]
    X = np.column_stack([now[:, 0], now[:, 1], lag, now[:, 2:7]])
    return X


def forbidden_pairs():
    return [(u, v) for u in range(12) for v in range(12) if TIER[u] > TIER[v]]


def positive_pairs():
    return [(IDX[u], IDX[v]) for u, v in POSITIVE]


def block_bootstrap(X, r, block=BLOCK):
    if r == 0:
        return X
    rng = np.random.RandomState(r)
    n = len(X)
    nb = int(np.ceil(n / block))
    starts = rng.randint(0, n - block + 1, size=nb)
    return np.vstack([X[s:s + block] for s in starts])[:n]


def background_metrics(S, topk=12):
    """S is [cause, effect]. Returns dict of primary and secondary metrics."""
    S = np.nan_to_num(np.abs(np.asarray(S, float)))
    np.fill_diagonal(S, 0.0)
    F, P = forbidden_pairs(), positive_pairs()
    pf = np.array([S[u, v] for u, v in F]); pp = np.array([S[u, v] for u, v in P])
    y = np.r_[np.ones(len(pp)), np.zeros(len(pf))]
    sc = np.r_[pp, pf]
    auroc = roc_auc_score(y, sc) if sc.max() > sc.min() else 0.5
    binary = set(np.unique(S)).issubset({0.0, 1.0})
    flat = [(S[u, v], u, v) for u in range(12) for v in range(12) if u != v]
    if binary:
        chosen = {(u, v) for s, u, v in flat if s > 0}
    else:
        chosen = {(u, v) for s, u, v in sorted(flat, reverse=True)[:topk]}
    return dict(bk_auroc=auroc, n_edges=len(chosen),
                n_forbidden=len(chosen & set(F)), n_positive=len(chosen & set(P)))


def main():
    from instantaneous_dag_benchmark import METHODS
    X0 = load_design()
    assert X0.shape == (730, 12), X0.shape
    os.makedirs(OUT_DIR, exist_ok=True)
    raw_path = os.path.join(OUT_DIR, "realworld_bike_raw.csv")
    npz_path = os.path.join(OUT_DIR, "realworld_bike_adj.npz")
    methods = list(METHODS.keys())
    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    if only:
        methods = [m for m in methods if m in only]
    rows, adj = [], {}
    for r in range(B_RESAMPLES + 1):
        X = block_bootstrap(X0, r)
        for m in methods:
            t0 = time.time()
            try:
                S = np.asarray(METHODS[m](X, r), float)
                met = background_metrics(S)
                rows.append(dict(resample=r, method=m, status="ok", time_s=round(time.time() - t0, 1), **met))
                adj[f"{m}|{r}"] = S
                print(f"[r{r}] {m:16s} bkAUROC={met['bk_auroc']:.3f} "
                      f"forbidden={met['n_forbidden']}/{met['n_edges']} pos={met['n_positive']}/7 "
                      f"t={time.time() - t0:.0f}s", flush=True)
            except Exception as e:
                rows.append(dict(resample=r, method=m, status=f"FAILED: {e}", time_s=round(time.time() - t0, 1)))
                print(f"[r{r}] {m} FAILED: {e}", flush=True)
            pd.DataFrame(rows).to_csv(raw_path, index=False)
            np.savez_compressed(npz_path, **adj)
    df = pd.DataFrame(rows)
    ok = df[df.status == "ok"]
    summ = ok.groupby("method").agg(
        bk_auroc_mean=("bk_auroc", "mean"), bk_auroc_std=("bk_auroc", "std"),
        forbidden_mean=("n_forbidden", "mean"), positive_mean=("n_positive", "mean"),
        edges_mean=("n_edges", "mean"), time_mean=("time_s", "mean"), n=("bk_auroc", "size"))
    summ.to_csv(os.path.join(OUT_DIR, "realworld_bike_summary.csv"))
    print(summ.round(3).to_string())
    print("failures:", int((df.status != "ok").sum()))


if __name__ == "__main__":
    main()
