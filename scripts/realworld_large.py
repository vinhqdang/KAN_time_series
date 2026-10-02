"""
Larger real-world, NON-financial validation (Revision 2): two hourly datasets.

  bike_hourly : UCI Bike Sharing, hourly file (Capital Bikeshare, Washington DC,
                2011-2012; 17,379 hourly records). Demand planning.
  beijing     : UCI Beijing PM2.5 (US Embassy Beijing, 2010-2014; 43,824 hourly
                records). Environmental monitoring / air-quality management.

Same idea as scripts/realworld_bike_sharing.py (daily data, n=730): no labelled graph
exists, so every method is scored against domain-certain BACKGROUND KNOWLEDGE.

=======================================================================
PRE-REGISTERED PROTOCOL  (written and committed BEFORE any method was run on
these two datasets; nothing below is changed after seeing results)
=======================================================================
Rows. Row t pairs the variables at hour t with lag-1 values from hour t-1. A row is
kept only if hour t-1 is present in the file at exactly one hour earlier (the bike
file omits hours with no rentals) and has no missing value. Count variables are
log1p-transformed (casual, registered, pm2.5); all columns are then z-scored.

bike_hourly (d=14)
  tier 0: yr, workingday, hr_sin, hr_cos, and lag-1 of temp, hum, windspeed, casual,
          registered
  tier 1: temp, hum, windspeed          tier 2: casual, registered
  POSITIVE (9): casual_lag->casual, registered_lag->registered, temp_lag->temp,
          temp->casual, temp->registered, workingday->casual, workingday->registered,
          hr_sin->registered, hr_cos->registered
beijing (d=14)
  tier 0: hr_sin, hr_cos, month_sin, month_cos, and lag-1 of DEWP, TEMP, PRES, Iws, pm
  tier 1: DEWP, TEMP, PRES, Iws         tier 2: pm   (pm = log1p pm2.5)
  (the categorical wind-direction column is not used)
  POSITIVE (10): pm_lag->pm, TEMP_lag->TEMP, DEWP_lag->DEWP, PRES_lag->PRES,
          Iws_lag->Iws (persistence); Iws->pm (wind disperses pollution);
          hr_sin, hr_cos, month_sin, month_cos -> TEMP (diurnal/seasonal cycle)
FORBIDDEN: every edge from a later tier to an earlier tier.
PRIMARY METRIC: background-knowledge AUROC separating POSITIVE (1) from FORBIDDEN (0)
  by each method's edge scores (binary-output methods: their binary edges).
SECONDARY: of each method's top-d edges (d = number of variables; all edges for
  binary-output methods), number forbidden / number positive recovered.
SECONDARY (SPADE only), sign/shape of the learned spline edge functions:
  bike_hourly: workingday->registered positive, workingday->casual negative;
  beijing: Iws->pm decreasing (phi(q90) < phi(q10) of the standardized cause).
  Each reported over all resamples, together with the edge's importance rank
  (a sign check on a near-zero edge is not evidence and is flagged as such).

Resampling / uncertainty.
  PRIMARY (all 9 methods): resamples r=0..5; resample r is a moving-block bootstrap
  (block 168 h = one week; RNG seed r) of n=2,000 rows from the full series, the same
  matrix for every method; method seed = r. The cubic-cost kernel methods (NoGAM)
  cannot run on the full series, which is why the common sample is n=2,000.
  SCALE CHECK (SPADE, NOTEARS-linear, DAGMA-linear, GOLEM, NOTEARS-MLP): r=0 is the
  full series, r=1,2 are full-length block bootstraps (same block length); reports
  whether conclusions change at n=17,303 / 41,543. Report mean +- std.
HYPERPARAMETERS: none tuned on these data. SPADE uses the settings of the headline
  synthetic benchmark (400 epochs, lr 5e-3, grid 8, lambda_g=0.01); baselines use their
  settings in scripts/instantaneous_dag_benchmark.py unchanged.
FAILURES are recorded in the status column and never dropped.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(ROOT, "experimental_results")
BLOCK, N_SUB, B_RES = 168, 2000, 5
SCALE_METHODS = ["SPADE", "NOTEARS-linear", "DAGMA-linear", "GOLEM", "NOTEARS-MLP"]


def load_bike_hourly():
    df = pd.read_csv(os.path.join(ROOT, "data", "bike_sharing_hourly", "hour.csv"))
    ts = pd.to_datetime(df["dteday"]) + pd.to_timedelta(df["hr"], unit="h")
    df["hr_sin"], df["hr_cos"] = np.sin(2 * np.pi * df.hr / 24), np.cos(2 * np.pi * df.hr / 24)
    df["casual"], df["registered"] = np.log1p(df.casual), np.log1p(df.registered)
    base = ["temp", "hum", "windspeed", "casual", "registered"]
    names = ["yr", "workingday", "hr_sin", "hr_cos"] + [b + "_lag" for b in base] + ["temp", "hum", "windspeed", "casual", "registered"]
    cur = df[["yr", "workingday", "hr_sin", "hr_cos"] + base].to_numpy(float)
    prev = np.vstack([np.full((1, 5), np.nan), df[base].to_numpy(float)[:-1]])
    keep = np.zeros(len(df), bool); keep[1:] = (ts.diff() == pd.Timedelta(hours=1)).to_numpy()[1:]
    X = np.column_stack([cur[:, :4], prev, cur[:, 4:]])[keep]
    tier = [0] * 9 + [1, 1, 1, 2, 2]
    pos = [("casual_lag", "casual"), ("registered_lag", "registered"), ("temp_lag", "temp"), ("temp", "casual"),
           ("temp", "registered"), ("workingday", "casual"), ("workingday", "registered"),
           ("hr_sin", "registered"), ("hr_cos", "registered")]
    return X[~np.isnan(X).any(1)], names, tier, pos


def load_beijing():
    df = pd.read_csv(os.path.join(ROOT, "data", "beijing_pm25", "PRSA_data_2010.1.1-2014.12.31.csv"))
    ts = pd.to_datetime(dict(year=df.year, month=df.month, day=df.day, hour=df.hour))
    df["hr_sin"], df["hr_cos"] = np.sin(2 * np.pi * df.hour / 24), np.cos(2 * np.pi * df.hour / 24)
    df["month_sin"], df["month_cos"] = np.sin(2 * np.pi * df.month / 12), np.cos(2 * np.pi * df.month / 12)
    df["pm"] = np.log1p(df["pm2.5"])
    base = ["DEWP", "TEMP", "PRES", "Iws", "pm"]
    names = ["hr_sin", "hr_cos", "month_sin", "month_cos"] + [b + "_lag" for b in base] + ["DEWP", "TEMP", "PRES", "Iws", "pm"]
    cur = df[["hr_sin", "hr_cos", "month_sin", "month_cos"] + base].to_numpy(float)
    prev = np.vstack([np.full((1, 5), np.nan), df[base].to_numpy(float)[:-1]])
    keep = np.zeros(len(df), bool); keep[1:] = (ts.diff() == pd.Timedelta(hours=1)).to_numpy()[1:]
    X = np.column_stack([cur[:, :4], prev, cur[:, 4:]])[keep]
    tier = [0] * 9 + [1, 1, 1, 1, 2]
    pos = [("pm_lag", "pm"), ("TEMP_lag", "TEMP"), ("DEWP_lag", "DEWP"), ("PRES_lag", "PRES"), ("Iws_lag", "Iws"),
           ("Iws", "pm"), ("hr_sin", "TEMP"), ("hr_cos", "TEMP"), ("month_sin", "TEMP"), ("month_cos", "TEMP")]
    return X[~np.isnan(X).any(1)], names, tier, pos


LOADERS = {"bike_hourly": load_bike_hourly, "beijing": load_beijing}


def block_bootstrap(X, r, n, block=BLOCK):
    rng = np.random.RandomState(r)
    nb = int(np.ceil(n / block)); starts = rng.randint(0, len(X) - block + 1, size=nb)
    return np.vstack([X[s:s + block] for s in starts])[:n]


def metrics(S, tier, pos, names):
    d = len(names); idx = {n: i for i, n in enumerate(names)}; tier = np.array(tier)
    S = np.nan_to_num(np.abs(np.asarray(S, float))); np.fill_diagonal(S, 0.0)
    F = [(u, v) for u in range(d) for v in range(d) if tier[u] > tier[v]]
    P = [(idx[u], idx[v]) for u, v in pos]
    sc = np.r_[[S[u, v] for u, v in P], [S[u, v] for u, v in F]]
    y = np.r_[np.ones(len(P)), np.zeros(len(F))]
    au = roc_auc_score(y, sc) if sc.max() > sc.min() else 0.5
    flat = [(S[u, v], u, v) for u in range(d) for v in range(d) if u != v]
    if set(np.unique(S)).issubset({0.0, 1.0}):
        chosen = {(u, v) for s, u, v in flat if s > 0}
    else:
        chosen = {(u, v) for s, u, v in sorted(flat, reverse=True)[:d]}
    return dict(bk_auroc=au, n_edges=len(chosen), n_forbidden=len(chosen & set(F)), n_positive=len(chosen & set(P)),
                n_pos_total=len(P))


def main():
    from instantaneous_dag_benchmark import METHODS
    ds = sys.argv[1]; part = sys.argv[2] if len(sys.argv) > 2 else "primary"
    X0, names, tier, pos = LOADERS[ds]()
    print(ds, "usable rows:", X0.shape, flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    tag = f"realworld_{ds}_{part}"
    if part == "primary":
        jobs = [(r, list(METHODS), block_bootstrap(X0, r, N_SUB)) for r in range(B_RES + 1)]
    else:
        jobs = [(r, SCALE_METHODS, X0 if r == 0 else block_bootstrap(X0, r, len(X0))) for r in range(3)]
    only = [a for a in sys.argv[3:]]
    rows, adj = [], {}
    raw_path, npz_path = os.path.join(OUT_DIR, tag + "_raw.csv"), os.path.join(OUT_DIR, tag + "_adj.npz")
    if os.path.exists(raw_path):                         # resume an interrupted run
        prev = pd.read_csv(raw_path); rows = prev[prev.status == "ok"].to_dict("records")
        adj = dict(np.load(npz_path)) if os.path.exists(npz_path) else {}
        print("resuming with", len(rows), "completed method-runs", flush=True)
    done = {(int(x["resample"]), x["method"]) for x in rows}
    for r, methods, X in jobs:
        for m in methods:
            if only and m not in only: continue
            if (r, m) in done: continue
            t0 = time.time()
            try:
                S = np.asarray(METHODS[m](X, r), float); met = metrics(S, tier, pos, names)
                rows.append(dict(dataset=ds, part=part, n=len(X), resample=r, method=m, status="ok",
                                 time_s=round(time.time() - t0, 1), **met)); adj[f"{m}|{r}"] = S
                print(f"[r{r}] {m:16s} bkAUROC={met['bk_auroc']:.3f} forb={met['n_forbidden']}/{met['n_edges']} "
                      f"pos={met['n_positive']}/{met['n_pos_total']} t={time.time() - t0:.0f}s", flush=True)
            except Exception as e:
                rows.append(dict(dataset=ds, part=part, n=len(X), resample=r, method=m, status=f"FAILED: {e}",
                                 time_s=round(time.time() - t0, 1)))
                print(f"[r{r}] {m} FAILED: {e}", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, tag + "_raw.csv"), index=False)
            np.savez_compressed(os.path.join(OUT_DIR, tag + "_adj.npz"), **adj)
    df = pd.DataFrame(rows); ok = df[df.status == "ok"]
    summ = ok.groupby("method").agg(bk_auroc_mean=("bk_auroc", "mean"), bk_auroc_std=("bk_auroc", "std"),
                                    forbidden_mean=("n_forbidden", "mean"), positive_mean=("n_positive", "mean"),
                                    edges_mean=("n_edges", "mean"), time_mean=("time_s", "mean"), n=("bk_auroc", "size"))
    summ.to_csv(os.path.join(OUT_DIR, tag + "_summary.csv")); print(summ.round(3).to_string())
    print("failures:", int((df.status != "ok").sum()))


if __name__ == "__main__":
    main()
