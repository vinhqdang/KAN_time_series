"""Stage 0 step 1: pools, subgraph sampling (seeds 5000..5029 per size and region), tau_e estimation, labels.
Run after load_raw.py and prep_pool.py."""
import numpy as np, json, networkx as nx
from scipy.signal import correlate
from common import *

SEEDS = list(range(5000, 5030)); SIZES = [5, 10]
MIN_WINDOW_DAYS = 365
MAXLAG = 14 * 96     # 14 days in 15-min steps

def run_len_best(mask):
    best = (0, 0); s = None
    for i, v in enumerate(np.append(mask, False)):
        if v and s is None: s = i
        if not v and s is not None:
            if i - s > best[1] - best[0]: best = (s, i)
            s = None
    return best

def day_valid(reg):
    """per station: day is valid if all 6h, 12h, 24h interpolated values on that day are present"""
    ok = None
    for rn, k in RES.items():
        a = np.load(f"{CACHE}/{reg}_agg_{rn}.npy")
        per = 96 // k
        v = ~np.isnan(a).reshape(-1, per, a.shape[1]).any(1)   # (days, N)
        ok = v if ok is None else ok & v
    return ok

def sample_subgraph(rng, Gu, pool, d, dv, ix):
    """random connected subgraph via random frontier growth, rejection if no common valid window >= 365 days"""
    pool = set(pool); nodes_l = sorted(pool)
    rej = 0
    while True:
        s = nodes_l[rng.integers(len(nodes_l))]
        S = [s]; front = set(n for n in Gu[s] if n in pool)
        while len(S) < d and front:
            fl = sorted(front); n = fl[rng.integers(len(fl))]
            S.append(n); front.discard(n)
            front |= set(m for m in Gu[n] if m in pool and m not in S)
        if len(S) < d: rej += 1; continue
        m = dv[:, [ix[n] for n in S]].all(1)
        a, b = run_len_best(m)
        if b - a >= MIN_WINDOW_DAYS: return sorted(S), (a, b), rej
        rej += 1

def ccf(x, y, L):
    """pairwise-complete Pearson corr of x[t], y[t+k] for k=-L..L (k>0: x leads). x,y 1-D with NaN."""
    mx, my = np.isfinite(x).astype(float), np.isfinite(y).astype(float)
    x0, y0 = np.nan_to_num(x), np.nan_to_num(y)
    N = len(x)
    def c(a, b):   # sum_t a[t] b[t+k], k=-L..L
        full = correlate(b, a, mode="full", method="fft")
        return full[N - 1 - L: N + L]
    n = c(mx, my); Sx = c(x0, my); Sy = c(mx, y0)
    Sxx = c(x0 * x0, my); Syy = c(mx, y0 * y0); Sxy = c(x0, y0)
    with np.errstate(all="ignore"):
        cov = Sxy / n - Sx / n * Sy / n
        vx = Sxx / n - (Sx / n) ** 2; vy = Syy / n - (Sy / n) ** 2
        r = cov / np.sqrt(vx * vy)
    r[n < 2000] = np.nan
    return np.arange(-L, L + 1), r

def main():
    out = {"subgraphs": []}; edges = []
    for reg in REGIONS:
        G = load_graph(reg); x, ids = load_x(reg); ix = {n: i for i, n in enumerate(ids)}
        dv = day_valid(reg)
        miss20 = np.ones(len(ids), bool)
        for rn, k in RES.items():     # 20% rule on the raw aggregated series
            a = aggregate(x, k); miss20 &= np.isnan(a).mean(0) <= 0.20
        pool = [int(n) for n in ids[miss20]]
        Gu = G.to_undirected()
        info = {"pool": len(pool)}
        for d in SIZES:
            used = set()
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                while True:     # avoid duplicate node sets within a cell
                    S, (a, b), rej = sample_subgraph(rng, Gu, pool, d, dv, ix)
                    if tuple(S) not in used: break
                used.add(tuple(S))
                sg = {"region": reg, "d": d, "seed": seed, "nodes": [int(n) for n in S],
                      "day0": int(a), "day1": int(b), "rejections": int(rej)}
                E = [(int(u), int(v)) for u, v in G.subgraph(S).edges]
                sg["edges"] = E
                out["subgraphs"].append(sg)
                s0, s1 = a * 96, b * 96
                xx = np.asarray(x[s0:s1], dtype=np.float64)
                for (u, v) in E:
                    du = np.diff(xx[:, ix[u]]); dvv = np.diff(xx[:, ix[v]])
                    lags, r = ccf(du, dvv, MAXLAG)
                    pos = lags >= 0
                    rp = np.where(pos, r, -np.inf); rp = np.where(np.isnan(rp), -np.inf, rp)
                    k = int(np.argmax(rp)); lag = int(lags[k]); rpk = float(r[k])
                    mir = float(r[len(lags) // 2 - lag]) if lag > 0 else float("nan")
                    edges.append({"region": reg, "d": d, "seed": seed, "u": u, "v": v, "tau_e": lag,
                                  "r_peak": rpk, "r_mirror": mir,
                                  "r_lag0": float(r[len(lags) // 2]),
                                  "n_window_days": b - a})
                print(reg, d, seed, S, b - a, "rej", rej, flush=True)
        out[reg] = info
    json.dump(out, open("subgraphs.json", "w"))
    import pandas as pd
    pd.DataFrame(edges).to_csv("edges.csv", index=False)

if __name__ == "__main__":
    main()
