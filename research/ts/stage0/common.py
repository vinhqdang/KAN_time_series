import numpy as np, pickle, networkx as nx
CACHE = "/tmp/claude-0/dl/cache"
PRODUCT = "/tmp/claude-0/dl/pz/product"
REGIONS = ["east_germany", "bavaria"]
RES = {"6h": 24, "12h": 48, "24h": 96}      # 15-min steps per aggregated step
MIN_FRAC = 0.5                               # agg step valid if >=50% of its 15-min values present

def load_x(reg):
    return np.load(f"{CACHE}/{reg}_x.npy", mmap_mode="r"), np.load(f"{CACHE}/{reg}_ids.npy")

def aggregate(x, k):
    T, N = x.shape
    r = np.asarray(x[: T // k * k]).reshape(T // k, k, N)
    cnt = np.isfinite(r).sum(1)
    s = np.nansum(r, 1)
    out = np.where(cnt >= MIN_FRAC * k, s / np.maximum(cnt, 1), np.nan)
    return out.astype(np.float64)

def fill_short_gaps(a, maxgap=2):
    """linear interpolation of interior NaN runs of length <= maxgap; other NaNs remain. a: (T,N)"""
    a = a.copy()
    T, N = a.shape
    for j in range(N):
        v = a[:, j]
        nan = np.isnan(v)
        if not nan.any():
            continue
        d = np.diff(np.concatenate([[0], nan.astype(int), [0]]))
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        for s, e in zip(starts, ends):
            if e - s <= maxgap and s > 0 and e < T:
                v[s:e] = np.interp(np.arange(s, e), [s - 1, e], [v[s - 1], v[e]])
    return a

def load_graph(reg):
    # pickle contains only networkx classes (checked with pickletools beforehand)
    return pickle.load(open(f"{PRODUCT}/rivers_{reg}.p", "rb"))
