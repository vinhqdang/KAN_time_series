import numpy as np, json, sys
from common import *
out = {}
for reg in REGIONS:
    x, ids = load_x(reg)
    ok_all = np.ones(len(ids), bool)
    stats = {"n_stations": len(ids)}
    for rn, k in RES.items():
        a = aggregate(x, k)
        miss = np.isnan(a).mean(0)
        pass20 = miss <= 0.20
        f = fill_short_gaps(a)
        complete = ~np.isnan(f).any(0)
        stats[rn] = {"pass_20pct": int(pass20.sum()), "complete_after_interp": int((pass20 & complete).sum())}
        ok_all &= pass20 & complete
        np.save(f"{CACHE}/{reg}_agg_{rn}.npy", f.astype(np.float32))
    stats["eligible_all_res"] = int(ok_all.sum())
    np.save(f"{CACHE}/{reg}_eligible.npy", ok_all)
    out[reg] = stats
    print(reg, stats, flush=True)
json.dump(out, open("pool_stats.json", "w"), indent=1)
