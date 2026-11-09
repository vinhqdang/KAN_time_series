"""Build the CausalRivers stage0 caches that empirical/ and empirical2/ read, at the locations of ../datapaths.py.

This is research/ts/stage0/load_raw.py followed by research/ts/stage0/prep_pool.py, with the output directory taken
from datapaths (THEORY2_CR_CACHE) instead of the hard-coded path in stage0/common.py.  The numerical steps are the
stage0 functions themselves (common.aggregate, common.fill_short_gaps, imported, not copied), so the arrays are the
ones the published analyses used.  Writes <region>_x.npy, <region>_ids.npy, <region>_agg_{6h,12h,24h}.npy and
<region>_eligible.npy for east_germany and bavaria, and prints the stage0 pool statistics.
Needs about 3 GB of RAM and a few minutes.  Usage:  OMP_NUM_THREADS=1 python3 -I build_cr_cache.py [x|agg|all]"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "ts", "stage0"))
import numpy as np
from datapaths import CR_CACHE, CR_PRODUCT
import common as stage0   # research/ts/stage0/common.py (REGIONS, RES, MIN_FRAC, aggregate, fill_short_gaps)


def build_x():
    """stage0/load_raw.py: 15-min float32 arrays and station ids from rivers_ts_<region>.csv."""
    import pandas as pd
    for reg in stage0.REGIONS:
        f = f"{CR_PRODUCT}/rivers_ts_{reg}.csv"
        cols = pd.read_csv(f, nrows=1).columns
        dt = {c: np.float32 for c in cols[1:]}
        df = pd.read_csv(f, dtype=dt, parse_dates=[0], index_col=0)
        ids = np.array([int(c) for c in df.columns])
        np.save(f"{CR_CACHE}/{reg}_x.npy", df.values.astype(np.float32))
        np.save(f"{CR_CACHE}/{reg}_ids.npy", ids)
        print(reg, df.shape, float(np.isnan(df.values).mean()), df.index[0], df.index[-1], flush=True)


def build_agg():
    """stage0/prep_pool.py: block means at 6 h / 12 h / 24 h, short-gap interpolation, eligibility mask."""
    for reg in stage0.REGIONS:
        x = np.load(f"{CR_CACHE}/{reg}_x.npy", mmap_mode="r"); ids = np.load(f"{CR_CACHE}/{reg}_ids.npy")
        ok_all = np.ones(len(ids), bool)
        stats = {"n_stations": len(ids)}
        for rn, k in stage0.RES.items():
            a = stage0.aggregate(x, k)
            pass20 = np.isnan(a).mean(0) <= 0.20
            f = stage0.fill_short_gaps(a)
            complete = ~np.isnan(f).any(0)
            stats[rn] = {"pass_20pct": int(pass20.sum()), "complete_after_interp": int((pass20 & complete).sum())}
            ok_all &= pass20 & complete
            np.save(f"{CR_CACHE}/{reg}_agg_{rn}.npy", f.astype(np.float32))
        stats["eligible_all_res"] = int(ok_all.sum())
        np.save(f"{CR_CACHE}/{reg}_eligible.npy", ok_all)
        print(reg, stats, flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("-h", "--help"):
        print(__doc__); sys.exit(0)
    os.makedirs(CR_CACHE, exist_ok=True)
    if what in ("x", "all"): build_x()
    if what in ("agg", "all"): build_agg()
