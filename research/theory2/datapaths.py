"""Locations of the external (non-committed) datasets used by the analysis scripts in research/theory2.

Raw data are not in the repository (see DATA.md for sources, versions and SHA-256 values).  Every analysis script
that reads raw data takes its locations from this module.  Resolution order for each location:

  1. a per-dataset environment variable (listed below), if set;
  2. otherwise  $THEORY2_DATA_ROOT/<default sub-path>;
  3. THEORY2_DATA_ROOT itself defaults to  research/theory2/data  (git-ignored).

  variable                 default sub-path under THEORY2_DATA_ROOT     contents
  THEORY2_CR_PRODUCT       causalrivers/product                         unzipped CausalRivers product.zip
                                                                        (rivers_<region>.p, rivers_ts_<region>.csv, ...)
  THEORY2_CR_CACHE         causalrivers/cache                           stage0 caches <region>_{x,ids,agg_<res>,eligible}.npy
                                                                        (built by research/ts/stage0/load_raw.py, prep_pool.py)
  THEORY2_CC_CRL           causal_chambers/lt_crl_benchmark_v1          citris_1.csv, buchholz_1_obs.csv
  THEORY2_CC_CAM           causal_chambers/lt_camera_v1                 scm_{1,2,4,5}_reference.csv
  THEORY2_CC_WT_WALKS      causal_chambers/wt_walks_v1                  actuators_random_walk_*.csv, regime_jumps_*.csv
  THEORY2_CC_LT_WALKS      causal_chambers/lt_walks_v1                  actuators_white.csv
  THEORY2_CC_B3            causal_chambers/b3                           the nine b3 held-out csv files (b3/fetch_b3.py output)

Scripts whose data directories are already command-line arguments (realdata/test3.py, b3/test_b3.py, ...) keep
their arguments; reproduce.sh passes these locations to them.

`python3 datapaths.py` prints the resolved locations and whether each exists (shell-friendly KEY=VALUE lines with
`--env`).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("THEORY2_DATA_ROOT") or os.path.join(HERE, "data")

_LAYOUT = {
    "THEORY2_CR_PRODUCT": ("causalrivers", "product"),
    "THEORY2_CR_CACHE": ("causalrivers", "cache"),
    "THEORY2_CC_CRL": ("causal_chambers", "lt_crl_benchmark_v1"),
    "THEORY2_CC_CAM": ("causal_chambers", "lt_camera_v1"),
    "THEORY2_CC_WT_WALKS": ("causal_chambers", "wt_walks_v1"),
    "THEORY2_CC_LT_WALKS": ("causal_chambers", "lt_walks_v1"),
    "THEORY2_CC_B3": ("causal_chambers", "b3"),
}


def path(var):
    """Resolved directory for one of the variables in _LAYOUT."""
    return os.environ.get(var) or os.path.join(DATA_ROOT, *_LAYOUT[var])


CR_PRODUCT = path("THEORY2_CR_PRODUCT")
CR_CACHE = path("THEORY2_CR_CACHE")
CC_CRL = path("THEORY2_CC_CRL")
CC_CAM = path("THEORY2_CC_CAM")
CC_WT_WALKS = path("THEORY2_CC_WT_WALKS")
CC_LT_WALKS = path("THEORY2_CC_LT_WALKS")
CC_B3 = path("THEORY2_CC_B3")


if __name__ == "__main__":
    env = "--env" in sys.argv[1:]
    if not env:
        print("THEORY2_DATA_ROOT =", DATA_ROOT)
    for var in _LAYOUT:
        p = path(var)
        print(f"{var}={p}" if env else f"{var:22s} {'ok     ' if os.path.isdir(p) else 'MISSING'} {p}")
