"""Shared loaders and helpers for make_figures.py and make_tables.py.

Everything is read from CSVs under research/ so the paper can be regenerated.
Run the scripts with python3.13 (pandas, numpy, scipy, matplotlib).
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../.."))
R = os.path.join(REPO, "research")

# Okabe-Ito colour-blind-safe palette
COL = {"SSC": "#0072B2", "SSR": "#E69F00", "SSA": "#009E73", "BASIL-W": "#D55E00", "BASIL": "#CC79A7",
       "GAM-ridge": "#56B4E9", "NOTEARS-linear": "#999999", "truth": "#000000"}
MARK = {"SSC": "o", "SSR": "s", "SSA": "^", "BASIL-W": "D", "BASIL": "v", "GAM-ridge": "P", "NOTEARS-linear": "X"}
LABEL = {"SSC": "SPADE-scaled", "SSR": "refit", "SSA": "adaptive GL", "BASIL-W": "BASIL-W"}

CELLS = [("C1", "d100 uniform"), ("C2", "d100 Laplace"), ("C3", "d100 hetero"), ("C4", "d100 Gaussian"),
         ("C5", "d100 n=500"), ("C6", "d100 n=2000"), ("C7", "d100 mech. 2"), ("C8", "d200 uniform")]
CELLNAME = dict(CELLS)
SHORT = {"C1": "unif", "C2": "Lap", "C3": "het", "C4": "Gau", "C5": "n500", "C6": "n2000", "C7": "mech2", "C8": "d200"}


def exists(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


def load_replication():
    """research/replication/results/all.csv (SSC, SSR, BASIL-W; 75 datasets)."""
    d = pd.read_csv(os.path.join(R, "replication/results/all.csv"))
    d = d.drop_duplicates(["cell", "seed", "M"])
    return d


def cell_of(row_d, noise, n, mech):
    if row_d == 200:
        return "C8"
    if mech == "m2":
        return "C7"
    if noise == "laplace":
        return "C2"
    if noise == "hetero":
        return "C3"
    if noise == "gauss":
        return "C4"
    if n == 500:
        return "C5"
    if n == 2000:
        return "C6"
    return "C1"


def load_adaptive():
    """research/adaptive/results/*.csv: SSA arm (and SSC again), de-duplicated by cell-seed-method."""
    fs = sorted(glob.glob(os.path.join(R, "adaptive/results/*.csv")))
    if not fs:
        return None
    a = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    a["cell"] = [cell_of(r.d, r.noise, r.n, r.mech) for r in a.itertuples()]
    a = a.drop_duplicates(["cell", "seed", "method"])
    a["M"] = a.method
    return a


def load_family(subdir, pattern="results/*.csv"):
    """Generic loader for a sibling experiment directory (replication2, adaptive2, semisyn).
    Returns None when nothing is there yet."""
    fs = sorted(glob.glob(os.path.join(R, subdir, pattern)))
    if not fs:
        return None
    try:
        return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    except Exception:
        return None


def boot_ci(x, f=np.mean, B=4000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.RandomState(seed)
    bs = [f(x[rng.randint(0, len(x), len(x))]) for _ in range(B)]
    return tuple(np.percentile(bs, [2.5, 97.5]))


def wilcoxon_p(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2 or np.all(x == 0):
        return np.nan
    return float(stats.wilcoxon(x, method="exact" if len(x) <= 25 and not (x == 0).any() else "auto").pvalue)


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for i, k in enumerate(o):
        run = max(run, (m - i) * p[k])
        adj[k] = min(1.0, run)
    return adj


def wide(df, col, cell, group="M"):
    return df[df.cell == cell].pivot(index="seed", columns=group, values=col)


def tex_escape(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def fmt(x, k=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.{k}f}"


def write_table(name, body):
    os.makedirs(os.path.join(HERE, "tables"), exist_ok=True)
    with open(os.path.join(HERE, "tables", name), "w") as fh:
        fh.write(body)
