"""Stage A design inspection of the experimenter-set inputs only (no sensor data, no correlations):
fraction of steps at which each input changes, median |step| when it changes, range used, number of
direction reversals.  Usage: python3 -I inspect_walks.py file.csv input1 input2 ..."""
import sys, csv
import numpy as np
f = sys.argv[1]
with open(f) as fh:
    r = csv.reader(fh); head = next(r); rows = list(r)
cols = {h: i for i, h in enumerate(head)}
for c in sys.argv[2:]:
    x = np.array([float(row[cols[c]]) for row in rows]); dx = np.diff(x); ch = dx != 0
    sg = np.sign(dx[ch]); rev = int((sg[1:] != sg[:-1]).sum())
    print(f"{c}: range [{x.min():g},{x.max():g}] changes {ch.mean():.3f} med|step| {np.median(np.abs(dx[ch])) if ch.any() else 0:g} reversals {rev} levels {len(np.unique(x))}")
