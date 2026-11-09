"""Merge the 8 B'1 test shards into results/b1x_test.csv.gz."""
import glob, os, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
fs = sorted(glob.glob(os.path.join(HERE, "results", "shards", "test_*.csv.gz")))
assert len(fs) == 8, fs
df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
df.sort_values(["family", "p", "seed", "T", "gen"]).to_csv(os.path.join(HERE, "results", "b1x_test.csv.gz"), index=False)
print(len(df), "rows")
