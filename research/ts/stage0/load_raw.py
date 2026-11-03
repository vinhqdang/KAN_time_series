"""Load CausalRivers CSVs once, store float32 15-min arrays outside the repo."""
import sys, numpy as np, pandas as pd
src, out = sys.argv[1], sys.argv[2]
for reg in ["east_germany", "bavaria"]:
    f = f"{src}/rivers_ts_{reg}.csv"
    cols = pd.read_csv(f, nrows=1).columns
    dt = {c: np.float32 for c in cols[1:]}
    df = pd.read_csv(f, dtype=dt, parse_dates=[0], index_col=0)
    ids = np.array([int(c) for c in df.columns])
    np.save(f"{out}/{reg}_x.npy", df.values.astype(np.float32))
    np.save(f"{out}/{reg}_ids.npy", ids)
    print(reg, df.shape, float(np.isnan(df.values).mean()), df.index[0], df.index[-1], df.index.to_series().diff().value_counts().head(2).to_dict())
