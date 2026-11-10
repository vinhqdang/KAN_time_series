import glob, sys, os
import pandas as pd
for tag, n_per in (("q2", 12), ("q5", 12)):
    fs = sorted(glob.glob(f"results/shards/{tag}/{tag}_test_*.csv"))
    done = 0; secs = 0.0; st = {}
    for f in fs:
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if "seed" not in d.columns: continue
        done += d.seed.nunique(); secs += float(d.secs.sum()); 
        for k, v in d.status.value_counts().items(): st[k] = st.get(k, 0) + int(v)
    total = (24 if tag == "q2" else 16) * n_per
    print(f"{tag}: shards started {len(fs)}, graph-seeds done {done}/{total}, fit CPU-h so far {secs/3600:.2f}, status {st}")
