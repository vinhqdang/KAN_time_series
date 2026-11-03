"""usage: run.py val|test [data...]  data in rivers netsim synth (default all). Output results_<phase>_<data>.csv"""
import sys, time, pandas as pd
from multiprocessing import Pool
import gapb

def job(a):
    meta, X, Y = a
    return gapb.run_dataset(meta, X, Y)

if __name__ == "__main__":
    phase = sys.argv[1]; which = sys.argv[2:] or ["synth", "rivers", "netsim"]
    seeds = range(6000, 6005) if phase == "val" else range(6005, 6030)
    for w in which:
        gen = {"rivers": lambda: gapb.rivers_datasets(set(seeds)), "synth": lambda: gapb.synth_datasets(seeds),
               "netsim": lambda: gapb.netsim_datasets(phase)}[w]()
        jobs = list(gen); R, N = [], []; t0 = time.time()
        with Pool(2) as pool:
            for i, (r, n) in enumerate(pool.imap_unordered(job, jobs, chunksize=1)):
                R += r; N += n
                if i % 10 == 0: print(w, i, len(jobs), round(time.time() - t0), flush=True)
        pd.DataFrame(R).to_csv(f"results_{phase}_{w}.csv", index=False)
        pd.DataFrame(N).to_csv(f"nec_{phase}_{w}.csv", index=False)
        print(w, "done", round(time.time() - t0), flush=True)
