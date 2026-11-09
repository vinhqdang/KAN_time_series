# Reproducing research/theory2

```bash
cd research/theory2
python3.13 -m pip install -r requirements.txt     # pinned versions actually used; Python 3.13.17
./reproduce.sh                                     # lean + graph-only checks (no data, no downloads)
./reproduce.sh sim                                 # simulation studies (hours)
export THEORY2_DATA_ROOT=/path/to/data             # optional; default research/theory2/data (git-ignored)
./reproduce.sh fetch realdata                      # download, verify checksums, run the real-data tests
./reproduce.sh all                                 # everything, in that order
```

`reproduce.sh` runs every Python script with `python3 -I`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` and `MKL_NUM_THREADS=1`. `-I` is isolated mode: no import from the current directory, no user site-packages. Captured stdout goes to `_repro_out/` (git-ignored). Where a committed copy of that output exists, the script compares the two and prints `identical` or `DIFFERS`. Scripts that write their own result files overwrite the committed files, so `git diff --stat` at the end lists every output that changed. The exit status is non-zero if any step failed or any checksum did not match.

## Environment

- **Python 3.13.17.** Package versions are pinned in `requirements.txt`: numpy 2.5.4, scipy 1.16.3, pandas 3.0.5, scikit-learn 1.9.1, networkx 3.7, matplotlib 3.11.2, causal-learn 0.1.4.8, flopsearch 0.3.0, gadjid 0.1.0. statsmodels, sympy and torch were installed but no theory2 script imports them.
- **Lean 4 v4.35.0** (`lean/lean-toolchain`), installed with elan. **Mathlib v4.35.0**, commit `56a005855ea17e0b0958f73ec801ae140b0a1679`. All Lean dependencies (batteries, aesop, Qq, proofwidgets, plausible, LeanSearchClient, importGraph, Cli) are pinned by commit in `lean/lake-manifest.json`. `lake exe cache get` downloads the prebuilt Mathlib.
- **g++ 13.3.0**, only needed to re-run `x1proof/verify/*.cpp`. Their raw outputs are committed. See the last section.
- **Tools for the fetch stage:** `curl`, `unzip`, `sha256sum`, `md5sum`.
- **Hardware:** 2 cores. All analyses were run on one or two cores.

## Stages

| stage | what it runs | data | run time (measured where stated) |
|---|---|---|---|
| `lean` | `lake exe cache get && lake build` in `lean/`, then `lake env lean Axioms*.lean` (axiom report; compare with `lean/AXIOMS.txt`), and a grep for `sorry` / `admit` / `axiom` | none | build from the Mathlib cache: tens of minutes on 2 cores (not timed here) |
| `graph` | `verify/` (T1 enumeration, corollaries, T2 aggregation and support); `ext/verify/` (X2 exhaustive enumeration p = 2, 3, 4; X2 examples; X3); summaries of the committed x1proof C++ outputs; graph-only predictions for B1, study 3 (`realdata/predict_graph.py`), study 4 (`icml_exp/predict4.py`) and B3 (`b3/predict_b3.py`) | none | minutes (each script measured below takes 0–45 s) |
| `sim` | `finite/f{1,2,3}_sim.py`, `finite/f_informative.py`, `ext/verify/x1_ens.py`; B1 (`b1/run_shards.sh`, about 2 h on 2 cores) and B1-NL (`b1nl/run_shards.sh`, about 2.2 h on 1 core), both resumable, then merged (`_merge_shards.py`) and analysed in `_repro_out/` | none | several hours |
| `fetch` | downloads CausalRivers `product.zip`, Causal Chambers `wt_walks_v1.zip` and `lt_walks_v1.zip`, the study-4 csv members (HTTP range requests, `icml_exp/remote_zip.py`) and the B3 sets (`b3/fetch_b3.py`); checks every MD5 / SHA-256 against `data_sha256/` (see `DATA.md`); builds the CausalRivers stage0 caches (`empirical/build_cr_cache.py`) | network | about 230 MB of downloads; cache build a few minutes, about 3 GB RAM |
| `realdata` | CausalRivers Phase 2 (`empirical/`: sample, test_p, explore_corr) and Phase E (`empirical2/`: sample2, test2, explore2); Causal Chambers study 3 (`realdata/test3.py`, `explore3.py`), study 4 (`icml_exp/test4.py main`, `summarise4.py`, `explore4.py`, `check_citris_design.py`) and B3 (`b3/test_b3.py`, `explore_b3.py`) | `fetch` | test3 about 40 s, test4 about 2 min, test_b3 reuses the committed per-dataset caches in `b3/results/`; the rest minutes each |

Logs that contain run times will show `DIFFERS` even when the numbers agree. Compare the JSON outputs (`git diff`) for those.

## Data locations

See `DATA.md`. All scripts take their data locations from `datapaths.py`: `THEORY2_DATA_ROOT`, or one variable per dataset such as `THEORY2_CR_CACHE` or `THEORY2_CC_WT_WALKS`. `python3 datapaths.py` shows what is resolved. If the data already sit somewhere else, symlink them into the layout or set the per-dataset variables, and skip `fetch`.

## What was verified when this was set up (2026-11-09)

The full stages were **not** re-run, because of the CPU budget. These checks were run:

- **Import / argument parsing.** Every refactored script byte-compiles. `empirical/test_p.py` imports without running its analysis. `icml_exp/test4.py` imports and resolves `DATA` through `datapaths`; `realdata/cc_graph.py` resolves through the new import. `empirical/build_cr_cache.py --help` and `_merge_shards.py --help` work.
- **Behaviour unchanged after removing `exec`.** `empirical/explore_corr.py` now imports `test_p` instead of exec-ing its source. Re-run on the original data (4 s), it reproduces `explore_corr.json` to within 7e-16; the last-digit differences are floating-point summation order. `icml_exp/dev4_lags.py` now imports `test4` (module-level definitions only, as before); it was not re-run (development script).
- **Graph-only / cheap outputs re-generated under `python3 -I`, byte-identical to the committed files:**
  - `b3/results/predict_b3_results.txt` (1 s), `realdata/predict_graph_results.txt` (<1 s), `icml_exp/predict4_results.txt` (1 s);
  - `icml_exp/summary4.txt` (from `results4.json`), `ext/verify/x2_examples_results.txt` (43 s), `b1/results/exact_predictions.txt` (1 s);
  - `x1proof/verify/results/{summary_p20000,summary_p100000,varscale_summary,cmp_trunc}.txt`.
- **Cache builder.** `empirical/build_cr_cache.py` reproduces `bavaria_agg_24h.npy` bit-identically from `bavaria_x.npy`.
- **`fetch` stage, run against the original copies through a symlinked `THEORY2_DATA_ROOT`** (no re-download): every MD5 and SHA-256 in `data_sha256/` matched. That covers product.zip and its 9 files, the 12 cache files, both Causal Chambers zips and the 19 used csv files in them, the 6 study-4 members and the 9 B3 files. `b3/fetch_b3.py` reproduced the CRC-32, SHA-256, row and column values of `b3/results/fetch_report.json`.

## What is not reproducible by one command, and why

- **`ext/verify/x1_er.py` tables** (`x1_er_p20000.txt`, `x1_er_p50000.txt`, `x1_er_scaling.txt`). The script takes `p c1,c2,... nseed`, and the number of seeds behind each table was not recorded (EXT.md gives 16 seeds for one row only). The script is deterministic given its arguments, but the exact calls cannot be reconstructed.
- **`ext/verify/x2_enum.py` random modes** (p = 5, 6) **and `x2_limit.py`.** The sample-size arguments of the random parts are not recorded in the outputs. The exhaustive p = 2, 3, 4 runs are in the `graph` stage, with arguments read from the output headers. `x2_limit.py limit 390 199` and the stationary run with 248 graphs follow the counts in EXT.md, but the committed `x2_limit_results.txt` mixes several invocations, so these runs are not wired in.
- **`x1proof/verify/*.cpp`** (x1check, gw_mc, gw_trunc). The compile flags and per-file arguments were not recorded. Each `x1check` record carries its own `p c seed`, so single records can be re-run (`g++ -O2 -std=c++17 x1check.cpp -o x1check && ./x1check p c seed`), but the full sweeps (hours of CPU) are not scripted. The `graph` stage re-derives every committed summary from the committed raw records.
- **`lit2/cands.py`** queries the arXiv API, so its output depends on the query date.
- **`b1x/`, `finite2/` and `lit3/`** were under active development by other work streams when this was written. They are not in `reproduce.sh`. Add them when they are final.
- **The exact merged B1 / B1-NL csv files.** The concatenation order used originally was not recorded. `_merge_shards.py` uses sorted file names, so the merged file can differ in row order. The analyses do not depend on row order.
- **External sources.** These can change or disappear. Downloads that do not match the recorded checksums are rejected, and the raw files must then come from a mirror. The SHA-256 values for the CausalRivers files and the study-3 archives were computed after those analyses ran (`DATA.md` §4).
