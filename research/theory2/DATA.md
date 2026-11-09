# External data used in research/theory2: sources, versions, checksums

No raw data are in the repository. The repository holds code, graph-only predictions, node ids, sampling windows and derived statistics. This file lists every external dataset the analyses read, where it came from, which version, and its checksums. The machine-readable manifests are in `data_sha256/`; `reproduce.sh fetch` downloads the data and checks them against those manifests.

The checksums below were computed on 2026-11-09 from the copies used in the analyses, which are still in the analysis session's scratch space (`/tmp/claude-0/dl`, `/tmp/claude-0/dl3`, `/tmp/claude-0/dl4` and `.../scratchpad/b3data`; none of these is a stable location). Where an earlier record exists, it agrees with them: the PLAN4 §0 SHA-256 prefixes, the PLAN3 MD5s, `b3/results/fetch_report.json`, and the `product.zip` byte count in `research/ts/01_literature.md`.

## Where the scripts look for the data

All locations come from `datapaths.py`. Nothing is hard-coded any more. Set either

- `THEORY2_DATA_ROOT` (default `research/theory2/data/`, git-ignored), using the layout below, or
- one of the per-dataset variables, which override the root.

| variable | default under `$THEORY2_DATA_ROOT` | contents | read by |
|---|---|---|---|
| `THEORY2_CR_PRODUCT` | `causalrivers/product/` | unzipped CausalRivers `product.zip` | `empirical/`, `empirical2/` (graph pickles); `empirical/build_cr_cache.py` (csv) |
| `THEORY2_CR_CACHE` | `causalrivers/cache/` | stage0 arrays `<region>_{x,ids,agg_6h,agg_12h,agg_24h,eligible}.npy` | `empirical/`, `empirical2/` |
| `THEORY2_CC_WT_WALKS` | `causal_chambers/wt_walks_v1/` | csv files of `wt_walks_v1.zip` | `realdata/test3.py`, `realdata/explore3.py`, `icml_exp/test4.py`, `icml_exp/dev4_lags.py` |
| `THEORY2_CC_LT_WALKS` | `causal_chambers/lt_walks_v1/` | csv files of `lt_walks_v1.zip` | `realdata/test3.py`, `icml_exp/test4.py`, `b3/dev_b3.py` |
| `THEORY2_CC_CRL` | `causal_chambers/lt_crl_benchmark_v1/` | `citris_1.csv`, `buchholz_1_obs.csv` | `icml_exp/test4.py`, `icml_exp/check_citris_design.py` |
| `THEORY2_CC_CAM` | `causal_chambers/lt_camera_v1/` | `scm_{1,2,4,5}_reference.csv` | `icml_exp/test4.py` |
| `THEORY2_CC_B3` | `causal_chambers/b3/` | the nine B3 held-out csv files | `b3/fetch_b3.py` (writes), `b3/test_b3.py`, `b3/explore_b3.py`, `b3/design_check2.py` |

`python3 datapaths.py` prints the resolved paths and whether each exists. Scripts that already took data directories as command-line arguments (`realdata/test3.py`, `b3/test_b3.py`, and others) still take them; `reproduce.sh` passes the `datapaths` locations.

## 1. CausalRivers (Stein et al., ICLR 2025)

- **Source:** GitHub release `First_release` of `CausalRivers/benchmark`. Asset URL: <https://github.com/CausalRivers/benchmark/releases/download/First_release/product.zip>. This is the URL in the benchmark README. The repository's own download record (`research/ts/01_literature.md`) says only "from the GitHub release". The file has the size recorded there.
- **Licence:** Datenlizenz Deutschland, Namensnennung 2.0 (dl-de/by-2-0). Cite Stein et al., ICLR 2025.
- **Version:** the zip members are dated 2025-02-12. It was downloaded on 2026-11-03, once, and reused by `research/ts/stage0` and by theory2 Phase 2 / Phase E.
- **Used:** RiversEastGermany and RiversBavaria, i.e. `rivers_ts_{east_germany,bavaria}.csv` (15-min series) and `rivers_{east_germany,bavaria}.p` (river graphs; networkx-only pickles, checked with pickletools in stage0). `rivers_*flood*` and `visualization/` are not used.

| file | bytes | MD5 | SHA-256 |
|---|---|---|---|
| `product.zip` | 133,228,548 | 62278bf7816c77389db6bc7130ff507d | 4f779fa9222653af092046eacba39a955cc65638bfb70fb7e31a32d9d8373b55 |
| `product/rivers_ts_east_germany.csv` | 618,673,429 | – | c0b2202df0c69eefd3174160d8d537aa4977452bd0545c77e3ef41052da8abc5 |
| `product/rivers_ts_bavaria.csv` | 466,018,856 | – | fae1fc5aac3d46c603d7f86950fba466022a970f2602e1b47d13dfb740508426 |
| `product/rivers_east_germany.p` | 107,954 | – | d6ff7df27325be8e1a8b7194ee2d1dd915a29e253561e389db66338a6711ade3 |
| `product/rivers_bavaria.p` | 81,237 | – | a798c76cba8ecd3e0971916783c94d29e56d81b4663b24c32fd1e7e81610d1f4 |

All nine files of `product/` are in `data_sha256/causalrivers_product.sha256`.

**Derived caches (stage0).** `research/ts/stage0/load_raw.py` and `prep_pool.py` built the 15-min float32 arrays and the 6 h / 12 h / 24 h block means (missing-value rule ≥ 50 % per block; linear interpolation of interior gaps of ≤ 2 steps; eligibility = ≤ 20 % missing and complete after interpolation). Every CausalRivers analysis in theory2 reads these caches, not the csv files. `empirical/build_cr_cache.py` rebuilds them at `THEORY2_CR_CACHE` by importing the stage0 functions. Check made on 2026-11-09: the rebuilt `bavaria_agg_24h.npy` is bit-identical to the cache used. The SHA-256 values of the cache files used are in `data_sha256/causalrivers_cache.sha256`. A rebuild with a different pandas or numpy version may not be bit-identical, so `reproduce.sh` reports a cache mismatch as a warning, not as an error.

## 2. Causal Chambers (Gamella, Peters, Bühlmann, Nature Machine Intelligence 2025)

- **Source:** the project's S3 bucket `https://causalchamber.s3.eu-central-1.amazonaws.com/downloadables/<dataset>.zip`. The URLs and MD5s are those in the dataset directory `directory.yaml` (SHA-256 of the copy used: c814acc3847a913fac25033aa884bc3457f022cf03f4210f47d341bb85582931). It is the dataset index used by the `causalchamber` package (<https://github.com/juangamella/causal-chamber-package>). The exact URL it was fetched from was not recorded.
- **Ground-truth graphs:** transcribed into `realdata/cc_graph.py`, `icml_exp/graphs4.py` and `b3/graphs_b3.py` from `causalchamber` 0.2.8 `ground_truth` (PLAN3). The package is not needed to run the code.
- **Licence:** CC BY 4.0 for the data; MIT for the package.
- **Downloaded** 2026-11-04 (studies 3 and 4), plus the B3 sets later. TLS verification was on throughout.

### 2.1 Whole archives (study 3 = `realdata/`, study 4 = `icml_exp/`)

| archive | bytes | MD5 (= directory.yaml) | SHA-256 |
|---|---|---|---|
| `wt_walks_v1.zip` | 46,471,335 | 19bb4e92cbe0b8dff49299b9b509ac36 | 1ef92d39fff7645a9e6e8cd41311e708362606491590a1534c6a5135d207736e |
| `lt_walks_v1.zip` | 2,003,074 | dcad019186661a56de7a2d1db97fc2f0 | 21fb3d0b2fad7d8ba544289ee7880785d1c35ae0b5d7f9302c794ca34d205a19 |

Used members:

- From `wt_walks_v1`: `actuators_random_walk_{1..16}.csv` (study 3: runs 2–16) and `regime_jumps_{single,multi}.csv` (study 4 E2).
- From `lt_walks_v1`: `actuators_white.csv` (study 3 replication; B3 development check).

The per-file SHA-256 values are in `data_sha256/cc_wt_walks_v1.sha256` and `data_sha256/cc_lt_walks_v1.sha256`. Two examples: `regime_jumps_single.csv` ae00d09adddde4db86f571d85f402adc959017f376848e14cc16625b02cd59e0 and `actuators_white.csv` 5f220699e4090fa6e723e5a369434b58647a84c179a12903e1f550f0a988c670.

`wt_changepoints_v1.zip` (SHA-256 a247ecd867aadb99f524f4cfcb22ded28e550b069ffd10f4df477c036e8a3979) was downloaded during Stage A but not analysed.

### 2.2 Single members of large image archives (study 4), fetched by HTTP range requests

These were fetched with `icml_exp/remote_zip.py`; `zipfile` checks the CRC-32 on extraction. The member path is `<archive>/<exp>/<exp>.csv`. The whole archives were never downloaded. Their directory.yaml MD5s are `lt_crl_benchmark_v1.zip` 8ec492afffd0142abd70b1fa7082b104 and `lt_camera_v1.zip` 9f4b52efdb1c08bd5ab615b6edc8ed95.

| file | archive | rows | SHA-256 |
|---|---|---|---|
| `citris_1.csv` | `lt_crl_benchmark_v1.zip` | 100,000 | 2291eb10b18b1740cc40ed91bb49e76df10e6c350520293a8c9eb41cf0b5c607 |
| `buchholz_1_obs.csv` | `lt_crl_benchmark_v1.zip` | 10,000 | 4844a04cef6a8097c22cd0628d8a24224c7be8718ed68909ea8a8bdde98c4a0a |
| `scm_1_reference.csv` | `lt_camera_v1.zip` | 5,000 | 16d7747601cae1b03dfca8526ff6a99f954c24c1bafcc1910ad905df46348c09 |
| `scm_2_reference.csv` | `lt_camera_v1.zip` | 10,000 | d6c652559a3770e9a75df85536c07a01d8c1d5e0c7efd70d6d9206996fecb08e |
| `scm_4_reference.csv` | `lt_camera_v1.zip` | 10,000 | df72c6dbb1c5e15a3db3981eebc3683b86ddcf6ea11c645841d85dbd88b2442d |
| `scm_5_reference.csv` | `lt_camera_v1.zip` | 10,000 | 827d045e91dc58a19ef0bb6b337d70b3625d303b98deb53935b896add8ea44f3 |

The first 16 hex digits agree with the values committed in `icml_exp/PLAN4.md` §0 before the analysis.

### 2.3 B3 held-out sets (`b3/`)

These were fetched by `b3/fetch_b3.py`. It uses range requests, checks the zip CRC-32 against the values fixed in `b3/PLAN.md` before download, and records SHA-256 and a design check in `b3/results/fetch_report.json`. The local copies were re-hashed on 2026-11-09, and all nine agree with `fetch_report.json`.

| file | archive (directory.yaml MD5) | CRC-32 | rows | SHA-256 |
|---|---|---|---|---|
| `ar_1_uniform_ref.csv` | `lt_camera_walks_v1.zip` (097f291680a9a57d2c62bd2bc2b78fce) | b153930d | 30,000 | c3f7ec56240db77b7281b836ec48c5bd89f420d98b4dfb8d7270b5cc84dcb53f |
| `ar_1_uniform_pol_1_90.csv` | same | 894684f7 | 30,000 | b6e9ca487670f88f550a6bbf8cbfaa0b2ebd037963bc0d5a2e8f6afbaef77477 |
| `ar_1_uniform_pol_12_30.csv` | same | dcab420f | 30,000 | 259d615b8f3cd4319246bcff1692ac092845eacfee024d357ea056d93ee135c7 |
| `scm_5_red.csv` | `lt_camera_v1.zip` (9f4b52efdb1c08bd5ab615b6edc8ed95) | a3d5a406 | 10,000 | 3c912d8a6676dc746a461814e2b29894ad0c30864b295a5d1938c8efe264ffb5 |
| `scm_5_blue.csv` | same | 08eb6c50 | 10,000 | c8e71cfb8c3d460da672c9cbec03cfeb372b93bcbcdd95c34bd48e61c53edd4f |
| `scm_5_green.csv` | same | 158bdeb3 | 10,000 | 4a00091387f9863c6ce8b018e27c72e7cc0fb89429cf365de14cb242c681bdff |
| `scm_4_green.csv` | same | de1c4f3d | 10,000 | 79d6250bbdc466b65702e658d551767ead7e613440beefd592ea3c654a1506af |
| `scm_4_blue.csv` | same | efc3eb05 | 10,000 | 4508d5e6620c162dfb7361695bb7e3c97bdc627f3c95abe0a069e8f2110e2310 |
| `scm_4_red.csv` | same | 9fdbe1ed | 10,000 | a6e4c479c02c246f5e99dc7d6df20e78f88956226b43ee0bc8d7a8826eef5623 |

## 3. Not data, but network access

`lit2/cands.py` queries the arXiv API for the literature scan. Its output depends on the date of the query and cannot be reproduced exactly. The query strings are in the script.

## 4. Known limits

- **CausalRivers asset URL.** It is taken from the CausalRivers README. The download log itself recorded only "GitHub release asset". The size and SHA-256 above identify the file, and `reproduce.sh fetch` refuses a download that does not match.
- **Source of `directory.yaml`.** Its URL was not recorded. Its SHA-256 is given above. The archive MD5s that matter are copied into this file and into `data_sha256/archives.txt`.
- **Hash timing.** The SHA-256 values for the CausalRivers files, the stage0 caches and the study-3 archives were first computed on 2026-11-09, after those analyses ran. Before that, the study-3 archives were identified only by their MD5 (PLAN3) and `product.zip` only by its size. The study-4 files have pre-analysis SHA-256 prefixes (PLAN4), and B3 has full pre-analysis SHA-256 values.
- **Mutable sources.** The S3 and GitHub sources can change or disappear. Mirrors must match the checksums above.
