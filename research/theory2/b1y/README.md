# b1y: decisive controls for the unified story

Status: **plan committed, no study run yet.** `PLAN.md` is the time-stamped plan (matched-correlation counterfactual start, variance-matched control with tuned learners, first-slice-excluded scoring as primary, run-level inference for the real-data tests, extra baselines, exact burn-in tightness). It is not a registration and has not been posted to OSF.

| file | what it is | state |
|---|---|---|
| `PLAN.md` | design, hypotheses, thresholds, budget, stop rules, seeds, multiplicity | committed before any run |
| `common_y.py` | counterfactual starts (CF1, CF2), exact window covariances, lam_star, variance matching, first-slice scoring | unit-tested |
| `seeds.py` | seeds, RNG keys, registry of earlier streams | disjointness tested |
| `test_b1y.py` | unit tests on tiny cases (population quantities and scoring conventions only) | all pass |
| `analyse_y.py` | decision rules (retention, dose, Holm) | `--selftest` passes on dummy results with injected truths |
| `q6_burnin_exact.py` | exact B_min versus Δ (graph-only) | **run**; `results/q6_burnin_exact.txt` |
| `q1_population_predictions.py`, `q1_dose_population.py` | graph-only oracle predictions for the Q1 arms and doses | **run**; `results/q1_*` |
| `run_q1.py` | Q1 simulation runner | written, not executed |
| `methods_y.py` | wrappers for DAGMA/NOTEARS with hyper-parameters, GES, GRaSP, BOSS, DirectLiNGAM, lag-1 OLS | untested on data |
| `run_q2.py`, `run_q5.py` | stubs | filled in after approval and step 0 |

Run the tests: `OMP_NUM_THREADS=1 /usr/bin/python3 test_b1y.py` and `/usr/bin/python3 analyse_y.py --selftest` (the system interpreter has the scientific stack; the default `python3` here does not).

Plan commit: `5c9f942`, pushed to `main` on 2026-11-10; the push command returned before 05:15:32 UTC by the local clock. The commit timestamp (05:15:29 UTC) is set by the client, and the hosting commit API returns that same value, so neither is a server push time. The hosting push-events feed did not yet list this push when queried (it lags); the server time of the push event is to be added here from `GET /repos/vinhqdang/KAN_time_series/events` before any run.
