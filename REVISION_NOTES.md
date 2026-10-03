# SPADE — Revision 2 (ARRAY-D-26-04878)

Reviewer 1 (round 2) asked for clearer exposition and real-world validation beyond finance;
Reviewer 3 requires no change; Reviewers 2 and 4 had no round-2 comments yet.

## What changed
1. **Method rewritten for readability** (manuscript/jmis_method.tex): roadmap, plain-language
   section, notation table, "in plain words" after each assumption/proposition; technical
   theorem, instantaneous-variant algorithm and sparsity-penalty choice moved to Appendix C.
2. **De-narration**: all "earlier draft / retracted / Reviewer N" text removed from Sections 1-5;
   the audit trail of corrections is Appendix D.
3. **Non-financial real-data validation (Section 3.8), protocols committed BEFORE running**:
   - daily bike sharing (n=730, d=12): SPADE BK-AUROC 0.810+/-0.061 (best); DAGMA-nonlinear 0.703
     (not significantly different, p=0.22); beats 7/8 baselines in all 6 resamples.
   - hourly bike sharing (n=17,303) and Beijing PM2.5 (n=41,543): on a common 2,000-row sample SPADE
     TIES linear NOTEARS (0.634 vs 0.649; 0.669 vs 0.669); on the full series SPADE leads in 3/3
     resamples (0.688 vs 0.622; 0.742 vs 0.658). Scale check ran SPADE on a Colab T4 GPU.
   - Spline-shape checks: working-day sign right 6/6 on daily data; temperature->casual edge nearly
     pruned (vacuous); at hourly resolution the checked edges are almost entirely pruned (vacuous).
4. Managerial illustration (Discussion 4.2), new abstract/contributions/conclusion, highlights, cover letter.
5. Response letter rewritten for Revision 2 (response_to_reviewers.*; built by make_response.py).

## Submission bundle
spade_latex_flat.zip = ONE main.tex (all sections, tables and the reference list inlined as thebibliography,
no main.bbl / .bib needed) + 8 PNG figures, no subfolders. Compiles with plain pdflatex x2-3, no BibTeX.
Rebuild: see the inline script used in the session (inline \\input files, replace \\bibliography with main.bbl content).

## Scripts (all committed)
scripts/realworld_bike_sharing.py, scripts/realworld_large.py (pre-registration in docstring),
scripts/realworld_bike_shapes.py, scripts/realworld_large_shapes.py, scripts/build_realworld_tables.py.
Raw results: experimental_results/realworld_*.

## Honest caveats
- SPADE does not win on small hourly samples; its advantage there needs the full series.
- Colab results for the scale check live in experimental_results/realworld_*_scale_log.txt and
  realworld_large_scale_raw.csv (parsed from run logs; the VM was reclaimed before files could be pulled).
- Wall-clock numbers in the hourly tables are omitted (shared-machine contention).

---

# SPADE — Revision 1 session notes (ARRAY-D-26-04878)

Summary of the work done in this revision pass, kept for reference.

## What changed, and why

1. **Instantaneous-DAG headline result (lambda_g retune).**
   Found the group-lasso weight was over-regularized (0.02); retuned via a
   held-out sweep to lambda_g=0.01. Final honest result (after a seed-leak
   bug fix — see below — and a 20-seed confirmatory sweep): SPADE AUROC
   0.947 / 0.926 / 0.915 at d=6/10/20. Beats SCORE at every width; beats
   NoGAM at d=6 and d=20, trails NoGAM at d=10 (0.926 vs 0.955). Trains
   ~275–930x faster than NoGAM (532x average).

2. **Forecast head undertrained.** SPADE's forecasting head trained only 30
   epochs vs baselines' 80. A held-out inner-validation sweep found ~300
   epochs is correct. MSE fell from 0.048 to 0.029 (now beats N-BEATS+RevIN
   and LSTM+RevIN, within 1.2x of the best). Trade-off: training time rose
   from 36s to 357s — SPADE lost its forecasting-speed advantage (it still
   holds for causal discovery, a different budget).

3. **Lagged non-linear benchmark: real data-generation bug.** The synthetic
   generator (`generate_nonlinear_scm`) could silently produce NaN or
   extreme-scale data for some (d, seed) draws — corrupting results across
   *every* method, not just SPADE. Fixed with a scoped finite/bounded-data
   check in `honest_causal_benchmark.py`. SPADE's non-linear AUROC there
   rose from 0.76 to 0.93 (was trailing every baseline; now on par with or
   ahead of the flexible neural/KAN family).

4. **d=50 linear gap, same root cause.** Same over-regularization pattern at
   d=50 (group-lasso penalty scales with d²-ish candidate-edge count).
   lambda_g=0.002 fixed it: AUROC 0.829 → 1.000, now tied with GOLEM.

5. **Critical finding from an independent 5-seat simulated review panel:**
   two reviewer seats (Methodology, Devil's Advocate) independently found
   that the *committed* lambda_g tuning script
   (`scripts/tune_instdag_lambda.py`) actually swept the same seeds used for
   the official reported numbers — contradicting the manuscript's claim of a
   disjoint held-out sweep. Verified true by reading the code directly. This
   is why the headline instantaneous-DAG number changed *twice*: the first
   "fix" (lambda_g=0.005) was itself leakage-tainted and had to be
   corrected a second time (to lambda_g=0.01, on a genuinely disjoint sweep,
   later expanded from 5 to 20 seeds at the user's request for statistical
   confidence — Wilcoxon p=0.39, confirming 0.01 sits in a broad plateau
   with 0.005, well above 0.02).

All of the above are disclosed transparently in the manuscript text itself
(not just corrected silently), in Appendix A.2 (Reproducibility), and in
`response_to_reviewers.tex`.

## What's still an honest, un-fixed weak point (by design — not gamed)

- Lagged **non-linear** aggregate still trails PCMCI/VAR-LiNGAM/VAR-Lasso
  (classical conditioning-based methods retain a real edge there).
- Forecasting MSE still trails the top two RevIN-matched baselines
  (TSMixer+RevIN, PatchTST+RevIN) by a narrow margin.
- Real financial data: the returns-based (stationary) causal analysis is a
  genuine negative result (graph collapses to fully-connected) — left as is,
  not something to "fix."

## Submission package delivered

- `manuscript/main.pdf` — full revised manuscript (compiles clean).
- `manuscript/response_to_reviewers.pdf` + `response_to_reviewers_plain.txt`
  — point-by-point response, PDF and plaintext (for pasting into Editorial
  Manager's text box).
- `manuscript/declarations.pdf` / `declarations.docx` — Funding (none), Data
  availability (public GitHub repo), and a generic AI-assisted-writing
  disclosure, separate from the existing Declaration of Competing Interests.
- Flat single-file LaTeX bundle (`main.tex` with everything inlined,
  `main.bbl` pre-compiled, 7 PNGs, no subfolders) — built for Elsevier's
  upload system; not committed to git (submission-only artifact). Verified
  to compile standalone with just those 9 files.
- `manuscript/*.zip`-equivalent modular LaTeX source (main.tex +
  jmis_*.tex + references.bib + figures/) also available for a
  from-source rebuild.

## Repro / audit trail

Every hyperparameter change in this session has a committed, runnable sweep
script and a real log, not just a claim:
- `scripts/tune_instdag_lambda.py` (instantaneous-DAG lambda_g, seeds 100–119)
- `scripts/tune_d50_lambda.py` (d=50 linear lambda_g, seeds 142–144)
- `scripts/tune_forecast_epochs.py` (forecast-head epoch count, inner
  train/val split of the training fold)

All work is committed and pushed to `main` on
`github.com/vinhqdang/KAN_time_series`.

Status: **revision complete, submission package ready.**
