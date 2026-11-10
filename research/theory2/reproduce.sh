#!/usr/bin/env bash
# One-command reproduction of research/theory2 (REPRODUCE.md explains each stage, its run time and its outputs).
#
#   ./reproduce.sh                 default: lean graph   (no data, no downloads)
#   ./reproduce.sh <stage> ...     any of: lean graph sim posthoc fetch realdata all
#
#   lean      build the Lean 4 / Mathlib proofs and print their axioms
#   graph     graph-only and exact checks (no data; fixed seeds where random), minutes
#   sim       the simulation studies (finite/, finite2/, ext/, b1/, b1nl/, b1x/), about 9 hours on 1-2 cores; resumable
#   posthoc   post hoc re-analyses of the committed B1 / B1-NL results, detectability calculation, Fig. oracle (b1x/)
#   fetch     download the external datasets into $THEORY2_DATA_ROOT, check MD5 / SHA-256 against data_sha256/
#   realdata  the real-data tests (CausalRivers; Causal Chambers studies 3, 4 and B3); needs fetch (or data in place)
#   all       lean graph sim posthoc fetch realdata
#
# Environment: PYTHON (default /usr/bin/python3 if present, else python3), THEORY2_DATA_ROOT and the per-dataset
# variables documented in datapaths.py / DATA.md.  Captured stdout goes to _repro_out/ (git-ignored) and is compared
# with the committed file where one exists ("identical" / "DIFFERS").  Scripts that write their own result files
# overwrite the committed ones in place, so `git diff --stat` afterwards lists every output that changed.
set -u -o pipefail
cd "$(dirname "$0")"
T2="$(pwd)"
if [ -z "${PYTHON:-}" ]; then
  if [ -x /usr/bin/python3 ]; then PYTHON=/usr/bin/python3; else PYTHON=python3; fi
fi
PY="$PYTHON -I"                       # isolated mode: never import modules from a data or current directory
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT="$T2/_repro_out"; mkdir -p "$OUT"
FAIL=0

# resolved data locations (datapaths.py is the single source of truth)
eval "$($PY "$T2/datapaths.py" --env)"
DATA_ROOT="${THEORY2_DATA_ROOT:-$T2/data}"
S3=https://causalchamber.s3.eu-central-1.amazonaws.com/downloadables

log() { printf '\n=== %s\n' "$*"; }
# run NAME REF DIR CMD... : run CMD inside DIR; stdout -> _repro_out/NAME.out; compare with DIR/REF unless REF is "-"
run() {
  local name=$1 ref=$2 dir=$3; shift 3
  local t0=$SECONDS rc res
  ( cd "$dir" && "$@" ) > "$OUT/$name.out" 2> "$OUT/$name.err"; rc=$?
  if [ $rc -ne 0 ]; then res="FAILED (rc=$rc, see _repro_out/$name.err)"; FAIL=1
  elif [ "$ref" = "-" ]; then res="ok"
  elif cmp -s "$OUT/$name.out" "$dir/$ref"; then res="ok, identical to $dir/$ref"
  else res="ok, DIFFERS from $dir/$ref (diff _repro_out/$name.out $dir/$ref)"; fi
  printf '%-24s %6ss  %s\n' "$name" $((SECONDS - t0)) "$res"
}

stage_lean() {
  log "lean: $(cat lean/lean-toolchain); Mathlib and all Lean dependencies pinned in lean/lake-manifest.json"
  if ! command -v lake >/dev/null; then echo "lake not found: install elan (https://github.com/leanprover/elan)"; FAIL=1; return; fi
  ( cd lean && lake exe cache get && LEAN_NUM_THREADS="${LEAN_NUM_THREADS:-2}" lake build ) || { FAIL=1; return; }
  for f in lean/Axioms*.lean; do run "lean_$(basename "$f" .lean)" - lean lake env lean "$(basename "$f")"; done
  echo "compare _repro_out/lean_Axioms*.out with lean/AXIOMS.txt (expected axioms: propext, Classical.choice, Quot.sound)"
  if grep -rnwE 'sorry|admit' lean/RelSort lean/RelSort.lean || grep -rnE '^axiom ' lean/RelSort lean/RelSort.lean; then
    echo "found sorry / admit / axiom in the Lean sources"; FAIL=1
  else echo "no sorry / admit / axiom in the Lean sources"; fi
}

stage_graph() {
  log "graph: exact / graph-only checks"
  run t1_enum            -                            verify         $PY t1_enum.py          # writes t1_enum_results.txt
  run t1_corollaries     -                            verify         $PY t1_corollaries.py   # writes t1_corollaries_results.txt
  run t2_aggregation     -                            verify         $PY t2_aggregation.py   # writes t2_aggregation_results.txt
  run t2_support         t2_support_results.txt       verify         $PY t2_support.py
  run x2_enum_p2         x2_enum_p2.txt               ext/verify     $PY x2_enum.py exh 2 2,3,4,5,6,7,8,9
  run x2_enum_p3         x2_enum_p3.txt               ext/verify     $PY x2_enum.py exh 3 2,3,4,5,6,7,9
  run x2_enum_p4         x2_enum_p4.txt               ext/verify     $PY x2_enum.py exh 4 3,5,8
  run x2_examples        x2_examples_results.txt      ext/verify     $PY x2_examples.py
  run x3_bayes           x3_bayes_results.txt         ext/verify     $PY x3_bayes.py
  run x1_summary_p20000  results/summary_p20000.txt   x1proof/verify $PY summarize.py results/x1check_p20000.txt
  run x1_summary_p100000 results/summary_p100000.txt  x1proof/verify $PY summarize.py results/x1check_p100000.txt
  run x1_varscale        results/varscale_summary.txt x1proof/verify $PY varscale.py
  run x1_cmp_trunc       results/cmp_trunc.txt        x1proof/verify $PY cmp_trunc.py
  run b1_predict         -                            b1             $PY exact_curves.py predict results
  run b1_curves          -                            b1             $PY exact_curves.py curves results
  run b1_delta           -                            b1             $PY exact_curves.py delta results
  run realdata_predict   predict_graph_results.txt    realdata       $PY predict_graph.py .
  run icml_predict4      predict4_results.txt         icml_exp       $PY predict4.py
  run b3_predict         results/predict_b3_results.txt b3           $PY predict_b3.py
}

stage_sim() {
  log "sim: simulation studies (long; fixed seeds)"
  run f1_sim             f1_results.txt               finite         $PY f1_sim.py
  run f2_sim             f2_results.txt               finite         $PY f2_sim.py
  run f3_sim             f3_results.txt               finite         $PY f3_sim.py
  run f_informative      f_informative_results.txt    finite         $PY f_informative.py
  run x1_ens             x1_ens_results.txt           ext/verify     $PY x1_ens.py
  # B1 (about 2 h on 2 cores) and B1-NL (about 2.2 h on 1 core): resumable shards under results/shards/ (git-ignored),
  # merged and analysed in _repro_out/.  The committed merged files are b1/results/b1_test.csv.gz and
  # b1nl/results/b1nl_test.csv.gz; analyse_b1*.py can be pointed at those directly.
  mkdir -p "$OUT/b1" "$OUT/b1nl"
  run b1_shards          -  b1    bash run_shards.sh
  run b1_merge           -  b1    $PY "$T2/_merge_shards.py" "results/shards/test_*.csv.gz" "$OUT/b1_test.csv.gz"
  run b1_analyse         -  b1    $PY analyse_b1.py "$OUT/b1_test.csv.gz" "$OUT/b1"
  run b1nl_shards        -  b1nl  bash run_shards.sh
  run b1nl_merge         -  b1nl  $PY "$T2/_merge_shards.py" "results/shards/test_*.csv.gz" "$OUT/b1nl_test.csv.gz"
  run b1nl_analyse       -  b1nl  $PY analyse_b1nl.py "$OUT/b1nl_test.csv.gz" "$OUT/b1nl"
  echo "compare _repro_out/b1/analysis.txt with b1/results/analysis.txt, _repro_out/b1nl/analysis.txt with b1nl/results/analysis.txt"
  # Learner extension (b1x; DAGMA, NOTEARS, GOLEM, PCMCI+; about 4.2 h on 1 core): resumable shards under
  # b1x/results/shards/ (git-ignored). merge_shards.py rewrites b1x/results/b1x_test.csv.gz in place; the analysis
  # goes to _repro_out/b1x/ (compare analysis.txt with b1x/results/analysis.txt).
  mkdir -p "$OUT/b1x"
  run b1x_shards         -  b1x   bash run_shards.sh
  run b1x_merge          -  b1x   $PY merge_shards.py
  run b1x_analyse        -  b1x   $PY analyse_b1x.py results/b1x_test.csv.gz "$OUT/b1x"
  # finite2: plug-in effective sample size, fixed-K size, recovery probability (stdout vs sim_finite2_results.txt
  # by eye); ER c = 1 convergence at p = 20,000 and 200,000 (compare with er_c1_conv_results.txt).
  run finite2_sim        -  finite2 $PY sim_finite2.py
  run finite2_er_c1_2e4  -  finite2 $PY er_c1_conv.py 20000 6
  run finite2_er_c1_2e5  -  finite2 $PY er_c1_conv.py 200000 6     # the single p = 10^6 seed (16 min) is not scripted
}

stage_posthoc() {
  log "posthoc: re-analyses of the committed B1 / B1-NL results and the detectability calculation (no new data)"
  # These scripts write into b1x/results/ in place (git diff shows any change). They need only the committed
  # b1/results/b1_test.csv.gz, b1nl/results/b1nl_test.csv.gz, b3 and b1x analysis files.
  run b1x_orders         -  b1x   $PY reanalysis_orders.py results/orders_x1.csv.gz
  run b1x_reanalysis     -  b1x   $PY reanalysis_b2.py results
  run b1x_oracle         -  b1x   $PY oracle_b3.py results
  run b1x_multiplicity   -  b1x   $PY multiplicity.py results
  run fig_oracle         -  icml_paper/figs $PY make_fig_oracle.py
}

archive() { awk -v z="$1" -v k="$2" '$1==z {print $k}' "$T2/data_sha256/archives.txt"; }   # NAME COLUMN (2 md5, 3 sha256, 5 url)
fetch_url() {  # fetch_url URL DEST SHA256 [MD5]
  local url=$1 dst=$2 sha=$3 md5=${4:-}
  mkdir -p "$(dirname "$dst")"
  if [ ! -f "$dst" ]; then
    if curl -fL --retry 3 -o "$dst.part" "$url"; then mv "$dst.part" "$dst"
    else echo "download failed: $url"; FAIL=1; return 1; fi
  fi
  if [ -n "$md5" ]; then echo "$md5  $dst" | md5sum -c - || { FAIL=1; return 1; }; fi
  echo "$sha  $dst" | sha256sum -c - || { FAIL=1; return 1; }
}
fetch_member() {  # fetch_member ZIP EXP DESTDIR SHA256 : one csv member by HTTP range requests (zip CRC-32 checked)
  local zip=$1 exp=$2 dir=$3 sha=$4
  mkdir -p "$dir"
  if [ ! -f "$dir/$exp.csv" ]; then
    $PY icml_exp/remote_zip.py get "$S3/$zip.zip" "$zip/$exp/$exp.csv" "$dir/$exp.csv" || { FAIL=1; return 1; }
  fi
  echo "$sha  $dir/$exp.csv" | sha256sum -c - || FAIL=1
}
check_dir() {  # check_dir DIR MANIFEST [warn] : sha256sum -c of the files listed in MANIFEST, inside DIR
  if ( cd "$1" && sha256sum -c --quiet "$T2/data_sha256/$2" ); then echo "$2: all files match"
  elif [ "${3:-}" = warn ]; then echo "WARNING: $2 differs (derived files; see DATA.md)"
  else echo "$2: MISMATCH"; FAIL=1; fi
}

stage_fetch() {
  log "fetch: external data into $DATA_ROOT"
  command -v curl >/dev/null && command -v unzip >/dev/null || { echo "need curl and unzip"; FAIL=1; return; }
  # CausalRivers (Stein et al., ICLR 2025; dl-de/by-2-0)
  local cr="$DATA_ROOT/causalrivers"
  if fetch_url "$(archive causalrivers_product.zip 5)" "$cr/product.zip" "$(archive causalrivers_product.zip 3)" \
               "$(archive causalrivers_product.zip 2)"; then
    [ -d "$THEORY2_CR_PRODUCT" ] || unzip -q -o "$cr/product.zip" -d "$(dirname "$THEORY2_CR_PRODUCT")"
    check_dir "$THEORY2_CR_PRODUCT" causalrivers_product.sha256
    [ -f "$THEORY2_CR_CACHE/bavaria_agg_24h.npy" ] || run build_cr_cache - empirical $PY build_cr_cache.py all
    check_dir "$THEORY2_CR_CACHE" causalrivers_cache.sha256 warn
  fi
  # Causal Chambers (Gamella, Peters, Buehlmann 2025; CC BY 4.0); MD5 as listed in the project's directory.yaml
  local cc="$DATA_ROOT/causal_chambers" z
  for z in wt_walks_v1 lt_walks_v1; do
    if fetch_url "$(archive "cc_$z.zip" 5)" "$cc/zips/$z.zip" "$(archive "cc_$z.zip" 3)" "$(archive "cc_$z.zip" 2)"; then
      [ -d "$cc/$z" ] || unzip -q -o "$cc/zips/$z.zip" -d "$cc"
    fi
  done
  check_dir "$THEORY2_CC_WT_WALKS" cc_wt_walks_v1.sha256
  check_dir "$THEORY2_CC_LT_WALKS" cc_lt_walks_v1.sha256
  local zip exp sha d
  while read -r zip exp sha; do
    case "$zip" in lt_crl_benchmark_v1) d=$THEORY2_CC_CRL;; lt_camera_v1) d=$THEORY2_CC_CAM;; *) continue;; esac
    fetch_member "$zip" "$exp" "$d" "$sha"
  done < data_sha256/cc_members.txt
  # B3 held-out sets: the pre-registered fetch script (zip CRC-32, SHA-256, design check), report to _repro_out
  run fetch_b3 - b3 $PY fetch_b3.py "$THEORY2_CC_B3" "$OUT/fetch_report_b3.json"
  check_dir "$THEORY2_CC_B3" cc_b3.sha256
}

stage_realdata() {
  log "realdata: tests on the downloaded data (pre-registered analyses; deterministic)"
  $PY datapaths.py
  # CausalRivers: Phase 2 (empirical/) and Phase E (empirical2/)
  run cr_sample          -                 empirical  $PY sample.py           # rewrites subgraphs_7000.json
  run cr_test_p          test_p.log        empirical  $PY test_p.py           # rewrites results_p.json
  run cr_explore_corr    -                 empirical  $PY explore_corr.py     # rewrites explore_corr.json
  run cr_sample2         -                 empirical2 $PY sample2.py          # rewrites subgraphs_8000.json
  run cr_test2           test2.log         empirical2 $PY test2.py            # rewrites results2.json
  run cr_explore2        explore2.log      empirical2 $PY explore2.py         # rewrites explore2.json
  # Causal Chambers: study 3 (realdata/), study 4 (icml_exp/), B3 (b3/)
  run cc_test3           test3.log         realdata   $PY test3.py . "$THEORY2_CC_WT_WALKS" "$THEORY2_CC_LT_WALKS"
  run cc_explore3        explore3.log      realdata   $PY explore3.py . "$THEORY2_CC_WT_WALKS"
  run cc_test4           -                 icml_exp   $PY test4.py . main results4.json
  run cc_summarise4      summary4.txt      icml_exp   $PY summarise4.py results4.json
  run cc_explore4        explore4.log      icml_exp   $PY explore4.py results4.json
  run cc_citris_design   check_citris_design_results.txt icml_exp $PY check_citris_design.py "$THEORY2_CC_CRL/citris_1.csv"
  run b3_test            -                 b3         $PY test_b3.py "$THEORY2_CC_B3" results
  run b3_explore         results/explore_b3.log b3    $PY explore_b3.py results "$THEORY2_CC_B3"
  # post hoc: sensitivity of the B3 pair decisions to the truncation lag K (finite2/)
  run b3_k_sensitivity   k_sensitivity_b3_results.txt finite2 $PY k_sensitivity_b3.py "$THEORY2_CC_B3" "$T2/b3"
}

stages=("$@"); [ ${#stages[@]} -eq 0 ] && stages=(lean graph)
for s in "${stages[@]}"; do
  case "$s" in
    lean) stage_lean;; graph) stage_graph;; sim) stage_sim;; posthoc) stage_posthoc;; fetch) stage_fetch;; realdata) stage_realdata;;
    all) stage_lean; stage_graph; stage_sim; stage_posthoc; stage_fetch; stage_realdata;;
    -h|--help) sed -n '2,18p' "$0"; exit 0;;
    *) echo "unknown stage: $s (lean graph sim posthoc fetch realdata all)"; exit 2;;
  esac
done
log "done (exit status $FAIL); committed outputs changed by this run:"
git -C "$T2" diff --stat -- . 2>/dev/null | tail -n 25
exit $FAIL
