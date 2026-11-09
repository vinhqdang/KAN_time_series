#!/bin/bash
# Resumable test run: one shard per family, written atomically to results/shards/; completed shards are skipped.
# Re-invoke after an interruption; then concatenate with: /usr/bin/python3 -c "..." (see README).
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p results/shards
for fam in symER ring chain outtree intree ER1 ER2; do
  out=results/shards/test_$fam.csv.gz
  [ -f "$out" ] && { echo "skip $fam"; continue; }
  /usr/bin/python3 run_b1.py test "$out.part" 2 "$fam" > "results/shards/test_$fam.log" 2>&1 && mv "$out.part" "$out" && echo "done $fam" || { echo "FAIL $fam"; exit 1; }
done
echo ALLDONE
