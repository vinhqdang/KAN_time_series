#!/bin/bash
# Resumable B'1 test run on one core: one shard per (family, p); rows are appended as data sets finish, so a
# re-invocation continues where it stopped. Merge afterwards with merge_shards.py.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p results/shards
for fam in ring chain intree ER2; do for p in 5 10; do
  out=results/shards/test_${fam}_${p}.csv.gz
  [ -f "$out" ] && { echo "skip $fam $p"; continue; }
  /usr/bin/python3 run_b1x.py test "$out" $fam $p >> "results/shards/test_${fam}_${p}.log" 2>&1 && echo "done $fam $p" || { echo "FAIL $fam $p"; exit 1; }
done; done
echo ALLDONE
