#!/bin/bash
# Q5 test run: 16 shards (ring, chain, intree, ER2 x p in {5, 10} x T in {3, 5}); NPAR parallel workers (default 3).
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/home/user/venv312/bin/python
mkdir -p results/shards/q5
{
for fam in ring chain intree ER2; do for p in 5 10; do for T in 3 5; do echo "$fam $p $T"; done; done; done
} | xargs -P ${NPAR:-3} -L 1 bash -c 'nice -n 10 '$PY' run_q5.py test results/shards/q5/q5_test_$0_$1_$2.csv $0 $1 $2 >> logs/q5_test_$0_$1_$2.log 2>&1'
echo done > logs/q5_test.done
