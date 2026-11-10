#!/bin/bash
# Q2: usage  run_q2_all.sh {tune|test|val}  ; 4 parallel workers over (family, p, T) cells; shards resumable per graph seed.
cd "$(dirname "$0")"
MODE=$1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/home/user/venv312/bin/python
mkdir -p results/shards/q2
{
for fam in chain intree ring ER2; do
  for cell in "5 3" "5 5" "5 10" "10 3" "10 5" "20 3"; do echo "$fam $cell"; done
done
} | xargs -P 4 -L 1 bash -c 'nice -n 10 '$PY' run_q2.py '$MODE' results/shards/q2/q2_'$MODE'_$0_$1_$2.csv $0 $1 $2 >> logs/q2_'$MODE'_$0_$1_$2.log 2>&1'
echo done > logs/q2_$MODE.done
