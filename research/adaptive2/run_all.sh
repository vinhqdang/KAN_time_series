#!/bin/bash
# sequential, 1 core; resume-safe (bench.py skips finished (seed,method) rows)
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NJOBS=1
mkdir -p results logs
while read -r line; do
  set -- $line
  if [[ "$1" == NODO=* ]]; then export NODO=1; shift; else unset NODO; fi
  for t in 1 2; do python3.13 bench.py "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" >> logs/$(basename $1 .csv).log 2>&1 && break; done
done < jobs.txt
echo ALLDONE > logs/ALLDONE
