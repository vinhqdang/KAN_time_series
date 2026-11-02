#!/bin/bash
export OMP_NUM_THREADS=1
cd "$(dirname "$0")"
for g in std old; do for d in 50 100; do
  python3.13 run_pilot.py test --gen $g --d $d --seed 800 --methods SPOT,SPOT-oracle-order,NOTEARS-linear,DAGMA-linear
done; done
python3.13 run_pilot.py test --gen std --d 200 --seed 800 --methods SPOT,SPOT-oracle-order,NOTEARS-linear
timeout 480 python3.13 run_pilot.py test --gen std --d 50 --seed 800 --methods SCORE --out results/test_score_d50.csv
timeout 400 python3.13 run_pilot.py test --gen std --d 50 --seed 800 --methods SPADE-screened --out results/test_spadescr_d50.csv
