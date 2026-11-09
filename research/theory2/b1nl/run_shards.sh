#!/bin/bash
# Resumable B1-NL test run on one core: shards per (config, family, p) in results/shards/, completed shards skipped.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p results/shards
/usr/bin/python3 run_b1nl.py test results/shards NL,SGN,SGNU >> results/shards/driver.log 2>&1
echo "exit $?" >> results/shards/driver.log
