#!/bin/bash
cd "$(dirname "$0")/.."
echo "[finalize] waiting for forecast leak CSV..."
until [ -f experimental_results/honest_forecast_leak.csv ]; do sleep 15; done
echo "[finalize] forecast done. running ablation + scalability sweep..."
python scripts/ablation_scale.py
echo "[finalize] building tables/figures/NUMBERS.json..."
python scripts/build_results.py > /dev/null 2>&1
echo "[finalize] DONE"
