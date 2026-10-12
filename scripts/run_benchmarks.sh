#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/run_benchmarks.sh [CUDA architecture]
# CUDA architecture examples: T4=75, P100=60, L4=89, A100=80.
ARCH="${1:-75}"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$ARCH"
cmake --build build --config Release -j
mkdir -p results
./build/cuda_bench --sizes 128,256,512 --vector-elements 16777216 --repeats 20 \
  --output results/benchmark_results.csv
python3 scripts/plot_results.py --input results/benchmark_results.csv --output-dir results
python3 scripts/generate_report.py --input results/benchmark_results.csv --output results/REPORT.md
