param(
  [string]$Architecture = "75"
)

$ErrorActionPreference = "Stop"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CUDA_ARCHITECTURES=$Architecture"
cmake --build build --config Release --parallel
New-Item -ItemType Directory -Force -Path results | Out-Null

$exe = if (Test-Path "build\Release\cuda_bench.exe") { "build\Release\cuda_bench.exe" } else { "build\cuda_bench.exe" }
& $exe --sizes 128,256,512 --vector-elements 16777216 --repeats 20 --output results/benchmark_results.csv
python scripts/plot_results.py --input results/benchmark_results.csv --output-dir results
python scripts/generate_report.py --input results/benchmark_results.csv --output results/REPORT.md
