# CUDA C++ GPU Kernel Optimization and Benchmarking

A beginner-friendly, resume-ready CUDA C++ project that moves from a CPU reference to a CUDA vector-add kernel, then compares naive and shared-memory tiled matrix multiplication. It uses CUDA events for GPU kernel timing and never depends on NVIDIA Nsight Compute.

## What this demonstrates

- CUDA C++ fundamentals: grids, blocks, threads, bounds checks, host/device memory transfers, and CUDA error handling.
- A CPU numerical reference plus element-wise correctness checks for every CUDA output.
- Two matrix multiplication approaches: a simple global-memory baseline and a shared-memory tiled kernel.
- Repeated CUDA-event timing, CSV result recording, plots, and a Markdown report that are created only from an actual GPU run.

### The idea in plain English

In matrix multiplication, each output number needs a row from A and a column from B. The naive GPU kernel repeatedly reads these values from global memory. The tiled kernel lets a group of threads load small patches into faster shared memory once, then reuse those patches. That reuse is the optimization being measured.

## Project layout

```
src/main.cu                 CUDA implementation and benchmark driver
scripts/run_benchmarks.sh   reproducible build, run, plot, and report command
scripts/run_benchmarks.ps1  equivalent runner for a local Windows CUDA setup
scripts/plot_results.py     turns real CSV measurements into PNG charts
scripts/generate_report.py  turns real CSV measurements into REPORT.md
scripts/validate_project.py no-GPU structural/syntax validation
data/                       explains why no made-up CSV is included
results/                    created on a real CUDA run
```

## Free Google Colab walkthrough

1. In Google Colab, choose **Runtime → Change runtime type → T4 GPU** (the free tier’s availability varies). Verify the allocation with `!nvidia-smi`.
2. Upload `cuda-kernel-optimization.zip` in the Files panel, then run the following in a new Colab cell:

```bash
!unzip -q cuda-kernel-optimization.zip -d /content
%cd /content/cuda-kernel-optimization
!nvcc --version
!bash scripts/run_benchmarks.sh 75
```

3. View the generated results:

```bash
!cat results/REPORT.md
from IPython.display import Image, display
display(Image("results/matmul_time_ms.png"))
display(Image("results/matmul_gflops.png"))
```

`75` is the compute capability of a T4. If Colab gives you another GPU, run `!nvidia-smi --query-gpu=name --format=csv,noheader` and use the matching architecture (examples: P100 = `60`, V100 = `70`, A100 = `80`, L4 = `89`). The code is still the same; this setting just makes compilation targeted and reproducible.

No paid NVIDIA software, paid cloud account, or profiling tool is required. Colab’s temporary GPU and the CUDA toolkit supplied by its image are sufficient when available.

## Local Linux build

Prerequisites: NVIDIA driver compatible with your CUDA toolkit, CUDA Toolkit (includes `nvcc`), CMake 3.18+, Python 3, and `matplotlib` (`python3 -m pip install matplotlib`).

```bash
bash scripts/run_benchmarks.sh 75
```

On Windows PowerShell (with the CUDA Toolkit, CMake, Python, and `matplotlib` installed), use:

```powershell
.\scripts\run_benchmarks.ps1 -Architecture 75
```

The default run uses matrix sizes `128,256,512`, 20 timed repetitions, and a vector of 16,777,216 elements. For a faster smoke test:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build -j
mkdir -p results
./build/cuda_bench --sizes 64,128 --vector-elements 1048576 --repeats 5 --output results/smoke.csv
```

## Output and measurement rules

`results/benchmark_results.csv` records workload, dimension/element count, implementation, mean kernel time in milliseconds, matrix GFLOP/s, correctness status, repeat count, and GPU name.

GPU timing uses CUDA events surrounding the kernel launches. One warm-up launch is excluded; host/device copies, CPU reference calculation, and correctness comparisons are outside the GPU timing interval. CPU timings use a host steady clock and are included only as a practical comparison, not as a hardware-independent performance claim.

The `plot_results.py` script refuses to plot a failed correctness run. There are deliberately no CSV, chart, or performance numbers in this repository: none were measured on the machine that created this package.

## How to explain it in an interview or resume review

> Built a CUDA C++ benchmark that validates vector addition and matrix multiplication against CPU references, comparing naive global-memory and shared-memory tiled kernels with CUDA-event timing across multiple matrix dimensions.

Be precise about your own results: name the GPU, sizes, repeat count, CUDA version, and measured speedup. Do not claim a specific speedup until you have run the project yourself.

## Troubleshooting

- **`nvcc: command not found`**: use a CUDA-enabled Colab runtime or install the CUDA Toolkit locally.
- **`No CUDA-capable GPU detected`**: GPU access was not enabled; the program exits without creating measurements.
- **CMake architecture error**: pass the correct `CMAKE_CUDA_ARCHITECTURES` value for the allocated GPU, or configure without that option to let your local toolkit choose.
- **Colab session disconnects**: rerun the one command in the walkthrough; measurements are intentionally ephemeral unless downloaded.

## Validation status of this delivered package

Offline structural and Python syntax validation can be run with:

```bash
python3 scripts/validate_project.py
```

GPU compilation and tests are **unrun in this delivery environment** because it does not provide `nvcc` or a CUDA GPU. Therefore, no benchmark results are claimed.
