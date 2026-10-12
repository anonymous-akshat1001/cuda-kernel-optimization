#!/usr/bin/env python3
"""Create plots only from an actual CUDA benchmark CSV; no synthetic data is used."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("The benchmark CSV has no rows. Run cuda_bench on a CUDA GPU first.")
    if any(row["correct"] != "PASS" for row in rows):
        raise SystemExit("Refusing to plot: one or more correctness checks failed.")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rows = read_rows(args.input)
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Install matplotlib: pip install matplotlib") from exc

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    matrix_rows = [r for r in rows if r["workload"] == "matrix_multiply"]
    if not matrix_rows:
        raise SystemExit("No matrix_multiply rows in the benchmark CSV.")

    grouped = defaultdict(list)
    for row in matrix_rows:
        grouped[row["implementation"]].append((int(row["size"]), float(row["time_ms"])))
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, values in sorted(grouped.items()):
        values.sort()
        ax.plot([v[0] for v in values], [v[1] for v in values], marker="o", label=name)
    ax.set_title("Matrix multiplication time (measured CUDA benchmark)")
    ax.set_xlabel("Matrix dimension N (N×N)")
    ax.set_ylabel("Mean time per run (ms)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "matmul_time_ms.png", dpi=160)
    plt.close(fig)

    gpu_rows = [r for r in matrix_rows if r["implementation"].startswith("cuda_")]
    grouped.clear()
    for row in gpu_rows:
        grouped[row["implementation"]].append((int(row["size"]), float(row["gflops"])))
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, values in sorted(grouped.items()):
        values.sort()
        ax.plot([v[0] for v in values], [v[1] for v in values], marker="o", label=name)
    ax.set_title("CUDA matrix multiplication throughput (measured)")
    ax.set_xlabel("Matrix dimension N (N×N)")
    ax.set_ylabel("GFLOP/s")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "matmul_gflops.png", dpi=160)
    print(f"Created plots in {output}")


if __name__ == "__main__":
    main()
