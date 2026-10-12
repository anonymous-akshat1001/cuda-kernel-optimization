#!/usr/bin/env python3
"""Write a transparent Markdown summary from real benchmark output."""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.input, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("No benchmark rows found; report not created.")
    device = rows[0]["device"]
    status = "PASS" if all(row["correct"] == "PASS" for row in rows) else "FAIL"
    lines = [
        "# CUDA Kernel Optimization Benchmark Report",
        "",
        "> This report is generated from the accompanying CSV. It contains no estimated or invented measurements.",
        "",
        f"- GPU reported by CUDA: `{device}`",
        f"- Correctness status: **{status}**",
        f"- Repetitions per timed measurement: {rows[0]['repeats']}",
        "- Timing method: CUDA events around kernel launches; host/device transfers excluded from kernel timing.",
        "",
        "## Measurements",
        "",
        "| Workload | Size | Implementation | Mean time (ms) | GFLOP/s | Correct |",
        "|---|---:|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['workload']} | {row['size']} | {row['implementation']} | "
            f"{float(row['time_ms']):.6f} | {float(row['gflops']):.3f} | {row['correct']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "Compare `cuda_naive` with `cuda_tiled_shared_memory` at the same size. The tiled kernel reuses blocks of A and B from shared memory, reducing repeated global-memory traffic. Results depend on the exact GPU, CUDA version, clock state, and matrix size.",
        "",
        "## Reproducibility",
        "",
        "Inputs use a fixed random seed (`20261013`). Rebuild in Release mode and rerun `scripts/run_benchmarks.sh` to regenerate this report and the plots.",
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Created {output}")


if __name__ == "__main__":
    main()
