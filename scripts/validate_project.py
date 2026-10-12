#!/usr/bin/env python3
"""Offline validation for machines without CUDA: verifies source and script contracts."""
from pathlib import Path
import ast

root = Path(__file__).resolve().parents[1]
required = ["CMakeLists.txt", "src/main.cu", "scripts/plot_results.py", "scripts/generate_report.py", "README.md"]
for name in required:
    path = root / name
    assert path.is_file(), f"Missing required file: {name}"
for name in ["scripts/plot_results.py", "scripts/generate_report.py"]:
    ast.parse((root / name).read_text(encoding="utf-8"))
source = (root / "src/main.cu").read_text(encoding="utf-8")
for marker in ["vector_add_kernel", "matmul_naive_kernel", "matmul_tiled_kernel", "cudaEventElapsedTime", "close_enough"]:
    assert marker in source, f"Source lacks {marker}"
print("PASS: project structure and Python script syntax are valid. CUDA compilation was not attempted by this validator.")
