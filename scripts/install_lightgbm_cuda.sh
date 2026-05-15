#!/usr/bin/env bash
# Rebuild LightGBM from source against CUDA 12.8, targeting sm_120 (RTX 5070 Ti).
# Run this after any `uv sync` that pulls in the PyPI lightgbm wheel.
#
# Why not pin this in pyproject.toml: that would force every environment
# (including CI) to source-build with CUDA, and CI does not have nvcc.

set -euo pipefail

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"

if ! command -v "$CUDA_HOME/bin/nvcc" >/dev/null; then
    echo "nvcc not found at $CUDA_HOME/bin/nvcc" >&2
    echo "Install: sudo apt install cuda-toolkit-12-8 (after adding NVIDIA repo)" >&2
    exit 1
fi

if ! command -v cmake >/dev/null; then
    echo "cmake not found" >&2
    echo "Install: sudo apt install cmake" >&2
    exit 1
fi

uv pip uninstall lightgbm

PATH="$CUDA_HOME/bin:$PATH" \
CUDAToolkit_ROOT="$CUDA_HOME" \
CMAKE_ARGS="-DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120" \
    uv pip install --no-binary lightgbm "lightgbm==4.6.0"

uv run python - <<'PY'
import lightgbm as lgb
import numpy as np
X = np.random.rand(1000, 20).astype("float32")
y = (np.random.rand(1000) > 0.5).astype(int)
m = lgb.LGBMClassifier(device="cuda", n_estimators=10, verbosity=-1)
m.fit(X, y)
print(f"OK: lightgbm {lgb.__version__} CUDA fit succeeded")
PY
