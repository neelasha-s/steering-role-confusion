#!/usr/bin/env bash
set -euo pipefail

# Run this from the repository root inside a CUDA RunPod pod.
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "No NVIDIA driver detected. Use a CUDA-enabled RunPod pod." >&2
  exit 1
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip

# Install the CUDA 12.8 PyTorch build used by the original H200 run.
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0
.venv/bin/pip install -r requirements.txt

.venv/bin/python -m ipykernel install --user \
  --name steering-role-confusion \
  --display-name "Python 3.11 (steering-role-confusion)"

echo "Run with: .venv/bin/jupyter lab --ip=0.0.0.0 --allow-root"
