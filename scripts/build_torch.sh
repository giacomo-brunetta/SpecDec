#!/usr/bin/env bash
set -euo pipefail

: "${CONDA_PREFIX:?Activate the vLLM_latest environment first}"
ROOT="${SPECDEC_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
SRC="${PYTORCH_SRC:-$HOME/src/pytorch}"
JOBS="${MAX_JOBS:-32}"
COMMIT="c0d2d8b86be633695307778a013631d3f1d30d20"

if [[ ! -d "$SRC/.git" ]]; then
  mkdir -p "$(dirname "$SRC")"
  git clone https://github.com/pytorch/pytorch.git "$SRC"
fi
cd "$SRC"
git fetch origin "$COMMIT"
git checkout --detach "$COMMIT"
git submodule sync --recursive
git submodule update --init --recursive

export CUDA_HOME="$CONDA_PREFIX"
export CMAKE_PREFIX_PATH="$CONDA_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export TORCH_CUDA_ARCH_LIST="8.0"
export USE_CUDA=1
export USE_CUDNN=1
export BUILD_TEST=0
export MAX_JOBS="$JOBS"
# Prevent a system CUDA 12.4 path from overriding the conda CUDA 12.6 runtime.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python -m pip install -r requirements.txt
mkdir -p "$ROOT/dist/torch"
python -m pip wheel . --no-build-isolation --no-deps --wheel-dir "$ROOT/dist/torch" -v
python -m pip install --no-deps "$ROOT"/dist/torch/torch-*.whl
