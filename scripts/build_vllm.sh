#!/usr/bin/env bash
set -euo pipefail

: "${CONDA_PREFIX:?Activate the vLLM_latest environment first}"
ROOT="${SPECDEC_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
SRC="${VLLM_SRC:-$HOME/src/vllm}"
JOBS="${MAX_JOBS:-32}"
COMMIT="e811e502efb2b77c7d664b20b68e8dc27e726067"

if [[ ! -d "$SRC/.git" ]]; then
  mkdir -p "$(dirname "$SRC")"
  git clone https://github.com/vllm-project/vllm.git "$SRC"
fi
cd "$SRC"
git fetch origin "$COMMIT"
git checkout --detach "$COMMIT"
if git apply --reverse --check "$ROOT/patches/vllm-unpin-torch.patch" >/dev/null 2>&1; then
  echo "vLLM torch-unpin patch is already applied"
else
  git apply "$ROOT/patches/vllm-unpin-torch.patch"
fi

export CUDA_HOME="$CONDA_PREFIX"
export CMAKE_PREFIX_PATH="$CONDA_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS="$JOBS"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Install the pinned CUDA runtime/build requirements, keeping the locally
# compiled torch in place (the patch removes vLLM's torch==2.14 pin).
python -m pip install -r requirements/build/cuda.txt
python -m pip install -r requirements/cuda.txt
mkdir -p "$ROOT/dist/vllm"
python -m pip wheel . --no-build-isolation --no-deps --wheel-dir "$ROOT/dist/vllm" -v
python -m pip install --no-deps "$ROOT"/dist/vllm/vllm-*.whl
