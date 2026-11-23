#!/usr/bin/env bash
# Create a private Python overlay for replay on the pinned vLLM pipeline runner.
set -euo pipefail

if [[ $# != 1 ]]; then
    echo "Usage: $0 NEW_OVERLAY_DIRECTORY" >&2
    exit 2
fi

project=$(cd "$(dirname "$0")/.." && pwd)
package=$(python -c 'from importlib.util import find_spec; from pathlib import Path; print(Path(find_spec("vllm").origin).parent)')
overlay=$1
mkdir -- "$overlay"
cp -as -- "$package" "$overlay/vllm"

# Replace the symlinks with private copies before applying the patch.
for module in v1/worker/gpu_model_runner.py config/vllm.py; do
    cp --remove-destination -- "$package/$module" "$overlay/vllm/$module"
done
patch --batch --forward --fuzz=0 --directory "$overlay" -p1 < "$project/patches/vllm-pp-replay.patch"
printf 'Set PYTHONPATH to include %s before launching vLLM.\n' "$(cd "$overlay" && pwd)"
