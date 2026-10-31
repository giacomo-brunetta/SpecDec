# Reproduce the Gemma 4 vLLM runs

This records the environment used on 2026-10-30, plus the commands to rebuild
PyTorch and vLLM from the exact source snapshots. These are source builds, not
PyPI vLLM installs. The produced local wheels are installed after compilation.

## Hardware and software used

- Ubuntu 22.04, Linux `aarch64` (ARM64), Python 3.12
- 2 × NVIDIA A100-PCIE-40GB, compute capability 8.0; PCIe interconnect, no NVLink
- NVIDIA driver 550.67; CUDA toolkit/runtime 12.6.2 (nvcc 12.6.77)
- GCC/G++ 13.4; CMake 4.4.4; Ninja
- PyTorch source: `pytorch/pytorch` commit `c0d2d8b86be633695307778a013631d3f1d30d20`
  (`2.16.0a0+gitc0d2d8b`)
- vLLM source: `vllm-project/vllm` commit `e811e502efb2b77c7d664b20b68e8dc27e726067`
  (`0.1.dev1+ge811e502e.d20261030.cu126`)
- vLLM was built against the locally compiled torch. The small patch in
  `patches/vllm-unpin-torch.patch` removes upstream's exact torch 2.14 build and
  runtime pins so this torch snapshot remains installed.

The local test environment had CUDA 12.4 in the system library path as well as
CUDA 12.6 in Conda. Loading torch without correcting library lookup can fail
with an undefined `cudaGetDriverEntryPointByVersion` symbol. The build scripts
prepend `$CONDA_PREFIX/lib` to `LD_LIBRARY_PATH`; use the same setting when
running vLLM. The installed vLLM build also required
`VLLM_USE_FLASHINFER_SAMPLER=0` because its FlashInfer sampler JIT passed an
`nvcc` option unsupported by the installed CUDA 12.6 compiler. Do not set this
unless the same compiler error occurs on the new server.

## Create the build environment

Install Miniforge/Conda and system build utilities (compiler toolchain, git,
CMake/Ninja if not provided through Conda). Then:

```bash
conda env create -f environment.yml
conda activate vLLM_latest
```

On this ARM64 system, Conda supplied GCC/G++ 13 and the CUDA 12.6.2 toolkit.
`environment.yml` captures the key toolchain dependencies; Conda may resolve
newer transitive package builds. For byte-for-byte package reproduction, use
this exact Linux ARM64 platform and pin the package lock separately.

## Build PyTorch from source

```bash
conda activate vLLM_latest
./scripts/build_torch.sh
```

The script fetches the pinned PyTorch commit and submodules, targets only A100
(SM 8.0), builds a wheel with CUDA enabled and tests disabled, and installs the
locally produced wheel. Override `MAX_JOBS` if needed, for example
`MAX_JOBS=16 ./scripts/build_torch.sh`.

## Build vLLM against that PyTorch

```bash
conda activate vLLM_latest
./scripts/build_vllm.sh
```

This checks out the pinned vLLM commit, applies the torch-unpin patch, installs
its CUDA build/runtime dependencies, builds its wheel locally, and installs
that wheel. It does not fetch or install a prebuilt vLLM wheel. The vLLM source
checkout should be clean before running; the script checks out the recorded
revision and applies the project patch.

To launch vLLM in this environment:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLLM_USE_FLASHINFER_SAMPLER=0
```

Checkpoints are separate from this repository. Hugging Face access to Google
Gemma repositories requires an authorized account and acceptance of the model
terms. Authenticate on the new server with `hf auth login`, then download the
needed repositories to a local model directory:

```bash
hf download google/gemma-4-31B-it --local-dir "$HOME/models/gemma-4-31B-it"
hf download google/gemma-4-E2B-it --local-dir "$HOME/models/gemma-4-E2B-it"
hf download google/gemma-4-E4B-it --local-dir "$HOME/models/gemma-4-E4B-it"
hf download google/gemma-4-12B-it --local-dir "$HOME/models/gemma-4-12B-it"
hf download google/gemma-4-31B-it-assistant --local-dir "$HOME/models/gemma-4-31B-it-assistant"
hf download RedHatAI/gemma-4-31B-it-speculator.eagle3 --local-dir "$HOME/models/gemma-4-31B-it-speculator.eagle3"
hf download RedHatAI/gemma-4-31B-it-speculator.dspark --local-dir "$HOME/models/gemma-4-31B-it-speculator.dspark"
```

These are the requested speculation checkpoints. For the earlier single-GPU
quantized comparison, the host also had
`RedHatAI/gemma-4-31B-it-NVFP4` (local folder name
`Gemma-4-31B-IT-NVFP4`). Download it separately if repeating that measurement.
The repository stores only datasets, scripts, configs, and result metadata;
never add model weight files or Hugging Face credentials.

## vLLM-only minimum checklist

If the destination already has vLLM installed, PyTorch/vLLM source builds can be
skipped if all of these are true:

1. It is a CUDA-enabled vLLM build that supports Gemma 4 and the desired
   speculative decoding method/configuration. For acceptance measurements,
   `vllm serve --help` must include
   `--per-request-spec-decode-metrics` (or the installed version must expose
   equivalent per-request acceptance metrics).
2. The installed build supports the target/drafter model architectures and its
   CUDA kernels support the installed GPU, driver, and CUDA runtime combination.
3. The target and selected drafter checkpoints have been downloaded and are
   readable. Use identical model revisions and tokenizer files across runs.
4. For the full BF16 pipeline-parallel baseline recorded here: 2 × A100 40 GB
   are the tested configuration. A single A100 40 GB cannot hold the 31B BF16
   target. A quantized checkpoint and supported quantization kernel are needed
   for a single-card run.
5. The vLLM server is otherwise idle during a measurement. Do not share one of
   the required GPUs with another user's process. The tested pair is PCIe-only;
   no NVLink is assumed.

The code in `specdec/` uses only the Python standard library. It sends requests
to a running OpenAI-compatible vLLM endpoint, saves per-request results, and
records speculative decoding metrics if the server returns them. Use the same workload,
request count, prompt/output limits, and server settings for each compared
method.

## Launch and run the documented baseline

`configs/plan.example.json` is a starter plan configured for the tested PP=2
setup (`tensor_parallel_size=1`, `pipeline_parallel_size=2`, eager execution);
edit `server.model` to the checkpoint path and configure one scenario at a
time. With the environment
active and GPU pair idle:

```bash
cd SpecDec
python -m specdec.serve --plan configs/plan.example.json --scenario baseline
```

In another terminal, activate the same environment, set the library variables
above, then run the harness. Example: the recorded BF16 PP=2 eager runs used a
344-token repeated atmospheric-scattering prompt, 350 output-token cap, one
warmup request, and 4/6/8 concurrent measured requests. Existing datasets can
be replayed as follows:

```bash
python -m specdec.benchmark \
  --dataset datasets/prompt_350.jsonl \
  --model gemma-4-31B-it \
  --method pp2-bf16-eager-prompt350-c6 \
  --output-root results \
  --concurrency 6 \
  --warmup-requests 1 \
  --repeat 12 \
  --max-tokens 350
```

See `python -m specdec.benchmark --help` for all request and sampling options.
The run directory records config and per-request output. Compare summaries:

```bash
python -m specdec.summarize results/*/summary.json --output results/comparison.csv
```

## Results available

The reproducible two-GPU BF16 eager prompt workload completed successfully at
concurrency 4, 6, and 8. Observed aggregate generation throughput was 35.3,
55.5, and 67.0 tokens/s, respectively. At concurrency 8, the KV cache was
nearly saturated and vLLM deferred some requests. The detailed JSON is in the
corresponding `results/20261030-1852*`, `results/20261030-1854*`, and
`results/20261030-1855*` directories. These are baseline throughput results;
no drafter acceptance rates have been measured yet. Earlier FP8 and NVFP4
pipeline-parallel attempts loaded weights but failed on the first inference
with a CUDA illegal memory access, so they are not valid performance results.
