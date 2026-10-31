# SpecDec

Scaffold for comparing Gemma 4 31B generation with speculative decoding off and
with selected drafter methods. The benchmark matrix and workload are
intentionally left configurable until they are specified.

For exact reproduction on another two-A100 server, start with
[`docs/reproduce.md`](docs/reproduce.md). It documents the host, pinned PyTorch
and vLLM revisions, local CUDA builds, benchmark setup, and required model
downloads. A vLLM-only checklist is also included for a host with an already
compatible vLLM installation. Do not upload model checkpoints to this repo.

The host already has the target and candidate checkpoints under
`/home/gbrun/models`. `configs/plan.example.json` points to the target and shows
one disabled speculative scenario as a template. Duplicate and edit that
scenario for the methods and settings in the final experiment plan.

## Acceptance instrumentation

The installed vLLM build can return `metrics.speculative_decoding` per completed
request. The server must be started with
`--per-request-spec-decode-metrics summary` or `detailed` for speculative
scenarios. `detailed` additionally includes accepted and drafted counts for
each verify step. The runner records these fields:

- `num_accepted_draft_tokens / num_spec_steps`: average accepted draft tokens
  per verification step (bonus token excluded).
- `1 + num_accepted_draft_tokens / num_spec_steps`: mean acceptance length,
  including the target model's always-accepted bonus token.
- `num_accepted_draft_tokens / num_draft_tokens`: draft-token acceptance rate.
- `acceptance_histogram`: verify-step counts indexed by number of accepted
  draft tokens.

For aggregate cross-checks, the runner snapshots `/metrics` before and after
the measured requests and records deltas for `vllm:spec_decode_*` counters.
Run against an otherwise idle vLLM server so those deltas belong to this run.
Baseline runs naturally have no speculative-acceptance metrics.

## Launch a configured scenario

Run from the `vLLM_latest` environment:

```bash
source /home/gbrun/miniforge3/etc/profile.d/conda.sh
conda activate vLLM_latest
cd /home/gbrun/SpecDec
python -m specdec.serve --plan configs/plan.example.json --scenario baseline
```

For speculative scenarios, set `enabled` to `true`, provide the method's
`speculative_config`, and set `per_request_spec_decode_metrics` to `summary` or
`detailed`. This launcher starts one vLLM server in the foreground.

## Run a workload

```bash
python -m specdec.benchmark \
  --dataset datasets/workload.jsonl \
  --model gemma-4-31B-it \
  --method baseline \
  --output-root results \
  --concurrency 1 \
  --warmup-requests 2 \
  --max-tokens 256
```

Use identical dataset and request settings for each method in a comparison.
Results are written to a unique run directory containing `run.json`,
`requests.jsonl`, and `summary.json`. Generated text is not saved by default.
Each request record includes status, latency, token usage, and any acceptance
metrics returned by vLLM.

## Compare completed runs

```bash
python -m specdec.summarize results/*/summary.json --output results/comparison.csv
```

The scripts do not select datasets, tune speculative-token counts, run
benchmarks, or infer a preferred method. Those choices can be filled in once
the experiment plan is settled.
